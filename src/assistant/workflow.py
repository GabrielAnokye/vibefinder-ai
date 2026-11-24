"""
The assistant pipeline, a bounded agentic workflow:

  validate request -> interpret -> retrieve -> plan -> filter catalog -> check coverage
  -> rank (existing engine) -> validate results -> generate explanation -> validate explanation
  -> [one content correction] -> [template fallback] -> return

Decisions taken from intermediate results:
- invalid request                 -> error response (stop)
- conflicting / nothing usable    -> clarification (stop)
- hard constraint unsatisfiable   -> no-match (stop; nothing is substituted)
- generated explanation invalid   -> one correction request with the specific errors, if the
                                     external attempt budget allows; otherwise template fallback
- correction invalid              -> discard generated text, use the template
- provider/transport failure      -> template fallback (auth/config errors are never retried)

The existing engine (recommend_songs / recommend_diverse_songs / score_song) is the only
source of scores, reasons, and ranking order. Every stage is recorded in a Trace.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.assistant.catalog import vocabulary
from src.assistant.interpreter import Interpreter
from src.assistant.models import (
    EXPLICIT, INFERRED, PREFERENCE_FIELDS, AssistantResponse, Explanation, Passage, Plan, Pref,
    Recommendation,
)
from src.assistant.planner import FIELD_LABELS, build_plan, describe_value, filter_candidates, satisfies
from src.assistant.providers import (
    DEFAULT_PROMPT_STYLE, MAX_HTTP_ATTEMPTS, AttemptBudget, GenerationRequest, ProviderError,
    TemplateProvider,
)
from src.assistant.retrieval import KNOWLEDGE_DIR, Retriever, load_knowledge
from src.assistant.trace import Trace
from src.assistant.validation import check_recommendations, check_request_text, parse_explanation
from src.recommender import (
    REASON_SEPARATOR, active_max_score, load_songs, max_possible_score, normalize_label,
    recommend_diverse_songs, recommend_songs,
)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "songs.csv"

# Results below this share of the score available from the active preferences are not shown
# as recommendations. A heuristic design choice, not a learned or calibrated value
# (see evaluation/results/quality_floor.md).
QUALITY_FLOOR = 0.40


class Assistant:
    def __init__(self, songs: Optional[List[Dict]] = None, knowledge_dir: Path = KNOWLEDGE_DIR,
                 passages: Optional[Sequence[Passage]] = None, quality_floor: float = QUALITY_FLOOR):
        """`passages` replaces the default knowledge index (used for single- vs multi-source experiments)."""
        self.songs = songs if songs is not None else load_songs(str(DATA_PATH))
        self.songs_by_id = {song["id"]: song for song in self.songs}
        self.interpreter = Interpreter(vocabulary(self.songs), self.songs)
        self.retriever = Retriever(passages if passages is not None else load_knowledge(self.songs, knowledge_dir))
        self.quality_floor = quality_floor

    # ------------------------------------------------------------ main entry point

    def ask(self, text: str, provider=None, retrieval: bool = True, mode: Optional[str] = None,
            diverse: bool = False, provider_note: str = "", prompt_style: str = DEFAULT_PROMPT_STYLE,
            max_http_attempts: int = MAX_HTTP_ATTEMPTS) -> AssistantResponse:
        provider = provider or TemplateProvider()
        is_template = isinstance(provider, TemplateProvider)
        response = AssistantResponse(status="ok", request=text if isinstance(text, str) else "",
                                     retrieval_enabled=retrieval)
        trace = response.trace = Trace(
            request=response.request, retrieval=retrieval, provider=provider.label,
            provider_kind="template" if is_template else ("simulated" if provider.name == "mock" else "live"),
            prompt_style=None if is_template else prompt_style, max_http_attempts=max_http_attempts,
            quality_floor=self.quality_floor,
        )

        # 1. Validate the request.
        error = check_request_text(text)
        trace.add("validate_request", "check length and content",
                  input={"characters": len(text) if isinstance(text, str) else None},
                  validation="rejected" if error else "accepted", decision="stop: error response" if error else None)
        if error:
            return self._finish(response, "error", error)

        # 2. Interpret.
        interp = self.interpreter.interpret(text)
        response.interpretation = interp
        trace.add("interpret", "rule-based interpretation", result={
            "contexts": interp.context_names,
            "explicit": {f: describe_value(f, p.value) for f, p in interp.prefs.items() if p.known},
            "genre_requirement": interp.genre_requirement, "unavailable_genres": interp.unavailable_genres,
            "excluded_genres": interp.excluded_genres, "excluded_artists": interp.excluded_artists,
            "unique_artists": interp.unique_artists, "count": interp.count.value,
            "ambiguities": interp.ambiguities})

        # 3. Retrieve.
        query = ""
        if retrieval:
            query = " ".join([text, *interp.context_names, *interp.unavailable_genres,
                              *(interp.prefs["genres"].value or ()), *(interp.prefs["moods"].value or ())])
            response.retrieved = self.retriever.search(query)
        trace.add("retrieve", "TF-IDF search over guides and catalog facts" if retrieval else "retrieval disabled",
                  input={"enabled": retrieval, "index_size": len(self.retriever.passages)},
                  result={"scores": {r.passage.source_id: r.score for r in response.retrieved}},
                  sources=[r.passage.source_id for r in response.retrieved])

        # 4. Plan.
        plan, prefs = build_plan(interp, response.retrieved, self.songs, mode_override=mode,
                                 diverse_override=diverse,
                                 citable_ids=set(self.retriever.by_id) if retrieval else set())
        response.plan, response.preferences = plan, prefs
        trace.add("plan", "build recommendation plan", result={
            "status": plan.status, "mode": plan.mode, "mode_reason": plan.mode_reason,
            "advanced": plan.advanced, "diverse": plan.diverse, "k": plan.k,
            "hints_used": {h.field: h.source_id for h in plan.hints if h.used},
            "hints_not_used": {h.field: h.note for h in plan.hints if not h.used},
            "unknown": plan.unknown_fields,
            "engine_prefs": {k: repr(v) if not isinstance(v, (int, float, str, bool, list)) else v
                             for k, v in plan.engine_prefs.items()}},
            sources=sorted({h.source_id for h in plan.hints if h.used}),
            decision="stop: ask for clarification" if plan.status == "clarification" else None)
        if plan.status == "clarification":
            return self._finish(response, "clarification", plan.message)
        if plan.status == "no_match":
            trace.add("check_coverage", "check hard constraints against the catalog",
                      result={"limitations": plan.limitations}, sources=list(plan.citations),
                      validation="hard constraint cannot be satisfied",
                      decision="stop: no-match response, no substitution")
            return self._finish(response, "no_match", plan.message)

        # 5. Filter the catalog (hard genres and exclusions) and check coverage.
        candidates = filter_candidates(self.songs, plan)
        trace.add("filter_catalog", "apply hard genres and exclusions", input={
            "hard_genres": plan.hard_genres, "excluded_genres": plan.excluded_genres,
            "excluded_artists": plan.excluded_artists}, result={"candidates": len(candidates)})
        if not candidates:
            trace.add("check_coverage", "check hard constraints against the catalog",
                      validation="no candidates left", decision="stop: no-match response")
            return self._finish(response, "no_match",
                                "No catalog songs are left after applying your required genres and exclusions.")

        # 6. Rank with the existing engine and select.
        recs = self._rank_and_select(plan, prefs, candidates, interp)
        engine = "recommend_diverse_songs" if plan.diverse else "recommend_songs"
        trace.add("rank", f"call {engine}", input={"mode": plan.mode, "advanced": plan.advanced,
                                                   "candidates": len(candidates), "k": plan.k},
                  result={"selected": [r.song["id"] for r in recs],
                          "scores": [round(r.score, 4) for r in recs],
                          "active_max": round(recs[0].active_max, 4) if recs else None})
        trace.add("check_coverage", "compare stated preferences with the catalog and the selected results",
                  result={"limitations": list(plan.limitations),
                          "unmet_by_song": {r.song["id"]: r.unmet for r in recs if r.unmet}},
                  sources=[c for c in plan.citations if c.startswith("catalog#")],
                  validation="all stated preferences covered" if not plan.limitations else "limitations disclosed")
        if not recs:
            return self._finish(response, "no_match",
                                f"No song scored at least {self.quality_floor:.0%} of the points available from "
                                "your preferences, so none is presented as a recommendation.")
        response.recommendations = recs

        # 7. Validate results against the catalog and engine.
        errors = check_recommendations(recs, self.songs_by_id, plan)
        trace.add("validate_results", "re-score every result and check constraints",
                  validation="accepted" if not errors else "rejected: " + "; ".join(errors),
                  decision="stop: refuse unverified results" if errors else None)
        if errors:  # should never happen; refuse rather than show unverified results
            response.recommendations = []
            return self._finish(response, "error", "Internal consistency check failed: " + "; ".join(errors))

        # 8-10. Explain, validate, correct once, or fall back.
        payload = response.payload = self.build_payload(response, plan, prefs, interp.context_names, candidates)
        explanation, generator, status, attempts = self.explain(
            payload, provider, prompt_style, trace, provider_note, max_http_attempts)
        response.explanation, response.generator = explanation, generator
        response.generation_status, response.http_attempts = status, attempts
        return self._finish(response, "ok", None)

    def _finish(self, response: AssistantResponse, status: str, message: Optional[str]) -> AssistantResponse:
        response.status, response.message = status, message
        response.trace.add("return", "return response", result={
            "status": status, "song_ids": [r.song["id"] for r in response.recommendations],
            "generator": response.generator or None, "http_attempts": response.http_attempts,
            "message": message})
        return response

    # ------------------------------------------------------------ ranking with the existing engine

    def _rank_and_select(self, plan: Plan, prefs: Dict[str, Pref], candidates: List[Dict],
                         interp) -> List[Recommendation]:
        prefs_dict, everything = plan.engine_prefs, len(candidates)
        if plan.diverse:
            ranked = [(p.song, p.base_score, p.explanation, p)
                      for p in recommend_diverse_songs(prefs_dict, candidates, k=everything,
                                                       mode=plan.mode, advanced=plan.advanced)]
        else:
            ranked = [(song, score, explanation, None)
                      for song, score, explanation in recommend_songs(prefs_dict, candidates, k=everything,
                                                                      mode=plan.mode, advanced=plan.advanced)]
        active_max = active_max_score(prefs_dict, plan.mode, plan.advanced)
        stated = {f: prefs[f].value for f in PREFERENCE_FIELDS if prefs[f].provenance == EXPLICIT}
        # A requested genre that is not in the catalog is missed by every result (substitutes only).
        always_unmet = ([f"genre ({', '.join(interp.unavailable_genres)})"]
                        if interp.unavailable_genres and "genres" not in stated else [])

        selected: List[Recommendation] = []
        seen_artists = set()
        skipped_repeat = skipped_floor = 0
        for song, score, explanation, pick in ranked:
            if len(selected) == plan.k:
                break
            artist = normalize_label(song["artist"])
            if plan.unique_artists and artist in seen_artists:
                skipped_repeat += 1
                continue
            if active_max and score / active_max < self.quality_floor:
                skipped_floor += 1
                continue
            seen_artists.add(artist)
            selected.append(Recommendation(
                rank=len(selected) + 1, song=song, score=score,
                reasons=explanation.split(REASON_SEPARATOR), active_max=active_max,
                unmet=always_unmet + [FIELD_LABELS[f] for f, v in stated.items() if not satisfies(song, f, v)],
                adjusted_score=pick.adjusted_score if pick else None,
                diversity_note=pick.penalty_explanation() if pick else None, pick=pick,
            ))

        if 0 < len(selected) < plan.k:
            why = []
            if len(candidates) < plan.k:
                why.append(f"only {len(candidates)} catalog songs pass your filters")
            if skipped_repeat:
                why.append("repeated artists were left out as you asked")
            if skipped_floor:
                why.append(f"weaker matches (below {self.quality_floor:.0%} of the available points) were left out")
            plan.limitations.append(f"Returned {len(selected)} of the {plan.k} songs requested: " + "; ".join(why) + ".")
        for label in sorted({u for rec in selected for u in rec.unmet} - set(always_unmet)):
            if all(label in rec.unmet for rec in selected):
                plan.limitations.append(f"None of the results meets your stated {label}; "
                                        f"the catalog has no closer match among the candidates.")
        return selected

    # ------------------------------------------------------------ grounding payload

    def build_payload(self, response: AssistantResponse, plan: Plan, prefs: Dict[str, Pref],
                      contexts: Sequence[str], candidates: List[Dict]) -> Dict[str, Any]:
        """Everything a provider may use: request, plan, engine results, retrieved passages, limitations."""
        sources = [self.retriever.by_id[i] for i in plan.citations if i in self.retriever.by_id]
        inferred = {FIELD_LABELS[f]: p.source for f, p in prefs.items() if p.provenance == INFERRED}
        results = []
        for rec in response.recommendations:
            song = rec.song
            met_inferred = [FIELD_LABELS[f] for f, p in prefs.items()
                            if p.provenance == INFERRED and satisfies(song, f, p.value)]
            results.append({
                "song_id": song["id"], "rank": rec.rank, "title": song["title"], "artist": song["artist"],
                "genre": song["genre"], "mood": song["mood"],
                "features": {k: song.get(k) for k in ("energy", "acousticness", "instrumentalness",
                                                       "vocal_intensity", "popularity", "release_decade",
                                                       "mood_tags")},
                "score": rec.score, "active_max": rec.active_max,
                "theoretical_max": max_possible_score(plan.mode, plan.advanced),
                "reasons": rec.reasons, "unmet": rec.unmet, "inferred_used": met_inferred,
                "match_label": rec.match_label,
                "adjusted_score": rec.adjusted_score, "diversity_note": rec.diversity_note,
            })
        return {
            "request": response.request,
            "contexts": list(contexts),
            "stated": {FIELD_LABELS[f]: describe_value(f, p.value) for f, p in prefs.items()
                       if p.provenance == EXPLICIT},
            "inferred_sources": inferred,
            "unknown": plan.unknown_fields,
            "plan": {"mode": plan.mode, "mode_reason": plan.mode_reason, "advanced": plan.advanced,
                     "diverse": plan.diverse, "unique_artists": plan.unique_artists, "k": plan.k,
                     "hard_genres": plan.hard_genres, "excluded_genres": plan.excluded_genres,
                     "excluded_artists": plan.excluded_artists, "candidates": len(candidates)},
            "sources": [{"id": p.source_id, "title": p.title, "text": p.text} for p in sources],
            "results": results,
            "limitations": list(plan.limitations),
        }

    # ------------------------------------------------------------ explanation with validation, correction, fallback

    def explain(self, payload: Dict[str, Any], provider, prompt_style: str = DEFAULT_PROMPT_STYLE,
                trace: Optional[Trace] = None, provider_note: str = "",
                max_http_attempts: int = MAX_HTTP_ATTEMPTS) -> Tuple[Optional[Explanation], str, str, int]:
        """Return (explanation, generator label, status line, external HTTP attempts used)."""
        trace = trace or Trace()
        budget = AttemptBudget(max_http_attempts)
        sources = [s["id"] for s in payload["sources"]]
        status = provider_note

        if not isinstance(provider, TemplateProvider):
            history: List[str] = []
            request = GenerationRequest(payload, prompt_style)
            raw, failure = self._call_provider(provider, request, budget, trace, "generate_explanation")
            if failure is not None:
                history.append(failure)
            else:
                explanation, errors = parse_explanation(raw, payload, self.songs)
                can_correct = budget.remaining > 0
                trace.add("validate_explanation", "validate generated JSON against the payload",
                          sources=sources, validation="accepted" if explanation else "rejected: " + "; ".join(errors),
                          decision=None if explanation else ("request one correction" if can_correct
                                                             else "fall back to template (no attempts left)"))
                if explanation:
                    return explanation, provider.label, "LLM output validated on the first attempt", budget.used
                history.append("first answer rejected: " + "; ".join(errors))
                if not can_correct:
                    history.append("no external attempts left for a correction")
                else:
                    correction = GenerationRequest(payload, prompt_style, previous_output=raw, errors=errors)
                    raw2, failure = self._call_provider(provider, correction, budget, trace, "correct_explanation",
                                                        errors=errors)
                    if failure is not None:
                        history.append("correction request failed: " + failure)
                    else:
                        explanation, errors2 = parse_explanation(raw2, payload, self.songs)
                        trace.add("validate_explanation", "validate corrected JSON against the payload",
                                  sources=sources,
                                  validation="accepted" if explanation else "rejected: " + "; ".join(errors2),
                                  decision=None if explanation else "discard generated text; fall back to template")
                        if explanation:
                            return (explanation, provider.label,
                                    "LLM output accepted after one correction (" + history[0] + ")", budget.used)
                        history.append("correction rejected: " + "; ".join(errors2))
            status = f"{provider.label} output not used ({'; then '.join(history)}); fell back to the template"

        template = TemplateProvider()
        explanation, errors = parse_explanation(template.generate(GenerationRequest(payload)), payload, self.songs)
        trace.add("generate_explanation", "template explanation (deterministic, not an LLM)",
                  sources=sources, validation="accepted" if explanation else "rejected: " + "; ".join(errors),
                  decision=status or None)
        return (explanation, template.label,
                status or ("validated" if explanation else "template explanation failed validation: " + "; ".join(errors)),
                budget.used)

    @staticmethod
    def _call_provider(provider, request: GenerationRequest, budget: AttemptBudget, trace: Trace, stage: str,
                       errors: Sequence[str] = ()) -> Tuple[Optional[str], Optional[str]]:
        """Call the provider once (it may use one transport retry). Return (raw text, failure reason)."""
        before = budget.used
        try:
            raw = provider.generate(request, budget)
        except ProviderError as error:
            trace.add(stage, f"call {provider.label}",
                      input={"prompt_style": request.style, "correction": request.is_correction,
                             "errors_sent": list(errors)},
                      result={"http_attempts": budget.used - before, "error_kind": error.kind},
                      validation=f"provider failure: {error.kind}",
                      decision="fall back to template" + ("" if error.retryable else " (not retried: permanent error)"))
            return None, f"{error.kind}: {error}"
        trace.add(stage, f"call {provider.label}",
                  input={"prompt_style": request.style, "correction": request.is_correction,
                         "errors_sent": list(errors)},
                  result={"http_attempts": budget.used - before, "characters": len(raw)})
        return raw, None
