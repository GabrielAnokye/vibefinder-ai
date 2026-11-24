"""
The assistant pipeline: validate -> interpret -> retrieve -> plan -> filter -> rank (existing
engine) -> select -> validate results -> explain -> validate explanation (fallback) -> respond.

The existing engine (recommend_songs / recommend_diverse_songs / score_song) is the only
source of scores, reasons, and ranking order.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.assistant.catalog import vocabulary
from src.assistant.interpreter import Interpreter
from src.assistant.models import (
    EXPLICIT, INFERRED, PREFERENCE_FIELDS, AssistantResponse, Plan, Pref, Recommendation,
)
from src.assistant.planner import FIELD_LABELS, build_plan, describe_value, filter_candidates, satisfies
from src.assistant.providers import ProviderError, TemplateProvider
from src.assistant.retrieval import KNOWLEDGE_DIR, Retriever, load_knowledge
from src.assistant.validation import check_recommendations, check_request_text, parse_explanation
from src.recommender import (
    REASON_SEPARATOR, active_max_score, load_songs, max_possible_score, normalize_label,
    recommend_diverse_songs, recommend_songs,
)

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "songs.csv"

# Results below this share of the score available from the active preferences are not shown
# as recommendations. A design choice, not a tuned value.
QUALITY_FLOOR = 0.40


class Assistant:
    def __init__(self, songs: Optional[List[Dict]] = None, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.songs = songs if songs is not None else load_songs(str(DATA_PATH))
        self.songs_by_id = {song["id"]: song for song in self.songs}
        self.interpreter = Interpreter(vocabulary(self.songs), self.songs)
        self.retriever = Retriever(load_knowledge(self.songs, knowledge_dir))

    # ------------------------------------------------------------ main entry point

    def ask(self, text: str, provider=None, retrieval: bool = True, mode: Optional[str] = None,
            diverse: bool = False, provider_note: str = "") -> AssistantResponse:
        provider = provider or TemplateProvider()
        response = AssistantResponse(status="ok", request=text if isinstance(text, str) else "",
                                     retrieval_enabled=retrieval)
        log = response.steps.append

        error = check_request_text(text)
        log({"step": "validate_request", "status": "failed" if error else "ok"})
        if error:
            response.status, response.message = "error", error
            return response

        interp = self.interpreter.interpret(text)
        response.interpretation = interp
        log({"step": "interpret", "contexts": interp.context_names,
             "explicit": [f for f, p in interp.prefs.items() if p.known]})

        if retrieval:
            query = " ".join([text, *interp.context_names, *interp.unavailable_genres,
                              *(interp.prefs["genres"].value or ()), *(interp.prefs["moods"].value or ())])
            response.retrieved = self.retriever.search(query)
        log({"step": "retrieve", "enabled": retrieval,
             "sources": [r.passage.source_id for r in response.retrieved]})

        plan, prefs = build_plan(interp, response.retrieved, self.songs, mode_override=mode,
                                 diverse_override=diverse, cite_catalog=retrieval)
        response.plan, response.preferences = plan, prefs
        log({"step": "plan", "status": plan.status, "mode": plan.mode, "advanced": plan.advanced,
             "diverse": plan.diverse, "hints_used": [h.field for h in plan.hints if h.used]})
        if plan.status != "ok":
            response.status, response.message = plan.status, plan.message
            return response

        candidates = filter_candidates(self.songs, plan)
        log({"step": "filter", "candidates": len(candidates)})
        if not candidates:
            response.status = "no_match"
            response.message = "No catalog songs are left after applying your required genres and exclusions."
            return response

        recs = self._rank_and_select(plan, prefs, candidates, interp)
        log({"step": "rank", "engine": "recommend_diverse_songs" if plan.diverse else "recommend_songs",
             "selected": [r.song["id"] for r in recs]})
        if not recs:
            response.status = "no_match"
            response.message = (f"No song scored at least {QUALITY_FLOOR:.0%} of the points available from "
                                "your preferences, so none is presented as a recommendation.")
            return response
        response.recommendations = recs

        errors = check_recommendations(recs, self.songs_by_id, plan)
        log({"step": "validate_results", "status": "failed" if errors else "ok", "errors": errors})
        if errors:  # should never happen; refuse rather than show unverified results
            response.status, response.recommendations = "error", []
            response.message = "Internal consistency check failed: " + "; ".join(errors)
            return response

        self._explain(response, plan, prefs, interp.context_names, candidates, provider, provider_note)
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
            if active_max and score / active_max < QUALITY_FLOOR:
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
                why.append(f"weaker matches (below {QUALITY_FLOOR:.0%} of the available points) were left out")
            plan.limitations.append(f"Returned {len(selected)} of the {plan.k} songs requested: " + "; ".join(why) + ".")
        for label in sorted({u for rec in selected for u in rec.unmet} - set(always_unmet)):
            if all(label in rec.unmet for rec in selected):
                plan.limitations.append(f"None of the results meets your stated {label}; "
                                        f"the catalog has no closer match among the candidates.")
        return selected

    # ------------------------------------------------------------ explanation with validation and fallback

    def _payload(self, response: AssistantResponse, plan: Plan, prefs: Dict[str, Pref],
                 contexts: Sequence[str], candidates: List[Dict]) -> Dict[str, Any]:
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

    def _explain(self, response, plan, prefs, contexts, candidates, provider, provider_note) -> None:
        payload = self._payload(response, plan, prefs, contexts, candidates)
        template = TemplateProvider()
        status = provider_note
        if not isinstance(provider, TemplateProvider):
            try:
                raw = provider.generate(payload)
                explanation, errors = parse_explanation(raw, response.recommendations, plan.citations, self.songs)
            except ProviderError as error:
                explanation, errors = None, [f"{error.kind}: {error}"]
            response.steps.append({"step": "generate_explanation", "provider": provider.label,
                                   "status": "ok" if explanation else "rejected", "errors": errors})
            if explanation:
                response.explanation, response.generator = explanation, provider.label
                response.generation_status = "LLM output validated (song ids, citations, mentioned songs)"
                return
            status = f"{provider.label} output not used ({'; '.join(errors)}); fell back to the template"

        explanation, errors = parse_explanation(template.generate(payload), response.recommendations,
                                                plan.citations, self.songs)
        response.steps.append({"step": "generate_explanation", "provider": template.label,
                               "status": "ok" if explanation else "rejected", "errors": errors})
        response.explanation, response.generator = explanation, template.label
        response.generation_status = status or ("validated" if explanation else
                                                "template explanation failed validation: " + "; ".join(errors))
