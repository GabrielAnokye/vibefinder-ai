"""Plain-text rendering of an AssistantResponse for the CLI. Formatting only: never changes results."""

from dataclasses import replace
from typing import List

from src.assistant.models import EXPLICIT, INFERRED, PREFERENCE_FIELDS, AssistantResponse
from src.assistant.planner import FIELD_LABELS, describe_value
from src.recommender import REASON_SEPARATOR, max_possible_score
from src.table import diverse_table, recommendations_table

# Engine reasons appear in this fixed order (basic four, then the five advanced ones).
REASON_FIELDS = ("genres", "moods", "energy", "acoustic", "popularity", "decade",
                 "instrumentalness", "vocal_intensity", "mood_tags")


def is_unknown_reason(reason: str) -> bool:
    """The engine's zero-point line for a preference the listener never gave (not a mismatch)."""
    return reason.startswith("No ") and reason.endswith(" preference given (+0.00)")


def display_reasons(response: AssistantResponse, reasons: List[str]) -> List[str]:
    """Hide unknown-preference lines; mark reasons whose target was inferred, not stated."""
    shown = []
    for field, reason in zip(REASON_FIELDS, reasons):
        if is_unknown_reason(reason):
            continue
        pref = response.preferences.get(field)
        shown.append(f"{reason} [inferred]" if pref is not None and pref.provenance == INFERRED else reason)
    return shown


def _provenance(pref) -> str:
    return f"you said {pref.source}" if pref.provenance == EXPLICIT else f"inferred from {pref.source}"


def _interpretation_lines(response: AssistantResponse) -> List[str]:
    interp, plan = response.interpretation, response.plan
    lines = ["Interpreted request:"]
    if interp.contexts:
        lines.append("  Listening context: " + ", ".join(f'{c} ("{p}")' for c, p in interp.contexts))
    prefs = response.preferences or interp.prefs   # final values after retrieved hints
    for field in PREFERENCE_FIELDS:
        pref = prefs[field]
        if pref.known:
            lines.append(f"  {FIELD_LABELS[field].capitalize()}: {describe_value(field, pref.value)} ({_provenance(pref)})")
    if interp.genre_requirement == "hard":
        lines.append("  Genre requirement: only the named genre(s)")
    if interp.unavailable_genres:
        lines.append(f"  Not in catalog: {', '.join(interp.unavailable_genres)}")
    if interp.excluded_genres or interp.excluded_artists:
        lines.append(f"  Excluded: {', '.join(interp.excluded_genres + interp.excluded_artists)}")
    if interp.unique_artists:
        lines.append("  No repeated artists: required")
    if interp.diversity.known:
        lines.append(f"  Variety: yes ({_provenance(interp.diversity)})")
    count = interp.count
    lines.append(f"  Songs requested: {count.value}" + (f" (you said {count.source})" if count.known else " (default)"))
    if plan is not None and plan.unknown_fields:
        lines.append(f"  Not specified (no points either way): {', '.join(plan.unknown_fields)}")
    for note in interp.ambiguities:
        lines.append(f"  Note: {note}")
    return lines


def render(response: AssistantResponse, show_plan: bool = False, table: bool = False) -> str:
    lines = ["VibeFinder AI assistant", f'Request: "{response.request}"', ""]
    if response.status == "error":
        lines.append(f"Error: {response.message}")
        return "\n".join(lines)
    lines.extend(_interpretation_lines(response))
    plan = response.plan

    if response.retrieval_enabled:
        sources = ", ".join(f"{r.passage.source_id} ({r.score:.2f})" for r in response.retrieved)
        lines.append(f"Retrieved sources: {sources or 'none relevant'}")
    else:
        lines.append("Retrieved sources: retrieval disabled (--no-retrieval)")

    if response.status in ("clarification", "no_match"):
        heading = "I need a bit more information:" if response.status == "clarification" else "No match:"
        lines += ["", heading, f"  {response.message}"]
        for limitation in (plan.limitations if plan else []):
            lines.append(f"  - {limitation}")
        if show_plan:
            lines += _plan_lines(response)
        return "\n".join(lines)

    lines.append(
        f"Strategy: {plan.mode} ranking ({plan.mode_reason}); advanced features "
        f"{'on' if plan.advanced else 'off'}; diversity reranking {'on' if plan.diverse else 'off'}"
        + (f" ({plan.diverse_reason})" if plan.diverse else "")
    )
    recs = response.recommendations
    active_max = recs[0].active_max
    theoretical = max_possible_score(plan.mode, plan.advanced)
    lines += ["", f"Recommendations ({len(recs)} of {plan.k} requested). Scores are out of {active_max:g} points "
                  f"available from your preferences ({theoretical:g} if every preference were set). "
                  "[inferred] marks a target taken from a knowledge guide, not from you.", ""]

    if table:
        if plan.diverse:
            picks = [replace(r.pick, explanation=REASON_SEPARATOR.join(display_reasons(response, r.reasons)))
                     for r in recs]
            lines.append(diverse_table(picks, active_max))
        else:
            rows = [(r.song, r.score, REASON_SEPARATOR.join(display_reasons(response, r.reasons))) for r in recs]
            lines.append(recommendations_table(rows, active_max))
        for rec in recs:
            if rec.unmet:
                lines.append(f"  #{rec.rank} misses: {', '.join(rec.unmet)}")
    else:
        for rec in recs:
            song = rec.song
            lines.append(f"{rec.rank}. {song['title']} by {song['artist']}  [{song['genre']}, {song['mood']}]")
            label = f"{rec.score:.2f} / {active_max:g} ({rec.share_of_active_max:.0%}, {rec.match_label} match"
            label += f"; misses: {', '.join(rec.unmet)})" if rec.unmet else ")"
            lines.append(f"   Relevance: {label}")
            if rec.adjusted_score is not None:
                lines.append(f"   Diversity (ordering only): {rec.diversity_note}")
            lines.append(f"   Scoring (engine): {'; '.join(display_reasons(response, rec.reasons))}")
            if show_plan:
                lines.append(f"   All engine reasons: {'; '.join(rec.reasons)}")
            lines.append("")

    explanation = response.explanation
    if explanation:
        lines.append(f"Explanation: {explanation.summary}")
        for rec in recs:
            cited = explanation.citations.get(rec.song["id"]) or []
            suffix = f" [{', '.join(cited)}]" if cited else ""
            lines.append(f"  {rec.rank}. {explanation.items[rec.song['id']]}{suffix}")
    limitations = list(dict.fromkeys(plan.limitations + (explanation.limitations if explanation else [])))
    if limitations:
        lines += ["", "Limitations:"] + [f"  - {item}" for item in limitations]
    lines += ["", f"Explanation generated by: {response.generator}", f"Generation status: {response.generation_status}"]
    if response.http_attempts:
        lines.append(f"External requests used: {response.http_attempts}")
    if show_plan:
        lines += _plan_lines(response)
    return "\n".join(lines)


def _plan_lines(response: AssistantResponse) -> List[str]:
    plan = response.plan
    lines = ["", "--- Plan details (--show-plan) ---"]
    if plan is None:
        return lines
    lines.append(f"Status: {plan.status}")
    if plan.engine_prefs:
        lines.append("Engine preferences (passed unchanged to the recommender):")
        lines += [f"  {key}: {value!r}" for key, value in plan.engine_prefs.items()]
    lines.append(f"Mode: {plan.mode} ({plan.mode_reason or 'n/a'}); advanced: {plan.advanced}; diverse: {plan.diverse}")
    lines.append(f"Hard genres: {plan.hard_genres or 'none'}; excluded genres: {plan.excluded_genres or 'none'}; "
                 f"excluded artists: {plan.excluded_artists or 'none'}; unique artists: {plan.unique_artists}")
    lines.append("Retrieved hints:" if plan.hints else "Retrieved hints: none applied")
    for hint in plan.hints:
        state = "used" if hint.used else f"not used ({hint.note})"
        lines.append(f"  {hint.field} = {describe_value(hint.field, hint.value)} from {hint.source_id}: {state}")
    for r in response.retrieved:
        rejected = f"; rejected hints: {list(r.passage.rejected_hints)}" if r.passage.rejected_hints else ""
        lines.append(f"  source {r.passage.source_id} [{r.passage.origin}] score {r.score:.4f}{rejected}")
    if response.trace is not None:
        lines.append("Trace:")
        lines += [f"  {line}" for line in response.trace.summary_lines()]
    return lines
