"""
Planner: Interpretation + retrieved evidence -> executable Plan for the existing engine.

Decision rules (all deterministic, recorded in the plan):
- Hints come only from retrieved guide sections whose 'Applies when' condition matches the
  interpretation, and they only fill fields the user left unknown (explicit always wins).
- Unknown fields are sent to the engine as NO_PREFERENCE; nothing is invented.
- Mode: --mode flag > user-named strategy > mood stated without genre (mood-first) >
  energy as the only stated label-free preference (energy-focused) > retrieved hint > baseline.
- Advanced scoring is on when any advanced preference (instrumentalness, vocals, popularity,
  decade, mood tags) is known.
- Hard genre requirements and exclusions become candidate filters applied before ranking.
- Multiple genres or moods are passed as a list: a song matching any of them earns the
  single existing match contribution once.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.assistant.catalog import ACOUSTIC_THRESHOLD, slugify
from src.assistant.models import (
    ADVANCED_FIELDS, EXPLICIT, INFERRED, MAX_RESULT_COUNT, PREFERENCE_FIELDS, AppliedHint,
    Interpretation, Plan, Pref, RetrievedPassage,
)
from src.recommender import DEFAULT_MODE, NO_PREFERENCE, active_preference_fields, normalize_label

# Tolerances used to decide whether a song "meets" a stated preference (for disclosure only;
# they never change scores). Design choices, documented in docs/applied_ai_design.md.
ENERGY_TOLERANCE = 0.20
UNIT_TOLERANCE = 0.25
POPULARITY_TOLERANCE = 20
EPSILON = 1e-9   # tolerances are inclusive: 0.55 - 0.35 is 0.20000000000000007 in floating point

HINT_FIELDS = {
    "energy": "energy", "acoustic": "acoustic", "instrumentalness": "instrumentalness",
    "vocal_intensity": "vocal_intensity", "popularity": "popularity", "mood_tags": "mood_tags",
    "moods": "moods",
}
FIELD_LABELS = {
    "genres": "genre", "moods": "mood", "energy": "energy", "acoustic": "acoustic sound",
    "popularity": "popularity", "decade": "release decade", "instrumentalness": "instrumentalness",
    "vocal_intensity": "vocal intensity", "mood_tags": "mood tags",
}
CLARIFY_NOTHING_KNOWN = (
    "I couldn't find a preference I can use to rank songs. Could you say what you're doing "
    "(for example studying, working out, or relaxing), how energetic the music should be, "
    "or a genre or mood you like?"
)


def satisfies(song: Dict, field: str, value: Any) -> bool:
    """Whether a song meets one stated preference, using the documented tolerances."""
    if field == "genres":
        return normalize_label(song["genre"]) in {normalize_label(v) for v in value}
    if field == "moods":
        return normalize_label(song["mood"]) in {normalize_label(v) for v in value}
    if field == "energy":
        return abs(song["energy"] - value) <= ENERGY_TOLERANCE + EPSILON
    if field == "acoustic":
        return (song["acousticness"] >= ACOUSTIC_THRESHOLD) == value
    if field == "popularity":
        return abs(song["popularity"] - value) <= POPULARITY_TOLERANCE + EPSILON
    if field == "decade":
        return song["release_decade"] == value
    if field in ("instrumentalness", "vocal_intensity"):
        return abs(song[field] - value) <= UNIT_TOLERANCE + EPSILON
    if field == "mood_tags":
        return bool({normalize_label(t) for t in song["mood_tags"]} & set(value))
    raise ValueError(f"unknown preference field {field!r}")


def describe_value(field: str, value: Any) -> str:
    if isinstance(value, tuple):
        return " or ".join(value) if field in ("genres", "moods") else ", ".join(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _applies(passage, interp: Interpretation) -> bool:
    """A hint section applies only when every 'Applies when' condition matches."""
    if not passage.applies_when:
        return False
    for key, values in passage.applies_when:
        if key == "context" and not set(values) & set(interp.context_names):
            return False
        if key == "unavailable_genre":
            # Substitute features only make sense for a soft request; "only metal" gets no substitutes.
            if interp.genre_requirement == "hard" or not set(values) & set(interp.unavailable_genres):
                return False
    return True


def _engine_label(pref: Pref):
    if not pref.known:
        return NO_PREFERENCE
    values = tuple(pref.value)
    return values[0] if len(values) == 1 else list(values)


def _engine_value(pref: Pref):
    if not pref.known:
        return NO_PREFERENCE
    return list(pref.value) if isinstance(pref.value, tuple) else pref.value


def build_plan(
    interp: Interpretation,
    retrieved: Sequence[RetrievedPassage],
    songs: Sequence[Dict],
    mode_override: Optional[str] = None,
    diverse_override: bool = False,
    citable_ids: Optional[Set[str]] = None,
) -> Tuple[Plan, Dict[str, Pref]]:
    """
    Return the plan and the final preferences (explicit + inferred + unknown).

    citable_ids: source ids that exist in the active knowledge index. Limitations are always
    computed from the catalog, but they cite a catalog passage only when it is in this set.
    """
    citable_ids = citable_ids or set()
    prefs: Dict[str, Pref] = {name: Pref(p.value, p.provenance, p.source) for name, p in interp.prefs.items()}
    hints: List[AppliedHint] = []
    limitations: List[str] = []
    citations = [r.passage.source_id for r in retrieved]
    diversity = Pref(interp.diversity.value, interp.diversity.provenance, interp.diversity.source)
    mode_hint: Optional[Tuple[str, str]] = None

    def cite(source_id: str) -> str:
        if source_id not in citable_ids:
            return ""
        if source_id not in citations:
            citations.append(source_id)
        return f" [{source_id}]"

    # 1. Retrieved hints fill unknown fields only.
    for item in retrieved:
        passage = item.passage
        if not passage.hints or not _applies(passage, interp):
            continue
        for key, value in passage.hints:
            if key == "diversity":
                if diversity.known:
                    hints.append(AppliedHint("diversity", value, passage.source_id, False, "already set"))
                else:
                    diversity = Pref(value, INFERRED, passage.source_id)
                    hints.append(AppliedHint("diversity", value, passage.source_id, True))
                continue
            if key == "mode":
                used = mode_hint is None
                mode_hint = mode_hint or (value, passage.source_id)
                hints.append(AppliedHint("mode", value, passage.source_id, used, "" if used else "earlier source kept"))
                continue
            field = HINT_FIELDS[key]
            current = prefs[field]
            if current.provenance == EXPLICIT:
                hints.append(AppliedHint(field, value, passage.source_id, False, "your explicit value was kept"))
            elif current.known:
                hints.append(AppliedHint(field, value, passage.source_id, False, "earlier source kept"))
            else:
                prefs[field] = Pref(value, INFERRED, passage.source_id)
                hints.append(AppliedHint(field, value, passage.source_id, True))

    def make(status: str, message: Optional[str], **overrides) -> Tuple[Plan, Dict[str, Pref]]:
        plan = Plan(
            engine_prefs={}, mode=DEFAULT_MODE, mode_reason="", advanced=False, diverse=False,
            diverse_reason="", unique_artists=interp.unique_artists, k=0, hard_genres=[],
            excluded_genres=list(interp.excluded_genres), excluded_artists=list(interp.excluded_artists),
            hints=hints, unknown_fields=[FIELD_LABELS[f] for f in PREFERENCE_FIELDS if not prefs[f].known],
            limitations=limitations, citations=citations, status=status, message=message,
        )
        for key, value in overrides.items():
            setattr(plan, key, value)
        return plan, prefs

    # 2. Clarification and hard requirements that cannot be met.
    if interp.clarification:
        return make("clarification", interp.clarification)

    genres = prefs["genres"]
    hard = interp.genre_requirement == "hard"
    if interp.unavailable_genres:
        missing = ", ".join(interp.unavailable_genres)
        if hard and not genres.known:
            return make("no_match",
                        f"The catalog has no {missing} songs{cite('catalog#genre-coverage')}. You asked for only "
                        f"{missing}, so I did not substitute other genres.")
        if hard:
            limitations.append(f"The catalog has no {missing} songs{cite('catalog#genre-coverage')}; "
                               f"only the available requested genres are included.")
        else:
            used = [h for h in hints if h.used and h.source_id.endswith("#genres-not-in-catalog")]
            how = ("results are ranked on sound features associated with it, so they are substitutes, "
                   "not examples of that genre" if used else "results are ranked on your other preferences")
            limitations.append(f"The catalog has no {missing} songs{cite('catalog#genre-coverage')}; {how}.")

    # 3. Engine preferences (unknown -> NO_PREFERENCE).
    advanced = any(prefs[f].known for f in ADVANCED_FIELDS)
    engine_prefs: Dict[str, Any] = {
        "favorite_genre": _engine_label(genres),
        "favorite_mood": _engine_label(prefs["moods"]),
        "target_energy": _engine_value(prefs["energy"]),
        "likes_acoustic": _engine_value(prefs["acoustic"]),
    }
    if advanced:
        engine_prefs.update({
            "target_popularity": _engine_value(prefs["popularity"]),
            "preferred_decade": _engine_value(prefs["decade"]),
            "target_instrumentalness": _engine_value(prefs["instrumentalness"]),
            "target_vocal_intensity": _engine_value(prefs["vocal_intensity"]),
            "preferred_tags": _engine_value(prefs["mood_tags"]),
        })
    if not active_preference_fields(engine_prefs, advanced):
        return make("clarification", CLARIFY_NOTHING_KNOWN)

    # 4. Mode.
    explicit = {f for f in PREFERENCE_FIELDS if prefs[f].provenance == EXPLICIT}
    if mode_override:
        mode, mode_reason = mode_override, "set by the --mode flag"
    elif interp.strategy.known:
        mode, mode_reason = interp.strategy.value, f"you asked for it ({interp.strategy.source})"
    elif "moods" in explicit and "genres" not in explicit:
        mode, mode_reason = "mood-first", "you described a mood but no genre"
    elif "energy" in explicit and not explicit & ({"genres", "moods"} | set(ADVANCED_FIELDS)):
        mode, mode_reason = "energy-focused", "energy is the only genre, mood, or feature preference you stated"
    elif mode_hint:
        mode, mode_reason = mode_hint[0], f"suggested by {mode_hint[1]}"
    else:
        mode, mode_reason = DEFAULT_MODE, "default genre-first ranking"

    # 5. Diversity.
    if diverse_override:
        diverse, diverse_reason = True, "set by the --diverse flag"
    elif diversity.known and diversity.value:
        source = diversity.source
        diverse, diverse_reason = True, (f"you said {source}" if diversity.provenance == EXPLICIT
                                         else f"suggested by {source}")
    else:
        diverse, diverse_reason = False, "no variety requested"

    # 6. Result count.
    k = interp.count.value
    if k > MAX_RESULT_COUNT:
        limitations.append(f"You asked for {k} songs; the maximum per request is {MAX_RESULT_COUNT}.")
        k = MAX_RESULT_COUNT

    hard_genres = list(genres.value) if hard and genres.known else []

    # 7. Catalog coverage disclosure for the stated (explicit) preferences.
    stated = {f: prefs[f].value for f in PREFERENCE_FIELDS if prefs[f].provenance == EXPLICIT}
    if len(stated) >= 2 and not any(all(satisfies(s, f, v) for f, v in stated.items()) for s in songs):
        names = ", ".join(f"{FIELD_LABELS[f]} {describe_value(f, v)}" for f, v in stated.items())
        source = ""
        if stated.get("acoustic") is True and "energy" in stated and stated["energy"] >= 0.6:
            source = cite("catalog#energy-vs-acousticness")
        elif "moods" in stated:
            source = cite(f"catalog#mood-{slugify(stated['moods'][0])}")
        limitations.append(f"No catalog song meets all of your stated preferences at once ({names}){source}; "
                           f"the results are the closest available compromises.")

    return make(
        "ok", None, engine_prefs=engine_prefs, mode=mode, mode_reason=mode_reason, advanced=advanced,
        diverse=diverse, diverse_reason=diverse_reason, k=k, hard_genres=hard_genres,
    )


def filter_candidates(songs: Sequence[Dict], plan: Plan) -> List[Dict]:
    """Apply hard genre requirements and exclusions before ranking (the engine never sees the rest)."""
    hard = {normalize_label(g) for g in plan.hard_genres}
    excluded_genres = {normalize_label(g) for g in plan.excluded_genres}
    excluded_artists = {normalize_label(a) for a in plan.excluded_artists}
    return [
        song for song in songs
        if (not hard or normalize_label(song["genre"]) in hard)
        and normalize_label(song["genre"]) not in excluded_genres
        and normalize_label(song["artist"]) not in excluded_artists
    ]
