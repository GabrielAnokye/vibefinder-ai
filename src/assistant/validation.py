"""
Deterministic guardrails: request text, final recommendations, and generated explanations.

These checks prove consistency with the catalog, the engine, the plan, and the list of
retrieved source ids. They do NOT prove that every sentence of generated prose is true, or
that a cited passage actually supports the sentence that cites it; that needs human review.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.assistant.models import MAX_REQUEST_CHARS, Explanation, Plan, Recommendation
from src.assistant.providers import EXAMPLE_NAMES, EXPLANATION_JSON_SCHEMA, FEATURES
from src.recommender import normalize_label, score_song

MAX_RAW_CHARS = 8000
MAX_SUMMARY_CHARS = 600
MAX_ITEM_CHARS = 600
MAX_LIMITATIONS = 10
SCORE_TOLERANCE = 1e-9
NUMBER_TOLERANCE = 0.006       # a stated decimal must match an authoritative value to 2 decimals

TOP_LEVEL_KEYS = set(EXPLANATION_JSON_SCHEMA["properties"])
ITEM_KEYS = set(EXPLANATION_JSON_SCHEMA["properties"]["items"]["items"]["properties"])
OVERCLAIMS = ("perfect match", "guaranteed", "you will love", "you'll love", "100% match")
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")


def check_request_text(text: object) -> Optional[str]:
    """An error message for unusable input, or None."""
    if not isinstance(text, str):
        return "The request must be text."
    if not text.strip():
        return "The request is empty. Describe what you'd like to hear, for example \"calm music for studying\"."
    if len(text) > MAX_REQUEST_CHARS:
        return f"The request is {len(text)} characters long; please keep it under {MAX_REQUEST_CHARS}."
    return None


def check_recommendations(recs: Sequence[Recommendation], songs_by_id: Dict[int, Dict], plan: Plan) -> List[str]:
    """Every result must be a real catalog song with the engine's exact score and obey the plan's rules."""
    errors: List[str] = []
    if len(recs) > plan.k:
        errors.append(f"{len(recs)} results exceed the requested {plan.k}")
    seen_artists: Set[str] = set()
    hard = {normalize_label(g) for g in plan.hard_genres}
    for expected_rank, rec in enumerate(recs, start=1):
        song_id = rec.song.get("id")
        catalog_song = songs_by_id.get(song_id)
        if catalog_song is None:
            errors.append(f"song id {song_id!r} is not in the catalog")
            continue
        if rec.song != catalog_song:
            errors.append(f"song {song_id} does not match its catalog record")
        if rec.rank != expected_rank:
            errors.append(f"song {song_id} has rank {rec.rank}, expected {expected_rank}")
        if not isinstance(rec.score, (int, float)) or not math.isfinite(rec.score):
            errors.append(f"song {song_id} has a non-finite score {rec.score!r}")
            continue
        engine_score, engine_reasons = score_song(plan.engine_prefs, catalog_song, plan.mode, plan.advanced)
        if abs(engine_score - rec.score) > SCORE_TOLERANCE:
            errors.append(f"song {song_id} score {rec.score} differs from the engine's {engine_score}")
        if rec.reasons != engine_reasons:
            errors.append(f"song {song_id} reasons differ from the engine's")
        genre, artist = normalize_label(catalog_song["genre"]), normalize_label(catalog_song["artist"])
        if hard and genre not in hard:
            errors.append(f"song {song_id} genre {genre} violates the required genres")
        if genre in {normalize_label(g) for g in plan.excluded_genres}:
            errors.append(f"song {song_id} has excluded genre {genre}")
        if artist in {normalize_label(a) for a in plan.excluded_artists}:
            errors.append(f"song {song_id} has excluded artist {catalog_song['artist']}")
        if plan.unique_artists and artist in seen_artists:
            errors.append(f"artist {catalog_song['artist']} repeats although repeats were prohibited")
        seen_artists.add(artist)
    return errors


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else text


def _mentions(text: str, name: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(name.casefold()) + r"(?![a-z0-9])", text.casefold()) is not None


def _numbers(value: Any) -> List[float]:
    return [float(n) for n in DECIMAL.findall(json.dumps(value))]


def _unsupported_numbers(text: str, allowed: Sequence[float]) -> List[str]:
    return [n for n in DECIMAL.findall(text) if not any(abs(float(n) - a) <= NUMBER_TOLERANCE for a in allowed)]


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def parse_explanation(
    raw: str, payload: Dict[str, Any], catalog: Sequence[Dict]
) -> Tuple[Optional[Explanation], List[str]]:
    """Parse and validate provider JSON against the grounding payload; return (Explanation, []) or (None, errors)."""
    if not isinstance(raw, str):
        return None, ["response is not text"]
    if len(raw) > MAX_RAW_CHARS:
        return None, [f"response is longer than {MAX_RAW_CHARS} characters"]
    try:
        data = json.loads(_strip_fences(raw))
    except ValueError:
        return None, ["response is not valid JSON"]
    if not isinstance(data, dict):
        return None, ["response is not a JSON object"]
    errors: List[str] = []
    extra = sorted(set(data) - TOP_LEVEL_KEYS)
    if extra:
        errors.append(f"unexpected top-level field(s) {extra}; recommendations and scores cannot be overridden")
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > MAX_SUMMARY_CHARS:
        errors.append("summary must be a non-empty string under the length limit")
        summary = ""
    items = data.get("items")
    if not isinstance(items, list):
        return None, errors + ["items must be a list"]

    results = {r["song_id"]: r for r in payload["results"]}
    expected_ids = list(results)
    allowed_sources = {s["id"] for s in payload["sources"]}
    texts: Dict[int, str] = {}
    citations: Dict[int, List[str]] = {}
    features: Dict[int, List[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            errors.append("each item must be an object")
            continue
        song_id = item.get("song_id")
        if not isinstance(song_id, int) or isinstance(song_id, bool) or song_id not in results:
            errors.append(f"song_id {song_id!r} is not one of the recommended songs")
            continue
        if song_id in texts:
            errors.append(f"song_id {song_id} is explained twice")
            continue
        extra = sorted(set(item) - ITEM_KEYS)
        if extra:
            errors.append(f"song_id {song_id} has unexpected field(s) {extra}; titles, scores and ranks come "
                          "from the recommender")
        text = item.get("explanation")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_ITEM_CHARS:
            errors.append(f"song_id {song_id} needs a non-empty explanation under the length limit")
            continue
        cited = item.get("citations", [])
        referenced = item.get("features_referenced", [])
        unmet = item.get("unmet_preferences", [])
        if not (_string_list(cited) and _string_list(referenced) and _string_list(unmet)):
            errors.append(f"song_id {song_id} citations, features_referenced and unmet_preferences must be lists of strings")
            continue
        unsupported = [c for c in cited if c not in allowed_sources]
        if unsupported:
            errors.append(f"song_id {song_id} cites unsupported source(s): {', '.join(unsupported)}")
        unknown_features = [f for f in referenced if f not in FEATURES]
        if unknown_features:
            errors.append(f"song_id {song_id} references unknown feature(s): {', '.join(unknown_features)}")
        if sorted(unmet) != sorted(results[song_id]["unmet"]):
            errors.append(f"song_id {song_id} unmet_preferences {unmet} do not match the recommender's "
                          f"{results[song_id]['unmet']}")
        allowed_numbers = _numbers(results[song_id]) + _numbers(payload["limitations"])
        bad_numbers = _unsupported_numbers(text, allowed_numbers)
        if bad_numbers:
            errors.append(f"song_id {song_id} states number(s) {bad_numbers} that are not in its authoritative data")
        texts[song_id] = text.strip()
        citations[song_id] = list(dict.fromkeys(cited))
        features[song_id] = list(dict.fromkeys(referenced))
    missing = [i for i in expected_ids if i not in texts]
    if missing:
        errors.append(f"no explanation for song id(s) {missing}")

    limitations = data.get("limitations", [])
    if (not _string_list(limitations) or len(limitations) > MAX_LIMITATIONS
            or any(len(x) > MAX_ITEM_CHARS for x in limitations)):
        errors.append("limitations must be a short list of strings")
        limitations = []

    prose = " ".join([summary] + list(texts.values()) + list(limitations))
    bad_summary_numbers = _unsupported_numbers(summary, _numbers(payload))
    if bad_summary_numbers:
        errors.append(f"summary states number(s) {bad_summary_numbers} that are not in the authoritative data")
    result_artists = {normalize_label(r["artist"]) for r in payload["results"]}
    for song in catalog:
        if song["id"] not in results and _mentions(prose, song["title"]):
            errors.append(f"mentions song not in the results: {song['title']}")
        if normalize_label(song["artist"]) not in result_artists and _mentions(prose, song["artist"]):
            errors.append(f"mentions artist not in the results: {song['artist']}")
    for name in EXAMPLE_NAMES:
        if _mentions(prose, name):
            errors.append(f"mentions few-shot example name {name!r}, which is not a catalog song")
    for phrase in OVERCLAIMS:
        if phrase in prose.casefold():
            errors.append(f"overclaims with {phrase!r}")

    if errors:
        return None, list(dict.fromkeys(errors))
    return Explanation(summary=summary.strip(), items=texts, citations=citations,
                       limitations=[x.strip() for x in limitations if x.strip()], features=features), []
