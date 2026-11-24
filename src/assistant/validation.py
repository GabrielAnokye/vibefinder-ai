"""
Deterministic guardrails: request text, final recommendations, and generated explanations.

These checks prove consistency with the catalog, the engine, and the retrieved sources.
They do not prove that every sentence of generated prose is true.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.assistant.models import MAX_REQUEST_CHARS, Explanation, Plan, Recommendation
from src.recommender import normalize_label, score_song

MAX_SUMMARY_CHARS = 600
MAX_ITEM_CHARS = 600
MAX_LIMITATIONS = 10
SCORE_TOLERANCE = 1e-9


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


def parse_explanation(
    raw: str, recs: Sequence[Recommendation], allowed_sources: Sequence[str], catalog: Sequence[Dict]
) -> Tuple[Optional[Explanation], List[str]]:
    """Parse and validate provider JSON; return (Explanation, []) or (None, errors)."""
    try:
        data = json.loads(_strip_fences(raw))
    except (TypeError, ValueError):
        return None, ["response is not valid JSON"]
    if not isinstance(data, dict):
        return None, ["response is not a JSON object"]
    errors: List[str] = []
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > MAX_SUMMARY_CHARS:
        errors.append("summary must be a non-empty string under the length limit")
    items = data.get("items")
    if not isinstance(items, list):
        return None, errors + ["items must be a list"]

    expected_ids = [rec.song["id"] for rec in recs]
    allowed = set(allowed_sources)
    texts: Dict[int, str] = {}
    citations: Dict[int, List[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            errors.append("each item must be an object")
            continue
        song_id = item.get("song_id")
        if not isinstance(song_id, int) or isinstance(song_id, bool) or song_id not in expected_ids:
            errors.append(f"song_id {song_id!r} is not one of the recommended songs")
            continue
        if song_id in texts:
            errors.append(f"song_id {song_id} is explained twice")
            continue
        text = item.get("explanation")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_ITEM_CHARS:
            errors.append(f"song_id {song_id} needs a non-empty explanation under the length limit")
            continue
        cited = item.get("citations", [])
        if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
            errors.append(f"song_id {song_id} citations must be a list of strings")
            continue
        unsupported = [c for c in cited if c not in allowed]
        if unsupported:
            errors.append(f"song_id {song_id} cites unsupported source(s): {', '.join(unsupported)}")
        texts[song_id] = text.strip()
        citations[song_id] = list(dict.fromkeys(cited))
    missing = [i for i in expected_ids if i not in texts]
    if missing:
        errors.append(f"no explanation for song id(s) {missing}")

    limitations = data.get("limitations", [])
    if (not isinstance(limitations, list) or len(limitations) > MAX_LIMITATIONS
            or not all(isinstance(x, str) and len(x) <= MAX_ITEM_CHARS for x in limitations)):
        errors.append("limitations must be a short list of strings")
        limitations = []

    # Prose must not mention catalog songs or artists that are not in the results.
    prose = " ".join([summary if isinstance(summary, str) else ""] + list(texts.values()) + list(limitations))
    result_ids = set(expected_ids)
    result_artists = {normalize_label(rec.song["artist"]) for rec in recs}
    for song in catalog:
        if song["id"] not in result_ids and _mentions(prose, song["title"]):
            errors.append(f"mentions song not in the results: {song['title']}")
        if normalize_label(song["artist"]) not in result_artists and _mentions(prose, song["artist"]):
            errors.append(f"mentions artist not in the results: {song['artist']}")

    if errors:
        return None, list(dict.fromkeys(errors))
    return Explanation(summary=summary.strip(), items=texts, citations=citations,
                       limitations=[x.strip() for x in limitations if x.strip()]), []
