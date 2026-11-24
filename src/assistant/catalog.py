"""
Catalog-derived knowledge: vocabulary and retrievable facts computed from data/songs.csv.

Nothing here is typed in by hand, so the facts cannot drift from the CSV.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.assistant.models import Passage
from src.recommender import normalize_label

ACOUSTIC_THRESHOLD = 0.5
INSTRUMENTAL_THRESHOLD = 0.6


def slugify(text: str) -> str:
    """Stable id fragment: lower-case words joined by hyphens ('r&b' -> 'r-b')."""
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


@dataclass(frozen=True)
class CatalogVocabulary:
    genres: Tuple[str, ...]        # normalized, sorted
    moods: Tuple[str, ...]
    artists: Tuple[str, ...]       # display names, sorted


def vocabulary(songs: List[Dict]) -> CatalogVocabulary:
    return CatalogVocabulary(
        genres=tuple(sorted({normalize_label(s["genre"]) for s in songs})),
        moods=tuple(sorted({normalize_label(s["mood"]) for s in songs})),
        artists=tuple(sorted({s["artist"] for s in songs}, key=str.casefold)),
    )


def _counts(values) -> str:
    counter = Counter(values)
    return ", ".join(f"{name} ({n})" for name, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _song_line(song: Dict) -> str:
    return (f"{song['title']} by {song['artist']} (id {song['id']}, {song['genre']}, {song['mood']}, "
            f"energy {song['energy']:.2f}, acousticness {song['acousticness']:.2f})")


def catalog_passages(songs: List[Dict]) -> List[Passage]:
    """Facts about the catalog as retrievable passages with stable 'catalog#...' ids."""
    passages: List[Passage] = []

    def add(source_id: str, title: str, text: str) -> None:
        passages.append(Passage(source_id=f"catalog#{source_id}", title=title, text=text, origin="catalog"))

    genres = [normalize_label(s["genre"]) for s in songs]
    moods = [normalize_label(s["mood"]) for s in songs]
    single = sorted(g for g, n in Counter(genres).items() if n == 1)
    add("genre-coverage", "Genre coverage",
        f"The catalog has {len(songs)} songs in {len(set(genres))} genres: {_counts(genres)}. "
        f"Genres with only one song: {', '.join(single) or 'none'}. Any other genre, such as metal or "
        f"country, is not in the catalog.")
    add("mood-coverage", "Mood coverage",
        f"The catalog has {len(set(moods))} moods: {_counts(moods)}.")
    artists = [s["artist"] for s in songs]
    repeated = sorted(a for a, n in Counter(artists).items() if n > 1)
    add("artists", "Artists",
        f"The catalog has {len(set(artists))} artists. Artists with more than one song: "
        f"{', '.join(repeated) or 'none'}. All artists: {', '.join(sorted(set(artists), key=str.casefold))}.")

    acoustic = [s for s in songs if s["acousticness"] >= ACOUSTIC_THRESHOLD]
    if acoustic:
        top = max(acoustic, key=lambda s: s["energy"])
        add("energy-vs-acousticness", "Energy versus acousticness",
            f"{len(acoustic)} songs have acousticness of at least {ACOUSTIC_THRESHOLD}; the most energetic of "
            f"them is {top['title']} at energy {top['energy']:.2f}. High-energy acoustic songs are not in "
            f"this catalog.")
    if all(s.get("instrumentalness") is not None for s in songs):
        instrumental = sorted((s for s in songs if s["instrumentalness"] >= INSTRUMENTAL_THRESHOLD),
                              key=lambda s: s["id"])
        add("instrumental-songs", "Instrumental songs",
            f"{len(instrumental)} songs have instrumentalness of at least {INSTRUMENTAL_THRESHOLD}: "
            + "; ".join(f"{s['title']} ({s['instrumentalness']:.2f})" for s in instrumental) + ".")

    for genre in sorted(set(genres)):
        members = [s for s in songs if normalize_label(s["genre"]) == genre]
        add(f"genre-{slugify(genre)}", f"Genre {genre}",
            f"Genre {genre} has {len(members)} song(s): " + "; ".join(_song_line(s) for s in members) + ".")
    for mood in sorted(set(moods)):
        members = [s for s in songs if normalize_label(s["mood"]) == mood]
        energies = [s["energy"] for s in members]
        add(f"mood-{slugify(mood)}", f"Mood {mood}",
            f"Mood {mood} has {len(members)} song(s) with energy from {min(energies):.2f} to {max(energies):.2f}: "
            + "; ".join(_song_line(s) for s in members) + ".")
    for song in songs:
        tags = ", ".join(song.get("mood_tags") or ()) or "none"
        add(f"song-{song['id']}", song["title"],
            f"{_song_line(song)}; tempo {song['tempo_bpm']:g} BPM; valence {song['valence']:.2f}; "
            f"danceability {song['danceability']:.2f}; mood tags: {tags}.")
    return passages
