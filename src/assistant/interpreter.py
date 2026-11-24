"""
Deterministic rule-based interpreter: request text -> Interpretation.

This is conservative keyword and phrase matching, not general language understanding.

How it works:
1. Tokenize (lower-case words; 'hip-hop', 'lo-fi', 'r&b', "don't", '90s' stay single tokens;
   . , ; ! ? are kept as separate tokens so clauses can be told apart).
2. Match phrases from a fixed lexicon, longest first, on whole tokens only. 'popularity'
   is one token, so it never matches the genre 'pop'.
3. Check negation and 'only' markers by looking back a few tokens inside the same clause.
4. Record each preference with its provenance (explicit) and the phrase that produced it.

Supported vocabulary is listed in the tables below (see also supported_vocabulary()).
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.assistant.catalog import CatalogVocabulary
from src.assistant.models import (
    EXPLICIT, MAX_RESULT_COUNT, Interpretation, Pref,
)
from src.recommender import MAX_DECADE, MIN_DECADE, MOOD_TAGS

# ---------------------------------------------------------------- vocabulary tables

GENRE_ALIASES = {
    "hip hop": "hip-hop", "hiphop": "hip-hop", "rap": "hip-hop", "lo-fi": "lofi", "lo fi": "lofi",
    "rnb": "r&b", "r and b": "r&b", "afrobeat": "afrobeats", "synth-wave": "synthwave",
    "synth wave": "synthwave", "indie-pop": "indie pop", "classical music": "classical",
}
# Recognized genres that are NOT in the catalog (so they can be reported, not silently ignored).
UNAVAILABLE_GENRES = (
    "metal", "heavy metal", "punk", "hard rock", "grunge", "country", "edm", "techno",
    "house music", "trance", "dubstep", "drum and bass", "blues", "gospel", "soul", "funk",
    "disco", "k-pop", "latin", "reggaeton", "salsa", "opera", "emo", "trap", "drill",
    "electronic", "bluegrass", "ska", "grime",
)
MOOD_WORDS = {
    "happy": "happy", "cheerful": "happy", "joyful": "happy", "feel-good": "happy", "feel good": "happy",
    "sad": "sad", "sadness": "sad", "heartbroken": "sad",
    "chill": "chill", "chilled": "chill", "chillout": "chill", "laid-back": "chill", "laid back": "chill",
    "relaxed": "relaxed", "intense": "intense", "moody": "moody", "dark": "moody",
    "romantic": "romantic", "date night": "romantic", "confident": "confident", "swagger": "confident",
}
TAG_WORDS = {
    "nostalgic": "nostalgic", "nostalgia": "nostalgic", "throwback": "nostalgic", "throwbacks": "nostalgic",
    "dreamy": "dreamy", "uplifting": "uplifting", "euphoric": "euphoric", "reflective": "reflective",
    "aggressive": "aggressive", "melancholic": "melancholic", "melancholy": "melancholic",
    "focused": "focused",
}
ENERGY_WORDS = {
    # very low
    "very calm": 0.15, "very quiet": 0.15, "very low energy": 0.15, "very low-energy": 0.15,
    "very mellow": 0.15,
    # low
    "calm": 0.30, "calming": 0.30, "quiet": 0.30, "mellow": 0.30, "gentle": 0.30, "soft": 0.30,
    "peaceful": 0.30, "soothing": 0.30, "low energy": 0.30, "low-energy": 0.30, "slow": 0.30,
    "tranquil": 0.30,
    # medium
    "moderate energy": 0.55, "medium energy": 0.55, "mid-tempo": 0.55,
    # high
    "energetic": 0.85, "high energy": 0.85, "high-energy": 0.85, "upbeat": 0.85, "pumped": 0.85,
    "hype": 0.85, "lively": 0.85, "fast": 0.85, "pump up": 0.85,
    # very high
    "very energetic": 0.95, "very high energy": 0.95, "very high-energy": 0.95,
    "extremely energetic": 0.95, "maximum energy": 0.95, "super energetic": 0.95,
}
ACOUSTIC_WORDS = {"acoustic": True, "unplugged": True}
VOCAL_PHRASES = {
    # phrase -> {field: value}; matched before negation, so "no vocals" is one phrase
    "instrumental": {"instrumentalness": 0.85},
    "instrumentals": {"instrumentalness": 0.85},
    "no vocals": {"instrumentalness": 0.85, "vocal_intensity": 0.05},
    "without vocals": {"instrumentalness": 0.85, "vocal_intensity": 0.05},
    "no lyrics": {"instrumentalness": 0.85, "vocal_intensity": 0.05},
    "without lyrics": {"instrumentalness": 0.85, "vocal_intensity": 0.05},
    "no singing": {"instrumentalness": 0.85, "vocal_intensity": 0.05},
    "without singing": {"instrumentalness": 0.85, "vocal_intensity": 0.05},
    "wordless": {"instrumentalness": 0.85, "vocal_intensity": 0.05},
    "without distracting vocals": {"vocal_intensity": 0.20},
    "no distracting vocals": {"vocal_intensity": 0.20},
    "soft vocals": {"vocal_intensity": 0.20}, "light vocals": {"vocal_intensity": 0.20},
    "quiet vocals": {"vocal_intensity": 0.20},
    "strong vocals": {"vocal_intensity": 0.80}, "powerful vocals": {"vocal_intensity": 0.80},
    "sing-along": {"vocal_intensity": 0.80}, "sing along": {"vocal_intensity": 0.80},
}
POPULARITY_WORDS = {
    "popular": 80, "hits": 80, "mainstream": 80, "well-known": 80, "well known": 80, "famous": 80,
    "obscure": 25, "underground": 25, "hidden gems": 25, "hidden gem": 25, "lesser-known": 25,
    "lesser known": 25, "deep cuts": 25, "niche": 25,
}
DECADE_WORDS = {"eighties": 1980, "nineties": 1990}
CONTEXT_WORDS = {
    "studying": ["study", "studying", "studies", "exam", "exams", "finals", "homework", "revision",
                 "revising", "reading", "concentrate", "concentrating", "concentration", "focus", "focusing"],
    "workout": ["workout", "workouts", "work out", "working out", "gym", "exercise", "exercising",
                "running", "jogging", "lifting", "training", "cardio", "fitness"],
    "relaxing": ["relax", "relaxing", "relaxation", "unwind", "unwinding", "wind down", "winding down",
                 "quiet evening", "evening", "night in", "quiet night"],
    "nostalgia": ["nostalgic", "nostalgia", "throwback", "throwbacks", "memories", "reminisce",
                  "reminiscing", "old times"],
    "discovery": ["discover", "discovery", "explore", "exploring", "new music", "something new",
                  "something different", "surprise me"],
}
DIVERSITY_WORDS = ["variety", "varied", "mix", "mixed", "mixture", "diverse", "diversity", "eclectic",
                   "assortment", "different genres", "different styles", "a range"]
STRATEGY_PHRASES = {
    "mood first": "mood-first", "mood matters most": "mood-first", "focus on mood": "mood-first",
    "prioritize mood": "mood-first", "energy first": "energy-focused",
    "energy matters most": "energy-focused", "focus on energy": "energy-focused",
    "prioritize energy": "energy-focused", "genre first": "baseline", "focus on genre": "baseline",
    "prioritize genre": "baseline",
}
NEGATORS = ["no", "not", "without", "except", "avoid", "exclude", "excluding", "skip", "never",
            "don't", "dont", "nothing", "hate", "dislike", "minus", "isn't", "aren't"]
HARD_MARKERS = ["only", "exclusively", "strictly", "solely", "nothing but", "must be"]
SOFT_MARKERS = ["preferably", "ideally", "maybe", "perhaps", "prefer", "if possible"]

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
COUNT_NOUNS = {"song", "songs", "track", "tracks", "tune", "tunes", "picks", "recommendations"}
# Words that end the search between a number and a song noun ("set k to 1000 and recommend song").
COUNT_GAP_STOPS = {",", "and", "or", "but", "to", "recommend", "give", "me", "play", "find", "set", "for",
                   "with", "by", "then", "please", "id"}
UNIQUE_ARTIST_CUES = {"different", "unique", "repeat", "repeated", "repeating", "repeats", "twice",
                      "distinct", "duplicate", "duplicates"}

# Tokens that may sit between a negator and what it negates ("no rock or pop", "don't want any pop").
CONNECTORS = {"or", "and", "nor", ",", "any", "more", "the", "a", "some", "of", "by", "songs", "music",
              "want", "like", "need", "include", "play", "give", "me", "too", "much"}
CLAUSE_BREAKS = {".", ";", "!", "?", "but"}
LOOKBACK = 5

TOKEN = re.compile(r"[a-z0-9&]+(?:[-'][a-z0-9&]+)*|[.,;!?]")
DECADE_TOKEN = re.compile(r"^(?:(19|20))?([0-9])0s$")
CONFLICT_GAP = 0.4


def tokenize(text: str) -> List[str]:
    return TOKEN.findall(text.casefold().replace("’", "'"))


def supported_vocabulary() -> Dict[str, List[str]]:
    """The phrases the interpreter understands, for documentation and --show-plan."""
    return {
        "contexts": [f"{c}: {', '.join(words)}" for c, words in CONTEXT_WORDS.items()],
        "energy": sorted(ENERGY_WORDS), "moods": sorted(MOOD_WORDS), "mood tags": sorted(TAG_WORDS),
        "acoustic": sorted(ACOUSTIC_WORDS), "vocals": sorted(VOCAL_PHRASES),
        "popularity": sorted(POPULARITY_WORDS), "variety": DIVERSITY_WORDS,
        "exclusions": NEGATORS, "hard requirements": HARD_MARKERS, "soft preferences": SOFT_MARKERS,
    }


# ---------------------------------------------------------------- phrase matching

@dataclass
class Match:
    start: int
    end: int
    phrase: str
    payloads: List[Tuple[str, Any]]

    def has(self, category: str) -> bool:
        return any(c == category for c, _ in self.payloads)

    def values(self, category: str) -> List[Any]:
        return [v for c, v in self.payloads if c == category]


class Lexicon:
    def __init__(self, vocab: CatalogVocabulary, songs: Sequence[Dict]):
        self.entries: Dict[Tuple[str, ...], List[Tuple[str, Any]]] = {}
        catalog_genres = set(vocab.genres)
        for genre in vocab.genres:
            self._add(genre, "genre", genre)
        for alias, genre in GENRE_ALIASES.items():
            if genre in catalog_genres:
                self._add(alias, "genre", genre)
        for genre in UNAVAILABLE_GENRES:
            if genre not in catalog_genres:
                self._add(genre, "unavailable_genre", genre.replace(" music", ""))
        catalog_moods = set(vocab.moods)
        for word, mood in MOOD_WORDS.items():
            if mood in catalog_moods:
                self._add(word, "mood", mood)
        for word, tag in TAG_WORDS.items():
            if tag in MOOD_TAGS:
                self._add(word, "tag", tag)
        for table, category in ((ENERGY_WORDS, "energy"), (ACOUSTIC_WORDS, "acoustic"),
                                (VOCAL_PHRASES, "vocal"), (POPULARITY_WORDS, "popularity"),
                                (DECADE_WORDS, "decade"), (STRATEGY_PHRASES, "strategy")):
            for phrase, value in table.items():
                self._add(phrase, category, value)
        for context, words in CONTEXT_WORDS.items():
            for word in words:
                self._add(word, "context", context)
        for words, category in ((DIVERSITY_WORDS, "diversity"), (NEGATORS, "negator"),
                                (HARD_MARKERS, "hard"), (SOFT_MARKERS, "soft")):
            for word in words:
                self._add(word, category, True)
        for artist in vocab.artists:
            self._add(artist, "artist", artist)
        for song in songs:
            self._add(song["title"], "title", song["title"])
        self.max_len = max(len(key) for key in self.entries)

    def _add(self, phrase: str, category: str, value: Any) -> None:
        key = tuple(tokenize(phrase))
        if key:
            self.entries.setdefault(key, []).append((category, value))

    def scan(self, tokens: List[str]) -> List[Match]:
        """Greedy longest-match, left to right, whole tokens only."""
        matches, i = [], 0
        while i < len(tokens):
            for length in range(min(self.max_len, len(tokens) - i), 0, -1):
                key = tuple(tokens[i:i + length])
                if key in self.entries:
                    matches.append(Match(i, i + length, " ".join(key), list(self.entries[key])))
                    i += length
                    break
            else:
                i += 1
        return matches


# ---------------------------------------------------------------- interpretation

class Interpreter:
    def __init__(self, vocab: CatalogVocabulary, songs: Sequence[Dict]):
        self.vocab = vocab
        self.lexicon = Lexicon(vocab, songs)

    def interpret(self, text: str) -> Interpretation:
        result = Interpretation(text=text)
        tokens = tokenize(text)
        matches = self.lexicon.scan(tokens)
        owner: List[Optional[int]] = [None] * len(tokens)
        for index, match in enumerate(matches):
            for position in range(match.start, match.end):
                owner[position] = index
        state = _State(result)

        for match in matches:
            negated = _negated(tokens, owner, matches, match.start)
            hard = (_marked(tokens, owner, matches, match.start, "hard")
                    or _followed_by_hard(tokens, owner, matches, match.end))
            for category, value in match.payloads:
                state.apply(category, value, match.phrase, negated, hard)

        _detect_unique_artists(tokens, result)
        _detect_count(tokens, result)
        _detect_decades(tokens, result)
        state.finish()
        return result


def _negated(tokens, owner, matches, start) -> bool:
    """
    True if a negator governs the phrase starting at `start`.

    Walks back over connectors ("or", "any", ",") and other genre/artist names, so that
    "no rock or pop" negates both genres; any other word ends the search.
    """
    j, steps = start - 1, 0
    while j >= 0 and steps < LOOKBACK:
        if tokens[j] in CLAUSE_BREAKS:
            return False
        index = owner[j]
        if index is not None:
            match = matches[index]
            if match.has("negator"):
                return True
            if not (match.has("genre") or match.has("unavailable_genre") or match.has("artist")):
                return False
            j, steps = match.start - 1, steps + 1
        elif tokens[j] in CONNECTORS:
            j, steps = j - 1, steps + 1
        else:
            return False
    return False


def _followed_by_hard(tokens, owner, matches, end) -> bool:
    """'rock only' / 'rock songs only': a hard marker right after the phrase (one filler word allowed)."""
    for j in range(end, min(end + 2, len(tokens))):
        index = owner[j]
        if index is not None and matches[index].has("hard"):
            return True
        if tokens[j] not in COUNT_NOUNS | {"music"}:
            return False
    return False


def _marked(tokens, owner, matches, start, category) -> bool:
    """True if a `category` marker ("only", ...) appears within LOOKBACK tokens in the same clause."""
    for j in range(start - 1, max(-1, start - 1 - LOOKBACK), -1):
        if tokens[j] in CLAUSE_BREAKS:
            return False
        index = owner[j]
        if index is not None and matches[index].has(category):
            return True
    return False


class _State:
    """Collects matches into the Interpretation and records conflicts."""

    def __init__(self, result: Interpretation):
        self.r = result
        self.energy_values: List[Tuple[float, str]] = []
        self.acoustic_values: List[Tuple[bool, str]] = []
        self.wanted_genres: List[Tuple[str, str]] = []
        self.hard = False
        self.conflicts: List[str] = []

    def _append(self, name: str, value: Any, phrase: str) -> None:
        pref = self.r.prefs[name]
        current = tuple(pref.value or ())
        if value not in current:
            pref.value = current + (value,)
            pref.provenance = EXPLICIT
            pref.source = ", ".join(filter(None, [pref.source, f'"{phrase}"']))

    def _set_once(self, name: str, value: Any, phrase: str) -> None:
        pref = self.r.prefs[name]
        if not pref.known:
            self.r.prefs[name] = Pref(value, EXPLICIT, f'"{phrase}"')
        elif pref.value != value:
            self.r.ambiguities.append(f'"{phrase}" also sets {name.replace("_", " ")}; kept the first value')

    def apply(self, category: str, value: Any, phrase: str, negated: bool, hard: bool) -> None:
        r = self.r
        if category in ("negator", "hard", "soft"):
            return
        if category in ("genre", "unavailable_genre"):
            if negated:
                if value not in r.excluded_genres:
                    r.excluded_genres.append(value)
                return
            self.hard = self.hard or hard
            if category == "genre":
                self.wanted_genres.append((value, phrase))
            elif value not in r.unavailable_genres:
                r.unavailable_genres.append(value)
            return
        if category == "artist":
            if negated:
                if value not in r.excluded_artists:
                    r.excluded_artists.append(value)
            else:
                r.ambiguities.append(f'Artist preferences are not supported, so "{value}" was not used to rank songs')
            return
        if category == "title":
            r.ambiguities.append(f'"Songs like {value}" requests are not supported; the title was ignored')
            return
        if category == "acoustic":
            self.acoustic_values.append((not value if negated else value, phrase))
            return
        if negated:
            r.ambiguities.append(f'Ignored "{phrase}" because it was negated; that preference stays unknown')
            return
        if category == "mood":
            self._append("moods", value, phrase)
        elif category == "tag":
            self._append("mood_tags", value, phrase)
        elif category == "energy":
            self.energy_values.append((value, phrase))
        elif category == "vocal":
            for name, number in value.items():
                self._set_once(name, number, phrase)
        elif category == "popularity":
            self._set_once("popularity", value, phrase)
        elif category == "decade":
            self._set_once("decade", value, phrase)
        elif category == "context":
            if value not in r.context_names:
                r.contexts.append((value, phrase))
        elif category == "diversity":
            if not r.diversity.known:
                r.diversity = Pref(True, EXPLICIT, f'"{phrase}"')
        elif category == "strategy":
            if not r.strategy.known:
                r.strategy = Pref(value, EXPLICIT, f'"{phrase}"')

    def finish(self) -> None:
        r = self.r
        # Energy: conflicting levels need clarification; close levels keep the first one.
        if self.energy_values:
            values = [v for v, _ in self.energy_values]
            if max(values) - min(values) >= CONFLICT_GAP:
                phrases = " and ".join(f'"{p}"' for _, p in self.energy_values)
                self.conflicts.append(f"The request asks for both {phrases} energy")
            else:
                value, phrase = self.energy_values[0]
                r.prefs["energy"] = Pref(value, EXPLICIT, f'"{phrase}"')
        if self.acoustic_values:
            if len({v for v, _ in self.acoustic_values}) > 1:
                self.conflicts.append("The request asks for both acoustic and non-acoustic music")
            else:
                value, phrase = self.acoustic_values[0]
                r.prefs["acoustic"] = Pref(value, EXPLICIT, f'"{phrase}"')
        for genre, phrase in self.wanted_genres:
            if genre in r.excluded_genres:
                self.conflicts.append(f"{genre} is both requested and excluded")
            else:
                self._append("genres", genre, phrase)
        if r.prefs["genres"].known or r.unavailable_genres:
            r.genre_requirement = "hard" if self.hard else "soft"
        inst, vocal = r.prefs["instrumentalness"], r.prefs["vocal_intensity"]
        if inst.known and vocal.known and inst.value >= 0.7 and vocal.value >= 0.7:
            self.conflicts.append("The request asks for instrumental music and strong vocals")
        if self.conflicts:
            r.ambiguities.extend(self.conflicts)
            r.clarification = ("Your request has conflicting requirements: " + "; ".join(self.conflicts)
                               + ". Which one should I follow?")


def _clauses(tokens: List[str]) -> List[List[str]]:
    clauses, current = [], []
    for token in tokens:
        if token in {".", ";", "!", "?"}:
            clauses.append(current)
            current = []
        else:
            current.append(token)
    clauses.append(current)
    return clauses


def _detect_unique_artists(tokens: List[str], result: Interpretation) -> None:
    for clause in _clauses(tokens):
        words = set(clause)
        if words & {"artist", "artists"} and (words & UNIQUE_ARTIST_CUES or "per" in words):
            result.unique_artists = True


def _number(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def _detect_count(tokens: List[str], result: Interpretation) -> None:
    found: List[Tuple[int, str]] = []
    for i, token in enumerate(tokens):
        n = _number(token)
        if n is None:
            continue
        if i > 0 and tokens[i - 1] == "top":
            found.append((n, f"top {token}"))
            continue
        for j in range(i + 1, min(i + 4, len(tokens))):
            if tokens[j] in COUNT_NOUNS:
                if j + 1 < len(tokens) and tokens[j + 1] == "per":
                    break  # "one song per artist" is a uniqueness rule, not a count
                found.append((n, " ".join(tokens[i:j + 1])))
                break
            if tokens[j] in CLAUSE_BREAKS or tokens[j] in COUNT_GAP_STOPS:
                break
    if not found:
        return
    value, phrase = found[0]
    if len({v for v, _ in found}) > 1:
        result.ambiguities.append(f'Several song counts were mentioned; using the first ("{phrase}")')
    result.count = Pref(value, EXPLICIT, f'"{phrase}"')
    if value < 1:
        result.clarification = "How many songs would you like? Please ask for at least one."
    elif value > MAX_RESULT_COUNT:
        result.ambiguities.append(f"{value} songs were requested; the maximum per request is {MAX_RESULT_COUNT}")


def _detect_decades(tokens: List[str], result: Interpretation) -> None:
    for token in tokens:
        m = DECADE_TOKEN.match(token)
        if not m:
            continue
        century = m.group(1) or ("20" if m.group(2) in "012" else "19")
        decade = int(f"{century}{m.group(2)}0")
        if MIN_DECADE <= decade <= MAX_DECADE and not result.prefs["decade"].known:
            result.prefs["decade"] = Pref(decade, EXPLICIT, f'"{token}"')
        elif not MIN_DECADE <= decade <= MAX_DECADE:
            result.ambiguities.append(f'"{token}" is outside the supported decades; ignored')
