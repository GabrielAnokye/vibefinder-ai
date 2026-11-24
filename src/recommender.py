import csv
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------- unknown preferences

class Preference(Enum):
    """
    Marker values for user preferences.

    NO_PREFERENCE means "the listener did not say". It is an Enum member, so only this exact
    object counts: strings such as "NO_PREFERENCE", "any" or "none" are never treated as the
    sentinel and keep being validated as ordinary values. None and missing fields are still
    rejected exactly as before.
    """
    NO_PREFERENCE = "no preference"

    def __repr__(self) -> str:
        return "NO_PREFERENCE"


NO_PREFERENCE = Preference.NO_PREFERENCE


def is_no_preference(value: object) -> bool:
    """True only for the NO_PREFERENCE sentinel itself."""
    return value is NO_PREFERENCE


# Scoring weights (finalized in Phase 2). Maximum possible score is 11.0.
GENRE_WEIGHT = 4.0
MOOD_WEIGHT = 3.0
ENERGY_WEIGHT = 3.0
ACOUSTIC_WEIGHT = 1.0
MAX_SCORE = GENRE_WEIGHT + MOOD_WEIGHT + ENERGY_WEIGHT + ACOUSTIC_WEIGHT


# ---------------------------------------------------------------- ranking modes (Strategy pattern)

@dataclass(frozen=True)
class ScoringWeights:
    """Immutable set of weights for the shared scoring formula in score_song."""
    genre: float
    mood: float
    energy: float
    acoustic: float

    @property
    def max_score(self) -> float:
        return self.genre + self.mood + self.energy + self.acoustic


@dataclass(frozen=True)
class ScoringStrategy:
    """A named ranking mode. Every mode uses score_song; only the weights differ."""
    name: str
    description: str
    get_weights: Callable[[], ScoringWeights]

    def weights(self) -> ScoringWeights:
        return self.get_weights()


def _baseline_weights() -> ScoringWeights:
    # Read the module constants at call time (not import time) so the baseline stays
    # exactly the Phase 2 rule and experiments/weight_experiment.py can still patch it.
    return ScoringWeights(GENRE_WEIGHT, MOOD_WEIGHT, ENERGY_WEIGHT, ACOUSTIC_WEIGHT)


MOOD_FIRST_WEIGHTS = ScoringWeights(genre=2.0, mood=4.0, energy=3.0, acoustic=1.0)
ENERGY_FOCUSED_WEIGHTS = ScoringWeights(genre=2.0, mood=3.0, energy=6.0, acoustic=1.0)

DEFAULT_MODE = "baseline"
STRATEGIES: Dict[str, ScoringStrategy] = {
    "baseline": ScoringStrategy(
        "baseline", "Genre-first: the original 4/3/3/1 weights", _baseline_weights
    ),
    "mood-first": ScoringStrategy(
        "mood-first", "Mood matters most; genre counts less", lambda: MOOD_FIRST_WEIGHTS
    ),
    "energy-focused": ScoringStrategy(
        "energy-focused", "Closeness to the target energy matters most", lambda: ENERGY_FOCUSED_WEIGHTS
    ),
}


def get_strategy(mode: str) -> ScoringStrategy:
    """Look up a ranking mode by name; raise a clear error for anything unknown."""
    if not isinstance(mode, str):
        raise TypeError(f"mode must be a string, got {mode!r}")
    if mode not in STRATEGIES:
        raise ValueError(f"unknown mode {mode!r}; choose from: {', '.join(STRATEGIES)}")
    return STRATEGIES[mode]

REQUIRED_COLUMNS = [
    "id", "title", "artist", "genre", "mood",
    "energy", "tempo_bpm", "valence", "danceability", "acousticness",
]
UNIT_RANGE_COLUMNS = ["energy", "valence", "danceability", "acousticness"]
REQUIRED_PREFERENCES = ["favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"]
REASON_SEPARATOR = "; "

# ---------------------------------------------------------------- optional advanced attributes
# Five extra song attributes (bonus challenge 1). The values in data/songs.csv are synthetic,
# like the songs themselves. They are only used when advanced scoring is explicitly requested.
ADVANCED_COLUMNS = ["popularity", "release_decade", "instrumentalness", "vocal_intensity", "mood_tags"]
ADVANCED_PREFERENCE_FIELDS = [
    "target_popularity", "preferred_decade", "target_instrumentalness",
    "target_vocal_intensity", "preferred_tags",
]
MOOD_TAGS = frozenset({
    "nostalgic", "euphoric", "reflective", "aggressive", "dreamy",
    "focused", "uplifting", "melancholic", "energetic",
})
MOOD_TAG_DELIMITER = "|"
MIN_DECADE, MAX_DECADE = 1900, 2020

# Advanced contributions, added on top of the mode's score (max 3.0 extra points).
POPULARITY_WEIGHT = 0.5
DECADE_WEIGHT = 0.5
INSTRUMENTALNESS_WEIGHT = 0.5
VOCAL_INTENSITY_WEIGHT = 0.5
MOOD_TAG_WEIGHT = 1.0
ADVANCED_MAX_SCORE = (
    POPULARITY_WEIGHT + DECADE_WEIGHT + INSTRUMENTALNESS_WEIGHT + VOCAL_INTENSITY_WEIGHT + MOOD_TAG_WEIGHT
)


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Optional advanced attributes; None means "not provided", never a real value.
    popularity: Optional[int] = None
    release_decade: Optional[int] = None
    instrumentalness: Optional[float] = None
    vocal_intensity: Optional[float] = None
    mood_tags: Optional[Tuple[str, ...]] = None

LabelPreference = Union[str, Sequence[str], Preference]


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py

    Any field may be NO_PREFERENCE (contributes 0 points to every song). favorite_genre and
    favorite_mood may also be a list of labels; a song matching any of them earns the usual
    single match contribution once.
    """
    favorite_genre: LabelPreference
    favorite_mood: LabelPreference
    target_energy: Union[float, Preference]
    likes_acoustic: Union[bool, Preference]
    # Optional advanced preferences; only read when advanced scoring is requested.
    target_popularity: Optional[Union[float, Preference]] = None
    preferred_decade: Optional[Union[int, Preference]] = None
    target_instrumentalness: Optional[Union[float, Preference]] = None
    target_vocal_intensity: Optional[Union[float, Preference]] = None
    preferred_tags: Optional[Union[Tuple[str, ...], Preference]] = None

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song], mode: str = DEFAULT_MODE, advanced: bool = False):
        get_strategy(mode)  # fail fast on an unknown mode
        _check_advanced_flag(advanced)
        self.songs = songs
        self.mode = mode
        self.advanced = advanced

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k Song objects, ranked by the shared functional recommender."""
        song_dicts = [asdict(song) for song in self.songs]
        # recommend_songs returns the same dict objects it was given, so map each back to its Song.
        song_for_dict = {id(song_dict): song for song_dict, song in zip(song_dicts, self.songs)}
        ranked = recommend_songs(asdict(user), song_dicts, k, mode=self.mode, advanced=self.advanced)
        return [song_for_dict[id(song_dict)] for song_dict, _, _ in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain a song's score using the same reasons score_song produces."""
        _, reasons = score_song(asdict(user), asdict(song), mode=self.mode, advanced=self.advanced)
        return REASON_SEPARATOR.join(reasons)


def normalize_label(text: str) -> str:
    """Trim, collapse repeated whitespace, and case-fold a genre or mood label."""
    return " ".join(text.split()).casefold()


def _parse_number(value: str, column: str, where: str) -> float:
    """Convert a CSV cell to a finite float, or raise ValueError naming the row."""
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{where}: {column} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{where}: {column} must be a finite number, got {value!r}")
    return number


def _parse_whole_number(value: str, column: str, where: str) -> int:
    """Convert a CSV cell to an int, or raise ValueError naming the row."""
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(f"{where}: {column} must be a whole number, got {value!r}") from None


def _normalize_tags(tags: List[object], where: str) -> List[str]:
    """Normalize mood tags and check each one against the MOOD_TAGS vocabulary."""
    normalized = []
    for tag in tags:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError(f"{where}: mood tags must be non-empty strings, got {tag!r}")
        label = normalize_label(tag)
        if label not in MOOD_TAGS:
            raise ValueError(
                f"{where}: unknown mood tag {tag!r}; choose from: {', '.join(sorted(MOOD_TAGS))}"
            )
        normalized.append(label)
    return normalized


def _check_decade(value: int, label: str) -> None:
    if value % 10 or not MIN_DECADE <= value <= MAX_DECADE:
        raise ValueError(
            f"{label} must be a decade from {MIN_DECADE} to {MAX_DECADE} (a multiple of 10), got {value!r}"
        )


def _parse_advanced(row: Dict[str, str], where: str) -> Dict:
    """Parse and validate the five advanced columns of one CSV row."""
    popularity = _parse_whole_number(row["popularity"], "popularity", where)
    if not 0 <= popularity <= 100:
        raise ValueError(f"{where}: popularity must be between 0 and 100, got {popularity}")
    decade = _parse_whole_number(row["release_decade"], "release_decade", where)
    _check_decade(decade, f"{where}: release_decade")
    advanced: Dict = {"popularity": popularity, "release_decade": decade}
    for col in ["instrumentalness", "vocal_intensity"]:
        advanced[col] = _parse_number(row[col], col, where)
        if not 0.0 <= advanced[col] <= 1.0:
            raise ValueError(f"{where}: {col} must be between 0 and 1, got {advanced[col]}")
    tags = _normalize_tags(row["mood_tags"].split(MOOD_TAG_DELIMITER), f"{where}: mood_tags")
    if len(set(tags)) != len(tags):
        raise ValueError(f"{where}: mood_tags has a repeated tag: {row['mood_tags']!r}")
    advanced["mood_tags"] = tuple(tags)
    return advanced


def load_songs(csv_path: str) -> List[Dict]:
    """Load and validate the song catalog from a CSV file into a list of dicts."""
    try:
        csv_file = open(csv_path, newline="", encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Song catalog not found: {csv_path}") from None

    songs: List[Dict] = []
    seen_ids = set()
    with csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: file is empty (no header row)")
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"{csv_path}: missing required column(s): {', '.join(missing_columns)}")
        # The advanced columns are optional as a group: all five, or none (an original-schema file).
        missing_advanced = [col for col in ADVANCED_COLUMNS if col not in reader.fieldnames]
        if 0 < len(missing_advanced) < len(ADVANCED_COLUMNS):
            raise ValueError(
                f"{csv_path}: has some advanced columns but is missing: {', '.join(missing_advanced)}"
            )
        has_advanced = not missing_advanced
        value_columns = REQUIRED_COLUMNS + (ADVANCED_COLUMNS if has_advanced else [])

        for row in reader:
            where = f"{csv_path}, line {reader.line_num}"
            if None in row:  # DictReader stores extra cells under the key None
                raise ValueError(f"{where}: row has more values than the header")
            for col in value_columns:
                if row[col] is None or row[col].strip() == "":
                    raise ValueError(f"{where}: missing value for {col}")

            try:
                song_id = int(row["id"])
            except ValueError:
                raise ValueError(f"{where}: id must be an integer, got {row['id']!r}") from None
            if song_id in seen_ids:
                raise ValueError(f"{where}: duplicate id {song_id}")
            seen_ids.add(song_id)

            song: Dict = {
                "id": song_id,
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
            }
            for col in ["energy", "tempo_bpm", "valence", "danceability", "acousticness"]:
                song[col] = _parse_number(row[col], col, where)
            for col in UNIT_RANGE_COLUMNS:
                if not 0.0 <= song[col] <= 1.0:
                    raise ValueError(f"{where}: {col} must be between 0 and 1, got {song[col]}")
            if song["tempo_bpm"] <= 0:
                raise ValueError(f"{where}: tempo_bpm must be positive, got {song['tempo_bpm']}")
            if has_advanced:
                song.update(_parse_advanced(row, where))
            songs.append(song)
    return songs


def _is_number(value: object) -> bool:
    """True for finite ints/floats, excluding bools."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_label_preference(key: str, value: object) -> None:
    """A label preference is NO_PREFERENCE, a non-empty string, or a non-empty list of them."""
    if value is NO_PREFERENCE:
        return
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{key} must be a non-empty string, got {value!r}")
        return
    if (isinstance(value, (list, tuple)) and value
            and all(isinstance(label, str) and label.strip() for label in value)):
        return
    raise ValueError(
        f"{key} must be a non-empty string, a non-empty list of strings, or NO_PREFERENCE, got {value!r}"
    )


def _validate_user_prefs(user_prefs: Dict) -> None:
    """Raise ValueError unless the profile has all four valid preference fields."""
    if not isinstance(user_prefs, dict):
        raise ValueError("user_prefs must be a dict")
    missing = [key for key in REQUIRED_PREFERENCES if key not in user_prefs]
    if missing:
        raise ValueError(f"user_prefs is missing required field(s): {', '.join(missing)}")
    for key in ["favorite_genre", "favorite_mood"]:
        _check_label_preference(key, user_prefs[key])
    target = user_prefs["target_energy"]
    if target is not NO_PREFERENCE and (not _is_number(target) or not 0.0 <= target <= 1.0):
        raise ValueError(f"target_energy must be a number between 0 and 1, got {target!r}")
    likes_acoustic = user_prefs["likes_acoustic"]
    if likes_acoustic is not NO_PREFERENCE and not isinstance(likes_acoustic, bool):
        raise ValueError(f"likes_acoustic must be True or False, got {likes_acoustic!r}")


def _validate_song(song: Dict) -> None:
    """Raise ValueError unless the song has the fields score_song needs."""
    for key in ["genre", "mood"]:
        if not isinstance(song.get(key), str):
            raise ValueError(f"song {key} must be a string, got {song.get(key)!r}")
    for key in ["energy", "acousticness"]:
        value = song.get(key)
        if not _is_number(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"song {key} must be a number between 0 and 1, got {value!r}")


# ---------------------------------------------------------------- advanced scoring (optional)

def _check_advanced_flag(advanced: bool) -> None:
    if not isinstance(advanced, bool):
        raise TypeError(f"advanced must be True or False, got {advanced!r}")


def _validate_advanced_prefs(user_prefs: Dict) -> Union[Tuple[str, ...], Preference]:
    """
    Check the five advanced preferences; return the unique, normalized preferred tags
    (or NO_PREFERENCE). None still counts as missing; only the sentinel means "no preference".
    """
    missing = [key for key in ADVANCED_PREFERENCE_FIELDS if user_prefs.get(key) is None]
    if missing:
        raise ValueError(f"advanced scoring needs these preferences, missing: {', '.join(missing)}")
    popularity = user_prefs["target_popularity"]
    if popularity is not NO_PREFERENCE and (not _is_number(popularity) or not 0 <= popularity <= 100):
        raise ValueError(f"target_popularity must be a number between 0 and 100, got {popularity!r}")
    decade = user_prefs["preferred_decade"]
    if decade is not NO_PREFERENCE:
        if not isinstance(decade, int) or isinstance(decade, bool):
            raise ValueError(f"preferred_decade must be a whole number, got {decade!r}")
        _check_decade(decade, "preferred_decade")
    for key in ["target_instrumentalness", "target_vocal_intensity"]:
        value = user_prefs[key]
        if value is not NO_PREFERENCE and (not _is_number(value) or not 0.0 <= value <= 1.0):
            raise ValueError(f"{key} must be a number between 0 and 1, got {value!r}")
    tags = user_prefs["preferred_tags"]
    if tags is NO_PREFERENCE:
        return NO_PREFERENCE
    if not isinstance(tags, (list, tuple, set, frozenset)) or not tags:
        raise ValueError(f"preferred_tags must be a non-empty list of mood tags, got {tags!r}")
    if isinstance(tags, (set, frozenset)):
        tags = sorted(tags, key=str)  # sets have no stable order; keep explanations deterministic
    # Each preferred tag counts once, in the order the user listed it.
    return tuple(dict.fromkeys(_normalize_tags(list(tags), "preferred_tags")))


def _validate_advanced_song(song: Dict) -> None:
    """Refuse to score a song whose advanced attributes are missing or invalid."""
    missing = [key for key in ADVANCED_COLUMNS if song.get(key) is None]
    if missing:
        raise ValueError(
            f"song {song.get('title', song.get('id'))!r} has no {', '.join(missing)}; "
            "advanced scoring needs a catalog with the advanced columns"
        )
    popularity = song["popularity"]
    if not _is_number(popularity) or not 0 <= popularity <= 100:
        raise ValueError(f"song popularity must be between 0 and 100, got {popularity!r}")
    decade = song["release_decade"]
    if not isinstance(decade, int) or isinstance(decade, bool):
        raise ValueError(f"song release_decade must be a whole number, got {decade!r}")
    _check_decade(decade, "song release_decade")
    for key in ["instrumentalness", "vocal_intensity"]:
        if not _is_number(song[key]) or not 0.0 <= song[key] <= 1.0:
            raise ValueError(f"song {key} must be a number between 0 and 1, got {song[key]!r}")
    if not isinstance(song["mood_tags"], (list, tuple)) or not song["mood_tags"]:
        raise ValueError(f"song mood_tags must be a non-empty list of tags, got {song['mood_tags']!r}")
    _normalize_tags(list(song["mood_tags"]), "song mood_tags")


def _no_preference_reason(feature: str) -> str:
    """Reason for a feature the listener left unspecified: no match or mismatch is claimed."""
    return f"No {feature} preference given (+0.00)"


def _advanced_points(
    user_prefs: Dict, song: Dict, preferred_tags: Union[Tuple[str, ...], Preference]
) -> Tuple[float, List[str]]:
    """The five advanced contributions (max ADVANCED_MAX_SCORE) and their reasons."""
    reasons: List[str] = []

    target = user_prefs["target_popularity"]
    if target is NO_PREFERENCE:
        popularity_points = 0.0
        reasons.append(_no_preference_reason("popularity"))
    else:
        popularity_points = POPULARITY_WEIGHT * (1.0 - abs(song["popularity"] - target) / 100.0)
        reasons.append(f"Popularity {song['popularity']} vs target {target:g} (+{popularity_points:.2f})")

    wanted_decade = user_prefs["preferred_decade"]
    if wanted_decade is NO_PREFERENCE:
        decade_points = 0.0
        reasons.append(_no_preference_reason("decade"))
    elif song["release_decade"] == wanted_decade:
        decade_points = DECADE_WEIGHT
        reasons.append(f"Decade match: {song['release_decade']}s (+{decade_points:.2f})")
    else:
        decade_points = 0.0
        reasons.append(f"Decade {song['release_decade']}s is not {wanted_decade}s (+{decade_points:.2f})")

    target = user_prefs["target_instrumentalness"]
    if target is NO_PREFERENCE:
        instrumental_points = 0.0
        reasons.append(_no_preference_reason("instrumentalness"))
    else:
        instrumental_points = INSTRUMENTALNESS_WEIGHT * (1.0 - abs(song["instrumentalness"] - target))
        reasons.append(
            f"Instrumentalness {song['instrumentalness']:.2f} vs target {target:.2f} (+{instrumental_points:.2f})"
        )

    target = user_prefs["target_vocal_intensity"]
    if target is NO_PREFERENCE:
        vocal_points = 0.0
        reasons.append(_no_preference_reason("vocal intensity"))
    else:
        vocal_points = VOCAL_INTENSITY_WEIGHT * (1.0 - abs(song["vocal_intensity"] - target))
        reasons.append(
            f"Vocal intensity {song['vocal_intensity']:.2f} vs target {target:.2f} (+{vocal_points:.2f})"
        )

    if preferred_tags is NO_PREFERENCE:
        tag_points = 0.0
        reasons.append(_no_preference_reason("mood-tag"))
    else:
        song_tags = {normalize_label(tag) for tag in song["mood_tags"]}
        matched = [tag for tag in preferred_tags if tag in song_tags]
        tag_points = MOOD_TAG_WEIGHT * len(matched) / len(preferred_tags)
        listed = f": {', '.join(matched)}" if matched else ""
        reasons.append(f"Mood tags {len(matched)}/{len(preferred_tags)} matched{listed} (+{tag_points:.2f})")

    total = popularity_points + decade_points + instrumental_points + vocal_points + tag_points
    return total, reasons


def max_possible_score(mode: str = DEFAULT_MODE, advanced: bool = False) -> float:
    """Highest score a song can get in this mode, including advanced points if enabled."""
    _check_advanced_flag(advanced)
    return get_strategy(mode).weights().max_score + (ADVANCED_MAX_SCORE if advanced else 0.0)


def _component_weights(mode: str, advanced: bool) -> Dict[str, float]:
    """Maximum contribution of each preference field (read at call time, like the scorer)."""
    w = get_strategy(mode).weights()
    weights = {
        "favorite_genre": w.genre, "favorite_mood": w.mood,
        "target_energy": w.energy, "likes_acoustic": w.acoustic,
    }
    if advanced:
        weights.update({
            "target_popularity": POPULARITY_WEIGHT, "preferred_decade": DECADE_WEIGHT,
            "target_instrumentalness": INSTRUMENTALNESS_WEIGHT,
            "target_vocal_intensity": VOCAL_INTENSITY_WEIGHT, "preferred_tags": MOOD_TAG_WEIGHT,
        })
    return weights


def active_preference_fields(user_prefs: Dict, advanced: bool = False) -> List[str]:
    """The preference fields that will be scored: every field except NO_PREFERENCE ones."""
    fields = REQUIRED_PREFERENCES + (ADVANCED_PREFERENCE_FIELDS if advanced else [])
    return [field for field in fields if user_prefs.get(field) is not NO_PREFERENCE]


def active_max_score(user_prefs: Dict, mode: str = DEFAULT_MODE, advanced: bool = False) -> float:
    """
    Highest score available from the preferences actually given.

    max_possible_score is the ceiling for a fully specified profile; NO_PREFERENCE fields add
    nothing to any song, so a partial profile is judged against this smaller ceiling instead.
    """
    _validate_request(user_prefs, 0, mode, advanced)
    weights = _component_weights(mode, advanced)
    return sum(weights[field] for field in active_preference_fields(user_prefs, advanced))


def _check_some_preference(user_prefs: Dict, advanced: bool) -> None:
    """Refuse to rank when every scored preference is NO_PREFERENCE (order would be by id only)."""
    if not active_preference_fields(user_prefs, advanced):
        raise ValueError("every preference is NO_PREFERENCE; at least one preference is needed to rank songs")


def _label_points(feature: str, song_label: str, wanted: object, weight: float) -> Tuple[float, str]:
    """
    Points and reason for a genre or mood preference.

    A single string behaves exactly as before. A list of labels earns the same single match
    contribution once if the song matches any of them (no double counting).
    """
    if wanted is NO_PREFERENCE:
        return 0.0, _no_preference_reason(feature)
    if isinstance(wanted, str):
        labels = [" ".join(wanted.split())]
    else:
        unique = {}
        for label in wanted:
            unique.setdefault(normalize_label(label), " ".join(label.split()))
        labels = list(unique.values())
    name = feature.capitalize()
    if normalize_label(song_label) in {normalize_label(label) for label in labels}:
        return weight, f"{name} match: {song_label} (+{weight:.2f})"
    return 0.0, f"{name} {song_label} is not {' or '.join(labels)} (+{0.0:.2f})"


def score_song(
    user_prefs: Dict, song: Dict, mode: str = DEFAULT_MODE, advanced: bool = False
) -> Tuple[float, List[str]]:
    """
    Score one song against a user profile with the given mode's weights; return (score, reasons).

    With advanced=True, the five advanced contributions are added on top of the same score.
    A NO_PREFERENCE field contributes 0 points to every song and says so in its reason.
    """
    weights = get_strategy(mode).weights()
    _check_advanced_flag(advanced)
    _validate_user_prefs(user_prefs)
    _validate_song(song)
    if advanced:
        preferred_tags = _validate_advanced_prefs(user_prefs)
        _validate_advanced_song(song)
    _check_some_preference(user_prefs, advanced)
    reasons: List[str] = []

    genre_points, reason = _label_points("genre", song["genre"], user_prefs["favorite_genre"], weights.genre)
    reasons.append(reason)

    mood_points, reason = _label_points("mood", song["mood"], user_prefs["favorite_mood"], weights.mood)
    reasons.append(reason)

    target = user_prefs["target_energy"]
    if target is NO_PREFERENCE:
        energy_points = 0.0
        reasons.append(_no_preference_reason("energy"))
    else:
        energy_similarity = max(0.0, 1.0 - abs(song["energy"] - target))
        energy_points = weights.energy * energy_similarity
        reasons.append(f"Energy {song['energy']:.2f} vs target {target:.2f} (+{energy_points:.2f})")

    if user_prefs["likes_acoustic"] is NO_PREFERENCE:
        acoustic_points = 0.0
        reasons.append(_no_preference_reason("acoustic"))
    elif user_prefs["likes_acoustic"]:
        acoustic_points = weights.acoustic * song["acousticness"]
        reasons.append(
            f"Acousticness {song['acousticness']:.2f}, you like acoustic (+{acoustic_points:.2f})"
        )
    else:
        acoustic_points = weights.acoustic * (1.0 - song["acousticness"])
        reasons.append(
            f"Acousticness {song['acousticness']:.2f}, you prefer less acoustic (+{acoustic_points:.2f})"
        )

    score = genre_points + mood_points + energy_points + acoustic_points
    if advanced:
        advanced_score, advanced_reasons = _advanced_points(user_prefs, song, preferred_tags)
        score += advanced_score
        reasons.extend(advanced_reasons)
    return score, reasons


def _validate_k(k: int) -> None:
    """Raise TypeError for a non-integer k and ValueError for a negative k."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k must be an integer, got {k!r}")
    if k < 0:
        raise ValueError(f"k must be 0 or greater, got {k}")


def _ranking_key(song: Dict, score: float) -> Tuple[float, int]:
    """Sort key: higher score first (rounded so float noise can't decide ties), then lower id."""
    song_id = song.get("id")
    if not isinstance(song_id, int) or isinstance(song_id, bool):
        raise ValueError(f"song id must be an integer, got {song_id!r}")
    return (-round(score, 6), song_id)


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = DEFAULT_MODE, advanced: bool = False
) -> List[Tuple[Dict, float, str]]:
    """Score every song with the chosen mode, rank by score (ties: lower id), and return the top k."""
    _validate_request(user_prefs, k, mode, advanced)
    # sorted() returns a new list, so the caller's catalog keeps its original order.
    scored = _score_all(user_prefs, songs, mode, advanced)
    ranked = sorted(scored, key=lambda item: _ranking_key(item[0], item[1]))
    return ranked[:k]


def _validate_request(user_prefs: Dict, k: int, mode: str, advanced: bool) -> None:
    """Validate a recommendation request up front, even if the catalog is empty."""
    _validate_k(k)
    get_strategy(mode)
    _check_advanced_flag(advanced)
    _validate_user_prefs(user_prefs)
    if advanced:
        _validate_advanced_prefs(user_prefs)
    _check_some_preference(user_prefs, advanced)


def _score_all(
    user_prefs: Dict, songs: List[Dict], mode: str, advanced: bool = False
) -> List[Tuple[Dict, float, str]]:
    """Score every song with score_song; return (song, score, explanation) in catalog order."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, mode=mode, advanced=advanced)
        scored.append((song, score, REASON_SEPARATOR.join(reasons)))
    return scored


# ---------------------------------------------------------------- diversity-aware reranking (optional)

# Design choices, not tuned or proven values. Change them here.
GENRE_REPEAT_PENALTY = 1.5   # subtracted once per already-selected song with the same genre
ARTIST_REPEAT_PENALTY = 2.0  # subtracted once if the artist has already been selected


@dataclass(frozen=True)
class DiversePick:
    """
    One diversity-reranked recommendation.

    base_score is the unmodified relevance score from score_song; adjusted_score is only
    used to choose the order and is never reported as relevance.
    """
    song: Dict
    base_score: float
    explanation: str          # score_song's reasons for base_score
    genre_repeats: int        # selected songs before this one with the same genre
    genre_penalty: float
    artist_repeat: bool       # artist already among the selected songs
    artist_penalty: float

    @property
    def adjusted_score(self) -> float:
        return self.base_score - self.genre_penalty - self.artist_penalty

    def penalty_explanation(self) -> str:
        """Plain-language account of the diversity penalties applied to this pick."""
        if not self.genre_penalty and not self.artist_penalty:
            return "No repetition penalty (adjusted score = relevance score)"
        parts = []
        if self.genre_penalty:
            parts.append(
                f"Genre {self.song['genre']} already picked {self.genre_repeats}x (-{self.genre_penalty:.2f})"
            )
        if self.artist_penalty:
            parts.append(f"Artist {self.song['artist']} already picked (-{self.artist_penalty:.2f})")
        parts.append(f"adjusted {self.adjusted_score:.2f} = relevance {self.base_score:.2f} minus penalties")
        return REASON_SEPARATOR.join(parts)


def _validate_penalty(name: str, value: float) -> None:
    if not _is_number(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")


def _diversity_labels(song: Dict) -> Tuple[str, str]:
    """Normalized (genre, artist) used to detect repeats."""
    for key in ["genre", "artist"]:
        if not isinstance(song.get(key), str):
            raise ValueError(f"song {key} must be a string, got {song.get(key)!r}")
    return normalize_label(song["genre"]), normalize_label(song["artist"])


def diversity_rerank(
    scored: List[Tuple[Dict, float, str]],
    k: int = 5,
    genre_penalty: Optional[float] = None,
    artist_penalty: Optional[float] = None,
) -> List[DiversePick]:
    """
    Greedily pick k songs from (song, score, explanation) tuples.

    At each step every remaining song gets
        adjusted = score - genre_penalty * (selected songs with its genre)
                         - artist_penalty * (1 if its artist was already selected else 0)
    and the highest adjusted score wins (ties: lower id). Penalties depend only on the songs
    already selected, so the first pick is always the most relevant song.
    Penalties default to the module constants, read at call time.
    """
    if genre_penalty is None:
        genre_penalty = GENRE_REPEAT_PENALTY
    if artist_penalty is None:
        artist_penalty = ARTIST_REPEAT_PENALTY
    _validate_k(k)
    _validate_penalty("genre_penalty", genre_penalty)
    _validate_penalty("artist_penalty", artist_penalty)

    remaining = list(scored)  # a copy: the caller's list and song dicts are never modified
    selected: List[DiversePick] = []
    genre_counts: Dict[str, int] = {}
    picked_artists = set()

    while remaining and len(selected) < k:
        candidates = []
        for song, base_score, explanation in remaining:
            genre, artist = _diversity_labels(song)
            repeats = genre_counts.get(genre, 0)
            artist_repeat = artist in picked_artists
            candidates.append(DiversePick(
                song=song,
                base_score=base_score,
                explanation=explanation,
                genre_repeats=repeats,
                genre_penalty=genre_penalty * repeats,
                artist_repeat=artist_repeat,
                artist_penalty=artist_penalty if artist_repeat else 0.0,
            ))
        best = min(range(len(candidates)),
                   key=lambda i: _ranking_key(candidates[i].song, candidates[i].adjusted_score))
        pick = candidates[best]
        selected.append(pick)
        del remaining[best]
        genre, artist = _diversity_labels(pick.song)
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        picked_artists.add(artist)
    return selected


def recommend_diverse_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = DEFAULT_MODE, advanced: bool = False
) -> List[DiversePick]:
    """Score every song with the chosen mode, then diversity-rerank the top k."""
    _validate_request(user_prefs, k, mode, advanced)
    return diversity_rerank(_score_all(user_prefs, songs, mode, advanced), k)
