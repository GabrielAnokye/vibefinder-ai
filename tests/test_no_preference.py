"""NO_PREFERENCE and multi-label preferences in the original engine (backward-compatible additions)."""

import copy
from dataclasses import asdict

import pytest

from src.main import DATA_PATH, PROFILES
from src.recommender import (
    NO_PREFERENCE, Recommender, Song, UserProfile, active_max_score, active_preference_fields,
    is_no_preference, load_songs, max_possible_score, recommend_diverse_songs, recommend_songs,
    score_song,
)

POP = PROFILES["High-Energy Pop"]
ADVANCED = {**POP, "target_popularity": 80, "preferred_decade": 2020, "target_instrumentalness": 0.1,
            "target_vocal_intensity": 0.7, "preferred_tags": ["uplifting", "euphoric"]}


@pytest.fixture(scope="module")
def catalog():
    return load_songs(str(DATA_PATH))


# ---------------------------------------------------------------- the sentinel itself

def test_sentinel_is_a_unique_object_not_a_string():
    assert is_no_preference(NO_PREFERENCE)
    assert repr(NO_PREFERENCE) == "NO_PREFERENCE"
    for lookalike in ["NO_PREFERENCE", "no preference", "any", "none", None, 0, False]:
        assert not is_no_preference(lookalike)


def test_sentinel_survives_dataclass_conversion_and_copies():
    profile = UserProfile(NO_PREFERENCE, "happy", NO_PREFERENCE, True)
    assert asdict(profile)["favorite_genre"] is NO_PREFERENCE
    assert copy.deepcopy(NO_PREFERENCE) is NO_PREFERENCE


def test_lookalike_strings_are_not_treated_as_the_sentinel(catalog):
    with pytest.raises(ValueError):
        score_song({**POP, "target_energy": "NO_PREFERENCE"}, catalog[0])
    with pytest.raises(ValueError):
        score_song({**POP, "likes_acoustic": "no preference"}, catalog[0])
    # As a genre, the string is an ordinary label that simply matches nothing.
    _, reasons = score_song({**POP, "favorite_genre": "NO_PREFERENCE"}, catalog[0])
    assert reasons[0] == "Genre pop is not NO_PREFERENCE (+0.00)"


# ---------------------------------------------------------------- original validation is unchanged

@pytest.mark.parametrize("key", ["favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"])
def test_none_and_missing_basic_fields_are_still_rejected(catalog, key):
    with pytest.raises(ValueError):
        score_song({**POP, key: None}, catalog[0])
    with pytest.raises(ValueError, match="missing required field"):
        score_song({k: v for k, v in POP.items() if k != key}, catalog[0])


@pytest.mark.parametrize("key", ["target_popularity", "preferred_decade", "target_instrumentalness",
                                 "target_vocal_intensity", "preferred_tags"])
def test_none_advanced_fields_are_still_missing(catalog, key):
    with pytest.raises(ValueError, match="missing"):
        score_song({**ADVANCED, key: None}, catalog[0], advanced=True)


def test_known_preferences_score_exactly_as_before(catalog):
    sunrise = catalog[0]
    score, reasons = score_song(POP, sunrise)
    assert score == pytest.approx(10.76, abs=0.005)
    assert reasons[0] == "Genre match: pop (+4.00)"


# ---------------------------------------------------------------- unknown fields contribute zero

BASIC_COMPONENT = {"favorite_genre": 0, "favorite_mood": 1, "target_energy": 2, "likes_acoustic": 3}
NO_PREF_REASON = {"favorite_genre": "No genre preference given (+0.00)",
                  "favorite_mood": "No mood preference given (+0.00)",
                  "target_energy": "No energy preference given (+0.00)",
                  "likes_acoustic": "No acoustic preference given (+0.00)"}


def _points(reason: str) -> float:
    return float(reason.rsplit("(+", 1)[1].rstrip(")"))


@pytest.mark.parametrize("key", list(BASIC_COMPONENT))
def test_unknown_basic_field_scores_zero_and_says_so(catalog, key):
    unknown = {**POP, key: NO_PREFERENCE}
    for song in catalog:
        full_score, full_reasons = score_song(POP, song)
        score, reasons = score_song(unknown, song)
        index = BASIC_COMPONENT[key]
        assert reasons[index] == NO_PREF_REASON[key]
        assert "match" not in reasons[index] and " is not " not in reasons[index]
        assert score == pytest.approx(full_score - _points(full_reasons[index]), abs=0.006)


@pytest.mark.parametrize("key, feature", [
    ("target_popularity", "popularity"), ("preferred_decade", "decade"),
    ("target_instrumentalness", "instrumentalness"), ("target_vocal_intensity", "vocal intensity"),
    ("preferred_tags", "mood-tag"),
])
def test_unknown_advanced_field_scores_zero_and_says_so(catalog, key, feature):
    unknown = {**ADVANCED, key: NO_PREFERENCE}
    for song in catalog:
        full_score, full_reasons = score_song(ADVANCED, song, advanced=True)
        score, reasons = score_song(unknown, song, advanced=True)
        removed = [r for r in full_reasons if r not in reasons]
        assert f"No {feature} preference given (+0.00)" in reasons
        assert score == pytest.approx(full_score - sum(_points(r) for r in removed), abs=0.006)


def test_unknown_field_does_not_change_relative_ranking(catalog):
    """Ranking with genre unknown equals ranking by the score minus the genre component."""
    unknown = {**POP, "favorite_genre": NO_PREFERENCE}
    ranked = [s["id"] for s, _, _ in recommend_songs(unknown, catalog, k=len(catalog))]
    expected = sorted(catalog, key=lambda s: (-round(score_song(POP, s)[0] - _points(score_song(POP, s)[1][0]), 6), s["id"]))
    assert ranked == [s["id"] for s in expected]


def test_all_unknown_preferences_are_refused(catalog):
    nothing = {k: NO_PREFERENCE for k in POP}
    with pytest.raises(ValueError, match="at least one preference"):
        recommend_songs(nothing, catalog)
    with pytest.raises(ValueError, match="at least one preference"):
        score_song(nothing, catalog[0])
    with pytest.raises(ValueError, match="at least one preference"):
        recommend_diverse_songs(nothing, catalog)
    # One known advanced preference is enough when advanced scoring is on.
    only_tags = {**nothing, **{k: NO_PREFERENCE for k in ADVANCED if k not in POP}, "preferred_tags": ["dreamy"]}
    assert recommend_songs(only_tags, catalog, k=3, advanced=True)


def test_active_max_score_distinguishes_partial_profiles():
    assert active_max_score(POP) == max_possible_score() == 11.0
    assert active_max_score({**POP, "favorite_genre": NO_PREFERENCE}) == 7.0
    assert active_max_score({**POP, "favorite_genre": NO_PREFERENCE}, mode="energy-focused") == 10.0
    partial_adv = {**ADVANCED, "target_popularity": NO_PREFERENCE, "preferred_decade": NO_PREFERENCE}
    assert active_max_score(partial_adv, advanced=True) == pytest.approx(11.0 + 2.0)
    assert active_preference_fields({**POP, "likes_acoustic": NO_PREFERENCE}) == [
        "favorite_genre", "favorite_mood", "target_energy"]


def test_oop_interface_and_diversity_accept_the_sentinel(catalog):
    songs = [Song(**s) for s in catalog]
    user = UserProfile(NO_PREFERENCE, "chill", 0.4, NO_PREFERENCE)
    top = Recommender(songs).recommend(user, k=3)
    assert all(song.mood == "chill" for song in top)
    assert "No genre preference given" in Recommender(songs).explain_recommendation(user, top[0])
    picks = recommend_diverse_songs(asdict(user), catalog, k=5)
    assert len(picks) == 5


# ---------------------------------------------------------------- multiple preferred labels

def test_multiple_genres_earn_the_single_genre_contribution_once(catalog):
    prefs = {**POP, "favorite_genre": ["rock", "hip-hop"]}
    by_id = {s["id"]: s for s in catalog}
    rock_score, rock_reasons = score_song(prefs, by_id[3])
    hiphop_score, hiphop_reasons = score_song(prefs, by_id[12])
    _, pop_reasons = score_song(prefs, by_id[1])
    assert rock_reasons[0] == "Genre match: rock (+4.00)"
    assert hiphop_reasons[0] == "Genre match: hip-hop (+4.00)"
    assert pop_reasons[0] == "Genre pop is not rock or hip-hop (+0.00)"
    single, _ = score_song({**POP, "favorite_genre": "rock"}, by_id[3])
    assert rock_score == pytest.approx(single)


def test_duplicate_labels_do_not_double_count(catalog):
    a, _ = score_song({**POP, "favorite_genre": ["rock", "Rock", " rock "]}, catalog[2])
    b, _ = score_song({**POP, "favorite_genre": "rock"}, catalog[2])
    assert a == pytest.approx(b)


def test_multiple_moods_work_the_same_way(catalog):
    prefs = {**POP, "favorite_mood": ("chill", "relaxed")}
    moods = {s["mood"] for s, _, _ in recommend_songs(prefs, catalog, k=5, mode="mood-first")[:3]}
    assert moods <= {"chill", "relaxed"}


@pytest.mark.parametrize("bad", [[], ["rock", ""], ["rock", 7], ("  ",), {"rock"}])
def test_invalid_label_lists_are_rejected(catalog, bad):
    with pytest.raises(ValueError):
        score_song({**POP, "favorite_genre": bad}, catalog[0])
