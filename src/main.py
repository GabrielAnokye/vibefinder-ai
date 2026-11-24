"""
Command line runner for the Music Recommender Simulation.

Run from the project root with:
    python -m src.main                  # default High-Energy Pop profile
    python -m src.main --all-profiles   # all four Phase 4 evaluation profiles
    python -m src.main --mode mood-first             # pick a ranking mode
    python -m src.main --all-profiles --mode energy-focused
    python -m src.main --diverse                     # diversity-aware reranking
    python -m src.main --table                       # ASCII table output
    python -m src.main --advanced                    # add the five advanced song features

VibeFinder AI assistant (natural-language requests):
    python -m src.main --ask "Recommend calm music for studying"
    python -m src.main --ask "..." --show-plan       # show the structured plan and steps
    python -m src.main --ask "..." --no-retrieval    # disable the knowledge guides
    python -m src.main --ask "..." --provider gemini       # optional LLM explanations
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from src import recommender
from src.recommender import (
    DEFAULT_MODE,
    STRATEGIES,
    get_strategy,
    load_songs,
    max_possible_score,
    recommend_diverse_songs,
    recommend_songs,
)
from src.table import diverse_table, recommendations_table

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
TOP_K = 5

# Hypothetical evaluation profiles (Phase 4). These are not real listener data.
PROFILES: Dict[str, Dict] = {
    "High-Energy Pop": {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.80,
        "likes_acoustic": False,
    },
    "Chill Lofi": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.40,
        "likes_acoustic": True,
    },
    "Intense Rock": {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.90,
        "likes_acoustic": False,
    },
    # Deliberately conflicting: a calm genre and a sad mood with a high energy target.
    "Adversarial Listener": {
        "favorite_genre": "lofi",
        "favorite_mood": "sad",
        "target_energy": 0.90,
        "likes_acoustic": False,
    },
}
DEFAULT_PROFILE = "High-Energy Pop"

# Example advanced preferences (bonus challenge 1), also hypothetical. Kept separate so the
# basic profiles and the default output stay exactly as they were; merged only with --advanced.
ADVANCED_PROFILE_PREFS: Dict[str, Dict] = {
    "High-Energy Pop": {
        "target_popularity": 80,
        "preferred_decade": 2020,
        "target_instrumentalness": 0.10,
        "target_vocal_intensity": 0.70,
        "preferred_tags": ["uplifting", "euphoric"],
    },
    "Chill Lofi": {
        "target_popularity": 50,
        "preferred_decade": 2020,
        "target_instrumentalness": 0.85,
        "target_vocal_intensity": 0.05,
        "preferred_tags": ["focused", "dreamy"],
    },
    "Intense Rock": {
        "target_popularity": 60,
        "preferred_decade": 2000,
        "target_instrumentalness": 0.10,
        "target_vocal_intensity": 0.85,
        "preferred_tags": ["aggressive", "energetic"],
    },
    # Still conflicting: melancholic but energetic.
    "Adversarial Listener": {
        "target_popularity": 30,
        "preferred_decade": 1990,
        "target_instrumentalness": 0.20,
        "target_vocal_intensity": 0.40,
        "preferred_tags": ["melancholic", "energetic"],
    },
}


def profile_prefs(name: str, advanced: bool = False) -> Dict:
    """A profile's preferences, with its example advanced preferences added when requested."""
    if advanced:
        return {**PROFILES[name], **ADVANCED_PROFILE_PREFS[name]}
    return PROFILES[name]


def print_recommendations(
    user_prefs: Dict, songs: List[Dict], k: int = TOP_K, mode: str = DEFAULT_MODE,
    diverse: bool = False, table: bool = False, advanced: bool = False,
) -> None:
    """Print a profile's preferences and its top-k recommendations with scores and reasons."""
    print("\nUser preferences:")
    for key, value in user_prefs.items():
        print(f"  {key}: {value}")

    max_score = max_possible_score(mode, advanced)

    if diverse:
        picks = recommend_diverse_songs(user_prefs, songs, k=k, mode=mode, advanced=advanced)
        if table:
            print("\nTop recommendations (diversity-reranked):\n")
            print(diverse_table(picks, max_score))
            return
        print("\nTop recommendations (diversity-reranked):\n")
        for rank, pick in enumerate(picks, start=1):
            song = pick.song
            print(f"{rank}. {song['title']} by {song['artist']}")
            print(f"   Genre: {song['genre']} | Mood: {song['mood']} | "
                  f"Relevance score: {pick.base_score:.2f} | Adjusted for diversity: {pick.adjusted_score:.2f}")
            print(f"   Because: {pick.explanation}")
            print(f"   Diversity: {pick.penalty_explanation()}")
            print()
        return

    recommendations = recommend_songs(user_prefs, songs, k=k, mode=mode, advanced=advanced)

    if table:
        print("\nTop recommendations:\n")
        print(recommendations_table(recommendations, max_score))
        return

    print("\nTop recommendations:\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']} by {song['artist']}")
        print(f"   Genre: {song['genre']} | Mood: {song['mood']} | Score: {score:.2f}")
        print(f"   Because: {explanation}")
        print()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    from src.assistant.providers import LIVE_PROVIDERS

    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="show recommendations for all four evaluation profiles",
    )
    parser.add_argument(
        "--mode",
        choices=list(STRATEGIES),
        default=None,  # resolved to DEFAULT_MODE later, so --ask can tell whether it was given
        help=f"ranking mode (default: {DEFAULT_MODE}; with --ask, overrides the assistant's choice)",
    )
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="rerank to reduce repeated genres and artists in the top results",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="show results as an ASCII table with the score breakdown",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="also score popularity, release decade, instrumentalness, vocal intensity, and mood tags",
    )
    assistant = parser.add_argument_group("VibeFinder AI assistant")
    assistant.add_argument("--ask", metavar="REQUEST", help="describe what you want to hear in plain language")
    assistant.add_argument(
        "--provider", choices=["template", *LIVE_PROVIDERS], default=None,
        help="who writes the explanation prose (default: template, offline)",
    )
    assistant.add_argument("--no-retrieval", action="store_true", help="do not use the knowledge guides")
    assistant.add_argument("--show-plan", action="store_true",
                           help="show the structured plan, all engine reasons, and the trace summary")
    assistant.add_argument(
        "--prompt-style", choices=["general", "specialized"], default=None,
        help="prompt for live providers (default: specialized); ignored by the template provider",
    )
    assistant.add_argument("--trace", action="store_true", help="print the execution trace after the answer")
    assistant.add_argument("--trace-file", metavar="PATH", help="write the execution trace as JSON to PATH")
    args = parser.parse_args(argv)
    if args.ask is None:
        used = [flag for flag, value in (("--provider", args.provider), ("--no-retrieval", args.no_retrieval),
                                         ("--show-plan", args.show_plan), ("--prompt-style", args.prompt_style),
                                         ("--trace", args.trace), ("--trace-file", args.trace_file)) if value]
        if used:
            parser.error(f"{', '.join(used)} can only be used with --ask")
        args.mode = args.mode or DEFAULT_MODE
    else:
        if args.all_profiles:
            parser.error("--all-profiles cannot be combined with --ask")
        if args.advanced:
            parser.error("--advanced cannot be combined with --ask; the assistant enables advanced "
                         "scoring when your request mentions an advanced feature (use --show-plan to see it)")
    return args


def run_assistant_cli(args: argparse.Namespace) -> int:
    """--ask: run the natural-language assistant. --mode and --diverse override its choices."""
    from src.assistant import Assistant, TemplateProvider, make_provider, render
    from src.assistant.providers import DEFAULT_PROMPT_STYLE
    from src.assistant.trace import redact

    provider, note = make_provider(args.provider or "template")
    if args.prompt_style and isinstance(provider, TemplateProvider):
        print(f"Note: --prompt-style {args.prompt_style} has no effect: the template provider is not an LLM "
              "and does not use prompts.")
    response = Assistant().ask(
        args.ask, provider=provider, retrieval=not args.no_retrieval,
        mode=args.mode, diverse=args.diverse, provider_note=note,
        prompt_style=args.prompt_style or DEFAULT_PROMPT_STYLE,
    )
    print(redact(render(response, show_plan=args.show_plan, table=args.table)))
    if args.trace:
        print("\nExecution trace:")
        print(redact("\n".join(f"  {line}" for line in response.trace.summary_lines())))
    if args.trace_file:
        path = Path(args.trace_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.trace.to_json(), encoding="utf-8")
        print(f"\nTrace written to {path}")
    return 1 if response.status == "error" else 0


def describe_mode(mode: str) -> str:
    """One-line summary of a ranking mode and its weights."""
    strategy = get_strategy(mode)
    w = strategy.weights()
    return (
        f"Ranking mode: {strategy.name} ({strategy.description}) - genre {w.genre:g}, "
        f"mood {w.mood:g}, energy up to {w.energy:g}, acoustic up to {w.acoustic:g}; max {w.max_score:g}"
    )


def describe_advanced(mode: str) -> str:
    """Explain where the advanced points come from."""
    r = recommender
    return (
        f"Advanced scoring: on - adds up to {r.ADVANCED_MAX_SCORE:g} points: popularity closeness "
        f"(up to {r.POPULARITY_WEIGHT:g}), decade match ({r.DECADE_WEIGHT:g}), instrumentalness closeness "
        f"(up to {r.INSTRUMENTALNESS_WEIGHT:g}), vocal intensity closeness (up to {r.VOCAL_INTENSITY_WEIGHT:g}), "
        f"mood-tag overlap (up to {r.MOOD_TAG_WEIGHT:g}); max {max_possible_score(mode, advanced=True):g}"
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Load the catalog and print the top 5 for the default profile, or for every profile."""
    args = parse_args(argv)
    if args.ask is not None:
        status = run_assistant_cli(args)
        if status:
            raise SystemExit(status)
        return
    songs = load_songs(str(DATA_PATH))
    print(f"Loaded songs: {len(songs)}")
    # The default (baseline) output is kept exactly as it was before ranking modes existed.
    if args.mode != DEFAULT_MODE:
        print(describe_mode(args.mode))
    if args.advanced:
        print(describe_advanced(args.mode))
    if args.diverse:
        print(
            f"Diversity reranking: on (-{recommender.GENRE_REPEAT_PENALTY:g} per earlier pick of the same genre, "
            f"-{recommender.ARTIST_REPEAT_PENALTY:g} if the artist was already picked)"
        )

    options = dict(mode=args.mode, diverse=args.diverse, table=args.table, advanced=args.advanced)
    if not args.all_profiles:
        print_recommendations(profile_prefs(DEFAULT_PROFILE, args.advanced), songs, **options)
        return

    for name in PROFILES:
        print()
        print("=" * 70)
        print(f"Profile: {name}")
        print("=" * 70)
        print_recommendations(profile_prefs(name, args.advanced), songs, **options)


if __name__ == "__main__":
    main()
