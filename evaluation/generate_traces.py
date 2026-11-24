"""
Regenerate the committed demonstration traces in evaluation/traces/.

Offline traces use the deterministic template provider. Traces whose name starts with
"simulated_" use a scripted MockProvider to show the correction and fallback paths; they are
NOT live model executions, and their metadata says so.

    python -m evaluation.generate_traces           # rewrite the files
    python -m evaluation.generate_traces --check   # exit 1 if a committed trace is out of date
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.run_evaluation import mock_for
from src.assistant import Assistant

TRACE_DIR = Path(__file__).parent / "traces"
SIMULATED_NOTE = ("SIMULATED provider: responses were scripted by evaluation/generate_traces.py to exercise "
                  "validation, correction and fallback. This is not a live model execution.")

DEMOS: List[Dict] = [
    {"file": "offline_study_success.json", "request": "I'm studying for finals. Recommend calm instrumental music with some variety."},
    {"file": "offline_workout_multi_genre.json", "request": "I need energetic music for a workout, preferably rock or hip-hop."},
    {"file": "offline_unique_artists.json", "request": "Chill lofi music, no repeated artists."},
    {"file": "offline_no_match_only_metal.json", "request": "Only recommend metal songs."},
    {"file": "offline_poor_coverage_acoustic_energy.json", "request": "Recommend acoustic music with very high energy."},
    {"file": "simulated_invalid_then_corrected.json", "request": "Music for studying.", "mock": ["bad_citation", "valid"]},
    {"file": "simulated_invalid_then_fallback.json", "request": "Music for studying.", "mock": ["unknown_song", "invalid_json"]},
]


def build_traces() -> Dict[str, str]:
    assistant = Assistant()
    traces = {}
    for demo in DEMOS:
        provider = mock_for(demo["mock"]) if "mock" in demo else None
        response = assistant.ask(demo["request"], provider=provider)
        if "mock" in demo:
            response.trace.metadata["note"] = SIMULATED_NOTE
            response.trace.metadata["mock_script"] = demo["mock"]
        traces[demo["file"]] = response.trace.to_json()
    return traces


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--check", action="store_true", help="verify committed traces instead of writing")
    args = parser.parse_args(argv)
    traces = build_traces()
    if args.check:
        stale = [name for name, text in traces.items()
                 if not (TRACE_DIR / name).exists() or (TRACE_DIR / name).read_text(encoding="utf-8") != text]
        print("All traces up to date." if not stale else f"Out of date: {', '.join(stale)}")
        return 1 if stale else 0
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in traces.items():
        (TRACE_DIR / name).write_text(text, encoding="utf-8")
        print(f"wrote {TRACE_DIR / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
