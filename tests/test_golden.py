"""The original CLI and weight experiment must print exactly what they printed before the assistant existed."""

from pathlib import Path

import pytest

from experiments import weight_experiment
from src.main import main

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.parametrize("argv, filename", [
    ([], "main_default.txt"),
    (["--all-profiles"], "main_all_profiles.txt"),
    (["--mode", "mood-first"], "main_mode_mood_first.txt"),
    (["--advanced"], "main_advanced.txt"),
    (["--diverse", "--table"], "main_diverse_table.txt"),
    (["--advanced", "--diverse", "--table"], "main_advanced_diverse_table.txt"),
])
def test_original_cli_output_is_unchanged(capsys, argv, filename):
    main(argv)
    assert capsys.readouterr().out == (GOLDEN / filename).read_text(encoding="utf-8")


def test_weight_experiment_output_is_unchanged(capsys):
    weight_experiment.main()
    assert capsys.readouterr().out == (GOLDEN / "weight_experiment.txt").read_text(encoding="utf-8")
