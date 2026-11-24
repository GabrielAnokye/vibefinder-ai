"""Evaluation harness, committed traces, experiment scripts, and the quality floor."""

import json
import urllib.request

import pytest

from evaluation import generate_traces, live_experiments, retrieval_comparison, run_evaluation
from evaluation.run_evaluation import evaluate_case, load_cases, summarize
from src.assistant import Assistant, MockProvider
from src.assistant.planner import satisfies
from src.assistant.providers import template_explanation


@pytest.fixture(scope="module")
def assistant():
    return Assistant()


# ---------------------------------------------------------------- cases and metrics

def test_cases_load_and_cover_the_required_scenarios():
    cases = load_cases()
    ids = {c["id"] for c in cases}
    assert len(cases) >= 14
    assert {"study-calm", "workout-energetic", "nostalgic-acoustic", "multiple-genres", "only-rock", "exclude-pop",
            "unique-artists", "exactly-three", "only-metal", "conflicting-hard", "acoustic-high-energy", "empty",
            "injection", "simulated-invalid-then-corrected"} <= ids
    assert all(c["id"].startswith("simulated-") for c in cases if "mock" in c)


def test_unknown_expectation_keys_are_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"id": "x", "request": "calm music", "expect": {"vibes": 1}}]}))
    with pytest.raises(ValueError, match="unknown expectations"):
        load_cases(path)


def test_pass_rates_are_calculated_from_checks():
    results = [
        {"id": "a", "required": True, "simulated": False, "passed": True,
         "checks": [{"passed": True}, {"passed": True}]},
        {"id": "b", "required": True, "simulated": True, "passed": False,
         "checks": [{"passed": True}, {"passed": False}, {"passed": False}]},
    ]
    s = summarize(results)
    assert (s["cases_passed"], s["case_pass_rate"]) == (1, 0.5)
    assert (s["checks_passed"], s["checks"], s["check_pass_rate"]) == (3, 5, 0.6)
    assert s["required_failures"] == ["b"] and s["simulated_cases"] == ["b"]


def test_failed_checks_are_reported_with_details(assistant, capsys):
    case = {"id": "wrong", "request": "Only rock songs please.", "expect": {"count": 5, "genres_only": ["pop"]}}
    result = evaluate_case(assistant, case)
    failed = {c["name"]: c["detail"] for c in result["checks"] if not c["passed"]}
    assert not result["passed"] and set(failed) == {"count", "genres_only"}
    assert "3 results" in failed["count"]
    run_evaluation.print_report("t", [result], summarize([result]))
    assert "FAIL  wrong" in capsys.readouterr().out


def test_a_crash_is_a_failed_case_not_a_crashed_harness(assistant, monkeypatch):
    monkeypatch.setattr(Assistant, "ask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = evaluate_case(assistant, {"id": "c", "request": "x", "expect": {"status": "ok"}})
    assert result["status"] == "crash" and result["checks"][0] == {"name": "no_crash", "passed": False,
                                                                   "detail": "RuntimeError: boom"}


def test_harness_calls_the_integrated_assistant(monkeypatch):
    calls = []
    original = Assistant.ask

    def spy(self, text, **kwargs):
        calls.append(text)
        return original(self, text, **kwargs)
    monkeypatch.setattr(Assistant, "ask", spy)
    cases = load_cases()[:3]
    run_evaluation.run(cases)
    assert calls == [c["request"] for c in cases]


def test_required_failure_gives_nonzero_exit(tmp_path, capsys):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [{"id": "impossible", "request": "Only rock songs please.",
                                           "expect": {"genres_only": ["jazz"]}}]}))
    assert run_evaluation.main(["--cases", str(path)]) == 1
    assert "Required failures: impossible" in capsys.readouterr().out


def test_full_offline_evaluation_passes_and_writes_json(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "results.json"
    assert run_evaluation.main(["--json", str(out), "--live"]) == 0
    data = json.loads(out.read_text())
    assert data["offline"]["summary"]["required_failures"] == []
    assert data["live"]["status"] == "pending"
    assert "Live evaluation pending" in capsys.readouterr().out


# ---------------------------------------------------------------- committed traces

def test_committed_traces_are_reproducible_and_labeled():
    for name, text in generate_traces.build_traces().items():
        committed = (generate_traces.TRACE_DIR / name).read_text(encoding="utf-8")
        assert committed == text, f"{name} is out of date; run python -m evaluation.generate_traces"
        metadata = json.loads(text)["metadata"]
        if name.startswith("simulated_"):
            assert metadata["provider_kind"] == "simulated" and "SIMULATED" in metadata["note"]
        else:
            assert metadata["provider_kind"] == "template"


# ---------------------------------------------------------------- experiments

def test_retrieval_turns_a_context_only_request_into_a_plan(assistant):
    row = retrieval_comparison.compare_on_off(assistant)[0]
    assert row["request"] == "Music for studying."
    assert row["on"]["status"] == "ok" and row["off"]["status"] == "clarification"
    assert any("listening_contexts#studying" in v for v in row["on"]["inferred"].values())


def test_explicit_preference_survives_retrieval(assistant):
    row = next(r for r in retrieval_comparison.compare_on_off(assistant) if "energetic" in r["request"])
    assert row["on"]["explicit"]["energy"] == row["off"]["explicit"]["energy"] == "0.85"
    assert row["on"]["hints_not_used"]["energy"] == "your explicit value was kept"


def test_metrics_on_valid_and_invalid_answers(assistant):
    payload = assistant.ask("Recommend acoustic music with very high energy.").payload
    good = live_experiments.explanation_metrics(json.dumps(template_explanation(payload)), payload, assistant.songs)
    assert good["schema_valid"] and good["coverage"] == 1.0 and good["unmet_disclosed (heuristic)"] == 1.0
    bad = live_experiments.explanation_metrics("nonsense", payload, assistant.songs)
    assert not bad["valid_json"] and not bad["schema_valid"]


def test_prompt_comparison_holds_everything_but_the_prompt_fixed(assistant):
    mock = MockProvider([lambda req: json.dumps(template_explanation(req.payload))] * 6)
    result = live_experiments.run_experiment("prompts", mock, assistant)
    assert result["simulated"] and len(result["runs"]) == 6
    for general, specialized in zip(mock.calls[0::2], mock.calls[1::2]):
        assert (general.style, specialized.style) == ("general", "specialized")
        assert general.payload == specialized.payload and not specialized.is_correction


def test_retrieval_generation_comparison_withholds_only_the_sources(assistant):
    mock = MockProvider([lambda req: json.dumps(template_explanation(req.payload))] * 6)
    live_experiments.run_experiment("retrieval", mock, assistant)
    supplied, withheld = mock.calls[0].payload, mock.calls[1].payload
    assert supplied["sources"] and withheld["sources"] == []
    assert {k: v for k, v in supplied.items() if k != "sources"} == {k: v for k, v in withheld.items() if k != "sources"}


def test_live_runner_makes_no_request_without_configuration(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert live_experiments.main(["prompts", "--yes"]) == 0
    assert "pending" in capsys.readouterr().out


def test_live_runner_requires_confirmation(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/model")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert live_experiments.main(["smoke"]) == 0
    assert "add --yes" in capsys.readouterr().out


# ---------------------------------------------------------------- quality floor and tolerances

def test_floor_removes_zero_point_results(assistant):
    no_floor = Assistant(quality_floor=0.0).ask("Give me 10 happy songs")
    assert len(no_floor.recommendations) == 10 and no_floor.recommendations[-1].score == 0
    default = assistant.ask("Give me 10 happy songs")
    assert len(default.recommendations) == 3 and all(x.song["mood"] == "happy" for x in default.recommendations)
    assert any("Returned 3 of the 10" in item for item in default.plan.limitations)


def test_tolerances_are_inclusive_at_the_boundary(assistant):
    island = next(s for s in assistant.songs if s["title"] == "Island Time Radio")   # energy 0.55
    assert satisfies(island, "energy", 0.35)        # gap exactly 0.20
    assert not satisfies(island, "energy", 0.30)    # gap 0.25
