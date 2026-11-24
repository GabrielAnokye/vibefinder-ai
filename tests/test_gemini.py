"""Gemini Developer API contract and workflow tests with an in-process HTTP opener."""

import json
import urllib.error
from io import BytesIO

import pytest

from evaluation import live_experiments, run_evaluation
from src.assistant import Assistant, GeminiProvider, TemplateProvider, make_provider
from src.assistant.providers import (
    EXPLANATION_JSON_SCHEMA, AttemptBudget, GenerationRequest, ProviderError, system_prompt, template_explanation,
)
from src.main import main as assistant_cli

REQUEST = "Music for studying"
MODEL = "gemini-3.5-flash-lite"
SERVED_MODEL = "gemini-3.5-flash-lite-001"
SECRET = "fake-gemini-key-for-tests"


def candidate(text, *, finish="STOP", model=SERVED_MODEL):
    return {"modelVersion": model, "candidates": [
        {"finishReason": finish, "content": {"parts": [{"text": text}]}},
    ]}


def accepted(body):
    payload = json.loads(body["contents"][0]["parts"][0]["text"])
    return candidate(json.dumps(template_explanation(payload)))


def http_error(code, **info):
    body = json.dumps({"error": {"message": f"upstream diagnostic {SECRET}", **info}}).encode()
    return urllib.error.HTTPError("https://generativelanguage.googleapis.com", code, "Gemini error", {}, BytesIO(body))


def make(outcomes, model=MODEL):
    pending = iter(outcomes)
    calls = []

    def opener(request, timeout):
        body = json.loads(request.data)
        calls.append((request, body, timeout))
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            outcome = outcome(body)
        return BytesIO(json.dumps(outcome).encode())

    return GeminiProvider(SECRET, model, opener=opener, sleep=lambda _: None), calls


def configure(monkeypatch, outcomes):
    provider, calls = make(outcomes)
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("GEMINI_MODEL", MODEL)
    monkeypatch.setattr(GeminiProvider, "from_env", classmethod(lambda cls, env=None: (provider, "")))
    return calls


@pytest.fixture(scope="module")
def assistant():
    return Assistant()


def test_configuration_requires_both_values_without_a_model_default():
    provider, note = make_provider("gemini", {})
    assert isinstance(provider, TemplateProvider)
    assert "Gemini not configured" in note and "GEMINI_API_KEY and GEMINI_MODEL" in note
    provider, note = make_provider("gemini", {"GEMINI_API_KEY": SECRET, "GEMINI_MODEL": f"models/{MODEL}"})
    assert isinstance(provider, GeminiProvider) and not note and provider.model == MODEL
    provider, note = make_provider("gemini", {"GEMINI_API_KEY": SECRET, "GEMINI_MODEL": "gemini-x?key=secret"})
    assert isinstance(provider, TemplateProvider) and "model ID" in note and "key=secret" not in note


@pytest.mark.parametrize("model,thinking", [
    ("gemini-2.5-flash", {"thinkingBudget": 0}),
    (MODEL, {"thinkingLevel": "MINIMAL"}),
    ("gemini-future", None),
])
def test_native_request_reuses_prompt_and_requests_bounded_structured_output(model, thinking):
    first = '{"summary":'
    rest = '"Plain text."}'
    response = candidate(first, model=f"{model}-001")
    response["candidates"][0]["content"]["parts"] += [
        {"thought": True, "text": "private thought"},
        {"text": rest},
    ]
    provider, calls = make([response], model)
    request = GenerationRequest({"results": []}, "general")
    assert provider.generate(request) == first + rest
    assert request.response_metadata == {"model": f"{model}-001", "finish_reason": "STOP"}
    sent, body, timeout = calls[0]
    assert sent.full_url == f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    assert sent.get_method() == "POST"
    assert sent.get_header("X-goog-api-key") == SECRET
    assert SECRET not in sent.full_url and SECRET.encode() not in sent.data and SECRET not in repr(provider)
    assert body["systemInstruction"]["parts"][0]["text"] == system_prompt("general")
    assert body["contents"] == [{"role": "user", "parts": [{"text": '{"results": []}'}]}]
    config = body["generationConfig"]
    assert config["responseJsonSchema"] == EXPLANATION_JSON_SCHEMA
    assert config["responseMimeType"] == "application/json"
    assert config["maxOutputTokens"] == 4096 and 0 < timeout <= 60
    assert config.get("thinkingConfig") == thinking and "temperature" not in config


@pytest.mark.parametrize("outcomes,kind,attempts", [
    ([http_error(401)], "auth", 1),
    ([http_error(400, status="INVALID_ARGUMENT", details=[{"reason": "API_KEY_INVALID"}])], "auth", 1),
    ([http_error(404)], "model_unavailable", 1),
    ([http_error(502, status="NOT_FOUND")], "model_unavailable", 1),
    ([http_error(402)], "quota", 1),
    ([http_error(429, details=[{"violations": [{"quotaId": "GenerateRequestsPerDay", "quotaValue": "10"}]}])],
     "quota", 1),
    ([http_error(429), http_error(429)], "rate_limit", 2),
    ([{"promptFeedback": {"blockReason": "SAFETY"}}], "blocked", 1),
    ([candidate("blocked content", finish="SAFETY")], "blocked", 1),
    ([candidate('{"summary":', finish="MAX_TOKENS")], "truncated", 1),
    ([candidate('{"summary":"looks complete"}', finish="MAX_TOKENS")], "truncated", 1),
    ([{}], "invalid_response", 1),
])
def test_gemini_failures_are_classified_and_keys_are_protected(outcomes, kind, attempts):
    provider, calls = make(outcomes)
    budget = AttemptBudget()
    with pytest.raises(ProviderError) as raised:
        provider.generate(GenerationRequest({}), budget)
    assert raised.value.kind == kind and budget.used == len(calls) == attempts
    assert SECRET not in str(raised.value)
    if kind == "truncated":
        assert raised.value.partial_output.startswith('{"summary":')
    if kind == "blocked":
        assert raised.value.partial_output is None


@pytest.mark.parametrize("corrects", [True, False])
def test_malformed_json_uses_workflow_correction_and_fallback(assistant, corrects):
    provider, calls = make([candidate("bad JSON"), accepted if corrects else candidate("still malformed")])
    response = assistant.ask(REQUEST, provider=provider)
    initial, retry = [call[1] for call in calls]
    assert initial["systemInstruction"] == retry["systemInstruction"]
    assert initial["generationConfig"] == retry["generationConfig"]
    assert retry["contents"][0] == initial["contents"][0]
    assert retry["contents"][1] == {"role": "model", "parts": [{"text": "bad JSON"}]}
    assert "not valid JSON" in retry["contents"][2]["parts"][0]["text"]
    assert response.http_attempts == 2 and response.explanation
    assert response.generator == (provider.label if corrects else TemplateProvider.label)
    assert response.trace.stages().count("correct_explanation") == 1


def test_rate_retry_and_correction_share_hard_three_attempt_limit(assistant):
    provider, calls = make([http_error(429), candidate("bad JSON"), http_error(503)])
    response = assistant.ask(REQUEST, provider=provider, max_http_attempts=10)
    assert len(calls) == response.http_attempts == response.trace.metadata["max_http_attempts"] == 3
    assert calls[0][1] == calls[1][1] and len(calls[2][1]["contents"]) == 3
    assert response.generator == TemplateProvider.label
    assert response.trace.stages().count("correct_explanation") == 1


def test_cli_selects_gemini_and_redacts_final_output_and_trace(monkeypatch, tmp_path, capsys):
    def echoed_key(body):
        result = accepted(body)
        turn = result["candidates"][0]["content"]["parts"][0]
        generated = json.loads(turn["text"])
        generated["summary"] = f"Study music. {SECRET}"
        turn["text"] = json.dumps(generated)
        return result

    calls = configure(monkeypatch, [echoed_key])
    trace_file = tmp_path / "trace.json"
    assistant_cli(["--ask", REQUEST, "--provider", "gemini", "--trace", "--trace-file", str(trace_file)])
    output = capsys.readouterr().out
    raw_trace = trace_file.read_text()
    assert len(calls) == 1 and f"Explanation generated by: gemini:{MODEL} (LLM)" in output
    assert SECRET not in output and SECRET not in raw_trace and "[redacted]" in output
    assert SERVED_MODEL in raw_trace


def test_gemini_rejects_a_provider_side_model_switch(assistant):
    provider, calls = make([candidate("{}", model="gemini-3.5-flash-lite")], model="gemini-2.5-flash")
    response = assistant.ask(REQUEST, provider=provider)
    assert len(calls) == 1 and response.generator == TemplateProvider.label
    assert "model_mismatch" in response.generation_status
    assert any(event.result.get("response_metadata", {}).get("model") == MODEL for event in response.trace.events)


def test_gemini_experiments_require_approval_and_save_each_arm(monkeypatch, tmp_path, capsys):
    calls = configure(monkeypatch, [accepted] * 13)
    monkeypatch.setattr(live_experiments, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(live_experiments, "TRACE_DIR", tmp_path / "traces")
    assert live_experiments.main(["all", "--provider", "gemini"]) == 0 and not calls
    assert "add --yes" in capsys.readouterr().out
    assert live_experiments.main(["all", "--provider", "gemini", "--yes"]) == 0
    assert len(calls) == 13
    assert len({req.full_url for req, _, _ in calls}) == 1
    assert all(body["generationConfig"] == calls[0][1]["generationConfig"] for _, body, _ in calls)
    for name, count in (("smoke", 1), ("retrieval", 6), ("prompts", 6)):
        file = f"live_gemini_{name}.json"
        result = json.loads((tmp_path / "results" / file).read_text())
        trace_file = tmp_path / "traces" / file
        assert trace_file.exists() and SECRET not in trace_file.read_text()
        assert result["reported_models"] == [SERVED_MODEL] and result["requested_model"] == MODEL
        assert result["logical_requests"] == result["http_attempts"] == count
        assert all(run["outcome"].endswith("_validated") for run in result["runs"])


def test_truncated_smoke_keeps_sanitized_partial_output_and_stops_all(monkeypatch, tmp_path):
    partial = '{"summary":"prefix ' + SECRET
    calls = configure(monkeypatch, [candidate(partial, finish="MAX_TOKENS")])
    monkeypatch.setattr(live_experiments, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(live_experiments, "TRACE_DIR", tmp_path / "traces")
    assert live_experiments.main(["all", "--provider", "gemini", "--yes"]) == 1
    saved = (tmp_path / "results/live_gemini_smoke.json").read_text()
    run = json.loads(saved)["runs"][0]
    assert len(calls) == 1 and run["outcome"] == "live_template_fallback"
    assert not run["initial"]["metrics"]["schema_valid"] and run["correction"] is None
    assert run["initial"]["raw"] == partial.replace(SECRET, "[redacted]") and SECRET not in saved
    assert not (tmp_path / "results/live_gemini_retrieval.json").exists()


def test_harness_counts_template_fallback_separately(monkeypatch, tmp_path):
    calls = configure(monkeypatch, [http_error(401)] + [accepted] * 4)
    destination = tmp_path / "evaluation.json"
    assert run_evaluation.main(["--live", "--provider", "gemini", "--yes", "--json", str(destination)]) == 0
    report = json.loads(destination.read_text())["live"]
    assert len(calls) == 5 and report["provider"] == f"gemini:{MODEL} (LLM)"
    assert report["generation"] == {"accepted": 4, "http_attempts": 5, "template_fallback": 1}
