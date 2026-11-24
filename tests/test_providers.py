"""Explanation providers, explanation validation, and fallback. No network access is used."""

import io
import json
import urllib.error

import pytest

from src.assistant import Assistant, MockProvider, OpenRouterProvider, TemplateProvider, make_provider
from src.assistant.providers import ProviderError

REQUEST = "Recommend three nostalgic acoustic songs for a quiet evening."
SECRET = "sk-or-test-SECRET-123"


@pytest.fixture(scope="module")
def assistant():
    return Assistant()


def explanation_for(payload, **overrides):
    """A valid model-style answer built from the payload, with optional tampering."""
    items = [{"song_id": r["song_id"], "explanation": f"Explains {r['title']}.",
              "citations": [payload["sources"][0]["id"]] if payload["sources"] else []}
             for r in payload["results"]]
    data = {"summary": "Grounded summary.", "items": items, "limitations": []}
    data.update(overrides)
    return json.dumps(data)


# ---------------------------------------------------------------- template and mock

def test_template_is_default_offline_and_validated(assistant, monkeypatch):
    for name in ["OPENROUTER_API_KEY", "OPENROUTER_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    r = assistant.ask(REQUEST)
    assert r.generator == "template (deterministic, not an LLM)" and r.generation_status == "validated"
    assert set(r.explanation.items) == {x.song["id"] for x in r.recommendations}
    assert not TemplateProvider.is_llm


def test_valid_mock_output_is_used_and_labeled_as_mock(assistant):
    mock = MockProvider([lambda payload: explanation_for(payload)])
    r = assistant.ask(REQUEST, provider=mock)
    assert r.generator == "mock (test double, not a live model)"
    assert r.generation_status.startswith("LLM output validated")
    assert r.explanation.summary == "Grounded summary."
    payload = mock.calls[0]
    assert {"request", "results", "sources", "plan", "limitations"} <= set(payload)
    assert payload["results"][0]["reasons"] == r.recommendations[0].reasons


@pytest.mark.parametrize("bad, expected", [
    (lambda p: "not json at all", "not valid JSON"),
    (lambda p: "[1, 2]", "not a JSON object"),
    (lambda p: explanation_for(p, items=[{"song_id": 999, "explanation": "x", "citations": []}]), "999"),
    (lambda p: explanation_for(p, items=[]), "no explanation for song id"),
    (lambda p: explanation_for(p, items=[{**i, "citations": ["made_up#source"]} for i in json.loads(explanation_for(p))["items"]]),
     "unsupported source"),
    (lambda p: explanation_for(p, summary="You will also love Storm Runner."), "not in the results: Storm Runner"),
    (lambda p: explanation_for(p, summary=""), "summary"),
])
def test_invalid_model_output_falls_back_to_template(assistant, bad, expected):
    r = assistant.ask(REQUEST, provider=MockProvider([bad]))
    assert r.generator == "template (deterministic, not an LLM)"
    assert "fell back to the template" in r.generation_status and expected in r.generation_status
    assert r.explanation is not None


def test_code_fenced_json_is_accepted(assistant):
    r = assistant.ask(REQUEST, provider=MockProvider([lambda p: "```json\n" + explanation_for(p) + "\n```"]))
    assert r.generation_status.startswith("LLM output validated")


def test_provider_failure_falls_back_to_template(assistant):
    r = assistant.ask(REQUEST, provider=MockProvider([ProviderError("network", "timed out", retryable=True)]))
    assert r.generator.startswith("template") and "network: timed out" in r.generation_status
    assert r.recommendations and r.status == "ok"


def test_model_cannot_change_scores_or_order(assistant):
    def tampered(payload):
        data = json.loads(explanation_for(payload))
        data["items"].reverse()
        return json.dumps(data)
    baseline = assistant.ask(REQUEST)
    r = assistant.ask(REQUEST, provider=MockProvider([tampered]))
    assert [(x.song["id"], x.score) for x in r.recommendations] == [(x.song["id"], x.score) for x in baseline.recommendations]


# ---------------------------------------------------------------- OpenRouter configuration

def test_openrouter_requires_key_and_model():
    provider, reason = OpenRouterProvider.from_env({})
    assert provider is None and "OPENROUTER_API_KEY" in reason and "OPENROUTER_MODEL" in reason
    provider, reason = OpenRouterProvider.from_env({"OPENROUTER_API_KEY": SECRET})
    assert provider is None and reason == "OPENROUTER_MODEL not set"
    provider, reason = OpenRouterProvider.from_env({"OPENROUTER_API_KEY": SECRET, "OPENROUTER_MODEL": "m",
                                                    "OPENROUTER_BASE_URL": "http://insecure"})
    assert provider is None and "https" in reason


def test_make_provider_falls_back_when_unconfigured(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider, note = make_provider("openrouter")
    assert isinstance(provider, TemplateProvider) and "not configured" in note


def test_timeout_and_max_tokens_are_bounded():
    provider, _ = OpenRouterProvider.from_env({"OPENROUTER_API_KEY": SECRET, "OPENROUTER_MODEL": "m",
                                               "OPENROUTER_TIMEOUT": "999", "OPENROUTER_MAX_TOKENS": "99999"})
    assert provider.timeout == 60.0 and provider.max_tokens == 2000


# ---------------------------------------------------------------- OpenRouter HTTP behaviour (fake transport)

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_opener(outcomes):
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(json.dumps(outcome).encode())
    return opener, calls


def http_error(code):
    return urllib.error.HTTPError("https://example.invalid", code, "error", {}, None)


def ok_body(content='{"summary": "s", "items": []}'):
    return {"choices": [{"message": {"content": content}}]}


def make(outcomes):
    opener, calls = fake_opener(list(outcomes))
    provider = OpenRouterProvider(SECRET, "test/model", opener=opener, sleep=lambda s: None, timeout=7)
    return provider, calls


def test_successful_request_shape():
    provider, calls = make([ok_body()])
    assert provider.generate({"results": []}) == '{"summary": "s", "items": []}'
    request, timeout = calls[0]
    body = json.loads(request.data)
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == f"Bearer {SECRET}"
    assert body["model"] == "test/model" and body["max_tokens"] == provider.max_tokens
    assert body["messages"][0]["role"] == "system" and timeout == 7


@pytest.mark.parametrize("outcomes, kind, attempts", [
    ([http_error(401)], "auth", 1),
    ([http_error(400)], "http", 1),
    ([http_error(500), http_error(503)], "server", 2),
    ([http_error(429), http_error(429)], "rate_limit", 2),
    ([urllib.error.URLError("timed out"), urllib.error.URLError("timed out")], "network", 2),
    ([{"unexpected": True}], "invalid_response", 1),
    ([ok_body("   ")], "invalid_response", 1),
])
def test_errors_are_categorized_with_at_most_one_retry(outcomes, kind, attempts):
    provider, calls = make(outcomes)
    with pytest.raises(ProviderError) as error:
        provider.generate({})
    assert error.value.kind == kind and len(calls) == attempts
    assert SECRET not in str(error.value)


def test_retry_succeeds_after_one_transient_failure():
    provider, calls = make([http_error(429), ok_body()])
    assert provider.generate({}) and len(calls) == 2


def test_credentials_are_never_exposed():
    provider, _ = make([])
    assert SECRET not in repr(provider) and SECRET not in provider.label
    provider, _ = make([urllib.error.URLError(f"proxy rejected {SECRET}")] * 2)
    with pytest.raises(ProviderError) as error:
        provider.generate({})
    assert SECRET not in str(error.value) and "[redacted]" in str(error.value)
