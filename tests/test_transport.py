"""Shared provider HTTPS trust and error diagnostics. Uses fake HTTP connections only."""

import ssl
import urllib.error
import urllib.request
from io import BytesIO
from types import SimpleNamespace

import pytest

from src.assistant import Assistant, GeminiProvider, OpenRouterProvider, TemplateProvider, providers
from src.assistant.providers import AttemptBudget, GenerationRequest, ProviderError


@pytest.fixture(params=[OpenRouterProvider, GeminiProvider])
def make_provider(request):
    def make(opener=None):
        return request.param("fake-provider-key", "gemini-test-model", timeout=7,
                             opener=opener, sleep=lambda _: None)

    return make


@pytest.mark.parametrize("with_override", [False, True])
def test_macos_system_fallback_preserves_verification_and_explicit_ca(
        make_provider, with_override, monkeypatch, tmp_path):
    bundle = tmp_path / "cert.pem"
    bundle.touch()
    monkeypatch.setattr(providers, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(providers, "MACOS_CA_BUNDLE", str(bundle))
    paths = ssl.DefaultVerifyPaths(None, None, "SSL_CERT_FILE", "/missing.pem", "SSL_CERT_DIR", "/missing/")
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: paths)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    if with_override:
        monkeypatch.setenv("SSL_CERT_FILE", "/explicitly-selected.pem")
    selected = []
    handlers = []
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    def make_context(*, cafile):
        selected.append(cafile)
        return context

    def build_opener(*registered):
        handlers.extend(registered)
        return SimpleNamespace(open=lambda *args, **kwargs: pytest.fail("network used"))

    monkeypatch.setattr(ssl, "create_default_context", make_context)
    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    provider = make_provider()
    https = [handler for handler in handlers if isinstance(handler, urllib.request.HTTPSHandler)]
    if with_override:
        assert not selected and not https
    else:
        assert selected == [str(bundle)] and len(https) == 1
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    if isinstance(provider, GeminiProvider):
        assert any(isinstance(h, urllib.request.HTTPRedirectHandler) for h in handlers)


def test_certificate_failure_is_actionable_without_retry(make_provider):
    cert_error = ssl.SSLCertVerificationError(1, "raw failure including fake-provider-key")
    cert_error.verify_code = 20

    def opener(request, timeout):
        raise urllib.error.URLError(cert_error)

    provider = make_provider(opener)
    budget = AttemptBudget()
    with pytest.raises(ProviderError) as caught:
        provider.generate(GenerationRequest({}), budget)
    error = caught.value
    assert error.kind == "tls_certificate" and budget.used == 1
    assert "URLError -> SSLCertVerificationError" in str(error)
    assert "verify_code=20" in str(error) and "unable to get local issuer certificate" in str(error)
    assert "fake-provider-key" not in str(error)


@pytest.mark.parametrize("phase,expected", [
    ("connect", "connecting or sending the request (URLError -> TimeoutError"),
    ("headers", "waiting for response headers (TimeoutError"),
    ("body", "reading the response body (TimeoutError"),
])
def test_timeout_phases_keep_retry_and_show_safe_trace_detail(make_provider, phase, expected):
    class ReadTimeout(BytesIO):
        def read(self, size=-1):
            raise TimeoutError("private response details")

    def opener(request, timeout):
        if phase == "connect":
            raise urllib.error.URLError(TimeoutError("private connection details"))
        if phase == "headers":
            raise TimeoutError("private response details")
        return ReadTimeout()

    response = Assistant().ask("Music for studying", provider=make_provider(opener))
    assert response.http_attempts == 2 and response.generator == TemplateProvider.label
    assert expected in response.generation_status and "timeout=7s" in response.generation_status
    event = next(event for event in response.trace.events if event.result.get("error_kind") == "network")
    assert expected in event.result["error_detail"]
    assert "private" not in response.trace.to_json()


def test_proxy_diagnostics_omit_credentials_and_report_status(make_provider):
    def opener(request, timeout):
        raise urllib.error.URLError(OSError("Tunnel connection failed: 407 http://proxy-user:proxy-pass@example.test"))

    budget = AttemptBudget()
    with pytest.raises(ProviderError) as caught:
        make_provider(opener).generate(GenerationRequest({}), budget)
    assert budget.used == 1 and "Proxy tunnel failed (URLError -> OSError, HTTP 407)" in str(caught.value)
    assert "proxy-user" not in str(caught.value) and "proxy-pass" not in str(caught.value)
