"""Tests for API key auth, rate limiting, and upload validation."""

import io
import zipfile

import pytest
from fastapi import HTTPException

from nexusrag.api import security
from nexusrag.config import Settings


class _FakeRequest:
    def __init__(self, host: str = "1.2.3.4") -> None:
        self.client = type("Client", (), {"host": host})()


def _settings(**api_overrides: object) -> Settings:
    s = Settings()
    for key, value in api_overrides.items():
        setattr(s.api, key, value)
    return s


class TestApiKey:
    async def test_no_key_configured_is_open(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings(api_key=""))
        assert await security.require_api_key(None) is None

    async def test_missing_key_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings(api_key="test-only-key"))
        with pytest.raises(HTTPException) as exc:
            await security.require_api_key(None)
        assert exc.value.status_code == 401

    async def test_wrong_key_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings(api_key="test-only-key"))
        with pytest.raises(HTTPException) as exc:
            await security.require_api_key("nope")
        assert exc.value.status_code == 401

    async def test_correct_key_accepted(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings(api_key="test-only-key"))
        assert await security.require_api_key("test-only-key") is None

    async def test_non_ascii_key_rejected_not_crash(self, monkeypatch) -> None:
        # non-ASCII must 401, not raise TypeError -> 500 (hmac.compare_digest)
        monkeypatch.setattr(security, "get_settings", lambda: _settings(api_key="test-only-key"))
        with pytest.raises(HTTPException) as exc:
            await security.require_api_key("sécrét-ключ-🔑")
        assert exc.value.status_code == 401


class TestRateLimiter:
    def test_limit_strings_track_settings(self, monkeypatch) -> None:
        monkeypatch.setattr(
            security,
            "get_settings",
            lambda: _settings(query_rate_per_minute=42, upload_rate_per_minute=7),
        )
        assert security.query_limit() == "42/minute"
        assert security.upload_limit() == "7/minute"

    def test_slowapi_returns_429_over_limit(self) -> None:
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        limiter = Limiter(key_func=get_remote_address)
        app = FastAPI()
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        @app.get("/x")
        @limiter.limit("2/minute")
        async def _x(request: Request) -> dict[str, bool]:
            return {"ok": True}

        client = TestClient(app)
        assert [client.get("/x").status_code for _ in range(3)] == [200, 200, 429]


class TestSameSiteGuard:
    @staticmethod
    def _request(headers: dict[str, str]):
        from starlette.datastructures import Headers

        req = _FakeRequest()
        req.headers = Headers(headers)
        return req

    async def test_cross_site_fetch_rejected(self) -> None:
        with pytest.raises(HTTPException) as exc:
            await security.require_same_site(self._request({"sec-fetch-site": "cross-site"}))
        assert exc.value.status_code == 403

    async def test_same_origin_fetch_allowed(self) -> None:
        assert (
            await security.require_same_site(self._request({"sec-fetch-site": "same-origin"}))
            is None
        )

    async def test_user_initiated_allowed(self) -> None:
        # Sec-Fetch-Site: none == typed URL / bookmark, not an attack surface.
        assert await security.require_same_site(self._request({"sec-fetch-site": "none"})) is None

    async def test_non_browser_client_allowed(self) -> None:
        # curl / the CLI send neither header; no ambient-credential CSRF risk.
        assert await security.require_same_site(self._request({})) is None

    async def test_foreign_origin_fallback_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings())
        with pytest.raises(HTTPException) as exc:
            await security.require_same_site(self._request({"origin": "http://evil.example"}))
        assert exc.value.status_code == 403

    async def test_allowlisted_origin_fallback_allowed(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings())
        allowed = _settings().api.cors_origins[0]
        assert await security.require_same_site(self._request({"origin": allowed})) is None


class TestValidateUpload:
    def test_pdf_magic_required(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings())
        with pytest.raises(HTTPException) as exc:
            security.validate_upload(".pdf", b"not a pdf", "application/pdf")
        assert exc.value.status_code == 415

    def test_pdf_magic_accepted(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings())
        security.validate_upload(".pdf", b"%PDF-1.4\nhi", "application/pdf")

    def test_disallowed_content_type(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings())
        with pytest.raises(HTTPException) as exc:
            security.validate_upload(".txt", b"hello", "application/x-sh")
        assert exc.value.status_code == 415

    def test_text_must_be_utf8(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings())
        with pytest.raises(HTTPException) as exc:
            security.validate_upload(".txt", b"\xff\xfe\x00binary", "text/plain")
        assert exc.value.status_code == 415

    def test_oversize_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings(max_upload_mb=0))
        with pytest.raises(HTTPException) as exc:
            security.validate_upload(".txt", b"x" * 10, "text/plain")
        assert exc.value.status_code == 413

    def test_invalid_docx_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings())
        with pytest.raises(HTTPException) as exc:
            # valid magic prefix but not a real zip
            security.validate_upload(".docx", b"PK\x03\x04garbage", "application/zip")
        assert exc.value.status_code == 415

    def test_valid_docx_accepted(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings())
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.xml", "<x/>")
        security.validate_upload(".docx", buf.getvalue(), "application/zip")

    def test_zip_bomb_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _settings(max_uncompressed_mb=0))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("big.txt", "A" * 100_000)
        with pytest.raises(HTTPException) as exc:
            security.validate_upload(".docx", buf.getvalue(), "application/zip")
        assert exc.value.status_code == 413


class TestAppConfig:
    def test_cors_default_is_not_wildcard(self) -> None:
        from nexusrag.config import APISettings

        origins = APISettings().cors_origins
        assert "*" not in origins
        assert origins and all(o.startswith("http://") for o in origins)

    def test_docs_disabled_when_api_key_set(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        import nexusrag.api as api

        monkeypatch.setattr(api, "get_settings", lambda: _settings(api_key="k"))
        client = TestClient(api.create_app())
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_docs_enabled_in_local_mode(self, monkeypatch) -> None:
        from fastapi.testclient import TestClient

        import nexusrag.api as api

        monkeypatch.setattr(api, "get_settings", lambda: _settings(api_key="", docs_enabled=True))
        client = TestClient(api.create_app())
        assert client.get("/docs").status_code == 200


class TestIdSanitizer:
    """Proves the SECURITY.md claims: traversal blocked, allowlist is linear-time."""

    def test_path_traversal_and_injection_rejected(self) -> None:
        from nexusrag.storage.vector_store import _sanitize_id

        for bad in ["../../etc/passwd", "a/b", "a\\b", "id; DROP TABLE x", "a' OR '1'='1", "a b"]:
            with pytest.raises(ValueError):
                _sanitize_id(bad)

    def test_safe_ids_accepted(self) -> None:
        from nexusrag.storage.vector_store import _sanitize_id

        for ok in ["doc_123", "chunk-abc", "AaZz09_-"]:
            assert _sanitize_id(ok) == ok

    def test_allowlist_regex_is_linear_time(self) -> None:
        # A pathological input must not cause catastrophic backtracking (ReDoS).
        import time

        from nexusrag.storage.vector_store import SAFE_ID_PATTERN

        adversarial = "a" * 200_000 + "!"  # long valid run then a rejecting char
        start = time.perf_counter()
        assert SAFE_ID_PATTERN.match(adversarial) is None
        assert time.perf_counter() - start < 0.1  # linear, well under a budget
