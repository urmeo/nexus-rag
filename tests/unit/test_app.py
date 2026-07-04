"""Tests for application-level routes."""

import logging

from fastapi.testclient import TestClient

from nexusrag.api import create_app
from nexusrag.config import get_settings


class TestHealthProbe:
    def test_health_returns_ok(self) -> None:
        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestLogLevel:
    def test_create_app_applies_log_level(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        get_settings.cache_clear()
        try:
            create_app()
            assert logging.getLogger("nexusrag").level == logging.WARNING
        finally:
            get_settings.cache_clear()
            logging.getLogger("nexusrag").setLevel(logging.NOTSET)
