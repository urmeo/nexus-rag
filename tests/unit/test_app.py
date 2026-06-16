"""Tests for application-level routes."""

from fastapi.testclient import TestClient

from nexusrag.api import create_app


class TestHealthProbe:
    def test_health_returns_ok(self) -> None:
        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
