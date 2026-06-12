"""Tests for application-level routes."""

from fastapi.testclient import TestClient

from nexusrag.api import create_app


class TestHealthProbe:
    """Tests for GET /health (app-level, no pipeline init)."""

    def test_health_returns_ok(self) -> None:
        """Test that /health returns status ok."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
