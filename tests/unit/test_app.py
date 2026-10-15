"""Tests for application-level routes."""

import logging
import re
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from scinexusrag.api import create_app
from scinexusrag.config import get_settings


class TestHealthProbe:
    def test_health_returns_ok(self) -> None:
        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPipelinePresentation:
    def test_pipeline_has_results_table_and_design_source(self) -> None:
        response = TestClient(create_app()).get("/pipeline")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        table = re.search(r"<table\b.*?</table>", response.text, re.DOTALL)
        assert table is not None
        assert len(re.findall(r"<th\b", table.group())) >= 3
        for value in ("SciFact", "NFCorpus", "0.708", "0.341", "0.704", "0.352", "0.703", "0.346"):
            assert value in table.group()
        assert "https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=29-2" in response.text

    def test_main_page_links_to_pipeline(self) -> None:
        response = TestClient(create_app()).get("/")

        assert response.status_code == 200
        assert re.search(r'<a\b[^>]*href="/pipeline"[^>]*>', response.text)

    @pytest.mark.parametrize(
        ("path", "content_type"),
        [
            ("/css/pipeline.css", "text/css"),
            ("/images/pipeline.svg", "image/svg+xml"),
            ("/images/pipeline-upload-arrow.svg", "image/svg+xml"),
            ("/images/pipeline-question-arrow.svg", "image/svg+xml"),
            ("/images/pipeline-index-arrow.svg", "image/svg+xml"),
        ],
    )
    def test_pipeline_assets_are_served(self, path: str, content_type: str) -> None:
        response = TestClient(create_app()).get(path)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(content_type)
        assert response.content
        if content_type == "image/svg+xml":
            assert ET.fromstring(response.content).tag == "{http://www.w3.org/2000/svg}svg"

    @pytest.mark.parametrize("path", ["/images/../config.py", "/images/%2e%2e/%2e%2e/config.py"])
    def test_images_cannot_serve_python_source(self, path: str) -> None:
        assert TestClient(create_app()).get(path).status_code == 404


class TestLogLevel:
    def test_create_app_applies_log_level(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        get_settings.cache_clear()
        try:
            create_app()
            assert logging.getLogger("scinexusrag").level == logging.WARNING
        finally:
            get_settings.cache_clear()
            logging.getLogger("scinexusrag").setLevel(logging.NOTSET)
