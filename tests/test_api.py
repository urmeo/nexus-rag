"""API checks."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import re
import sys
import threading
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import quote

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.datastructures import Headers
from starlette.requests import Request

import scinexusrag.api.app as api
from scinexusrag.api import metrics, middleware, routes, security, uploads
from scinexusrag.api.app import create_app
from scinexusrag.api.dependencies import get_nexusrag
from scinexusrag.api.metrics import MetricsCollector
from scinexusrag.api.routes import MAX_FILENAME_LENGTH, router
from scinexusrag.api.schemas import QueryRequest
from scinexusrag.application import DocumentInventory, IngestResult, NexusRAG, SystemStats
from scinexusrag.generation.orchestrator import RAGResponse
from scinexusrag.generation.synthesizer import Source
from scinexusrag.ingestion.embedder import Embedder, EmbeddingConfig, EmbeddingConfigError
from scinexusrag.settings import (
    APISettings,
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    RetrievalSettings,
    SelfCorrectionSettings,
    Settings,
    StorageSettings,
    get_settings,
)
from scinexusrag.storage.vector_store import VectorStoreError
from scinexusrag.version import __version__


class TestSingletonBehavior:
    def test_get_nexusrag_returns_same_instance(self, tmp_path):
        first = get_nexusrag(Settings(data_dir=tmp_path))
        assert get_nexusrag() is first

    def test_get_nexusrag_with_different_settings(self, tmp_path):
        first = get_nexusrag(Settings(data_dir=tmp_path / "first"))
        assert get_nexusrag(Settings(data_dir=tmp_path / "second")) is first


class TestHealthProbe:
    def test_health_returns_ok(self) -> None:
        app = create_app()
        client = TestClient(app, base_url="http://localhost")

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestPipelinePresentation:
    def test_pipeline_has_results_table_and_design_source(self) -> None:
        response = TestClient(create_app(), base_url="http://localhost").get("/pipeline")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        table = re.search(r"<table\b.*?</table>", response.text, re.DOTALL)
        assert table is not None
        assert len(re.findall(r"<th\b", table.group())) >= 3
        for value in ("SciFact", "NFCorpus", "0.708", "0.341", "0.704", "0.352", "0.703", "0.346"):
            assert value in table.group()
        assert "https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=29-2" in response.text

    def test_main_page_links_to_pipeline(self) -> None:
        response = TestClient(create_app(), base_url="http://localhost").get("/")

        assert response.status_code == 200
        assert re.search(r'<a\b[^>]*href="/pipeline"[^>]*>', response.text)

    @pytest.mark.parametrize(
        ("path", "content_type"),
        [
            ("/css/pipeline.css", "text/css"),
            ("/images/pipeline.svg", "image/svg+xml"),
        ],
    )
    def test_pipeline_assets_are_served(self, path: str, content_type: str) -> None:
        response = TestClient(create_app(), base_url="http://localhost").get(path)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(content_type)
        assert response.content
        if content_type == "image/svg+xml":
            assert ET.fromstring(response.content).tag == "{http://www.w3.org/2000/svg}svg"

    @pytest.mark.parametrize(
        "path", ["/images/../settings.py", "/images/%2e%2e/%2e%2e/settings.py"]
    )
    def test_images_cannot_serve_python_source(self, path: str) -> None:
        assert TestClient(create_app(), base_url="http://localhost").get(path).status_code == 404


class TestLogLevel:
    def test_create_app_applies_log_level(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        logger = logging.getLogger("scinexusrag")
        original_level = logger.level
        get_settings.cache_clear()
        try:
            create_app()
            assert logger.level == logging.WARNING
        finally:
            get_settings.cache_clear()
            logger.setLevel(original_level)

    @pytest.mark.parametrize("host_configured", [False, True])
    def test_create_app_prints_package_logs_once_unless_the_host_configured_logging(
        self, monkeypatch, capsys, host_configured
    ) -> None:
        logger = logging.getLogger("scinexusrag")
        original_level = logger.level
        monkeypatch.setattr(
            logging.getLogger(), "handlers", [logging.NullHandler()] if host_configured else []
        )
        monkeypatch.setattr(logger, "handlers", [])
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        get_settings.cache_clear()
        try:
            create_app()
            create_app()
            logging.getLogger("scinexusrag.ingestion").info("Indexed %d passages", 3)
        finally:
            get_settings.cache_clear()
            logger.setLevel(original_level)

        printed = capsys.readouterr().err.count("scinexusrag.ingestion: Indexed 3 passages")
        assert printed == (0 if host_configured else 1)


def _write_frontend(directory: Path, label: str) -> Path:
    directory.mkdir()
    (directory / "index.html").write_text(f"<html>{label} index</html>", encoding="utf-8")
    (directory / "pipeline.html").write_text(f"<html>{label} pipeline</html>", encoding="utf-8")
    (directory / "css").mkdir()
    (directory / "css" / "style.css").write_text(
        f":root {{ --interface: {label}; }}", encoding="utf-8"
    )
    return directory


class TestFrontendConfiguration:
    def test_factory_reads_current_environment_and_keeps_apps_isolated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        first = _write_frontend(tmp_path / "first", "first")
        second = _write_frontend(tmp_path / "second", "second")
        monkeypatch.setenv("NEXUSRAG_FRONTEND_DIR", str(first))
        first_app = create_app()
        monkeypatch.setenv("NEXUSRAG_FRONTEND_DIR", str(second))
        second_app = create_app()

        for app, label in ((first_app, "first"), (second_app, "second")):
            with TestClient(app, base_url="http://localhost") as client:
                assert client.get("/").text == f"<html>{label} index</html>"
                assert client.get("/pipeline").text == f"<html>{label} pipeline</html>"
                assert client.get("/css/style.css").text == f":root {{ --interface: {label}; }}"

    @pytest.mark.parametrize("page,filename", [("/", "index.html"), ("/pipeline", "pipeline.html")])
    @pytest.mark.parametrize("remove_after_creation", [False, True])
    def test_missing_entry_page_returns_404(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        page: str,
        filename: str,
        remove_after_creation: bool,
    ) -> None:
        frontend = _write_frontend(tmp_path / "frontend", "custom")
        monkeypatch.setenv("NEXUSRAG_FRONTEND_DIR", str(frontend))
        if not remove_after_creation:
            (frontend / filename).unlink()
        app = create_app()
        if remove_after_creation:
            (frontend / filename).unlink()

        with TestClient(app, base_url="http://localhost") as client:
            response = client.get(page)
            assert response.status_code == 404
            assert client.get("/health").json() == {"status": "ok"}

    @pytest.mark.parametrize("exists_as_file", [False, True])
    def test_non_directory_frontend_keeps_api_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exists_as_file: bool
    ) -> None:
        frontend = tmp_path / "frontend"
        if exists_as_file:
            frontend.write_text("Not a directory", encoding="utf-8")
        monkeypatch.setenv("NEXUSRAG_FRONTEND_DIR", str(frontend))

        with TestClient(create_app(), base_url="http://localhost") as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            assert client.get("/").status_code == 404
            assert client.get("/pipeline").status_code == 404

    def test_static_mount_ignores_regular_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        frontend = tmp_path / "frontend"
        frontend.mkdir()
        (frontend / "index.html").write_text("<html>Custom interface</html>", encoding="utf-8")
        (frontend / "css").write_text("Not a directory", encoding="utf-8")
        monkeypatch.setenv("NEXUSRAG_FRONTEND_DIR", str(frontend))

        with TestClient(create_app(), base_url="http://localhost") as client:
            assert client.get("/").text == "<html>Custom interface</html>"
            assert client.get("/css/style.css").status_code == 404
            assert client.get("/health").status_code == 200

    def test_relative_frontend_stays_bound_after_working_directory_changes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        initial, later = tmp_path / "initial", tmp_path / "later"
        initial.mkdir()
        later.mkdir()
        _write_frontend(initial / "frontend", "initial")
        _write_frontend(later / "frontend", "later")
        monkeypatch.chdir(initial)
        monkeypatch.setenv("NEXUSRAG_FRONTEND_DIR", "frontend")
        app = create_app()
        monkeypatch.chdir(later)

        with TestClient(app, base_url="http://localhost") as client:
            assert client.get("/").text == "<html>initial index</html>"
            assert client.get("/pipeline").text == "<html>initial pipeline</html>"
            assert client.get("/css/style.css").text == ":root { --interface: initial; }"

    @pytest.mark.parametrize("page,filename", [("/", "index.html"), ("/pipeline", "pipeline.html")])
    def test_html_symlink_cannot_escape_frontend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, page: str, filename: str
    ) -> None:
        frontend = _write_frontend(tmp_path / "frontend", "custom")
        secret = tmp_path / "private.html"
        secret.write_text("Private document contents", encoding="utf-8")
        (frontend / filename).unlink()
        (frontend / filename).symlink_to(secret)
        monkeypatch.setenv("NEXUSRAG_FRONTEND_DIR", str(frontend))

        with TestClient(create_app(), base_url="http://localhost") as client:
            response = client.get(page)
            assert response.status_code == 404
            assert "Private document contents" not in response.text


_ORIGIN = "http://localhost:3000"


@pytest.fixture
def cors_settings(monkeypatch, local_settings):
    cors_settings = local_settings
    cors_settings.api.cors_origins = [_ORIGIN]
    cors_settings.api.max_upload_mb = 1

    def unexpected_pipeline():
        pytest.fail("Rejected requests and CORS preflights must not load the pipeline.")

    monkeypatch.setattr(routes, "get_nexusrag", unexpected_pipeline)
    return cors_settings


@pytest.mark.parametrize("content_length,status", [("+1", 400), ("65537", 413)])
def test_body_rejection_keeps_allowed_origin_headers(cors_settings, content_length, status):
    with TestClient(api.create_app(), base_url="http://localhost") as client:
        response = client.post(
            "/api/query",
            content=b"x" if status == 400 else b"x" * 65537,
            headers={"Origin": _ORIGIN, "Content-Length": content_length},
        )

    assert response.status_code == status
    assert response.json()["detail"] == (
        "Invalid Content-Length" if status == 400 else "Request body too large"
    )
    assert response.headers["access-control-allow-origin"] == _ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "origin" in response.headers["vary"].lower()


async def test_streamed_body_rejection_keeps_cors_headers(cors_settings, asgi_request):
    chunks = [b'{"question":"', b"x" * 32768, b"x" * 32768, b'"}']
    messages, consumed = await asgi_request(
        api.create_app(),
        [
            (b"host", b"localhost"),
            (b"origin", _ORIGIN.encode()),
            (b"content-type", b"application/json"),
        ],
        chunks,
    )

    starts = [message for message in messages if message["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 413
    assert Headers(raw=starts[0]["headers"])["access-control-allow-origin"] == _ORIGIN
    assert consumed < len(chunks)


def test_allowed_preflight_reaches_cors_without_pipeline_access(cors_settings):
    with TestClient(api.create_app(), base_url="http://localhost") as client:
        response = client.options(
            "/api/query",
            headers={
                "Origin": _ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, X-API-Key",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == _ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]
    allowed_headers = response.headers["access-control-allow-headers"].lower()
    assert "content-type" in allowed_headers
    assert "x-api-key" in allowed_headers


@pytest.mark.parametrize(
    "hosts",
    [
        [],
        [b"rebind.example"],
        [b"localhost:0"],
        [b"localhost", b"rebind.example"],
        [b"localhost"] * 2,
    ],
)
async def test_invalid_preflight_hosts_are_rejected_before_cors(cors_settings, hosts, asgi_request):
    headers = [(b"host", host) for host in hosts]
    headers.extend([(b"origin", _ORIGIN.encode()), (b"access-control-request-method", b"POST")])

    messages, consumed = await asgi_request(api.create_app(), headers, method="OPTIONS")

    starts = [message for message in messages if message["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 400
    assert "access-control-allow-origin" not in Headers(raw=starts[0]["headers"])
    assert consumed == 0


@pytest.mark.parametrize("preflight", [False, True])
def test_wildcard_origins_do_not_enable_credentials(cors_settings, preflight):
    cors_settings.api.cors_origins = ["*"]
    with TestClient(api.create_app(), base_url="http://localhost") as client:
        if preflight:
            response = client.options(
                "/api/query",
                headers={
                    "Origin": "https://client.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
        else:
            response = client.get("/health", headers={"Origin": "https://client.example"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


def test_body_errors_do_not_allow_unconfigured_origins(cors_settings):
    with TestClient(api.create_app(), base_url="http://localhost") as client:
        response = client.post(
            "/api/query", content=b"x" * 65537, headers={"Origin": "https://foreign.example"}
        )

    assert response.status_code == 413
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("host,key", [("0.0.0.0", ""), ("127.0.0.1", "test-only-key")])
def test_explicit_network_configuration_keeps_public_hosts(cors_settings, host, key):
    cors_settings.api.host = host
    cors_settings.api.api_key = key
    with TestClient(api.create_app(), base_url="https://research.example") as client:
        health = client.get("/health", headers={"Origin": _ORIGIN})
        preflight = client.options(
            "/api/query",
            headers={"Origin": _ORIGIN, "Access-Control-Request-Method": "POST"},
        )

    for response in (health, preflight):
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == _ORIGIN


def test_document_page_uses_inventory_totals_without_separate_reads(monkeypatch):
    pipeline = MagicMock(spec=NexusRAG)
    pipeline.get_document_inventory.return_value = DocumentInventory(
        documents=[
            {"id": "first", "filename": "first.txt", "word_count": 4, "chunk_count": 2},
            {"id": "recovered", "filename": "recovered.txt", "word_count": 3},
            {"id": "last", "filename": "last.txt", "word_count": 6, "chunk_count": 3},
        ],
        total_chunks=7,
    )
    pipeline.get_stats.side_effect = AssertionError("Unexpected statistics read")
    pipeline.list_documents.side_effect = AssertionError("Unexpected document read")
    monkeypatch.setattr("scinexusrag.api.routes.get_nexusrag", lambda: pipeline)
    app = FastAPI()
    app.include_router(router)

    response = TestClient(app, base_url="http://localhost").get("/api/documents?limit=1&offset=1")

    assert response.status_code == 200
    page = response.json()
    assert [document["id"] for document in page["documents"]] == ["recovered"]
    assert page["total_documents"] == 3
    assert page["total_chunks"] == 7
    pipeline.get_document_inventory.assert_called_once_with()
    pipeline.get_stats.assert_not_called()
    pipeline.list_documents.assert_not_called()


class TestMetricsCollector:
    def test_initial_state(self) -> None:
        collector = MetricsCollector()
        snap = collector.snapshot()

        assert snap["total_queries"] == 0
        assert snap["total_ingestions"] == 0
        assert snap["total_deletions"] == 0
        assert snap["avg_query_time_ms"] == 0.0
        assert snap["uptime_seconds"] >= 0
        assert set(snap) == {
            "total_queries",
            "total_ingestions",
            "total_deletions",
            "avg_query_time_ms",
            "uptime_seconds",
        }
        for key in ("total_queries", "total_ingestions", "total_deletions"):
            assert type(snap[key]) is int
        for key in ("avg_query_time_ms", "uptime_seconds"):
            assert type(snap[key]) is float

    def test_record_query(self) -> None:
        collector = MetricsCollector()
        collector.record_query(100.0)
        collector.record_query(200.0)

        snap = collector.snapshot()
        assert snap["total_queries"] == 2
        assert snap["avg_query_time_ms"] == 150.0

    def test_record_ingest(self) -> None:
        collector = MetricsCollector()
        collector.record_ingest()
        collector.record_ingest()
        collector.record_ingest()

        snap = collector.snapshot()
        assert snap["total_ingestions"] == 3

    def test_record_delete(self) -> None:
        collector = MetricsCollector()
        collector.record_delete()

        snap = collector.snapshot()
        assert snap["total_deletions"] == 1

    def test_zero_duration_is_recorded(self) -> None:
        collector = MetricsCollector()
        collector.record_query(0)
        snap = collector.snapshot()
        assert snap["total_queries"] == 1
        assert snap["avg_query_time_ms"] == 0.0

    @pytest.mark.parametrize(
        "duration,error",
        [
            (True, TypeError),
            (False, TypeError),
            ("12.5", TypeError),
            (None, TypeError),
            (complex(1, 1), TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
            (-1.0, ValueError),
            (10**1000, ValueError),
        ],
    )
    def test_invalid_duration_preserves_statistics(self, duration, error):
        collector = MetricsCollector()
        collector.record_query(100.0)

        with pytest.raises(error):
            collector.record_query(duration)

        snap = collector.snapshot()
        assert snap["total_queries"] == 1
        assert snap["avg_query_time_ms"] == 100.0
        collector.record_query(200)
        assert collector.snapshot()["avg_query_time_ms"] == 150.0

    @pytest.mark.parametrize(
        "durations,expected",
        [
            ([sys.float_info.max] * 4, sys.float_info.max),
            ([0.0, sys.float_info.max], sys.float_info.max / 2),
            ([sys.float_info.max, 0.0], sys.float_info.max / 2),
        ],
    )
    def test_finite_durations_do_not_overflow_average(self, durations, expected):
        collector = MetricsCollector()
        for duration in durations:
            collector.record_query(duration)
        snap = collector.snapshot()
        assert snap["total_queries"] == len(durations)
        assert math.isfinite(snap["avg_query_time_ms"])
        assert snap["avg_query_time_ms"] == pytest.approx(expected, rel=1e-15)

    def test_varied_durations_match_arithmetic_mean(self):
        collector = MetricsCollector()
        durations = [0.0, 1.25, 200, 5.5, 0.125, 750.0, 12.75] * 20
        for duration in durations:
            collector.record_query(duration)
        snap = collector.snapshot()
        assert snap["total_queries"] == len(durations)
        assert snap["avg_query_time_ms"] == pytest.approx(
            math.fsum(durations) / len(durations), rel=1e-14
        )

    def test_snapshots_are_detached(self):
        collector = MetricsCollector()
        before = collector.snapshot()
        before["total_queries"] = 999
        collector.record_query(25.0)
        after = collector.snapshot()

        assert before["avg_query_time_ms"] == 0.0
        assert after["total_queries"] == 1
        assert after["avg_query_time_ms"] == 25.0
        after["avg_query_time_ms"] = -1.0
        assert collector.snapshot()["avg_query_time_ms"] == 25.0

    def test_overlapping_writers_and_snapshot_readers(self):
        collector = MetricsCollector()
        writers, readers, iterations = 4, 2, 200
        start = threading.Barrier(writers + readers, timeout=5)

        def write():
            start.wait()
            for _ in range(iterations):
                collector.record_query(12.5)
                collector.record_ingest()
                collector.record_delete()

        def read():
            start.wait()
            previous = dict.fromkeys(("total_queries", "total_ingestions", "total_deletions"), 0)
            for _ in range(iterations):
                snap = collector.snapshot()
                for key in previous:
                    assert previous[key] <= snap[key] <= writers * iterations
                    previous[key] = snap[key]
                assert snap["avg_query_time_ms"] == (12.5 if snap["total_queries"] else 0.0)

        with ThreadPoolExecutor(max_workers=writers + readers) as pool:
            jobs = [pool.submit(write) for _ in range(writers)]
            jobs.extend(pool.submit(read) for _ in range(readers))
            for job in jobs:
                job.result(timeout=10)

        snap = collector.snapshot()
        assert snap["total_queries"] == writers * iterations
        assert snap["total_ingestions"] == writers * iterations
        assert snap["total_deletions"] == writers * iterations
        assert snap["avg_query_time_ms"] == 12.5


def test_concurrent_first_access_constructs_one_singleton(monkeypatch):
    workers = 4
    start = threading.Barrier(workers + 1, timeout=5)
    entered = threading.Event()
    release = threading.Event()
    construction = []

    def create_collector():
        construction.append(1)
        entered.set()
        assert release.wait(timeout=5)
        return MetricsCollector()

    monkeypatch.setattr(metrics, "_collector", None)
    monkeypatch.setattr(metrics, "MetricsCollector", create_collector)

    def get_collector():
        start.wait()
        return metrics.get_metrics_collector()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(get_collector) for _ in range(workers)]
        start.wait()
        try:
            assert entered.wait(timeout=5)
        finally:
            release.set()
        collectors = [job.result(timeout=10) for job in jobs]

    assert len(construction) == 1
    assert all(collector is collectors[0] for collector in collectors)
    collectors[0].record_query(4.0)
    assert metrics.get_metrics_collector().snapshot()["total_queries"] == 1


def test_metrics_http_response_preserves_counts_types_and_rounding(monkeypatch, local_settings):
    collector = MetricsCollector()
    collector.record_query(12.34)
    collector.record_query(56.78)
    collector.record_ingest()
    collector.record_delete()
    statistics = SystemStats(
        total_documents=3,
        total_chunks=12,
        total_words=1200,
        llm_model="fixture-generator",
        embedding_model="fixture-embedder",
        llm_available=False,
    )
    monkeypatch.setattr(routes, "get_metrics_collector", lambda: collector)
    monkeypatch.setattr(
        routes, "get_nexusrag", lambda: SimpleNamespace(get_stats=lambda: statistics)
    )

    with TestClient(api.create_app(), base_url="http://localhost") as client:
        response = client.get("/api/metrics")

    assert response.status_code == 200
    data = response.json()
    expected = {
        "total_queries": 2,
        "total_ingestions": 1,
        "total_deletions": 1,
        "avg_query_time_ms": 34.6,
        "total_documents": 3,
        "total_chunks": 12,
        "version": __version__,
    }
    assert set(data) == {"uptime_seconds", *expected}
    for key, value in expected.items():
        assert data[key] == value
        assert type(data[key]) is type(value)
    assert type(data["uptime_seconds"]) is float
    assert data["uptime_seconds"] >= 0


@pytest.fixture
def request_settings(local_settings):
    request_settings = local_settings
    request_settings.api.max_upload_mb = 1
    return request_settings


@pytest.mark.parametrize(
    "host",
    [
        "rebind.example",
        "localhost.evil.example",
        "evil@localhost",
        "localhost:not-a-port",
        "[::1",
        "[::1]evil.example",
        "[::1]foo:8000",
        "localhost:",
        "127.0.0.1:",
        "[::1]:",
        "localhost:0",
        "[::1]:65536",
    ],
)
def test_local_api_rejects_foreign_hosts(request_settings, monkeypatch, host):
    def unexpected_pipeline():
        pytest.fail("Rejected hosts must not read the document collection.")

    monkeypatch.setattr(routes, "get_nexusrag", unexpected_pipeline)
    response = TestClient(api.create_app(), base_url="http://localhost").get(
        "/api/documents", headers={"Host": host, "Sec-Fetch-Site": "same-origin"}
    )
    assert response.status_code == 400


def test_loopback_host_rejects_unicode_case_folding():
    assert not middleware.is_loopback_host("localhoſt")


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST:8000",
        "127.0.0.1",
        "127.0.0.1:8000",
        "[::1]",
        "[::1]:65535",
    ],
)
def test_local_api_accepts_loopback_hosts(request_settings, host):
    response = TestClient(api.create_app(), base_url="http://localhost").get(
        "/health", headers={"Host": host}
    )
    assert response.status_code == 200


def test_configured_key_preserves_public_hosts(request_settings):
    request_settings.api.api_key = "test-only-key"
    response = TestClient(api.create_app(), base_url="https://research.example").get("/health")
    assert response.status_code == 200


def test_explicit_network_bind_preserves_public_hosts(request_settings):
    request_settings.api.host = "0.0.0.0"
    response = TestClient(api.create_app(), base_url="https://research.example").get("/health")
    assert response.status_code == 200


@pytest.mark.parametrize("content_length", [None, b"1", b"2097152"])
@pytest.mark.parametrize("mount_path", ["", "/research"])
def test_oversized_multipart_stops_before_reading_all_bytes(
    request_settings, monkeypatch, content_length, mount_path, asgi_request
):
    import starlette.formparsers

    temporary_files = []
    create_temporary_file = starlette.formparsers.SpooledTemporaryFile

    def tracked_temporary_file(*args, **kwargs):
        file = create_temporary_file(*args, **kwargs)
        temporary_files.append(file)
        return file

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", tracked_temporary_file)
    boundary = b"security-boundary"
    prefix = (
        b"--" + boundary + b'\r\nContent-Disposition: form-data; name="file"; '
        b'filename="large.txt"\r\nContent-Type: text/plain\r\n\r\n'
    )
    chunks = [prefix] + [b"x" * 65536] * 32 + [b"\r\n--" + boundary + b"--\r\n"]
    headers = [
        (b"host", b"localhost"),
        (b"content-type", b"multipart/form-data; boundary=" + boundary),
    ]
    if content_length is not None:
        headers.append((b"content-length", content_length))
    messages, consumed = asyncio.run(
        asgi_request(
            api.create_app(),
            headers,
            chunks,
            path=mount_path + "/api/ingest",
            root_path=mount_path,
        )
    )

    assert (
        next(message["status"] for message in messages if message["type"] == "http.response.start")
        == 413
    )
    assert consumed < len(chunks)
    assert all(file.closed for file in temporary_files)


def test_upload_trailing_slash_preserves_redirect(request_settings):
    response = TestClient(api.create_app(), base_url="http://localhost").post(
        "/api/ingest/",
        files={"file": ("normal.txt", b"x" * (128 * 1024), "text/plain")},
        follow_redirects=False,
    )
    assert response.status_code == 307
    assert response.headers["location"] == "http://localhost/api/ingest"


def test_normal_upload_reaches_ingestion(request_settings, monkeypatch):
    ingested = []

    def ingest(data, filename, extension):
        ingested.append((data, filename, extension))
        return SimpleNamespace(
            success=True,
            document_id="test-doc",
            filename=filename,
            chunk_count=1,
            word_count=2,
            error=None,
        )

    monkeypatch.setattr(routes, "get_nexusrag", lambda: SimpleNamespace(ingest_bytes=ingest))
    response = TestClient(api.create_app(), base_url="http://localhost").post(
        "/api/ingest", files={"file": ("small.txt", b"safe text", "text/plain")}
    )
    assert response.status_code == 200
    assert ingested == [(b"safe text", "small.txt", ".txt")]


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"host", b"localhost"), (b"host", b"evil.example")],
        [(b"host", b"evil.example"), (b"host", b"localhost")],
        [(b"host", b"localhost"), (b"host", b"localhost")],
    ],
)
def test_local_api_rejects_missing_or_duplicate_hosts_before_reading_body(
    request_settings, monkeypatch, headers, asgi_request
):
    def unexpected_pipeline():
        pytest.fail("Invalid host headers must not reach the document pipeline.")

    monkeypatch.setattr(routes, "get_nexusrag", unexpected_pipeline)
    messages, consumed = asyncio.run(asgi_request(api.create_app(), headers))
    assert [
        message["status"] for message in messages if message["type"] == "http.response.start"
    ] == [400]
    assert consumed == 0


@pytest.mark.parametrize(
    "content_lengths, expected_status",
    [
        ([b""], 400),
        ([b"-1"], 400),
        ([b"+1"], 400),
        ([b" 1"], 400),
        ([b"1 "], 400),
        ([b"1_000"], 400),
        ([b"1.0"], 400),
        ([b"\xb2"], 400),
        ([b"1, 1"], 400),
        ([b"1", b"1"], 400),
        ([b"1", b"65537"], 400),
        ([b"65537"], 413),
        ([b"9" * 5000], 413),
    ],
)
def test_api_rejects_invalid_or_oversized_declared_lengths_before_reading_body(
    request_settings, monkeypatch, content_lengths, expected_status, asgi_request
):
    def unexpected_pipeline():
        pytest.fail("Rejected request lengths must not reach the document pipeline.")

    monkeypatch.setattr(routes, "get_nexusrag", unexpected_pipeline)
    headers = [(b"host", b"localhost")]
    headers.extend((b"content-length", value) for value in content_lengths)
    messages, consumed = asyncio.run(asgi_request(api.create_app(), headers))
    assert [
        message["status"] for message in messages if message["type"] == "http.response.start"
    ] == [expected_status]
    assert consumed == 0


async def _body_size_app(scope, receive, send):
    size = 0
    while True:
        message = await receive()
        size += len(message.get("body", b""))
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": str(size).encode()})


@pytest.mark.parametrize("content_length", [None, b"1", b"00065536", b"0" * 5000])
@pytest.mark.parametrize("overflow", [False, True])
def test_api_stream_limit_counts_actual_bytes(content_length, overflow, asgi_request):
    headers = [(b"host", b"localhost")]
    if content_length is not None:
        headers.append((b"content-length", content_length))
    chunks = [b"x" * 32768, b"x" * 32768]
    if overflow:
        chunks.extend([b"x", b"unread"])
    app = middleware.LoopbackHostMiddleware(
        middleware.RequestBodyLimitMiddleware(_body_size_app, max_upload_bytes=1)
    )

    messages, consumed = asyncio.run(asgi_request(app, headers, chunks))

    assert [
        message["status"] for message in messages if message["type"] == "http.response.start"
    ] == [413 if overflow else 200]
    assert consumed == (3 if overflow else 2)
    if not overflow:
        assert messages[-1]["body"] == b"65536"


def test_concurrent_requests_have_independent_body_limits(asgi_request):
    app = middleware.LoopbackHostMiddleware(
        middleware.RequestBodyLimitMiddleware(_body_size_app, max_upload_bytes=1)
    )

    async def requests():
        return await asyncio.gather(
            *(
                asgi_request(app, [(b"host", b"localhost")], [b"x" * 32768, b"x" * 32768])
                for _ in range(2)
            )
        )

    for messages, consumed in asyncio.run(requests()):
        assert messages[0]["status"] == 200
        assert messages[-1]["body"] == b"65536"
        assert consumed == 2


@pytest.mark.parametrize("scope", [{"type": "lifespan"}, {"type": "websocket"}])
def test_non_http_scope_passes_through_unchanged(scope):
    calls = []

    async def receive():
        pytest.fail("Middleware must not consume non-HTTP messages.")

    async def send(message):
        pytest.fail("Middleware must not send non-HTTP messages.")

    async def app(received_scope, received_receive, received_send):
        calls.append((received_scope, received_receive, received_send))

    guarded = middleware.LoopbackHostMiddleware(
        middleware.RequestBodyLimitMiddleware(app, max_upload_bytes=1)
    )
    asyncio.run(guarded(scope, receive, send))
    assert len(calls) == 1
    assert calls[0][0] is scope
    assert calls[0][1] is receive
    assert calls[0][2] is send


@pytest.fixture
def local_client(local_settings):
    with TestClient(api.create_app(), base_url="http://localhost") as local_client:
        yield local_client


@pytest.mark.parametrize(
    "document_id",
    [
        "bad id",
        "bad.id",
        "bad$id",
        "paper\\notes",
        "pápér",
        "bad\x00id",
        "a" * 65,
        "_index",
        "_INDEX",
        "_InDeX",
    ],
)
def test_invalid_document_ids_do_not_reach_pipeline(local_client, monkeypatch, document_id):
    def unexpected_pipeline():
        pytest.fail("Invalid document IDs must be rejected before loading the pipeline.")

    monkeypatch.setattr(routes, "get_nexusrag", unexpected_pipeline)

    response = local_client.delete(f"/api/documents/{quote(document_id, safe='')}")

    assert response.status_code == 400
    assert "Invalid document ID" in response.json()["detail"]


async def test_empty_document_id_does_not_reach_pipeline(monkeypatch):
    def unexpected_pipeline():
        pytest.fail("An empty document ID must not load the pipeline.")

    monkeypatch.setattr(routes, "get_nexusrag", unexpected_pipeline)

    with pytest.raises(HTTPException) as error:
        await routes.delete_document("")

    assert error.value.status_code == 400


def test_corrupt_metadata_is_reported_as_server_error(local_client, monkeypatch, tmp_path):
    settings = Settings(data_dir=tmp_path)
    settings.storage.lancedb_path = tmp_path / "lancedb"
    rag = NexusRAG(settings)
    index_path = rag.document_store.path / "_index.json"
    corrupt_index = '{"private_document_metadata":'
    index_path.write_text(corrupt_index, encoding="utf-8")
    monkeypatch.setattr(routes, "get_nexusrag", lambda: rag)

    response = local_client.delete("/api/documents/valid_document")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to delete document"}
    assert index_path.read_text(encoding="utf-8") == corrupt_index
    assert rag._embedder is None
    assert rag._vector_store is None
    assert rag._bm25 is None
    assert rag._llm is None
    assert not settings.storage.lancedb_path.exists()


@pytest.fixture
def failure_settings(tmp_path, local_settings):
    failure_settings = local_settings
    failure_settings.data_dir = tmp_path
    failure_settings.storage.lancedb_path = tmp_path / "lancedb"
    return failure_settings


@pytest.fixture
def failure_client(failure_settings):
    with TestClient(api.create_app(), base_url="http://localhost") as failure_client:
        yield failure_client


@pytest.mark.parametrize("path", ["/api/health", "/api/metrics"])
def test_corrupt_document_index_is_not_reported_as_available(
    failure_settings, failure_client, monkeypatch, path
):
    documents = failure_settings.data_dir / "documents"
    documents.mkdir()
    (documents / "_index.json").write_text('{"private_document":', encoding="utf-8")
    pipeline = NexusRAG(failure_settings)
    monkeypatch.setattr(routes, "get_nexusrag", lambda: pipeline)

    response = failure_client.get(path)

    if path == "/api/health":
        assert response.status_code == 503
        assert response.json()["status"] == "error"
        assert response.json()["llm_available"] is False
    else:
        assert response.status_code == 500
        assert response.json() == {"detail": "Failed to retrieve metrics"}
    assert "private_document" not in response.text
    assert str(failure_settings.data_dir) not in response.text


@pytest.mark.parametrize("path", ["/api/health", "/api/metrics"])
@pytest.mark.parametrize("failure_stage", ["initialization", "statistics"])
def test_unreadable_pipeline_does_not_return_success(
    failure_client, monkeypatch, path, failure_stage
):
    def fail():
        raise OSError("private storage details")

    pipeline = SimpleNamespace(get_stats=fail)
    factory = fail if failure_stage == "initialization" else lambda: pipeline
    monkeypatch.setattr(routes, "get_nexusrag", factory)

    response = failure_client.get(path)

    if path == "/api/health":
        assert response.status_code == 503
        assert response.json()["status"] == "error"
        assert response.json()["llm_available"] is False
    else:
        assert response.status_code == 500
        assert response.json() == {"detail": "Failed to retrieve metrics"}
    assert "private storage details" not in response.text


def test_metrics_collector_failure_returns_generic_error(failure_client, monkeypatch):
    def fail():
        raise RuntimeError("private collector details")

    monkeypatch.setattr(routes, "get_metrics_collector", lambda: SimpleNamespace(snapshot=fail))

    response = failure_client.get("/api/metrics")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to retrieve metrics"}
    assert "private collector details" not in response.text


def test_unavailable_llm_keeps_valid_pipeline_health_visible(failure_client, monkeypatch):
    statistics = SystemStats(
        total_documents=3,
        total_chunks=12,
        total_words=1200,
        llm_model="fixture-generator",
        embedding_model="fixture-embedder",
        llm_available=False,
    )
    monkeypatch.setattr(
        routes, "get_nexusrag", lambda: SimpleNamespace(get_stats=lambda: statistics)
    )

    response = failure_client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm_available": False,
        "llm_model": "fixture-generator",
        "embedding_model": "fixture-embedder",
        "total_documents": 3,
        "total_chunks": 12,
        "total_words": 1200,
    }


@pytest.fixture
def pipeline(monkeypatch):
    pipeline = MagicMock(spec=NexusRAG)
    pipeline.query.return_value = RAGResponse(
        answer="Answer [2].",
        sources=[Source(index=2, chunk_id="chunk", document_id="doc", content="Evidence.")],
        confidence=0.625,
        processing_time_ms=12.36,
        warnings=["Check the cited passage."],
    )
    monkeypatch.setattr(routes, "get_nexusrag", lambda: pipeline)
    return pipeline


@pytest.fixture
def source_client(pipeline, local_settings):
    with TestClient(api.create_app(), base_url="http://localhost") as source_client:
        yield source_client


@pytest.mark.parametrize(
    "source_name,expected",
    [
        ("embedded.pdf", "embedded.pdf"),
        ("papers/embedded.pdf", "embedded.pdf"),
        (r"C:\papers/mixed\embedded.pdf", "embedded.pdf"),
        ("/", "Unknown"),
        ("", "Unknown"),
    ],
)
def test_source_filename_is_the_passage_document_basename(
    source_client, pipeline, source_name, expected
):
    pipeline.query.return_value.sources[0].document_name = source_name

    response = source_client.post("/api/query", json={"question": "Evidence?"})

    assert response.status_code == 200
    assert response.json()["sources"][0]["filename"] == expected
    pipeline.list_documents.assert_not_called()


@pytest.mark.parametrize("length", [500, 501])
def test_source_preview_preserves_wire_fields_and_truncation_boundary(
    source_client, pipeline, length
):
    source = pipeline.query.return_value.sources[0]
    source.content = "x" * length
    source.document_name = "paper.pdf"
    source.score = 0.87654

    response = source_client.post("/api/query", json={"question": "Evidence?"})

    assert response.status_code == 200
    preview = "x" * 500 + ("..." if length > 500 else "")
    assert response.json() == {
        "answer": "Answer [2].",
        "confidence": 0.625,
        "sources": [
            {
                "index": 2,
                "content": preview,
                "truncated": length > 500,
                "filename": "paper.pdf",
                "document_id": "doc",
                "section_title": "",
                "page_number": None,
                "score": 0.877,
            }
        ],
        "processing_time_ms": 12.4,
        "warnings": ["Check the cited passage."],
    }


def test_sources_keep_sparse_citation_indices_and_page_numbers(source_client, pipeline):
    pipeline.query.return_value.answer = "Answer [2] [4]."
    pipeline.query.return_value.sources.append(
        Source(
            index=4,
            chunk_id="another-chunk",
            document_id="another-doc",
            content="Additional evidence.",
            document_name="another.pdf",
            section_title="Results",
            page_number=7,
        )
    )

    response = source_client.post("/api/query", json={"question": "Evidence?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Answer [2] [4]."
    assert [source["index"] for source in data["sources"]] == [2, 4]
    assert data["sources"][1]["page_number"] == 7
    assert data["sources"][1]["section_title"] == "Results"


@pytest.fixture
def upload_app(monkeypatch, local_settings):
    settings = local_settings
    settings.api.max_upload_mb = 1
    monkeypatch.setattr(uploads, "_MAGIC", None)

    def ingest(data, filename, extension):
        return IngestResult(
            document_id="uploaded_document",
            filename=filename,
            chunk_count=1,
            word_count=1,
            success=True,
        )

    pipeline = Mock(spec=NexusRAG)
    pipeline.ingest_bytes.side_effect = ingest
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(routes, "get_nexusrag", factory)
    return SimpleNamespace(app=api.create_app(), pipeline=pipeline, factory=factory)


@pytest.mark.parametrize("control", [b"\x00", b"\t", b"\x1f", b"\x7f"])
def test_control_characters_in_upload_names_are_rejected(upload_app, control):
    body = (
        b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="bad'
        + control
        + b'.txt"\r\nContent-Type: text/plain\r\n\r\nEvidence.\r\n--boundary--\r\n'
    )
    with TestClient(upload_app.app, base_url="http://localhost") as client:
        response = client.post(
            "/api/ingest",
            content=body,
            headers={"Content-Type": "multipart/form-data; boundary=boundary"},
        )

    assert response.status_code == 400
    upload_app.factory.assert_not_called()


@pytest.mark.parametrize(
    "filename",
    ["txt", "md", ".txt", "paper.exe", "folder/", "C:\\folder\\", "a" * 256 + ".txt"],
)
def test_invalid_upload_names_do_not_reach_pipeline(upload_app, filename):
    with TestClient(upload_app.app, base_url="http://localhost") as client:
        response = client.post(
            "/api/ingest", files={"file": (filename, b"Evidence.", "text/plain")}
        )

    assert response.status_code == 400
    upload_app.factory.assert_not_called()


@pytest.mark.parametrize(
    "filename, basename, extension",
    [
        ("../../papers/Study.TXT", "Study.TXT", ".txt"),
        (r"C:\papers\Study.MD", "Study.MD", ".md"),
    ],
)
def test_upload_paths_use_basename_and_case_insensitive_extension(
    upload_app, filename, basename, extension
):
    with TestClient(upload_app.app, base_url="http://localhost") as client:
        response = client.post(
            "/api/ingest", files={"file": (filename, b"Evidence.", "text/plain")}
        )

    assert response.status_code == 200
    assert response.json()["filename"] == basename
    upload_app.pipeline.ingest_bytes.assert_called_once_with(b"Evidence.", basename, extension)


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_upload_size_boundary(upload_app, extra_bytes):
    content = b"x" * (1024 * 1024 + extra_bytes)
    with TestClient(upload_app.app, base_url="http://localhost") as client:
        response = client.post("/api/ingest", files={"file": ("paper.txt", content, "text/plain")})

    if extra_bytes:
        assert response.status_code == 413
        upload_app.factory.assert_not_called()
    else:
        assert response.status_code == 200
        upload_app.pipeline.ingest_bytes.assert_called_once_with(content, "paper.txt", ".txt")


async def test_upload_validation_does_not_block_health_requests(upload_app, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def validate(*args):
        entered.set()
        release.wait()

    monkeypatch.setattr(routes, "validate_upload", validate)
    fail_safe = threading.Timer(5, release.set)
    fail_safe.daemon = True
    fail_safe.start()
    transport = httpx.ASGITransport(app=upload_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        upload = asyncio.create_task(
            client.post("/api/ingest", files={"file": ("paper.txt", b"Evidence.", "text/plain")})
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert not release.is_set()

            health = await asyncio.wait_for(client.get("/health"), timeout=1)

            assert health.status_code == 200
            assert health.json() == {"status": "ok"}
            assert not upload.done()
            upload_app.factory.assert_not_called()
        finally:
            release.set()
            fail_safe.cancel()
            response = await asyncio.wait_for(upload, timeout=2)

    assert response.status_code == 200
    upload_app.pipeline.ingest_bytes.assert_called_once_with(b"Evidence.", "paper.txt", ".txt")


MAX_QUERY_LENGTH = get_settings().retrieval.max_query_length


def _docx_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<?xml version='1.0'?><Types/>")
    return buf.getvalue()


PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntest content"


DOCX_BYTES = _docx_bytes()


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def route_client(app):
    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def route_rag(monkeypatch):
    mock = MagicMock(spec=NexusRAG)

    mock.get_stats.return_value = SystemStats(
        total_documents=2,
        total_chunks=10,
        total_words=5000,
        llm_model="llama3.2:3b",
        embedding_model="all-MiniLM-L6-v2",
        llm_available=True,
    )

    mock.list_documents.return_value = []

    def document_inventory():
        documents = mock.list_documents()
        return DocumentInventory(
            documents=documents,
            total_chunks=sum(doc.get("chunk_count", 0) for doc in documents),
        )

    mock.get_document_inventory.side_effect = document_inventory
    monkeypatch.setattr("scinexusrag.api.routes.get_nexusrag", lambda: mock)

    return mock


@pytest.fixture
def upload_limit_bytes(monkeypatch):
    monkeypatch.setattr(get_settings().api, "max_upload_mb", 1)
    return 1024 * 1024


class TestHealthCheck:
    def test_health_check_success(self, route_client, route_rag):
        response = route_client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert data["llm_available"] is True
        assert data["llm_model"] == "llama3.2:3b"
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        assert data["total_documents"] == 2
        assert data["total_chunks"] == 10

    def test_health_check_with_error(self, route_client):
        with patch("scinexusrag.api.routes.get_nexusrag", side_effect=Exception("Test error")):
            response = route_client.get("/api/health")

            assert response.status_code == 503
            data = response.json()

            assert data["status"] == "error"
            assert data["llm_available"] is False


class TestIngestDocument:
    @pytest.mark.parametrize(
        "extension,content_type,content",
        [
            (".pdf", "application/pdf", PDF_BYTES),
            (".txt", "text/plain", b"Document text."),
            (
                ".docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                DOCX_BYTES,
            ),
            (".md", "text/markdown", b"# Document\n\nDocument text."),
        ],
    )
    def test_supported_upload_reaches_ingestion(
        self, route_client, route_rag, extension, content_type, content
    ):
        filename = f"paper{extension}"
        route_rag.ingest_bytes.return_value = IngestResult(
            document_id="uploaded_document",
            filename=filename,
            chunk_count=5,
            word_count=1000,
            success=True,
        )

        response = route_client.post(
            "/api/ingest", files={"file": (filename, content, content_type)}
        )

        assert response.status_code == 200
        assert response.json() == {
            "document_id": "uploaded_document",
            "filename": filename,
            "chunk_count": 5,
            "word_count": 1000,
            "success": True,
            "error": None,
        }
        route_rag.ingest_bytes.assert_called_once_with(content, filename, extension)

    @pytest.mark.parametrize(
        "filename, content, content_type, status, message",
        [
            ("", b"content", "text/plain", 422, None),
            (
                "a" * (MAX_FILENAME_LENGTH + 1) + ".txt",
                b"content",
                "text/plain",
                400,
                "Filename too long",
            ),
            ("test.xyz", b"content", "application/octet-stream", 400, "Unsupported file type"),
            ("test.txt", b"", "text/plain", 400, "File is empty"),
        ],
        ids=["missing-name", "long-name", "unsupported-format", "empty-file"],
    )
    def test_invalid_upload_is_rejected(
        self, route_client, route_rag, filename, content, content_type, status, message
    ):
        response = route_client.post(
            "/api/ingest", files={"file": (filename, content, content_type)}
        )

        assert response.status_code == status
        if message is not None:
            assert message in response.json()["detail"]
        route_rag.ingest_bytes.assert_not_called()

    def test_ingest_file_too_large(self, route_client, route_rag, upload_limit_bytes):
        oversized_content = b"x" * (upload_limit_bytes + 1)

        response = route_client.post(
            "/api/ingest",
            files={"file": ("large.pdf", oversized_content, "application/pdf")},
        )

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]
        assert "1MB" in response.json()["detail"]

    def test_ingest_file_at_size_limit(self, route_client, route_rag, upload_limit_bytes):
        route_rag.ingest_bytes.return_value = IngestResult(
            document_id="doc_large",
            filename="large.pdf",
            chunk_count=100,
            word_count=50000,
            success=True,
        )

        content_at_limit = b"%PDF-1.4\n" + b"x" * (upload_limit_bytes - 9)

        response = route_client.post(
            "/api/ingest",
            files={"file": ("large.pdf", content_at_limit, "application/pdf")},
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_ingest_path_traversal_in_filename(self, route_client, route_rag):
        route_rag.ingest_bytes.return_value = IngestResult(
            document_id="doc_traversal",
            filename="test.txt",
            chunk_count=1,
            word_count=100,
            success=True,
        )

        response = route_client.post(
            "/api/ingest",
            files={"file": ("../../etc/passwd.txt", b"content", "text/plain")},
        )

        assert response.status_code == 200

        route_rag.ingest_bytes.assert_called_once_with(b"content", "passwd.txt", ".txt")

    def test_ingest_ingestion_failure(self, route_client, route_rag):
        route_rag.ingest_bytes.return_value = IngestResult(
            document_id="",
            filename="test.txt",
            chunk_count=0,
            word_count=0,
            success=False,
            error="Failed to parse document",
        )

        response = route_client.post(
            "/api/ingest",
            files={"file": ("test.txt", b"invalid content", "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "Failed to parse document"


class TestQueryDocuments:
    def test_query_success(self, route_client, route_rag):
        source = Source(
            index=1,
            chunk_id="chunk_1",
            document_id="doc_123",
            document_name="test.pdf",
            content="Relevant content about the topic",
            section_title="Introduction",
            page_number=1,
            score=0.95,
        )

        route_rag.query.return_value = RAGResponse(
            answer="Paris is the capital of France.",
            sources=[source],
            confidence=0.92,
            processing_time_ms=150.5,
        )
        route_rag.list_documents.return_value = [{"id": "doc_123", "filename": "test.pdf"}]

        response = route_client.post(
            "/api/query", json={"question": "What is the capital of France?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "Paris is the capital of France."
        assert data["confidence"] == 0.92
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "test.pdf"
        assert data["processing_time_ms"] == 150.5
        route_rag.query.assert_called_once_with("What is the capital of France?")

    def test_query_surfaces_warnings(self, route_client, route_rag):
        route_rag.query.return_value = RAGResponse(
            answer="Ungrounded answer.",
            sources=[],
            confidence=0.3,
            processing_time_ms=12.0,
            warnings=["answer cites no sources"],
        )
        route_rag.list_documents.return_value = []

        data = route_client.post("/api/query", json={"question": "q?"}).json()

        assert data["warnings"] == ["answer cites no sources"]

    @pytest.mark.parametrize(
        "payload, message",
        [
            ({"question": ""}, "cannot be empty"),
            ({"question": "   \n\t   "}, None),
            ({"question": "a" * (MAX_QUERY_LENGTH + 1)}, "too long"),
            ({}, None),
        ],
        ids=["empty", "whitespace", "too-long", "missing"],
    )
    def test_invalid_query_is_rejected(self, route_client, route_rag, payload, message):
        response = route_client.post("/api/query", json=payload)

        assert response.status_code == 422
        if message is not None:
            assert message in response.json()["detail"][0]["msg"].lower()
        route_rag.query.assert_not_called()

    def test_query_at_max_length(self, route_client, route_rag):
        max_length_question = "a" * MAX_QUERY_LENGTH

        route_rag.query.return_value = RAGResponse(
            answer="Answer",
            sources=[],
            confidence=0.5,
            processing_time_ms=100.0,
        )

        response = route_client.post("/api/query", json={"question": max_length_question})

        assert response.status_code == 200

    def test_query_no_documents(self, route_client, route_rag):
        route_rag.query.return_value = RAGResponse(
            answer="No documents have been uploaded yet.",
            sources=[],
            confidence=0.0,
            processing_time_ms=10.0,
        )
        route_rag.list_documents.return_value = []

        response = route_client.post("/api/query", json={"question": "What is the answer?"})

        assert response.status_code == 200
        data = response.json()

        assert data["sources"] == []
        assert data["processing_time_ms"] == 10.0

    def test_query_multiple_sources(self, route_client, route_rag):
        sources = [
            Source(
                index=1,
                chunk_id="chunk_1",
                document_id="doc_1",
                document_name="paper1.pdf",
                content="Content from paper 1",
                section_title="Methods",
                page_number=5,
                score=0.95,
            ),
            Source(
                index=2,
                chunk_id="chunk_2",
                document_id="doc_2",
                document_name="paper2.pdf",
                content="Content from paper 2",
                section_title="Results",
                page_number=10,
                score=0.87,
            ),
        ]

        route_rag.query.return_value = RAGResponse(
            answer="Combined answer from both sources",
            sources=sources,
            confidence=0.91,
            processing_time_ms=200.0,
        )
        route_rag.list_documents.return_value = [
            {"id": "doc_1", "filename": "paper1.pdf"},
            {"id": "doc_2", "filename": "paper2.pdf"},
        ]

        response = route_client.post("/api/query", json={"question": "What did the research show?"})

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["score"] == 0.95
        assert data["sources"][1]["score"] == 0.87

    def test_query_preserves_source_indices(self, route_client, route_rag):
        sources = [
            Source(index=index, chunk_id=f"chunk_{index}", document_id="doc", content="Evidence.")
            for index in (2, 4)
        ]
        route_rag.query.return_value = RAGResponse("Answer [2] [4].", sources, 0.5)

        response = route_client.post("/api/query", json={"question": "What was measured?"})

        assert response.status_code == 200
        assert [source["index"] for source in response.json()["sources"]] == [2, 4]

    def test_query_long_source_content_truncated(self, route_client, route_rag):
        long_content = "x" * 1000
        source = Source(
            index=1,
            chunk_id="chunk_1",
            document_id="doc_123",
            document_name="test.pdf",
            content=long_content,
            section_title="Section",
            page_number=1,
            score=0.9,
        )

        route_rag.query.return_value = RAGResponse(
            answer="Answer",
            sources=[source],
            confidence=0.9,
            processing_time_ms=100.0,
        )
        response = route_client.post("/api/query", json={"question": "What is this?"})

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"][0]["content"]) <= 503
        assert data["sources"][0]["content"].endswith("...")


class TestListDocuments:
    def test_list_documents_empty(self, route_client, route_rag):
        route_rag.list_documents.return_value = []

        response = route_client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()

        assert data["documents"] == []
        assert data["total_documents"] == 0
        assert data["total_chunks"] == 0

    def test_list_documents_with_documents(self, route_client, route_rag):
        route_rag.list_documents.return_value = [
            {
                "id": "doc_1",
                "filename": "paper1.pdf",
                "word_count": 1000,
                "chunk_count": 5,
                "file_type": "pdf",
                "uploaded_at": "2024-01-15T10:30:00",
            },
            {
                "id": "doc_2",
                "filename": "paper2.pdf",
                "word_count": 800,
                "chunk_count": 4,
                "file_type": "pdf",
                "uploaded_at": "2024-01-15T11:00:00",
            },
        ]

        response = route_client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()

        assert len(data["documents"]) == 2
        assert data["total_documents"] == 2
        assert data["total_chunks"] == 9
        assert data["documents"][0]["id"] == "doc_1"
        assert data["documents"][0]["word_count"] == 1000
        assert data["documents"][1]["id"] == "doc_2"

    def test_list_documents_pagination(self, route_client, route_rag):
        route_rag.list_documents.return_value = [
            {"id": f"doc_{i}", "filename": f"p{i}.pdf", "word_count": 10} for i in range(5)
        ]

        page = route_client.get("/api/documents?limit=2&offset=2").json()

        assert [d["id"] for d in page["documents"]] == ["doc_2", "doc_3"]
        assert page["total_documents"] == 5

        assert route_client.get("/api/documents?limit=0").status_code == 422
        assert route_client.get("/api/documents?offset=-1").status_code == 422


class TestHealthStatistics:
    def test_health_reports_collection_totals(self, route_client, route_rag):
        response = route_client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert data["total_documents"] == 2
        assert data["total_chunks"] == 10
        assert data["total_words"] == 5000
        assert data["llm_available"] is True
        assert data["llm_model"] == "llama3.2:3b"
        assert data["embedding_model"] == "all-MiniLM-L6-v2"

    def test_status_route_is_gone(self, route_client):
        assert route_client.get("/api/status").status_code == 404


class TestDeleteDocument:
    def test_delete_document_success(self, route_client, route_rag):
        route_rag.delete_document.return_value = True

        response = route_client.delete("/api/documents/doc_123")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["message"] == "Document deleted"
        route_rag.delete_document.assert_called_once_with("doc_123")

    def test_delete_document_not_found(self, route_client, route_rag):
        route_rag.delete_document.return_value = False

        response = route_client.delete("/api/documents/nonexistent")

        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_trailing_slash_uses_clear_all_route(self, route_client, route_rag):
        response = route_client.delete("/api/documents/")

        assert response.status_code == 200
        route_rag.clear_all.assert_called_once()

    def test_delete_document_id_too_long(self, route_client, route_rag):
        long_id = "a" * 65

        response = route_client.delete(f"/api/documents/{long_id}")

        assert response.status_code == 400
        assert "Invalid document ID" in response.json()["detail"]


class TestClearAllDocuments:
    def test_clear_all_documents_success(self, route_client, route_rag):
        response = route_client.delete("/api/documents")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["message"] == "All documents cleared"
        route_rag.clear_all.assert_called_once()


class TestPydanticValidation:
    @pytest.mark.parametrize(
        "question, message",
        [
            ("", "Question cannot be empty"),
            ("   \n\t   ", "Question cannot be empty"),
            ("a" * (MAX_QUERY_LENGTH + 1), "too long"),
        ],
        ids=["empty", "whitespace", "too-long"],
    )
    def test_invalid_query_request(self, question, message):
        with pytest.raises(ValidationError, match=message):
            QueryRequest(question=question)

    @pytest.mark.parametrize(
        "question, expected",
        [
            ("What is the meaning of life?", "What is the meaning of life?"),
            ("  What is this?  \n", "What is this?"),
        ],
        ids=["unchanged", "trimmed"],
    )
    def test_valid_query_request(self, question, expected):
        assert QueryRequest(question=question).question == expected


class TestRouteErrors:
    @pytest.mark.parametrize(
        "method, path, operation, error, message",
        [
            (
                "POST",
                "/api/ingest",
                "ingest_bytes",
                Exception("Database error"),
                "Failed to process document",
            ),
            ("POST", "/api/query", "query", Exception("LLM error"), "Failed to process query"),
            (
                "GET",
                "/api/documents",
                "list_documents",
                Exception("Storage error"),
                "Failed to retrieve documents",
            ),
            (
                "DELETE",
                "/api/documents/doc_123",
                "delete_document",
                Exception("Deletion error"),
                "Failed to delete document",
            ),
            (
                "DELETE",
                "/api/documents/invalid-id-format",
                "delete_document",
                ValueError("Invalid stored metadata"),
                "Failed to delete document",
            ),
            (
                "DELETE",
                "/api/documents",
                "clear_all",
                Exception("Clear error"),
                "Failed to clear documents",
            ),
        ],
        ids=["ingest", "query", "inventory", "delete", "invalid-metadata", "clear"],
    )
    def test_backend_errors_are_reported(
        self, route_client, route_rag, method, path, operation, error, message
    ):
        getattr(route_rag, operation).side_effect = error
        kwargs = {}
        if operation == "ingest_bytes":
            kwargs["files"] = {"file": ("test.txt", b"content", "text/plain")}
        elif operation == "query":
            kwargs["json"] = {"question": "What is the answer?"}

        response = route_client.request(method, path, **kwargs)

        assert response.status_code == 500
        assert response.json()["detail"] == message

    def test_missing_content_type_header(self, route_client, route_rag):
        route_rag.ingest_bytes.return_value = IngestResult(
            document_id="doc_1",
            filename="test.txt",
            chunk_count=1,
            word_count=100,
            success=True,
        )

        response = route_client.post(
            "/api/ingest",
            content=(
                b'--boundary\r\nContent-Disposition: form-data; name="file"; '
                b'filename="test.txt"\r\n\r\ncontent\r\n--boundary--\r\n'
            ),
            headers={"Content-Type": "multipart/form-data; boundary=boundary"},
        )

        assert response.status_code == 200
        route_rag.ingest_bytes.assert_called_once_with(b"content", "test.txt", ".txt")

    def test_invalid_json_in_query(self, route_client, route_rag):
        response = route_client.post(
            "/api/query",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


@pytest.fixture
def application_client(route_rag):
    return TestClient(create_app(), base_url="http://localhost")


class TestEmbeddingConfigErrors:
    @staticmethod
    def assert_conflict(response):
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "EMBEDDING_REVISION" in detail
        assert "DATA_DIR" in detail
        assert "LANCEDB_PATH" in detail
        assert "original documents" in detail
        assert "private embedding metadata" not in detail

    @pytest.mark.parametrize("path", ["/api/health", "/api/documents", "/api/metrics"])
    def test_count_conflict_is_not_reported_as_empty_or_healthy(
        self, application_client, route_rag, path
    ):
        method = "get_document_inventory" if path == "/api/documents" else "get_stats"
        getattr(route_rag, method).side_effect = EmbeddingConfigError("private embedding metadata")

        self.assert_conflict(application_client.get(path))

    def test_query_conflict_is_actionable_and_does_not_expose_metadata(
        self, application_client, route_rag
    ):
        route_rag.query.side_effect = EmbeddingConfigError("private embedding metadata")

        response = application_client.post("/api/query", json={"question": "What was measured?"})

        self.assert_conflict(response)

    def test_ingest_conflict_returns_409(self, application_client, route_rag):
        route_rag.ingest_bytes.side_effect = EmbeddingConfigError("private embedding metadata")

        response = application_client.post(
            "/api/ingest", files={"file": ("paper.txt", b"Sensor readings.", "text/plain")}
        )

        self.assert_conflict(response)

    def test_real_ingest_rejects_same_dimension_model_change(self, tmp_path):
        settings = Settings(
            data_dir=tmp_path,
            storage=StorageSettings(lancedb_path=tmp_path / "lancedb"),
            embedding=EmbeddingSettings(model="test-model", revision="a" * 40),
        )
        config = EmbeddingConfig("test-model", "a" * 40, "", "", True, 4)
        original_embedder = MagicMock(spec=Embedder)
        original_embedder.embedding_config = config
        original_embedder.embed.return_value = np.array([[1, 0, 0, 0]], dtype=np.float32)
        original = NexusRAG(settings)
        original._embedder = original_embedder
        result = original.ingest_bytes(
            b"Sensor A records temperature every thirty seconds.", "original.txt", ".txt"
        )
        assert result.success
        assert original.vector_store.count() == 1

        changed = NexusRAG(settings)
        changed_embedder = MagicMock(spec=Embedder)
        changed_embedder.embedding_config = replace(config, model_name="different-model")
        changed._embedder = changed_embedder
        with (
            patch("scinexusrag.api.routes.get_nexusrag", return_value=changed),
            TestClient(create_app(), base_url="http://localhost") as route_client,
        ):
            self.assert_conflict(
                route_client.post(
                    "/api/ingest",
                    files={"file": ("new.txt", b"Sensor B records pressure.", "text/plain")},
                )
            )
            self.assert_conflict(
                route_client.post("/api/query", json={"question": "What did the sensor record?"})
            )

        changed_embedder.embed.assert_not_called()
        changed_embedder.embed_query.assert_not_called()
        assert changed._llm is None
        reopened = NexusRAG(settings)
        reopened._embedder = original_embedder
        assert reopened.vector_store.count() == 1
        assert [document["id"] for document in reopened.list_documents()] == [result.document_id]

    def test_delete_conflict_returns_409(self, application_client, route_rag):
        route_rag.delete_document.side_effect = EmbeddingConfigError("private embedding metadata")

        self.assert_conflict(application_client.delete("/api/documents/paper"))

    def test_explicit_clear_does_not_require_compatible_stats(self, application_client, route_rag):
        route_rag.get_stats.side_effect = EmbeddingConfigError("private embedding metadata")

        response = application_client.delete("/api/documents")

        assert response.status_code == 200
        route_rag.clear_all.assert_called_once()
        route_rag.get_stats.assert_not_called()

    def test_auth_runs_before_index_checks(self, application_client, route_rag):
        route_rag.get_stats.side_effect = EmbeddingConfigError("private embedding metadata")
        with patch(
            "scinexusrag.api.security.get_settings", return_value=_route_settings(api_key="secret")
        ):
            response = application_client.get("/api/health")

        assert response.status_code == 401
        route_rag.get_stats.assert_not_called()


class TestStorageErrors:
    @staticmethod
    def assert_unavailable(response):
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "vector index is unavailable" in detail
        assert "server logs" in detail
        assert "private database details" not in detail
        assert "No documents" not in detail

    @pytest.mark.parametrize(
        ("method", "path", "failed_method"),
        [
            ("GET", "/api/health", "get_stats"),
            ("GET", "/api/documents", "get_document_inventory"),
            ("GET", "/api/metrics", "get_stats"),
            ("POST", "/api/query", "query"),
            ("DELETE", "/api/documents/paper", "delete_document"),
            ("DELETE", "/api/documents", "clear_all"),
        ],
    )
    def test_storage_failure_returns_503(
        self, application_client, route_rag, method, path, failed_method
    ):
        getattr(route_rag, failed_method).side_effect = VectorStoreError("private database details")

        response = application_client.request(
            method, path, json={"question": "What was measured?"} if method == "POST" else None
        )

        self.assert_unavailable(response)

    def test_ingest_storage_failure_returns_503(self, application_client, route_rag):
        route_rag.ingest_bytes.side_effect = VectorStoreError("private database details")

        response = application_client.post(
            "/api/ingest", files={"file": ("paper.txt", b"Sensor readings.", "text/plain")}
        )

        self.assert_unavailable(response)

    def test_failed_real_index_count_is_distinct_from_an_empty_corpus(self, tmp_path, monkeypatch):
        route_rag = NexusRAG(
            Settings(
                data_dir=tmp_path,
                storage=StorageSettings(lancedb_path=tmp_path / "lancedb"),
            )
        )
        embedder = MagicMock(spec=Embedder)
        embedder.embedding_config = EmbeddingConfig("fixture", "a" * 40, "", "", True, 4)
        route_rag._embedder = embedder
        with (
            patch("scinexusrag.api.routes.get_nexusrag", return_value=route_rag),
            TestClient(create_app(), base_url="http://localhost") as route_client,
        ):
            empty = route_client.post("/api/query", json={"question": "What was measured?"})
            assert empty.status_code == 200
            assert "No documents" in empty.json()["answer"]

            def fail_count():
                raise OSError("private database details")

            monkeypatch.setattr(route_rag.vector_store.table, "count_rows", fail_count)
            self.assert_unavailable(
                route_client.post("/api/query", json={"question": "What was measured?"})
            )

        embedder.embed_query.assert_not_called()
        assert route_rag._orchestrator is None


def _route_settings(**api_overrides) -> Settings:
    settings = Settings()
    for key, value in api_overrides.items():
        setattr(settings.api, key, value)
    return settings


class TestApiKeyAuth:
    @pytest.mark.parametrize("key", [None, "wrong"], ids=["missing", "wrong"])
    def test_invalid_key_is_rejected(self, route_client, route_rag, key):
        with patch(
            "scinexusrag.api.security.get_settings", return_value=_route_settings(api_key="secret")
        ):
            response = route_client.get("/api/health", headers={"X-API-Key": key} if key else {})

        assert response.status_code == 401
        route_rag.get_stats.assert_not_called()

    def test_correct_key_allowed(self, route_client, route_rag):
        with patch(
            "scinexusrag.api.security.get_settings", return_value=_route_settings(api_key="secret")
        ):
            response = route_client.get("/api/health", headers={"X-API-Key": "secret"})
            assert response.status_code == 200


class TestUploadValidationThroughRoute:
    def test_wrong_magic_bytes_rejected(self, route_client, route_rag):
        files = {"file": ("bad.pdf", b"this is not a pdf at all", "application/pdf")}
        response = route_client.post("/api/ingest", files=files)
        assert response.status_code == 415

    def test_zip_bomb_rejected(self, route_client, route_rag):

        with patch(
            "scinexusrag.api.uploads.get_settings",
            return_value=_route_settings(max_uncompressed_mb=0),
        ):
            files = {
                "file": (
                    "doc.docx",
                    DOCX_BYTES,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            }
            response = route_client.post("/api/ingest", files=files)
            assert response.status_code == 413


def _security_settings(**api_overrides: object) -> Settings:
    settings = Settings()
    for key, value in api_overrides.items():
        setattr(settings.api, key, value)
    return settings


class _MagicDatabaseError(Exception):
    pass


class TestMimeDetectorLoading:
    def test_missing_optional_package_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "magic", None)
        assert uploads._load_magic() is None

    @pytest.mark.parametrize("error", [OSError, _MagicDatabaseError])
    def test_unavailable_native_detector_returns_none(self, monkeypatch, error):
        def create_detector(**kwargs):
            raise error("Native MIME detector unavailable")

        monkeypatch.setitem(
            sys.modules,
            "magic",
            SimpleNamespace(Magic=create_detector, MagicException=_MagicDatabaseError),
        )
        assert uploads._load_magic() is None

    def test_successful_detector_is_returned(self, monkeypatch):
        detector = SimpleNamespace(from_buffer=lambda data: "text/plain")
        monkeypatch.setitem(
            sys.modules,
            "magic",
            SimpleNamespace(Magic=lambda **kwargs: detector, MagicException=_MagicDatabaseError),
        )
        assert uploads._load_magic() is detector

    def test_unexpected_initialization_error_is_not_suppressed(self, monkeypatch):
        def create_detector(**kwargs):
            raise RuntimeError("Detector initialization failed")

        monkeypatch.setitem(
            sys.modules,
            "magic",
            SimpleNamespace(Magic=create_detector, MagicException=_MagicDatabaseError),
        )
        with pytest.raises(RuntimeError, match="Detector initialization failed"):
            uploads._load_magic()


class TestApiKey:
    async def test_no_key_configured_is_open(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _security_settings(api_key=""))
        assert await security.require_api_key(None) is None

    @pytest.mark.parametrize("key", [None, "nope"], ids=["missing", "wrong"])
    async def test_invalid_key_is_rejected(self, monkeypatch, key):
        monkeypatch.setattr(
            security, "get_settings", lambda: _security_settings(api_key="test-only-key")
        )
        with pytest.raises(HTTPException) as exc:
            await security.require_api_key(key)
        assert exc.value.status_code == 401
        assert exc.value.headers == {"WWW-Authenticate": "ApiKey"}

    async def test_correct_key_accepted(self, monkeypatch) -> None:
        monkeypatch.setattr(
            security, "get_settings", lambda: _security_settings(api_key="test-only-key")
        )
        assert await security.require_api_key("test-only-key") is None

    async def test_non_ascii_key_rejected_not_crash(self, monkeypatch) -> None:
        monkeypatch.setattr(
            security, "get_settings", lambda: _security_settings(api_key="test-only-key")
        )
        with pytest.raises(HTTPException) as exc:
            await security.require_api_key("sécrét-ключ-🔑")
        assert exc.value.status_code == 401


class TestRateLimiter:
    def test_limit_strings_track_settings(self, monkeypatch) -> None:
        monkeypatch.setattr(
            security,
            "get_settings",
            lambda: _security_settings(query_rate_per_minute=42, upload_rate_per_minute=7),
        )
        assert security.query_limit() == "42/minute"
        assert security.upload_limit() == "7/minute"

    def test_slowapi_returns_429_over_limit(self) -> None:
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

        client = TestClient(app, base_url="http://localhost")
        assert [client.get("/x").status_code for _ in range(3)] == [200, 200, 429]


class TestSameSiteGuard:
    @staticmethod
    def _request(headers: dict[str, str], *, scheme: str = "http", root_path: str = "") -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "scheme": scheme,
                "server": ("localhost", 8000),
                "path": f"{root_path}/api/query",
                "root_path": root_path,
                "query_string": b"",
                "headers": [
                    (key.lower().encode("ascii"), value.encode("ascii"))
                    for key, value in headers.items()
                ],
            }
        )

    @pytest.mark.parametrize("fetch_site", ["cross-site", "unknown", ""])
    async def test_unsafe_fetch_metadata_rejected(self, fetch_site) -> None:
        with pytest.raises(HTTPException) as exc:
            await security.require_same_site(self._request({"sec-fetch-site": fetch_site}))
        assert exc.value.status_code == 403

    @pytest.mark.parametrize("fetch_site", ["same-origin", "same-site", "none", "Same-Origin"])
    async def test_safe_fetch_metadata_without_origin_allowed(self, fetch_site) -> None:
        assert (
            await security.require_same_site(self._request({"sec-fetch-site": fetch_site})) is None
        )

    async def test_non_browser_client_allowed(self) -> None:
        assert await security.require_same_site(self._request({})) is None

    async def test_foreign_origin_fallback_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _security_settings())
        with pytest.raises(HTTPException) as exc:
            await security.require_same_site(self._request({"origin": "http://evil.example"}))
        assert exc.value.status_code == 403

    async def test_allowlisted_origin_fallback_allowed(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _security_settings())
        allowed = _security_settings().api.cors_origins[0]
        assert await security.require_same_site(self._request({"origin": allowed})) is None

    @pytest.mark.parametrize("fetch_site", ["same-origin", "same-site", "none"])
    async def test_safe_fetch_metadata_does_not_allow_foreign_origin(
        self, monkeypatch, fetch_site
    ) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _security_settings())
        with pytest.raises(HTTPException) as exc:
            await security.require_same_site(
                self._request({"sec-fetch-site": fetch_site, "origin": "http://evil.example"})
            )
        assert exc.value.status_code == 403

    async def test_unsafe_fetch_metadata_rejects_allowlisted_origin(self, monkeypatch) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _security_settings())
        with pytest.raises(HTTPException) as exc:
            await security.require_same_site(
                self._request(
                    {
                        "sec-fetch-site": "cross-site",
                        "origin": _security_settings().api.cors_origins[0],
                    }
                )
            )
        assert exc.value.status_code == 403

    @pytest.mark.parametrize(
        "scheme,host,root_path",
        [
            ("http", "localhost:9123", ""),
            ("https", "research.example", ""),
            ("https", "research.example:9443", "/research"),
            ("http", "[::1]:9123", "/research"),
        ],
    )
    async def test_request_origin_allowed_with_custom_host_and_mount(
        self, monkeypatch, scheme, host, root_path
    ) -> None:
        monkeypatch.setattr(security, "get_settings", lambda: _security_settings(cors_origins=[]))
        request = self._request(
            {"host": host, "origin": f"{scheme}://{host}"}, scheme=scheme, root_path=root_path
        )
        assert await security.require_same_site(request) is None

    @pytest.mark.parametrize(
        "origin", ["", "http://localhost:9123", "https://localhost:8000", "null"]
    )
    async def test_different_port_scheme_or_opaque_origin_rejected(self, monkeypatch, origin):
        monkeypatch.setattr(security, "get_settings", lambda: _security_settings(cors_origins=[]))
        with pytest.raises(HTTPException) as exc:
            await security.require_same_site(self._request({"origin": origin}))
        assert exc.value.status_code == 403


class TestValidateUpload:
    @pytest.fixture(autouse=True)
    def upload_settings(self, monkeypatch):
        monkeypatch.setattr(uploads, "get_settings", lambda: _security_settings())
        monkeypatch.setattr(uploads, "_MAGIC", None)

    @pytest.fixture
    def contents(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("a.xml", "<x/>")
        return {
            ".pdf": b"%PDF-1.4\nhi",
            ".docx": buffer.getvalue(),
            ".txt": b"Document text.",
            ".md": b"# Document\n\nDocument text.",
        }

    def test_pdf_magic_required_without_mime_detector(self) -> None:
        with pytest.raises(HTTPException) as exc:
            uploads.validate_upload(".pdf", b"not a pdf", "application/pdf")
        assert exc.value.status_code == 415

    @pytest.mark.parametrize(
        "extension,content_type",
        [
            (".pdf", "application/pdf"),
            (".docx", "application/zip"),
            (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            (".txt", "text/plain"),
            (".txt", "text/markdown"),
            (".md", "text/plain"),
            (".md", "text/markdown"),
        ],
    )
    def test_matching_content_types_accepted(self, contents, extension, content_type):
        uploads.validate_upload(extension, contents[extension], content_type)

    @pytest.mark.parametrize("extension", [".pdf", ".docx", ".txt", ".md"])
    @pytest.mark.parametrize("content_type", [None, "", "application/octet-stream"])
    def test_generic_content_types_accepted(self, contents, extension, content_type):
        uploads.validate_upload(extension, contents[extension], content_type)

    @pytest.mark.parametrize(
        "extension,content_type",
        [
            (".pdf", "text/plain"),
            (".docx", "application/pdf"),
            (".txt", "application/zip"),
            (".md", "application/pdf"),
        ],
    )
    def test_content_type_for_another_extension_rejected(self, contents, extension, content_type):
        with pytest.raises(HTTPException) as exc:
            uploads.validate_upload(extension, contents[extension], content_type)
        assert exc.value.status_code == 415

    def test_content_type_parameters_and_case_accepted(self):
        uploads.validate_upload(".md", b"Document text.", " Text/Markdown ; charset=UTF-8 ")

    def test_extension_case_is_normalized(self, contents):
        uploads.validate_upload(".PDF", contents[".pdf"], "application/pdf")

    def test_unsupported_extension_rejected(self):
        with pytest.raises(HTTPException) as exc:
            uploads.validate_upload(".zip", b"Document text.", "application/octet-stream")
        assert exc.value.status_code == 415

    def test_disallowed_content_type(self) -> None:
        with pytest.raises(HTTPException) as exc:
            uploads.validate_upload(".txt", b"hello", "application/x-sh")
        assert exc.value.status_code == 415

    @pytest.mark.parametrize("extension", [".txt", ".md"])
    def test_text_must_be_utf8_without_mime_detector(self, extension) -> None:
        with pytest.raises(HTTPException) as exc:
            uploads.validate_upload(extension, b"\xff\xfe\x00binary", "text/plain")
        assert exc.value.status_code == 415

    @pytest.mark.parametrize("extra_byte", [False, True])
    def test_file_size_boundary_counts_utf8_bytes(self, monkeypatch, extra_byte) -> None:
        monkeypatch.setattr(uploads, "get_settings", lambda: _security_settings(max_upload_mb=1))
        text = "é" * (1024 * 1024 // 2) + ("x" if extra_byte else "")
        if extra_byte:
            with pytest.raises(HTTPException) as exc:
                uploads.validate_upload(".txt", text, "text/plain")
            assert exc.value.status_code == 413
        else:
            uploads.validate_upload(".txt", text, "text/plain")

    def test_invalid_docx_rejected_without_mime_detector(self) -> None:
        with pytest.raises(HTTPException) as exc:
            uploads.validate_upload(".docx", b"PK\x03\x04garbage", "application/zip")
        assert exc.value.status_code == 415

    @pytest.mark.parametrize(
        "extension,detected",
        [
            (".pdf", "text/plain"),
            (".docx", "application/pdf"),
            (".txt", "application/zip"),
            (".md", "application/pdf"),
        ],
    )
    def test_sniffed_content_mismatch_rejected(self, monkeypatch, contents, extension, detected):
        monkeypatch.setattr(uploads, "_MAGIC", SimpleNamespace(from_buffer=lambda data: detected))
        with pytest.raises(HTTPException) as exc:
            uploads.validate_upload(extension, contents[extension], "application/octet-stream")
        assert exc.value.status_code == 415

    @pytest.mark.parametrize("detected", ["text/x-rst", "TEXT/PLAIN", "inode/x-empty"])
    def test_sniffed_text_variants_accepted(self, monkeypatch, detected):
        monkeypatch.setattr(uploads, "_MAGIC", SimpleNamespace(from_buffer=lambda data: detected))
        data = b"" if detected == "inode/x-empty" else b"Document text."
        uploads.validate_upload(".md", data, "text/markdown")

    def test_mime_detector_failure_is_not_suppressed(self, monkeypatch):
        def fail(_data):
            raise OSError("MIME detector failed")

        monkeypatch.setattr(uploads, "_MAGIC", SimpleNamespace(from_buffer=fail))
        with pytest.raises(OSError, match="MIME detector failed"):
            uploads.validate_upload(".txt", b"Document text.", "text/plain")

    @pytest.mark.parametrize("extra_byte", [False, True])
    def test_docx_size_boundary_sums_archive_members(self, monkeypatch, extra_byte):
        monkeypatch.setattr(
            uploads, "get_settings", lambda: _security_settings(max_uncompressed_mb=1)
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("a.xml", b"A" * (1024 * 1024 // 2))
            archive.writestr("b.xml", b"B" * (1024 * 1024 // 2 + int(extra_byte)))
        if extra_byte:
            with pytest.raises(HTTPException) as exc:
                uploads.validate_upload(".docx", buffer.getvalue(), "application/zip")
            assert exc.value.status_code == 413
        else:
            uploads.validate_upload(".docx", buffer.getvalue(), "application/zip")


class TestAppConfig:
    def test_cors_default_is_not_wildcard(self) -> None:

        origins = APISettings().cors_origins
        assert "*" not in origins
        assert origins and all(o.startswith("http://") for o in origins)

    def test_docs_disabled_when_api_key_set(self, monkeypatch) -> None:

        monkeypatch.setattr(api, "get_settings", lambda: _security_settings(api_key="k"))
        client = TestClient(api.create_app(), base_url="http://localhost")
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_docs_enabled_in_local_mode(self, monkeypatch) -> None:

        monkeypatch.setattr(
            api, "get_settings", lambda: _security_settings(api_key="", docs_enabled=True)
        )
        client = TestClient(api.create_app(), base_url="http://localhost")
        assert client.get("/docs").status_code == 200


class TestSettings:
    def test_api_keys_are_excluded_from_settings_representations(self, monkeypatch):

        key = "test-only-private-key"
        monkeypatch.setenv("NEXUSRAG_API_KEY", key)
        settings = Settings()

        assert settings.api.api_key == key
        assert key not in repr(settings)
        assert key not in repr(settings.api)
        assert key not in settings.model_dump_json()
        assert "api_key" not in settings.model_dump()["api"]

    def test_default_settings(self):

        settings = Settings()

        assert settings.llm.model == "llama3.2:3b"
        assert settings.llm.base_url == "http://localhost:11434"
        assert settings.embedding.model == "BAAI/bge-small-en-v1.5"
        assert settings.ingestion.chunk_size == 1200
        assert settings.retrieval.top_k == 8
        assert settings.self_correction.enabled is True
        assert settings.log_level == "INFO"

    def test_nested_settings_types(self):

        settings = Settings()

        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.embedding, EmbeddingSettings)
        assert isinstance(settings.ingestion, IngestionSettings)
        assert isinstance(settings.retrieval, RetrievalSettings)
        assert isinstance(settings.self_correction, SelfCorrectionSettings)
        assert isinstance(settings.storage, StorageSettings)
        assert isinstance(settings.api, APISettings)

    def test_env_override_llm(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:8080")
        monkeypatch.setenv("LLM_MODEL", "mistral:7b")

        llm = LLMSettings()

        assert llm.base_url == "http://custom:8080"
        assert llm.model == "mistral:7b"

    def test_env_override_embedding(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding-model")
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")

        embedding = EmbeddingSettings()

        assert embedding.model == "custom-embedding-model"
        assert embedding.device == "cuda"

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = Settings()

        assert settings.log_level == "DEBUG"

    def test_storage_path_types(self):

        settings = Settings()

        assert isinstance(settings.storage.lancedb_path, Path)
        assert isinstance(settings.data_dir, Path)

    def test_api_settings(self):

        api = APISettings()

        assert api.host == "127.0.0.1"
        assert api.port == 8000

    def test_ingestion_settings_values(self):

        ingestion = IngestionSettings()

        assert ingestion.chunk_size > 0
        assert ingestion.chunk_overlap >= 0
        assert ingestion.chunk_overlap < ingestion.chunk_size
        assert ingestion.min_chunk_size > 0

    def test_retrieval_settings_values(self):

        retrieval = RetrievalSettings()

        assert retrieval.top_k > 0
        assert retrieval.max_query_length > 0

    def test_self_correction_settings(self):

        correction = SelfCorrectionSettings()

        assert correction.enabled is True
        assert 0 <= correction.confidence_tau <= 1
        assert correction.feedback_docs > 0 and correction.feedback_terms > 0

    def test_get_settings_caching(self):

        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_nested_llm_settings_read_environment(self, monkeypatch):

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-file:1234")
        monkeypatch.setenv("LLM_MODEL", "from-env-file")

        settings = Settings()

        assert settings.llm.base_url == "http://env-file:1234"
        assert settings.llm.model == "from-env-file"


@pytest.mark.parametrize("guard,status", [("auth", 401), ("origin", 403), ("rate", 429)])
async def test_upload_guards_run_before_multipart_parsing(
    local_settings, route_rag, monkeypatch, asgi_request, guard, status
):
    import starlette.formparsers

    local_settings.api.api_key = "test-only-key"
    local_settings.api.upload_rate_per_minute = 1
    request = httpx.Request(
        "POST", "http://localhost/api/ingest", files={"file": ("paper.txt", b"Evidence.")}
    )
    body = request.read()
    headers = [
        (b"host", b"localhost"),
        (b"content-type", request.headers["content-type"].encode()),
    ]
    if guard != "auth":
        headers.append((b"x-api-key", b"test-only-key"))
    if guard == "origin":
        headers.append((b"origin", b"https://foreign.example"))
    application = api.create_app()
    monkeypatch.setattr(security.limiter, "enabled", True)
    security.limiter.reset()
    try:
        if guard == "rate":
            route_rag.ingest_bytes.return_value = IngestResult("paper", "paper.txt", 1, 1, True)
            messages, consumed = await asgi_request(
                application, headers, [body], path="/api/ingest"
            )
            assert messages[0]["status"] == 200
            assert consumed == 1
            route_rag.ingest_bytes.reset_mock()
        spool = Mock(side_effect=AssertionError("Rejected upload created a temporary file"))
        monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", spool)
        messages, consumed = await asgi_request(application, headers, [body], path="/api/ingest")
        assert messages[0]["status"] == status
        assert consumed == 0
        spool.assert_not_called()
        route_rag.ingest_bytes.assert_not_called()
    finally:
        security.limiter.reset()


@pytest.mark.parametrize(
    "outcome,status",
    [
        ("success", 200),
        ("filename", 400),
        ("size", 413),
        ("failure", 500),
        ("field", 422),
        ("extra", 400),
    ],
)
def test_multipart_files_close_on_every_exit(
    application_client, route_rag, upload_limit_bytes, monkeypatch, outcome, status
):
    import starlette.formparsers

    temporary_files = []
    create_file = starlette.formparsers.SpooledTemporaryFile

    def track_file(*args, **kwargs):
        file = create_file(*args, **kwargs)
        temporary_files.append(file)
        return file

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", track_file)
    route_rag.ingest_bytes.return_value = IngestResult("paper", "paper.txt", 1, 1, True)
    if outcome == "failure":
        route_rag.ingest_bytes.side_effect = RuntimeError("Ingestion failed")
    content = b"x" * (upload_limit_bytes + 1) if outcome == "size" else b"Evidence."
    filename = "paper.xyz" if outcome == "filename" else "paper.txt"
    field = "other" if outcome == "field" else "file"
    files = [(field, (filename, content, "text/plain"))]
    if outcome == "extra":
        files.append(("file", ("extra.txt", b"More evidence.", "text/plain")))

    response = application_client.post("/api/ingest", files=files)

    assert response.status_code == status
    assert temporary_files
    assert all(file.closed for file in temporary_files)


def test_ingest_openapi_retains_multipart_upload(application_client):
    operation = application_client.get("/openapi.json").json()["paths"]["/api/ingest"]["post"]
    body = operation["requestBody"]
    schema = body["content"]["multipart/form-data"]["schema"]
    assert body["required"] is True
    assert schema["required"] == ["file"]
    assert schema["properties"]["file"]["type"] == "string"
    assert schema["properties"]["file"]["format"] == "binary"


@pytest.mark.parametrize("binary", [False, True])
def test_missing_file_error_excludes_unrelated_form_content(application_client, route_rag, binary):
    marker = "private-form-marker"
    response = application_client.post(
        "/api/ingest",
        files={"other": ("payload.bin", b"\xff" * (1024 * 1024 + 1))} if binary else None,
        data=None if binary else {"other": marker},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "file"]
    assert all("input" not in error and "ctx" not in error for error in detail)
    assert marker not in response.text
    assert "SpooledTemporaryFile" not in response.text
    route_rag.ingest_bytes.assert_not_called()
