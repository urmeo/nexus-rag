"""Shared fixtures."""

import asyncio
from pathlib import Path
from queue import Queue
from threading import Lock, RLock, get_ident

import pytest


@pytest.fixture(autouse=True)
def _disable_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    from scinexusrag.api.security import limiter

    monkeypatch.setattr(limiter, "enabled", False)


@pytest.fixture(autouse=True)
def _reset_scinexusrag_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    import scinexusrag.api.dependencies as dependencies

    monkeypatch.setattr(dependencies, "_instance", None)


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.txt"
    content = """INTRODUCTION

This parser fixture contains temperature readings from two sensors.
Both sensors report values in degrees Celsius.

METHODOLOGY

Read both sensors at the same interval.
Store one row per timestamp.

RESULTS

The fixture contains timestamped readings.
These values are test data, not study results.

CONCLUSION

Both sensor columns use the same units.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_markdown_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.md"
    content = """# Research Paper Title

## Abstract

This test fixture contains two sensor readings per timestamp.

## Introduction

Both sensors record temperature in degrees Celsius.
The reference sensor is labeled A; the second sensor is labeled B.

## Methods

### Data Collection

Read both sensors at thirty-second intervals.

### Analysis

Compare readings with matching timestamps.

## Results

The fixture contains one row per timestamp.

## Conclusion

Both sensor columns use the same units.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def parsed_document(sample_text_file):
    from scinexusrag.ingestion.parser import DocumentParser

    parser = DocumentParser()
    return parser.parse(sample_text_file)


class ObservedLock:
    def __init__(self, lock):
        self.lock = lock
        self.attempts = Queue()

    def __enter__(self):
        self.attempts.put(get_ident())
        if not self.lock.acquire(timeout=5):
            raise TimeoutError("Test lock was not released")
        return self

    def __exit__(self, *_args):
        self.lock.release()

    def wait_for_attempts(self, count):
        return [self.attempts.get(timeout=5) for _ in range(count)]

    def wait_for_threads(self, count):
        threads = set()
        while len(threads) < count:
            threads.add(self.attempts.get(timeout=5))
        return threads


@pytest.fixture
def observed_lock():
    return ObservedLock(Lock())


@pytest.fixture
def observed_write_lock():
    return ObservedLock(RLock())


@pytest.fixture
def local_settings(monkeypatch):
    import scinexusrag.api.app as api
    from scinexusrag.api import routes, security
    from scinexusrag.settings import Settings

    settings = Settings()
    settings.api.host = "127.0.0.1"
    settings.api.api_key = ""
    for module in (api, routes, security):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def asgi_request():
    async def request(app, headers, chunks=(), *, method="POST", path="/api/query", root_path=""):
        messages = []
        consumed = 0

        async def receive():
            nonlocal consumed
            if consumed >= len(chunks):
                pytest.fail("The application read beyond the supplied request body.")
            body = chunks[consumed]
            consumed += 1
            await asyncio.sleep(0)
            return {"type": "http.request", "body": body, "more_body": consumed < len(chunks)}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "root_path": root_path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("localhost", 8000),
        }
        await app(scope, receive, send)
        return messages, consumed

    return request
