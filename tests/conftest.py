"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _disable_rate_limits() -> Generator[None, None, None]:
    """Disable the slowapi limiter so route tests are not throttled."""
    from scinexusrag.api.security import limiter

    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture(autouse=True)
def _reset_scinexusrag_singleton() -> Generator[None, None, None]:
    """Keep the global singleton from leaking a stale instance across tests."""
    import scinexusrag.pipeline as pipeline

    pipeline._instance = None
    yield
    pipeline._instance = None


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    file_path = temp_dir / "sample.txt"
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
def sample_markdown_file(temp_dir: Path) -> Path:
    file_path = temp_dir / "sample.md"
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
def empty_file(temp_dir: Path) -> Path:
    file_path = temp_dir / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def parsed_document(sample_text_file):
    from scinexusrag.ingestion import DocumentParser

    parser = DocumentParser()
    return parser.parse(sample_text_file)
