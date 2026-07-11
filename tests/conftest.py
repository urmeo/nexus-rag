"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _disable_rate_limits() -> Generator[None, None, None]:
    """Disable the slowapi limiter so route tests are not throttled."""
    from nexusrag.api.security import limiter

    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture(autouse=True)
def _reset_nexusrag_singleton() -> Generator[None, None, None]:
    """Keep the global singleton from leaking a stale instance across tests."""
    import nexusrag.pipeline as _p

    _p._instance = None
    yield
    _p._instance = None


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file."""
    file_path = temp_dir / "sample.txt"
    content = """INTRODUCTION

This is the introduction section of the document.
It contains multiple paragraphs of text.

METHODOLOGY

The methodology section describes the approach used.
We employed various techniques for analysis.

RESULTS

The results show significant findings.
Statistical analysis confirms the hypothesis.

CONCLUSION

In conclusion, the study demonstrates important outcomes.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_markdown_file(temp_dir: Path) -> Path:
    """Create a sample markdown file."""
    file_path = temp_dir / "sample.md"
    content = """# Research Paper Title

## Abstract

This paper presents novel findings in the field.

## Introduction

Background information and context for the research.
Multiple sentences provide detailed explanation.

## Methods

### Data Collection

Data was collected from various sources.

### Analysis

Statistical methods were applied to the dataset.

## Results

The analysis revealed significant patterns.

## Conclusion

The findings support the initial hypothesis.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def empty_file(temp_dir: Path) -> Path:
    """Create an empty text file."""
    file_path = temp_dir / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def parsed_document(sample_text_file):
    """Provide a parsed document for testing."""
    from nexusrag.ingestion import DocumentParser

    parser = DocumentParser()
    return parser.parse(sample_text_file)
