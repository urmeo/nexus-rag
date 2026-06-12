"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


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
def sample_pdf_path(temp_dir: Path) -> Path:
    """Create a minimal PDF file for testing."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
307
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def empty_file(temp_dir: Path) -> Path:
    """Create an empty text file."""
    file_path = temp_dir / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def sample_text_chunks() -> list[str]:
    """Provide sample text chunks for testing retrieval."""
    return [
        "The study found significant improvements in treatment outcomes.",
        "Methodology included randomized controlled trials with 500 participants.",
        "Results showed a 35% reduction in symptoms after 8 weeks.",
        "Statistical analysis used ANOVA with post-hoc Tukey tests.",
        "Limitations include small sample size and short follow-up period.",
    ]


@pytest.fixture
def mock_embeddings() -> np.ndarray:
    """Provide deterministic mock embedding vectors."""
    np.random.seed(42)
    return np.random.rand(5, 384).astype(np.float32)


@pytest.fixture
def mock_query_embedding() -> np.ndarray:
    """Provide a mock query embedding."""
    np.random.seed(123)
    return np.random.rand(384).astype(np.float32)


@pytest.fixture
def mock_embedding_model():
    """Provide a mock SentenceTransformer model."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384

    def mock_encode(texts, **kwargs):
        if isinstance(texts, str):
            np.random.seed(hash(texts) % 2**32)
            return np.random.rand(384).astype(np.float32)
        np.random.seed(42)
        return np.random.rand(len(texts), 384).astype(np.float32)

    model.encode = mock_encode
    return model


@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client."""
    client = MagicMock()
    client.generate.return_value = "This is a mock LLM response."
    client.is_available.return_value = True
    return client


@pytest.fixture
def sample_document_metadata() -> dict:
    """Provide sample document metadata."""
    return {
        "filename": "research_paper.pdf",
        "extension": ".pdf",
        "size_bytes": 12345,
        "created_at": "2025-01-15T10:30:00+00:00",
        "modified_at": "2025-01-15T10:30:00+00:00",
    }


@pytest.fixture
def parsed_document(sample_text_file):
    """Provide a parsed document for testing."""
    from nexusrag.ingestion import DocumentParser

    parser = DocumentParser()
    return parser.parse(sample_text_file)


@pytest.fixture
def sample_chunks(parsed_document):
    """Provide sample chunks from a parsed document."""
    from nexusrag.ingestion import SemanticChunker

    chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=500)
    return chunker.chunk(parsed_document)


@pytest.fixture
def env_override(monkeypatch):
    """Fixture factory for setting environment variables."""

    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)

    return _set_env
