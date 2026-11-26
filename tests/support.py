"""Shared test doubles and helpers."""

from __future__ import annotations

from dataclasses import replace

from scinexusrag.ingestion.embedder import EmbeddingConfig
from scinexusrag.models import ParsedDocument
from scinexusrag.storage.vector_store import VectorStore


def document(doc_id="paper"):
    return ParsedDocument(doc_id, "Two words.", {"filename": "paper.txt", "file_type": "txt"})


PIN = "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"


def make_config(dim: int = 4, **changes) -> EmbeddingConfig:
    return replace(EmbeddingConfig("test-model", "a" * 40, "", "", True, dim), **changes)


def make_store(tmp_path, dim: int = 4, **changes) -> VectorStore:
    return VectorStore(path=tmp_path / "lancedb", embedding_config=make_config(dim, **changes))


def sparse_index_loaded(rag) -> bool:
    return rag._collection is not None and rag._collection._sparse is not None
