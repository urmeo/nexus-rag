"""Vector retrieval."""

from dataclasses import dataclass
from math import inf
from numbers import Real
from typing import Protocol

from scinexusrag.ingestion.chunker import Chunk
from scinexusrag.ingestion.embedder import Embedder
from scinexusrag.storage.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str = "dense"


class DenseSearcher(Protocol):
    """Vector search."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...


class DenseRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if type(top_k) is not int:
            raise ValueError("top_k must be an integer")
        if document_id is not None and (
            not isinstance(document_id, str) or not document_id.strip()
        ):
            raise ValueError("document_id must be a nonempty string or None")
        if not query.strip() or top_k <= 0:
            return []
        self.vector_store.validate_embedding_config(self.embedder.embedding_config)
        query_embedding = self.embedder.embed_query(query)

        if document_id is None:
            results = self.vector_store.search(query_embedding, top_k)
        else:
            results = self.vector_store.search_by_document(query_embedding, document_id, top_k)

        return [RetrievalResult(chunk=r.chunk, score=r.score, source="dense") for r in results]

    def retrieve_with_threshold(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[RetrievalResult]:
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, Real)
            or not -inf < min_score < inf
        ):
            raise ValueError("min_score must be a finite number")
        results = self.retrieve(query, top_k)
        return [r for r in results if r.score >= min_score]
