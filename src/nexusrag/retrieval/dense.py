"""Dense retrieval over an embedding store."""

from dataclasses import dataclass
from typing import Protocol

from nexusrag.ingestion import Chunk, Embedder
from nexusrag.storage import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str = "dense"


class DenseSearcher(Protocol):
    """What the hybrid retriever needs from a dense backend."""

    embedder: Embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...


class DenseRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed_query(query)

        if document_id:
            results = self.vector_store.search_by_document(query_embedding, document_id, top_k)
        else:
            results = self.vector_store.search(query_embedding, top_k)

        return [RetrievalResult(chunk=r.chunk, score=r.score, source="dense") for r in results]

    def retrieve_with_threshold(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[RetrievalResult]:
        results = self.retrieve(query, top_k)
        return [r for r in results if r.score >= min_score]
