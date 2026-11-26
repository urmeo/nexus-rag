"""Vector retrieval."""

from dataclasses import dataclass
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

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if type(top_k) is not int:
            raise ValueError("top_k must be an integer")
        if not query.strip() or top_k <= 0:
            return []
        self.vector_store.validate_embedding_config(self.embedder.embedding_config)
        results = self.vector_store.search(self.embedder.embed_query(query), top_k)
        return [RetrievalResult(chunk=r.chunk, score=r.score, source="dense") for r in results]
