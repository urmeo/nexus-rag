"""Vector retrieval."""

from scinexusrag.ingestion.embedder import Embedder
from scinexusrag.retrieval.base import RetrievalResult
from scinexusrag.storage.vector_store import VectorStore


class DenseRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip() or top_k <= 0:
            return []
        self.vector_store.validate_embedding_config(self.embedder.embedding_config)
        results = self.vector_store.search(self.embedder.embed_query(query), top_k)
        return [RetrievalResult(chunk=r.chunk, score=r.score, source="dense") for r in results]
