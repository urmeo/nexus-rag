"""Exact in-memory dense index for deterministic evaluation."""

from __future__ import annotations

import numpy as np

from nexusrag.ingestion import Chunk, Embedder
from nexusrag.retrieval.dense import RetrievalResult


def corpus_to_chunks(corpus_text: dict[str, str]) -> list[Chunk]:
    return [
        Chunk(id=doc_id, content=text, document_id=doc_id, metadata={"doc_id": doc_id})
        for doc_id, text in corpus_text.items()
    ]


class ExactDenseRetriever:
    """Brute-force cosine search over precomputed embeddings, for reproducible eval."""

    def __init__(self, embedder: Embedder, chunks: list[Chunk], batch_size: int = 64):
        self.embedder = embedder
        self.chunks = chunks
        self.matrix = embedder.embed(
            [c.content for c in chunks], batch_size=batch_size, show_progress=False
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        qv = self.embedder.embed_query(query)
        scores = self.matrix @ qv
        k = min(top_k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]), source="dense")
            for i in top
        ]
