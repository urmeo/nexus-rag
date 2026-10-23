"""Exact retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from scinexusrag.ingestion.chunker import Chunk
from scinexusrag.ingestion.embedder import Embedder
from scinexusrag.retrieval.dense import RetrievalResult


def unique_document_ids(results: Sequence[RetrievalResult]) -> list[str]:
    return list(dict.fromkeys(result.chunk.document_id for result in results))


def corpus_to_chunks(corpus_text: dict[str, str]) -> list[Chunk]:
    return [
        Chunk(id=doc_id, content=text, document_id=doc_id, metadata={"doc_id": doc_id})
        for doc_id, text in corpus_text.items()
    ]


class ExactDenseRetriever:
    """Cosine search."""

    def __init__(self, embedder: Embedder, chunks: list[Chunk], batch_size: int = 64):
        if not embedder.normalize:
            raise ValueError("ExactDenseRetriever needs a normalized embedder for cosine scoring")
        self.embedder = embedder
        self.chunks = deepcopy(chunks)
        self.matrix = embedder.embed(
            [chunk.content for chunk in self.chunks], batch_size=batch_size, show_progress=False
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.chunks or top_k <= 0:
            return []
        query_vector = self.embedder.embed_query(query)
        scores = self.matrix @ query_vector
        k = min(top_k, len(self.chunks))
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]), source="dense")
            for i in top
        ]
