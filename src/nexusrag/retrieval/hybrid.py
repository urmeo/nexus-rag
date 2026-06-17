"""Hybrid dense + sparse retrieval with reciprocal rank fusion."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from nexusrag.retrieval.dense import DenseSearcher, RetrievalResult
from nexusrag.retrieval.sparse import BM25Retriever

logger = logging.getLogger(__name__)


def rrf_fuse(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    weights: Sequence[float],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Weighted reciprocal rank fusion of ranked result lists."""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievalResult] = {}
    for results, w in zip(ranked_lists, weights, strict=True):
        for rank, r in enumerate(results, start=1):
            scores[r.chunk.id] += w / (k + rank)
            best.setdefault(r.chunk.id, r)

    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    if top_k is not None:
        order = order[:top_k]
    norm = max(scores.values()) if scores else 1.0
    return [
        RetrievalResult(chunk=best[cid].chunk, score=scores[cid] / norm, source="hybrid")
        for cid in order
    ]


class HybridRetriever:
    """Fuses a dense and a BM25 retriever with reciprocal rank fusion."""

    def __init__(
        self,
        dense_retriever: DenseSearcher,
        sparse_retriever: BM25Retriever,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("weights must be non-negative")
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def _run(
        self, query: str, dw: float, sw: float, top_k: int, depth: int
    ) -> list[RetrievalResult]:
        dense = self.dense.retrieve(query, depth)
        sparse = self.sparse.retrieve(query, depth)
        return rrf_fuse([dense, sparse], [dw, sw], self.rrf_k, top_k)

    def retrieve(self, query: str, top_k: int = 10, depth: int = 50) -> list[RetrievalResult]:
        return self._run(query, self.dense_weight, self.sparse_weight, top_k, depth)

    def retrieve_dense_only(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self.dense.retrieve(query, top_k)

    def retrieve_sparse_only(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self.sparse.retrieve(query, top_k)


class AdaptiveHybridRetriever(HybridRetriever):
    """Hybrid retriever that shifts fusion weight by query shape.

    Short or notation-heavy queries lean lexical; long natural-language
    queries lean dense. Falls back to the base split in between.
    """

    def __init__(
        self,
        dense_retriever: DenseSearcher,
        sparse_retriever: BM25Retriever,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        shift: float = 0.2,
    ):
        super().__init__(dense_retriever, sparse_retriever, dense_weight, sparse_weight, rrf_k)
        self.shift = shift

    def retrieve(self, query: str, top_k: int = 10, depth: int = 50) -> list[RetrievalResult]:
        dw, sw = self._weights(query)
        return self._run(query, dw, sw, top_k, depth)

    def _weights(self, query: str) -> tuple[float, float]:
        words = query.split()
        lexical = len(words) <= 4 or any(_looks_technical(w) for w in words)
        semantic = len(words) >= 12
        if lexical and not semantic:
            return max(0.0, self.dense_weight - self.shift), self.sparse_weight + self.shift
        if semantic and not lexical:
            return self.dense_weight + self.shift, max(0.0, self.sparse_weight - self.shift)
        return self.dense_weight, self.sparse_weight


def _looks_technical(word: str) -> bool:
    return word.isupper() or "_" in word or any(c.isdigit() for c in word) or len(word) > 14
