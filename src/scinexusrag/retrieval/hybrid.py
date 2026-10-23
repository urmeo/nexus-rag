"""Rank fusion."""

from __future__ import annotations

import string
from collections import defaultdict
from collections.abc import Sequence
from heapq import nlargest
from numbers import Real
from sys import float_info

from scinexusrag.retrieval.dense import DenseSearcher, RetrievalResult
from scinexusrag.retrieval.sparse import BM25Retriever


def _fusion_weights(weights: Sequence[float], k: int) -> list[float]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("RRF k must be a nonnegative integer.")
    values: list[float] = []
    for weight in weights:
        if (
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not 0 <= weight <= float_info.max
        ):
            raise ValueError("Fusion weights must be finite nonnegative numbers.")
        values.append(float(weight))
    return values


def _search_requested(query: str, top_k: int, depth: int) -> bool:
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer.")
    if top_k <= 0 or not query.strip():
        return False
    if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
        raise ValueError("depth must be a positive integer.")
    return True


def rrf_fuse(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    weights: Sequence[float],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Rank fusion."""
    values = _fusion_weights(weights, k)
    if len(ranked_lists) != len(values):
        raise ValueError("Each ranked list must have one fusion weight.")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer or None.")
        if top_k <= 0:
            return []
    peak = max(values, default=0.0)
    scale = peak if peak > 1.0 or 0 < peak < float_info.min else 1.0
    scores: dict[str, float] = defaultdict(float)
    results_by_id: dict[str, RetrievalResult] = {}
    for results, weight in zip(ranked_lists, values, strict=True):
        seen: set[str] = set()
        rank = 0
        for result in results:
            chunk_id = result.chunk.id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            scores[chunk_id] += (weight / scale) / (k + rank)
            results_by_id.setdefault(chunk_id, result)

    order = nlargest(len(scores) if top_k is None else top_k, scores, key=scores.__getitem__)
    normalizer = max(scores.values(), default=0.0) or 1.0
    return [
        RetrievalResult(
            chunk=results_by_id[chunk_id].chunk,
            score=scores[chunk_id] / normalizer,
            source="hybrid",
        )
        for chunk_id in order
    ]


class HybridRetriever:
    """Hybrid retrieval."""

    def __init__(
        self,
        dense_retriever: DenseSearcher,
        sparse_retriever: BM25Retriever,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        _fusion_weights([dense_weight, sparse_weight], rrf_k)
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def _weights(self, _query: str) -> tuple[float, float]:
        return self.dense_weight, self.sparse_weight

    def _run(self, query: str, top_k: int, depth: int) -> tuple[list[RetrievalResult], float]:
        if not _search_requested(query, top_k, depth):
            return [], 0.0
        dense_weight, sparse_weight = self._weights(query)
        dense = self.dense.retrieve(query, depth)
        sparse = self.sparse.retrieve(query, depth)
        fused = rrf_fuse([dense, sparse], [dense_weight, sparse_weight], self.rrf_k, top_k)
        return fused, (dense[0].score if dense else 0.0)

    def retrieve(self, query: str, top_k: int = 10, depth: int = 50) -> list[RetrievalResult]:
        return self._run(query, top_k, depth)[0]

    def retrieve_with_dense_top(
        self, query: str, top_k: int = 10, depth: int = 50
    ) -> tuple[list[RetrievalResult], float]:
        """Dense confidence."""
        return self._run(query, top_k, depth)


class AdaptiveHybridRetriever(HybridRetriever):
    """Query-aware fusion."""

    def __init__(
        self,
        dense_retriever: DenseSearcher,
        sparse_retriever: BM25Retriever,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        shift: float = 0.2,
    ) -> None:
        super().__init__(dense_retriever, sparse_retriever, dense_weight, sparse_weight, rrf_k)
        if (
            isinstance(shift, bool)
            or not isinstance(shift, Real)
            or not 0 <= shift <= float_info.max
        ):
            raise ValueError("Adaptive shift must be a finite nonnegative number.")
        self.shift: float = shift

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
    word = word.strip(string.punctuation)
    return bool(word) and (
        word.isupper() or "_" in word or any(c.isdigit() for c in word) or len(word) > 14
    )
