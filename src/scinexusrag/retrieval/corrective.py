"""Corrective retrieval."""

from __future__ import annotations

from collections import Counter
from numbers import Real

from scinexusrag.retrieval.dense import RetrievalResult
from scinexusrag.retrieval.hybrid import HybridRetriever, _search_requested, rrf_fuse


class CorrectiveRetriever:
    """Conditional feedback."""

    def __init__(
        self,
        base: HybridRetriever,
        tau: float = 0.55,
        feedback_docs: int = 5,
        feedback_terms: int = 10,
        enabled: bool = True,
    ) -> None:
        if isinstance(tau, bool) or not isinstance(tau, Real) or not 0 <= tau <= 1:
            raise ValueError("Corrective threshold must be finite and between 0 and 1.")
        for name, value in {
            "feedback_docs": feedback_docs,
            "feedback_terms": feedback_terms,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean.")
        self.base: HybridRetriever = base
        self.tau: float = tau
        self.feedback_docs: int = feedback_docs
        self.feedback_terms: int = feedback_terms
        self.enabled: bool = enabled

    def expand(self, query: str, results: list[RetrievalResult]) -> str:
        query_terms = set(self.base.sparse.tokenize(query))
        counts = Counter(
            term
            for result in results[: self.feedback_docs]
            for term in self.base.sparse.tokenize(result.chunk.content)
            if term not in query_terms
        )
        extra = " ".join(term for term, _ in counts.most_common(self.feedback_terms))
        return f"{query} {extra}".strip()

    def retrieve_traced(
        self, query: str, top_k: int = 10, depth: int = 50
    ) -> tuple[list[RetrievalResult], bool]:
        if not _search_requested(query, top_k, depth):
            return [], False
        first, dense_top = self.base.retrieve_with_dense_top(query, top_k=depth, depth=depth)
        if not self.enabled or not first or dense_top >= self.tau:
            return first[:top_k], False
        expanded = self.expand(query, first)
        if expanded == query.strip():
            return first[:top_k], False
        second = self.base.retrieve(expanded, top_k=depth, depth=depth)
        fused = rrf_fuse([first, second], [1.0, 1.0], self.base.rrf_k, top_k)
        return fused, True

    def retrieve(self, query: str, top_k: int = 10, depth: int = 50) -> list[RetrievalResult]:
        return self.retrieve_traced(query, top_k, depth)[0]
