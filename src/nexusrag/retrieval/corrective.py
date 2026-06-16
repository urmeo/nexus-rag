"""Confidence-gated corrective retrieval via pseudo-relevance feedback."""

from __future__ import annotations

from collections import Counter

from nexusrag.retrieval.dense import RetrievalResult
from nexusrag.retrieval.hybrid import HybridRetriever, rrf_fuse


class CorrectiveRetriever:
    """Re-retrieves with feedback terms when the first pass looks weak.

    Confidence is the top dense cosine similarity. Below ``tau`` the query
    is expanded with frequent terms from the first-pass documents and the
    two passes are fused, so the cost is paid only on hard queries.
    """

    def __init__(
        self,
        base: HybridRetriever,
        tau: float = 0.55,
        feedback_docs: int = 5,
        feedback_terms: int = 10,
    ):
        self.base = base
        self.tau = tau
        self.feedback_docs = feedback_docs
        self.feedback_terms = feedback_terms

    def confidence(self, query: str) -> float:
        top = self.base.dense.retrieve(query, 1)
        return top[0].score if top else 0.0

    def expand(self, query: str, results: list[RetrievalResult]) -> str:
        qterms = set(self.base.sparse.tokenize(query))
        counts: Counter[str] = Counter()
        for r in results[: self.feedback_docs]:
            for t in self.base.sparse.tokenize(r.chunk.content):
                if t not in qterms:
                    counts[t] += 1
        extra = [t for t, _ in counts.most_common(self.feedback_terms)]
        return f"{query} {' '.join(extra)}".strip()

    def retrieve_traced(
        self, query: str, top_k: int = 10, depth: int = 50
    ) -> tuple[list[RetrievalResult], bool]:
        first = self.base.retrieve(query, top_k=depth, depth=depth)
        if not first or self.confidence(query) >= self.tau:
            return first[:top_k], False
        expanded = self.expand(query, first)
        second = self.base.retrieve(expanded, top_k=depth, depth=depth)
        fused = rrf_fuse([first, second], [1.0, 1.0], self.base.rrf_k, top_k)
        return fused, True

    def retrieve(self, query: str, top_k: int = 10, depth: int = 50) -> list[RetrievalResult]:
        return self.retrieve_traced(query, top_k, depth)[0]
