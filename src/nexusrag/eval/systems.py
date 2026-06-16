"""The additive retrieval ladder for the ablation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from nexusrag.eval.indexes import ExactDenseRetriever
from nexusrag.ingestion import Chunk, Embedder
from nexusrag.retrieval import (
    AdaptiveHybridRetriever,
    BM25Retriever,
    CorrectiveRetriever,
    HybridRetriever,
    Reranker,
)
from nexusrag.retrieval.dense import RetrievalResult

RetrieveFn = Callable[[str, int], list[str]]


def _ids(results: Sequence[RetrievalResult]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        d = r.chunk.document_id
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def build_systems(
    chunks: list[Chunk],
    embedder: Embedder,
    include_rerank: bool = False,
    tau: float = 0.55,
) -> dict[str, RetrieveFn]:
    """Each rung adds exactly one component over the previous."""
    dense = ExactDenseRetriever(embedder, chunks)
    bm25 = BM25Retriever()
    bm25.add(chunks)

    hybrid = HybridRetriever(dense, bm25, 0.5, 0.5)
    adaptive = AdaptiveHybridRetriever(dense, bm25, 0.5, 0.5)
    corrective = CorrectiveRetriever(adaptive, tau=tau)

    def depth(k: int) -> int:
        return max(k, 50)

    systems: dict[str, RetrieveFn] = {
        "BM25": lambda q, k: _ids(bm25.retrieve(q, top_k=depth(k))),
        "Dense": lambda q, k: _ids(dense.retrieve(q, top_k=depth(k))),
        "Hybrid (RRF)": lambda q, k: _ids(hybrid.retrieve(q, top_k=k, depth=depth(k))),
        "+ Adaptive weights": lambda q, k: _ids(adaptive.retrieve(q, top_k=k, depth=depth(k))),
        "+ Corrective PRF": lambda q, k: _ids(corrective.retrieve(q, top_k=k, depth=depth(k))),
    }

    if include_rerank:
        reranker = Reranker(device="cpu")

        def with_rerank(q: str, k: int) -> list[str]:
            cands = adaptive.retrieve(q, top_k=depth(k), depth=depth(k))
            return _ids(reranker.rerank(q, cands, top_k=k))

        systems["+ Rerank (cross-enc)"] = with_rerank

    return systems
