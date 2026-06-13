"""Retrieval system ladder for ablation experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from nexusrag.eval.indexes import ExactDenseRetriever
from nexusrag.ingestion import Chunk, Embedder
from nexusrag.retrieval import (
    AdaptiveHybridRetriever,
    BM25Retriever,
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


def _with_rerank(base: HybridRetriever, reranker: Reranker) -> RetrieveFn:
    def fn(query: str, k: int) -> list[str]:
        candidates = base.retrieve(query, top_k=max(k, 50))
        reranked = reranker.rerank(query, candidates, top_k=k)
        return _ids(reranked)

    return fn


def build_systems(
    chunks: list[Chunk],
    embedder: Embedder,
    include_rerank: bool = True,
) -> dict[str, RetrieveFn]:
    """The additive ablation ladder.

    Each entry adds exactly one component over the previous.
    """
    dense = ExactDenseRetriever(embedder, chunks)
    bm25 = BM25Retriever()
    bm25.add(chunks)

    rrf = HybridRetriever(dense, bm25, 0.7, 0.3, use_mmr=False, use_keyword_boost=False)
    adaptive = AdaptiveHybridRetriever(
        dense, bm25, 0.7, 0.3, use_mmr=False, use_keyword_boost=False
    )
    boosted = AdaptiveHybridRetriever(dense, bm25, 0.7, 0.3, use_mmr=False, use_keyword_boost=True)
    full = AdaptiveHybridRetriever(dense, bm25, 0.7, 0.3, use_mmr=True, use_keyword_boost=True)

    systems: dict[str, RetrieveFn] = {
        "BM25": lambda q, k: _ids(bm25.retrieve(q, top_k=k)),
        "Dense (MiniLM)": lambda q, k: _ids(dense.retrieve(q, top_k=k)),
        "Hybrid-RRF": lambda q, k: _ids(rrf.retrieve(q, top_k=k)),
        "+ Adaptive weights": lambda q, k: _ids(adaptive.retrieve(q, top_k=k)),
        "+ Keyword boost": lambda q, k: _ids(boosted.retrieve(q, top_k=k)),
    }

    if include_rerank:
        reranker = Reranker()
        systems["+ Rerank (cross-enc)"] = _with_rerank(boosted, reranker)
        systems["+ MMR (full)"] = _with_rerank(full, reranker)

    return systems
