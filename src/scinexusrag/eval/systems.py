"""Build retrieval baselines and optional retrieval stages for evaluation."""

from __future__ import annotations

from collections.abc import Callable

from scinexusrag.eval.indexes import ExactDenseRetriever, unique_document_ids
from scinexusrag.ingestion import Chunk, Embedder
from scinexusrag.retrieval import (
    AdaptiveHybridRetriever,
    BM25Retriever,
    CorrectiveRetriever,
    HybridRetriever,
    Reranker,
)

RetrieveFn = Callable[[str, int], list[str]]


def build_systems(
    chunks: list[Chunk],
    embedder: Embedder,
    include_rerank: bool = False,
    include_splade: bool = False,
    tau: float = 0.55,
    splade_model: str | None = None,
    splade_revision: str | None = None,
) -> dict[str, RetrieveFn]:
    """Compare sparse and dense retrieval, fusion, and optional refinements."""
    dense = ExactDenseRetriever(embedder, chunks)
    bm25 = BM25Retriever()
    bm25.add(chunks)

    hybrid = HybridRetriever(dense, bm25, 0.5, 0.5)
    adaptive = AdaptiveHybridRetriever(dense, bm25, 0.5, 0.5)
    corrective = CorrectiveRetriever(adaptive, tau=tau)

    def candidate_depth(k: int) -> int:
        return max(k, 50)

    systems: dict[str, RetrieveFn] = {
        "BM25": lambda q, k: unique_document_ids(bm25.retrieve(q, top_k=k)),
        "Dense": lambda q, k: unique_document_ids(dense.retrieve(q, top_k=k)),
        "Hybrid (RRF)": lambda q, k: unique_document_ids(
            hybrid.retrieve(q, top_k=k, depth=candidate_depth(k))
        ),
        "+ Adaptive weights": lambda q, k: unique_document_ids(
            adaptive.retrieve(q, top_k=k, depth=candidate_depth(k))
        ),
        "+ Corrective PRF": lambda q, k: unique_document_ids(
            corrective.retrieve(q, top_k=k, depth=candidate_depth(k))
        ),
    }

    if include_rerank:
        reranker = Reranker(device="cpu")

        def with_rerank(q: str, k: int) -> list[str]:
            candidates = corrective.retrieve(q, top_k=candidate_depth(k), depth=candidate_depth(k))
            return unique_document_ids(reranker.rerank(q, candidates, top_k=k))

        systems["+ Rerank (cross-enc)"] = with_rerank

    if include_splade:
        from scinexusrag.retrieval.splade import DEFAULT_MODEL, SpladeRetriever

        splade = SpladeRetriever(
            chunks,
            model_name=splade_model or DEFAULT_MODEL,
            revision=splade_revision,
            device="cpu",
        )
        systems["SPLADE"] = lambda q, k: unique_document_ids(splade.retrieve(q, top_k=k))

    return systems
