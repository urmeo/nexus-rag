"""Cross-encoder reranking."""

from typing import Any, Literal

from nexusrag.retrieval.dense import RetrievalResult


class Reranker:
    """Re-scores a shortlist with a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Literal["cpu", "cuda", "mps"] | None = None,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.content) for r in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Min-max normalize; when every score is equal (incl. a single result),
        # map to a neutral 1.0 rather than 0.0 so the top hit is not reported
        # as irrelevant.
        min_score, max_score = min(scores), max(scores)
        span = max_score - min_score

        reranked = [
            RetrievalResult(
                chunk=r.chunk,
                score=(s - min_score) / span if span else 1.0,
                source=f"{r.source}+rerank",
            )
            for r, s in zip(results, scores, strict=True)
        ]

        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked
