"""Cross-encoder reranking for improved relevance."""

from typing import Any, Literal

from nexusrag.retrieval.dense import RetrievalResult


class Reranker:
    """Cross-encoder reranker for result refinement."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Literal["cpu", "cuda", "mps"] | None = None,
        batch_size: int = 8,  # Reduced for 8GB RAM systems
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazy load cross-encoder model."""
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
        """
        Rerank results using cross-encoder.

        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of results to return (None = all)

        Returns:
            Reranked results sorted by cross-encoder score
        """
        if not results:
            return []

        pairs = [(query, r.chunk.content) for r in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Normalize scores to 0-1
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        reranked = [
            RetrievalResult(
                chunk=r.chunk,
                score=(s - min_score) / score_range,
                source=f"{r.source}+rerank",
            )
            for r, s in zip(results, scores, strict=True)
        ]

        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def score(self, query: str, text: str) -> float:
        """Get relevance score for a single query-text pair."""
        return float(self.model.predict([(query, text)])[0])
