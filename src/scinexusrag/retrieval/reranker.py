"""Passage reranking."""

from threading import Lock
from typing import Any, Literal

import numpy as np

from scinexusrag.retrieval.dense import RetrievalResult
from scinexusrag.settings import HF_REVISIONS


class Reranker:
    """Cross-encoder reranking."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Literal["cpu", "cuda", "mps"] | None = None,
        batch_size: int = 8,
        revision: str | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("reranker model name must be a nonempty string")
        if device not in (None, "cpu", "cuda", "mps"):
            raise ValueError("reranker device must be cpu, cuda, mps or None")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("reranker batch_size must be a positive integer")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("reranker revision must be a nonempty string or None")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.revision = HF_REVISIONS.get(model_name) if revision is None else revision
        self._model: Any = None
        self._model_lock = Lock()

    @property
    def model(self) -> Any:
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    revision=self.revision,
                )
            return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str):
            raise ValueError("reranker query must be a string")
        if top_k is not None and type(top_k) is not int:
            raise ValueError("reranker top_k must be an integer or None")
        if not results or not query.strip() or (top_k is not None and top_k <= 0):
            return []

        pairs = [(query, r.chunk.content) for r in results]
        scores = np.asarray(self.model.predict(pairs, batch_size=self.batch_size))
        if scores.shape == (len(results), 1):
            scores = scores[:, 0]
        if scores.shape != (len(results),) or scores.dtype.kind not in "iuf":
            raise ValueError("reranker needs one numeric score per result")
        scores = scores.astype(np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("reranker scores must be finite")
        min_score, max_score = float(scores.min()), float(scores.max())
        span = max_score - min_score
        if np.isinf(span):
            scores /= max(abs(min_score), abs(max_score))
            min_score, max_score = float(scores.min()), float(scores.max())
            span = max_score - min_score

        scores = (scores - min_score) / span if span else np.ones_like(scores)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            RetrievalResult(
                chunk=results[index].chunk,
                score=float(scores[index]),
                source=f"{results[index].source}+rerank",
            )
            for index in order
        ]
