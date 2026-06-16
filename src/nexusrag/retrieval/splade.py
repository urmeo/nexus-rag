"""Learned sparse retrieval with SPLADE."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import sparse

from nexusrag.ingestion import Chunk
from nexusrag.retrieval.dense import RetrievalResult

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


class SpladeRetriever:
    """Encodes text to sparse term weights over the vocabulary, scored by dot product."""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        batch_size: int = 16,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.chunks = chunks
        self._tok: Any = None
        self._model: Any = None
        self.matrix = self._encode([c.content for c in chunks])

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def _encode(self, texts: list[str]) -> sparse.csr_matrix:
        import torch

        self._load()
        blocks = []
        for i in range(0, len(texts), self.batch_size):
            enc = self._tok(
                texts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            weighted = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
            vecs = weighted.max(dim=1).values.cpu().numpy()
            blocks.append(sparse.csr_matrix(vecs))
        return sparse.vstack(blocks).tocsr()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        qv = self._encode([query])
        scores = np.asarray((self.matrix @ qv.T).todense()).ravel()
        k = min(top_k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]), source="splade")
            for i in top
        ]
