"""SPLADE retrieval."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any, Literal

import numpy as np
from scipy import sparse

from scinexusrag.ingestion.chunker import Chunk
from scinexusrag.retrieval.dense import RetrievalResult
from scinexusrag.settings import HF_REVISIONS

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


class SpladeRetriever:
    """Sparse retrieval."""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        batch_size: int = 16,
        max_length: int = 256,
        revision: str | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("SPLADE model name must be a nonempty string")
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("SPLADE device must be cpu, cuda or mps")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("SPLADE batch_size must be a positive integer")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("SPLADE max_length must be a positive integer")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("SPLADE revision must be a nonempty string or None")
        if not isinstance(chunks, list) or any(
            not isinstance(chunk, Chunk) or not isinstance(chunk.content, str) for chunk in chunks
        ):
            raise ValueError("SPLADE chunks must be a list of chunks with text content")
        self.model_name = model_name
        self.revision = revision if revision is not None else HF_REVISIONS.get(model_name)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.chunks = deepcopy(chunks)
        self._tok: Any = None
        self._model: Any = None
        self._model_lock = Lock()
        self._index_config = self._encoding_config
        self.matrix = self._encode([chunk.content for chunk in self.chunks])

    @property
    def _encoding_config(self) -> tuple[str, str | None, str, int]:
        return self.model_name, self.revision, self.device, self.max_length

    def _load(self) -> None:
        with self._model_lock:
            if self._model is None:
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
                model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name, revision=self.revision
                ).to(self.device)
                model.eval()
                self._tok, self._model = tokenizer, model

    def _encode(self, texts: list[str]) -> sparse.csr_matrix:
        if not texts:
            return sparse.csr_matrix((0, 0))
        import torch

        self._load()
        blocks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            if (
                logits.ndim != 3
                or logits.shape[:2] != enc["attention_mask"].shape
                or logits.shape[0] != len(batch)
                or logits.shape[1] == 0
                or logits.shape[2] == 0
            ):
                raise ValueError("SPLADE predictions must align with tokenized passages")
            weighted = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
            vecs = weighted.max(dim=1).values.cpu().numpy()
            if not np.isfinite(vecs).all():
                raise ValueError("SPLADE vocabulary weights must be finite")
            blocks.append(sparse.csr_matrix(vecs))
        return sparse.vstack(blocks).tocsr()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not isinstance(query, str):
            raise ValueError("SPLADE query must be a string")
        if type(top_k) is not int:
            raise ValueError("SPLADE top_k must be an integer")
        if not self.chunks or not query.strip() or top_k <= 0:
            return []
        if self._encoding_config != self._index_config:
            raise ValueError("SPLADE encoding settings changed; rebuild the retriever")
        qv = self._encode([query])
        if self.matrix.shape[0] != len(self.chunks) or qv.shape != (1, self.matrix.shape[1]):
            raise ValueError("SPLADE vectors must align with the indexed corpus")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.asarray((self.matrix @ qv.T).todense()).ravel()
        if not np.isfinite(scores).all():
            raise ValueError("SPLADE retrieval scores must be finite")
        k = min(top_k, len(self.chunks))
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievalResult(chunk=deepcopy(self.chunks[i]), score=float(scores[i]), source="splade")
            for i in top
        ]
