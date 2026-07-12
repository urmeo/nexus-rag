"""Text embedding with sentence-transformers."""

import gc
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from scinexusrag.config import HF_REVISIONS

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def default_prefixes(model_name: str) -> tuple[str, str]:
    name = model_name.lower()
    if "e5" in name:
        return "query: ", "passage: "
    if "bge" in name and "-en" in name:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


class Embedder:
    """Sentence-transformer embeddings with optional query/passage prefixes."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Literal["cpu", "cuda", "mps"] | None = None,
        normalize: bool = True,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        revision: str | None = None,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self._model: SentenceTransformer | None = None
        self._device = device
        self.revision = revision or HF_REVISIONS.get(model_name)
        qp, dp = default_prefixes(model_name)
        self.query_prefix = qp if query_prefix is None else query_prefix
        self.doc_prefix = dp if doc_prefix is None else doc_prefix

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=self._device, revision=self.revision
            )
        return self._model

    def unload(self) -> None:
        """Release the underlying model; it lazy-loads again on next use."""
        self._model = None
        gc.collect()

    @property
    def dimension(self) -> int:
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(get_dim())

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        if self.doc_prefix:
            texts = [self.doc_prefix + t for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        result: NDArray[np.float32] = embeddings.astype(np.float32)
        if len(texts) > 100:
            gc.collect()
        return result

    def embed_query(self, query: str) -> NDArray[np.float32]:
        embedding = self.model.encode(
            self.query_prefix + query,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        result: NDArray[np.float32] = embedding.astype(np.float32)
        return result

    def similarity(
        self,
        query_embedding: NDArray[np.float32],
        document_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        if self.normalize:
            result: NDArray[np.float32] = np.dot(document_embeddings, query_embedding)
            return result
        qn = np.linalg.norm(query_embedding)
        q = query_embedding / qn if qn else query_embedding
        dn = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        dn[dn == 0] = 1.0  # zero vector -> 0 similarity, not NaN
        d = document_embeddings / dn
        result2: NDArray[np.float32] = np.dot(d, q)
        return result2
