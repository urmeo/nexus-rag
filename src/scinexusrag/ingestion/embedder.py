"""Text embedding with sentence-transformers."""

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from scinexusrag.config import HF_REVISIONS

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingConfigError(RuntimeError):
    """A persistent index cannot verify its embedding configuration."""


def _pinned_revision(revision: str | None) -> str:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise EmbeddingConfigError(
            "Persistent indexes require a full model commit SHA in EMBEDDING_REVISION."
        )
    return revision.lower()


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    revision: str
    query_prefix: str
    doc_prefix: str
    normalize: bool
    dimension: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise EmbeddingConfigError("Embedding model name must be a nonempty string.")
        if not isinstance(self.query_prefix, str) or not isinstance(self.doc_prefix, str):
            raise EmbeddingConfigError("Embedding prefixes must be strings.")
        if type(self.normalize) is not bool:
            raise EmbeddingConfigError("Embedding normalization must be a boolean.")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise EmbeddingConfigError("Embedding dimension must be a positive integer.")
        object.__setattr__(self, "revision", _pinned_revision(self.revision))


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
        self._loaded_model_source: tuple[str, str | None] | None = None
        self._device = device
        self.revision = revision or HF_REVISIONS.get(model_name)
        default_query_prefix, default_doc_prefix = default_prefixes(model_name)
        self.query_prefix = default_query_prefix if query_prefix is None else query_prefix
        self.doc_prefix = default_doc_prefix if doc_prefix is None else doc_prefix

    @property
    def model(self) -> Any:
        source = (self.model_name, self.revision)
        if self._model is not None and self._loaded_model_source != source:
            raise EmbeddingConfigError(
                "Model settings changed after loading. Unload the model first."
            )
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=self._device, revision=self.revision
            )
            self._loaded_model_source = source
        return self._model

    @property
    def embedding_config(self) -> EmbeddingConfig:
        if (
            Path(self.model_name).is_absolute()
            or self.model_name.startswith(".")
            or Path(self.model_name).exists()
        ):
            raise EmbeddingConfigError(
                "Persistent indexes require a repository model with a pinned commit, not local files."
            )
        return EmbeddingConfig(
            model_name=self.model_name,
            revision=_pinned_revision(self.revision),
            query_prefix=self.query_prefix,
            doc_prefix=self.doc_prefix,
            normalize=self.normalize,
            dimension=self.dimension,
        )

    def unload(self) -> None:
        """Release the underlying model; it lazy-loads again on next use."""
        self._model = None
        self._loaded_model_source = None
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
        if not self.normalize:
            query_norm = np.linalg.norm(query_embedding)
            if query_norm:
                query_embedding = query_embedding / query_norm
            document_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            document_norms[document_norms == 0] = 1.0
            document_embeddings = document_embeddings / document_norms
        similarities: NDArray[np.float32] = np.dot(document_embeddings, query_embedding)
        return similarities
