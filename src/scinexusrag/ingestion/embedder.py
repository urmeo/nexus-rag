"""Text embeddings."""

import gc
import re
import threading
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from scinexusrag.settings import HF_REVISIONS

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingConfigError(RuntimeError):
    """Incompatible embeddings."""


def _pinned_revision(revision: str | None) -> str:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise EmbeddingConfigError(
            "Persistent indexes require a full model commit SHA in EMBEDDING_REVISION."
        )
    return revision.lower()


def _validate_settings(
    model_name: str, query_prefix: str, doc_prefix: str, normalize: bool
) -> None:
    if not isinstance(model_name, str) or not model_name.strip():
        raise EmbeddingConfigError("Embedding model name must be a nonempty string.")
    if not isinstance(query_prefix, str) or not isinstance(doc_prefix, str):
        raise EmbeddingConfigError("Embedding prefixes must be strings.")
    if type(normalize) is not bool:
        raise EmbeddingConfigError("Embedding normalization must be a boolean.")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    revision: str
    query_prefix: str
    doc_prefix: str
    normalize: bool
    dimension: int

    def __post_init__(self) -> None:
        _validate_settings(self.model_name, self.query_prefix, self.doc_prefix, self.normalize)
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


def _model_dimension(model: Any) -> int:
    get_dimension = getattr(model, "get_embedding_dimension", None)
    if not callable(get_dimension):
        get_dimension = getattr(model, "get_sentence_embedding_dimension", None)
    if not callable(get_dimension):
        raise EmbeddingConfigError("Model does not expose its embedding dimension.")
    dimension = get_dimension()
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
        raise EmbeddingConfigError("Model embedding dimension must be a positive integer.")
    return int(dimension)


def _embedding_array(values: Any, shape: tuple[int, ...]) -> NDArray[np.float32]:
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"Embedding model must return numeric vectors with shape {shape}.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Embedding model returned nonfinite vectors.")
    return result


class Embedder:
    """Sentence embeddings."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Literal["cpu", "cuda", "mps"] | None = None,
        normalize: bool = True,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        revision: str | None = None,
    ) -> None:
        _validate_settings(
            model_name,
            "" if query_prefix is None else query_prefix,
            "" if doc_prefix is None else doc_prefix,
            normalize,
        )
        if device not in (None, "cpu", "cuda", "mps"):
            raise ValueError("Embedding device must be cpu, cuda, mps or None.")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise EmbeddingConfigError("Embedding revision must be a nonempty string or None.")
        default_query_prefix, default_doc_prefix = default_prefixes(model_name)
        self.query_prefix = default_query_prefix if query_prefix is None else query_prefix
        self.doc_prefix = default_doc_prefix if doc_prefix is None else doc_prefix
        self.model_name = model_name
        self.normalize = normalize
        self._model: SentenceTransformer | None = None
        self._loaded_model_source: tuple[str, str | None] | None = None
        self._model_lock = threading.Lock()
        self._device = device
        self.revision = HF_REVISIONS.get(model_name) if revision is None else revision

    @property
    def model(self) -> Any:
        with self._model_lock:
            source = (self.model_name, self.revision)
            if self._model is not None and self._loaded_model_source != source:
                raise EmbeddingConfigError(
                    "Model settings changed after loading. Unload the model first."
                )
            if self._model is None:
                model_name, revision = source
                device = self._device
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name, device=device, revision=revision)
                self._loaded_model_source = source
            return self._model

    @property
    def embedding_config(self) -> EmbeddingConfig:
        _validate_settings(self.model_name, self.query_prefix, self.doc_prefix, self.normalize)
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
        with self._model_lock:
            self._model = None
            self._loaded_model_source = None
        gc.collect()

    @property
    def dimension(self) -> int:
        return _model_dimension(self.model)

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise ValueError("Embedding texts must be a list of strings.")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Embedding batch size must be a positive integer.")
        if type(show_progress) is not bool:
            raise ValueError("Embedding progress flag must be a boolean.")
        model = self.model
        dimension = _model_dimension(model)
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)

        if self.doc_prefix:
            texts = [self.doc_prefix + t for t in texts]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        result = _embedding_array(embeddings, (len(texts), dimension))
        if len(texts) > 100:
            gc.collect()
        return result

    def embed_query(self, query: str) -> NDArray[np.float32]:
        if not isinstance(query, str):
            raise ValueError("Embedding query must be a string.")
        model = self.model
        dimension = _model_dimension(model)
        embedding = model.encode(
            self.query_prefix + query,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return _embedding_array(embedding, (dimension,))

    def similarity(
        self,
        query_embedding: NDArray[np.float32],
        document_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        if self.normalize:
            return np.asarray(np.dot(document_embeddings, query_embedding), dtype=np.float32)
        query = query_embedding.astype(np.float64)
        documents = document_embeddings.astype(np.float64)
        query_norm = np.linalg.norm(query)
        if query_norm:
            query = query / query_norm
        document_norms = np.linalg.norm(documents, axis=1, keepdims=True)
        document_norms[document_norms == 0] = 1.0
        documents = documents / document_norms
        return np.asarray(np.dot(documents, query), dtype=np.float32)
