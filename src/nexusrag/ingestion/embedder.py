"""Text embedding using sentence-transformers."""

import gc
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper for sentence-transformer embedding models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Literal["cpu", "cuda", "mps"] | None = None,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self._model: SentenceTransformer | None = None
        self._device = device

    @property
    def model(self) -> Any:
        """Lazy load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        # newer sentence-transformers renamed the method
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(get_dim())

    def embed(
        self,
        texts: list[str],
        batch_size: int = 16,  # Reduced default for 8GB RAM systems
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (default 16 for memory efficiency)
            show_progress: Show progress bar

        Returns:
            Array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        result: NDArray[np.float32] = embeddings.astype(np.float32)

        # Memory cleanup for large batches
        if len(texts) > 100:
            gc.collect()

        return result

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """
        Embed a single query string.

        Args:
            query: Query text

        Returns:
            1D array of shape (dimension,)
        """
        embedding = self.model.encode(
            query,
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
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Shape (dimension,)
            document_embeddings: Shape (n_docs, dimension)

        Returns:
            Similarity scores of shape (n_docs,)
        """
        if self.normalize:
            result: NDArray[np.float32] = np.dot(document_embeddings, query_embedding)
            return result

        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        result2: NDArray[np.float32] = np.dot(doc_norms, query_norm)
        return result2
