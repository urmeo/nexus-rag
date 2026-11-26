"""Keeps the document store, vector index and BM25 index describing the same passages."""

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from scinexusrag.models import Chunk, ParsedDocument
from scinexusrag.retrieval.sparse import BM25Retriever
from scinexusrag.storage.document_store import DocumentStore
from scinexusrag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Collection:
    """Documents and their passages across the three stores.

    A write reaches every store or is rolled back. The BM25 index lives in memory and is rebuilt
    from the vector index; when a failed write leaves it unrepairable it is dropped, and
    ``on_sparse_reset`` lets the owner discard retrievers built on it. Callers serialize access.
    """

    def __init__(
        self,
        documents: DocumentStore,
        vectors: VectorStore,
        on_sparse_reset: Callable[[], None] = lambda: None,
    ) -> None:
        self.documents = documents
        self.vectors = vectors
        self._sparse: BM25Retriever | None = None
        self._on_sparse_reset = on_sparse_reset

    @property
    def sparse(self) -> BM25Retriever:
        return self._sparse if self._sparse is not None else self.rebuild_sparse()

    def rebuild_sparse(self) -> BM25Retriever:
        chunks = self.vectors.get_all_chunks()
        sparse = self._sparse if self._sparse is not None else BM25Retriever()
        sparse.rebuild(chunks)
        self._sparse = sparse
        return sparse

    def add(
        self, document: ParsedDocument, chunks: list[Chunk], embeddings: NDArray[np.float32]
    ) -> None:
        self.documents.add(document)
        try:
            self.vectors.add(chunks, embeddings)
            self.sparse.add(chunks)
            self.documents.update_metadata(document.id, "chunk_count", len(chunks))
            self._check_in_step("persist")
        except Exception:
            try:
                self.vectors.delete_by_document(document.id)
                self.rebuild_sparse()
                self.documents.delete(document.id)
            except Exception:
                logger.warning("Ingestion rollback did not finish", exc_info=True)
                self._repair_sparse("failed ingestion rollback")
            raise

    def delete(self, document_id: str) -> bool:
        if not self.documents.exists(document_id):
            return False
        sparse = self.sparse
        chunk_ids = self.vectors.get_chunk_ids_by_document(document_id)
        try:
            self.vectors.delete_by_document(document_id)
            if chunk_ids:
                sparse.remove(chunk_ids)
            self._check_in_step("delete")
            self.documents.delete(document_id)
        except Exception:
            self._repair_sparse("failed deletion")
            raise
        return True

    def clear(self) -> None:
        self.vectors.clear()
        if self._sparse is not None:
            self._sparse.clear()
        self.documents.clear()

    def _check_in_step(self, operation: str) -> None:
        sparse_count, vector_count = self.sparse.count(), self.vectors.count()
        if sparse_count != vector_count:
            raise RuntimeError(
                f"index desync after {operation}: bm25={sparse_count} vector={vector_count}"
            )

    def _repair_sparse(self, operation: str) -> None:
        try:
            self.rebuild_sparse()
        except Exception:
            self._sparse = None
            self._on_sparse_reset()
            logger.exception("Could not rebuild the sparse index after %s", operation)
