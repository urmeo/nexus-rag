"""Vector storage using LanceDB."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from nexusrag.ingestion import Chunk

logger = logging.getLogger(__name__)

# Pattern for safe IDs (alphanumeric, underscore, hyphen only)
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _sanitize_id(value: str) -> str:
    """Sanitize ID to prevent SQL injection."""
    if not value:
        raise ValueError("ID cannot be empty")
    if not SAFE_ID_PATTERN.match(value):
        raise ValueError(f"Invalid ID format: {value[:20]}")
    if len(value) > 128:
        raise ValueError("ID too long")
    return value


@dataclass
class SearchResult:
    """A search result with chunk and similarity score."""

    chunk: Chunk
    score: float


class VectorStore:
    """LanceDB-backed vector store for chunk embeddings."""

    def __init__(
        self,
        path: str | Path = "./data/lancedb",
        table_name: str = "chunks",
        embedding_dim: int = 384,
    ):
        self.path = Path(path)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    @property
    def db(self) -> lancedb.DBConnection:
        """Lazy database connection."""
        if self._db is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.path))
        return self._db

    @property
    def table(self) -> lancedb.table.Table:
        """Get or create the chunks table."""
        if self._table is None:
            if self.table_name in self.db.list_tables().tables:
                self._table = self.db.open_table(self.table_name)
            else:
                self._table = self._create_table()
        return self._table

    def _create_table(self) -> lancedb.table.Table:
        """Create table with schema."""
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
            ]
        )
        return self.db.create_table(self.table_name, schema=schema)

    def add(
        self,
        chunks: list[Chunk],
        embeddings: NDArray[np.float32],
    ) -> int:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects
            embeddings: Array of shape (len(chunks), embedding_dim)

        Returns:
            Number of chunks added
        """
        if len(chunks) == 0:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings")

        import json

        records = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "metadata": json.dumps(chunk.metadata),
                "vector": embedding.tolist(),
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        self.table.add(records)
        return len(records)

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            top_k: Number of results to return
            filter_expr: Optional SQL filter (e.g., "document_id = 'abc'")

        Returns:
            List of SearchResult sorted by similarity (descending)
        """
        import json

        query = self.table.search(query_embedding.tolist()).limit(top_k)

        if filter_expr:
            query = query.where(filter_expr)

        results = query.to_list()

        return [
            SearchResult(
                chunk=Chunk(
                    id=row["id"],
                    content=row["content"],
                    document_id=row["document_id"],
                    metadata=json.loads(row["metadata"]),
                ),
                score=1.0 - row["_distance"],  # Convert distance to similarity
            )
            for row in results
        ]

    def search_by_document(
        self,
        query_embedding: NDArray[np.float32],
        document_id: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search within a specific document."""
        safe_doc_id = _sanitize_id(document_id)
        return self.search(
            query_embedding,
            top_k=top_k,
            filter_expr=f"document_id = '{safe_doc_id}'",
        )

    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by their IDs."""
        import json

        if not chunk_ids:
            return []

        # Sanitize all IDs to prevent SQL injection
        safe_ids = [_sanitize_id(cid) for cid in chunk_ids]
        ids_str = ", ".join(f"'{cid}'" for cid in safe_ids)
        results = self.table.search().where(f"id IN ({ids_str})").limit(len(chunk_ids)).to_list()

        return [
            Chunk(
                id=row["id"],
                content=row["content"],
                document_id=row["document_id"],
                metadata=json.loads(row["metadata"]),
            )
            for row in results
        ]

    def delete(self, chunk_ids: list[str]) -> int:
        """
        Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted
        """
        if not chunk_ids:
            return 0

        # Sanitize all IDs to prevent SQL injection
        safe_ids = [_sanitize_id(cid) for cid in chunk_ids]
        ids_str = ", ".join(f"'{cid}'" for cid in safe_ids)
        initial_count = self.count()
        self.table.delete(f"id IN ({ids_str})")
        return initial_count - self.count()

    def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document."""
        safe_doc_id = _sanitize_id(document_id)
        initial_count = self.count()
        self.table.delete(f"document_id = '{safe_doc_id}'")
        return initial_count - self.count()

    def count(self) -> int:
        """Total number of chunks in the store."""
        try:
            return int(self.table.count_rows())
        except (OSError, ValueError):
            return 0

    def count_by_document(self, document_id: str) -> int:
        """Count chunks for a specific document."""
        safe_doc_id = _sanitize_id(document_id)
        results = (
            self.table.search().where(f"document_id = '{safe_doc_id}'").limit(100000).to_list()
        )
        return len(results)

    def list_documents(self) -> list[str]:
        """Get all unique document IDs."""
        try:
            results: list[str] = self.table.to_pandas()["document_id"].unique().tolist()
            return results
        except Exception:
            logger.debug("Failed to list documents from vector store", exc_info=True)
            return []

    def clear(self) -> None:
        """Remove all data from the store."""
        if self.table_name in self.db.list_tables().tables:
            self.db.drop_table(self.table_name)
        self._table = None

    def optimize(self) -> None:
        """Optimize the index for better search performance."""
        if self.count() > 0:
            self.table.create_index(
                metric="cosine",
                num_partitions=min(256, max(1, self.count() // 100)),
                num_sub_vectors=min(96, self.embedding_dim // 4),
            )

    def get_all_chunks(self) -> list[Chunk]:
        """Retrieve all chunks from the store."""
        import json

        try:
            if self.count() == 0:
                return []

            results = self.table.to_pandas()
            return [
                Chunk(
                    id=row["id"],
                    content=row["content"],
                    document_id=row["document_id"],
                    metadata=json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"],
                )
                for _, row in results.iterrows()
            ]
        except Exception:
            logger.debug("Failed to get all chunks", exc_info=True)
            return []

    def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        """Retrieve all chunks for a specific document."""
        import json

        try:
            safe_doc_id = _sanitize_id(document_id)
            results = (
                self.table.search().where(f"document_id = '{safe_doc_id}'").limit(100000).to_list()
            )
            return [
                Chunk(
                    id=row["id"],
                    content=row["content"],
                    document_id=row["document_id"],
                    metadata=json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"],
                )
                for row in results
            ]
        except Exception:
            logger.debug("Failed to get chunks by document", exc_info=True)
            return []
