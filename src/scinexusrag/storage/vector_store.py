"""Vector storage using LanceDB."""

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.query import LanceVectorQueryBuilder
from numpy.typing import NDArray

from scinexusrag.ingestion import Chunk, EmbeddingConfig, EmbeddingConfigError

logger = logging.getLogger(__name__)


SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
EMBEDDING_CONFIG_KEY = b"nexusrag.embedding_config"


def _sanitize_id(value: str) -> str:
    """Sanitize ID to prevent SQL injection."""
    if not value:
        raise ValueError("ID cannot be empty")
    if not SAFE_ID_PATTERN.fullmatch(value):
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
        *,
        embedding_config: EmbeddingConfig,
    ):
        if not isinstance(embedding_config, EmbeddingConfig):
            raise EmbeddingConfigError("VectorStore requires an explicit EmbeddingConfig.")
        self.path = Path(path)
        self.table_name = table_name
        self.embedding_config = embedding_config
        self.embedding_dim = embedding_config.dimension
        self._init_lock = threading.RLock()
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    @property
    def db(self) -> lancedb.DBConnection:
        """Lazy database connection."""
        with self._init_lock:
            if self._db is None:
                self.path.mkdir(parents=True, exist_ok=True)
                self._db = lancedb.connect(str(self.path), read_consistency_interval=timedelta(0))
            return self._db

    @property
    def table(self) -> lancedb.table.Table:
        """Get or create the chunks table."""
        with self._init_lock:
            try:
                table = self._table
                if table is None:
                    table = (
                        self.db.open_table(self.table_name)
                        if self._table_exists()
                        else self._create_table()
                    )
                self._validate_table(table)
            except EmbeddingConfigError:
                raise
            except Exception as exc:
                raise EmbeddingConfigError("Cannot open or validate the vector index.") from exc
            self._table = table
            return table

    def _config_metadata(self) -> dict[str, Any]:
        return {"version": 1, **asdict(self.embedding_config)}

    def _validate_table(self, table: lancedb.table.Table) -> None:
        schema = table.schema
        raw = (schema.metadata or {}).get(EMBEDDING_CONFIG_KEY)
        if raw is None:
            raise EmbeddingConfigError(
                "Vector index has no embedding configuration. Rebuild it in a new collection."
            )
        try:
            stored = json.loads(raw)
            actual = json.dumps(stored, sort_keys=True, allow_nan=False)
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise EmbeddingConfigError("Vector index embedding metadata is invalid.") from exc
        expected = json.dumps(self._config_metadata(), sort_keys=True, allow_nan=False)
        if actual != expected:
            raise EmbeddingConfigError(
                "Vector index embedding configuration differs. Restore the original settings "
                "or rebuild in a new collection."
            )
        if schema.field("vector").type != pa.list_(pa.float32(), self.embedding_dim):
            raise EmbeddingConfigError(
                "Vector index schema does not match its embedding dimension."
            )

    def validate_embedding_config(self, config: EmbeddingConfig) -> None:
        if config != self.embedding_config:
            raise EmbeddingConfigError(
                "Active embedder differs from the vector index configuration. "
                "Restore the original settings or rebuild in a new collection."
            )
        _ = self.table

    def _table_exists(self) -> bool:
        """Find the configured table with either LanceDB listing API."""
        list_tables = getattr(self.db, "list_tables", None)
        if list_tables is None:
            return self.table_name in self.db.table_names()

        page_token: str | None = None
        while True:
            page = list_tables(page_token=page_token)
            if self.table_name in page.tables:
                return True
            page_token = page.page_token
            if not page_token:
                return False

    def _create_table(self) -> lancedb.table.Table:
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
            ],
            metadata={
                EMBEDDING_CONFIG_KEY: json.dumps(
                    self._config_metadata(), sort_keys=True, allow_nan=False
                ).encode("utf-8")
            },
        )
        return self.db.create_table(self.table_name, schema=schema)

    def add(self, chunks: list[Chunk], embeddings: NDArray[np.float32]) -> int:
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings")

        records = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "metadata": json.dumps(
                    {
                        **chunk.metadata,
                        "document_name": chunk.document_name
                        or chunk.metadata.get("document_name", ""),
                        "section_title": chunk.section_title
                        or chunk.metadata.get("section_title", ""),
                        "page_number": chunk.page_number
                        if chunk.page_number is not None
                        else chunk.metadata.get("page_number"),
                        "chunk_index": chunk.chunk_index,
                        "context_before": chunk.context_before,
                        "context_after": chunk.context_after,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                    }
                ),
                "vector": embedding.tolist(),
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        self.table.add(records)
        return len(records)

    @staticmethod
    def _chunk_from_row(row: dict[str, Any]) -> Chunk:
        metadata = (
            json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        )
        return Chunk(
            id=row["id"],
            content=row["content"],
            document_id=row["document_id"],
            metadata=metadata,
            document_name=metadata.get("document_name", ""),
            section_title=metadata.get("section_title", ""),
            page_number=metadata.get("page_number"),
            chunk_index=metadata.get("chunk_index", 0),
            context_before=metadata.get("context_before", ""),
            context_after=metadata.get("context_after", ""),
            start_offset=metadata.get("start_offset"),
            end_offset=metadata.get("end_offset"),
        )

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:

        vector_query = cast(LanceVectorQueryBuilder, self.table.search(query_embedding.tolist()))
        query = vector_query.metric("cosine").limit(top_k)

        if filter_expr:
            query = query.where(filter_expr)

        results = query.to_list()

        return [
            SearchResult(
                chunk=self._chunk_from_row(row),
                score=max(0.0, 1.0 - row["_distance"]),
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

    def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0

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
        return int(self.table.count_rows())

    def list_documents(self) -> list[str]:
        """Get all unique document IDs."""
        try:
            return list(dict.fromkeys(self.table.to_arrow().column("document_id").to_pylist()))
        except EmbeddingConfigError:
            raise
        except Exception:
            logger.debug("Failed to list documents from vector store", exc_info=True)
            return []

    def clear(self) -> None:
        """Remove all data from the store."""
        with self._init_lock:
            if self._table_exists():
                self.db.drop_table(self.table_name)
            self._table = None

    def get_all_chunks(self) -> list[Chunk]:
        try:
            if self.count() == 0:
                return []

            return [self._chunk_from_row(row) for row in self.table.to_arrow().to_pylist()]
        except EmbeddingConfigError:
            raise
        except Exception:
            logger.debug("Failed to get all chunks", exc_info=True)
            return []

    def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        try:
            safe_doc_id = _sanitize_id(document_id)
            results = (
                self.table.search().where(f"document_id = '{safe_doc_id}'").limit(100000).to_list()
            )
            return [self._chunk_from_row(row) for row in results]
        except EmbeddingConfigError:
            raise
        except Exception:
            logger.debug("Failed to get chunks by document", exc_info=True)
            return []
