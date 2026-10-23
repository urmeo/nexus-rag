"""Vector storage."""

import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, cast

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.query import LanceVectorQueryBuilder
from numpy.typing import NDArray

from scinexusrag.ingestion.chunker import Chunk
from scinexusrag.ingestion.embedder import EmbeddingConfig, EmbeddingConfigError

SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
EMBEDDING_CONFIG_KEY = b"nexusrag.embedding_config"
SCAN_PAGE_SIZE = 10_000


class VectorStoreError(RuntimeError):
    """Storage failure."""


def _sanitize_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("ID must be a string")
    if not value:
        raise ValueError("ID cannot be empty")
    if len(value) > 128:
        raise ValueError("ID too long")
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid ID format: {value[:20]}")
    return value


def _cosine_vectors(values: NDArray[np.float32], shape: tuple[int, ...]) -> NDArray[np.float32]:
    vectors = np.asarray(values)
    if vectors.shape != shape or vectors.dtype.kind not in "iuf":
        raise ValueError(f"Embeddings must be real numeric vectors with shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        vectors = vectors.astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite float32 values")
    if not np.all(np.any(vectors != 0, axis=-1)):
        raise ValueError("Cosine embeddings must be nonzero")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=-1, keepdims=True)
    return np.asarray(vectors / norms, dtype=np.float32)


def _id_filter(safe_ids: list[str]) -> str:
    values = ", ".join(f"'{chunk_id}'" for chunk_id in safe_ids)
    return f"id IN ({values})"


def _metadata_json(metadata: dict[str, Any]) -> str:
    for field in ("document_name", "section_title", "context_before", "context_after"):
        if not isinstance(metadata.get(field, ""), str):
            raise ValueError(f"Passage {field} must be a string")
    for field in ("page_number", "start_offset", "end_offset"):
        value = metadata.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"Passage {field} must be a nonnegative integer or None")
    chunk_index = metadata.get("chunk_index", 0)
    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("Passage chunk_index must be a nonnegative integer")
    start, end = metadata.get("start_offset"), metadata.get("end_offset")
    if start is not None and end is not None and end < start:
        raise ValueError("Passage end_offset must not precede start_offset")
    try:
        return json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Passage metadata must contain finite JSON values") from exc


@dataclass
class SearchResult:
    """Retrieved passage."""

    chunk: Chunk
    score: float


class VectorStore:
    """Passage vectors."""

    def __init__(
        self,
        path: str | Path = "./data/lancedb",
        table_name: str = "chunks",
        *,
        embedding_config: EmbeddingConfig,
    ) -> None:
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
        """Database connection."""
        with self._init_lock:
            if self._db is None:
                self.path.mkdir(parents=True, exist_ok=True)
                self._db = lancedb.connect(str(self.path), read_consistency_interval=timedelta(0))
            return self._db

    @property
    def table(self) -> lancedb.table.Table:
        """Passage table."""
        with self._access("Cannot open or read the vector index."):
            table = self._table
            if table is None:
                table = (
                    self.db.open_table(self.table_name)
                    if self._table_exists()
                    else self._create_table()
                )
            self._validate_table(table)
            self._table = table
            return table

    @contextmanager
    def _access(self, message: str) -> Iterator[None]:
        try:
            with self._init_lock:
                yield
        except (EmbeddingConfigError, VectorStoreError):
            raise
        except Exception as exc:
            raise VectorStoreError(message) from exc

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
        """Locate table."""
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
        if not isinstance(chunks, list) or any(not isinstance(chunk, Chunk) for chunk in chunks):
            raise ValueError("chunks must be a list of Chunk objects")
        embeddings = _cosine_vectors(embeddings, (len(chunks), self.embedding_dim))
        if not chunks:
            return 0

        chunk_ids: set[str] = set()
        for chunk in chunks:
            chunk_id = _sanitize_id(chunk.id)
            _sanitize_id(chunk.document_id)
            if chunk_id in chunk_ids:
                raise ValueError(f"Duplicate chunk ID: {chunk_id}")
            chunk_ids.add(chunk_id)
            if not isinstance(chunk.content, str) or not isinstance(chunk.metadata, dict):
                raise ValueError("Chunks need text content and dictionary metadata")

        records = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "metadata": _metadata_json(
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

        with self._init_lock:
            self._check_new_ids([record["id"] for record in records])
            with self._access("Cannot write passages to the vector index."):
                self.table.add(records)
        return len(records)

    def _check_new_ids(self, chunk_ids: list[str]) -> None:
        with self._access("Cannot check existing IDs in the vector index."):
            table = self.table
            for offset in range(0, len(chunk_ids), SCAN_PAGE_SIZE):
                existing = (
                    table.search()
                    .select(["id"])
                    .where(_id_filter(chunk_ids[offset : offset + SCAN_PAGE_SIZE]))
                    .limit(1)
                    .to_list()
                )
                if existing:
                    break
            else:
                return
        raise ValueError(f"Chunk ID already exists: {existing[0]['id']}")

    @staticmethod
    def _chunk_from_row(row: dict[str, Any]) -> Chunk:
        try:
            metadata = (
                json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            )
            if not isinstance(metadata, dict):
                raise ValueError("Passage metadata must be a JSON object")
            _metadata_json(metadata)
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorStoreError("Cannot decode passage metadata from the vector index.") from exc
        try:
            _sanitize_id(row["id"])
            _sanitize_id(row["document_id"])
            if not isinstance(row["content"], str):
                raise ValueError("Passage content must be a string")
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorStoreError("Cannot decode passage fields from the vector index.") from exc
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
        if type(top_k) is not int:
            raise ValueError("top_k must be an integer")
        if top_k <= 0:
            return []
        query_embedding = _cosine_vectors(query_embedding, (self.embedding_dim,))
        if filter_expr is not None and not isinstance(filter_expr, str):
            raise ValueError("filter_expr must be a string or None")
        with self._access("Cannot search the vector index."):
            vector_query = cast(
                LanceVectorQueryBuilder, self.table.search(query_embedding.tolist())
            )
            query = vector_query.metric("cosine").limit(top_k)
            if filter_expr:
                query = query.where(filter_expr)
            results = query.to_list()

        matches = []
        for row in results:
            try:
                distance = row["_distance"]
                if (
                    isinstance(distance, bool)
                    or not isinstance(distance, Real)
                    or not isfinite(distance)
                ):
                    raise ValueError("Cosine distance must be a finite number")
                distance = float(distance)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise VectorStoreError(
                    "Cannot decode cosine distance from the vector index."
                ) from exc
            matches.append(
                SearchResult(
                    chunk=self._chunk_from_row(row),
                    score=float(min(1.0, max(0.0, 1.0 - distance))),
                )
            )
        return matches

    def search_by_document(
        self,
        query_embedding: NDArray[np.float32],
        document_id: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        safe_doc_id = _sanitize_id(document_id)
        return self.search(
            query_embedding,
            top_k=top_k,
            filter_expr=f"document_id = '{safe_doc_id}'",
        )

    def delete(self, chunk_ids: list[str]) -> int:
        if not isinstance(chunk_ids, list):
            raise ValueError("chunk_ids must be a list of strings")
        if not chunk_ids:
            return 0

        safe_ids = list(dict.fromkeys(_sanitize_id(chunk_id) for chunk_id in chunk_ids))
        return self._delete_where(_id_filter(safe_ids))

    def delete_by_document(self, document_id: str) -> int:
        safe_doc_id = _sanitize_id(document_id)
        return self._delete_where(f"document_id = '{safe_doc_id}'")

    def _delete_where(self, filter_expr: str) -> int:
        with self._access("Cannot delete passages from the vector index."):
            table = self.table
            initial_count = table.count_rows(filter_expr)
            if initial_count == 0:
                return 0
            table.delete(filter_expr)
            remaining_count = table.count_rows(filter_expr)
            if remaining_count > initial_count:
                raise VectorStoreError("Matching passages changed during deletion; retry it.")
            return int(initial_count - remaining_count)

    def count(self) -> int:
        with self._access("Cannot read the vector index row count."):
            return int(self.table.count_rows())

    def _read_rows(
        self, columns: list[str], filter_expr: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Snapshot scan."""
        with self._access("Cannot read passages from the vector index."):
            table = self.table
            completed = False
            try:
                table.checkout(table.version)
                offset = 0
                while True:
                    query = table.search().select(columns).offset(offset).limit(SCAN_PAGE_SIZE)
                    if filter_expr:
                        query = query.where(filter_expr)
                    rows = query.to_list()
                    yield from rows
                    if len(rows) < SCAN_PAGE_SIZE:
                        completed = True
                        return
                    offset += len(rows)
            finally:
                try:
                    table.checkout_latest()
                except Exception:
                    self._table = None
                    if completed:
                        raise

    def list_documents(self) -> list[str]:
        return list(dict.fromkeys(row["document_id"] for row in self._read_rows(["document_id"])))

    def clear(self) -> None:
        with self._init_lock:
            if self._table_exists():
                self.db.drop_table(self.table_name)
            self._table = None

    def get_all_chunks(self) -> list[Chunk]:
        return [
            self._chunk_from_row(row)
            for row in self._read_rows(["id", "content", "document_id", "metadata"])
        ]

    def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        safe_doc_id = _sanitize_id(document_id)
        return [
            self._chunk_from_row(row)
            for row in self._read_rows(
                ["id", "content", "document_id", "metadata"],
                f"document_id = '{safe_doc_id}'",
            )
        ]

    def get_chunk_ids_by_document(self, document_id: str) -> set[str]:
        safe_doc_id = _sanitize_id(document_id)
        return {row["id"] for row in self._read_rows(["id"], f"document_id = '{safe_doc_id}'")}
