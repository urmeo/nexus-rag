"""JSON-based storage for parsed documents."""

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexusrag.ingestion import ParsedDocument, Section
from nexusrag.utils.filenames import resolve_display_name

# Safe filename pattern - only alphanumeric and limited special chars
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_ID_LENGTH = 64


class DocumentStore:
    """File-based document storage using JSON."""

    def __init__(self, path: str | Path = "./data/documents"):
        self.path = Path(path).resolve()  # Resolve to absolute path
        self.path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.path / "_index.json"
        self._index: dict[str, dict[str, Any]] | None = None

    def _validate_doc_id(self, doc_id: str) -> str:
        """Validate and sanitize document ID to prevent path traversal."""
        if not doc_id:
            raise ValueError("Document ID cannot be empty")

        if len(doc_id) > MAX_ID_LENGTH:
            raise ValueError(f"Document ID too long (max {MAX_ID_LENGTH} chars)")

        if not SAFE_ID_PATTERN.match(doc_id):
            raise ValueError("Document ID contains invalid characters")

        # Additional safety: ensure no path components
        if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
            raise ValueError("Document ID contains invalid path characters")

        return doc_id

    @property
    def index(self) -> dict[str, dict[str, Any]]:
        """Lazy load document index."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load index from disk."""
        if self._index_path.exists():
            data: dict[str, dict[str, Any]] = json.loads(
                self._index_path.read_text(encoding="utf-8")
            )
            return data
        return {}

    def _save_index(self) -> None:
        """Persist index to disk atomically via write-to-temp-then-rename."""
        data = json.dumps(self.index, indent=2, ensure_ascii=False)
        fd, tmp_path = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_path, self._index_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _doc_path(self, doc_id: str) -> Path:
        """Get file path for a document (with path traversal protection)."""
        safe_id = self._validate_doc_id(doc_id)
        doc_path = (self.path / f"{safe_id}.json").resolve()

        # Final safety check: ensure path is within data directory
        if not str(doc_path).startswith(str(self.path)):
            raise ValueError("Invalid document path")

        return doc_path

    def add(self, document: ParsedDocument) -> str:
        """
        Store a parsed document.

        Args:
            document: ParsedDocument to store

        Returns:
            Document ID
        """
        doc_dict = {
            "id": document.id,
            "content": document.content,
            "metadata": document.metadata,
            "sections": [asdict(s) for s in document.sections],
        }

        # Atomic write: write to temp file then rename
        target = self._doc_path(document.id)
        data = json.dumps(doc_dict, indent=2, ensure_ascii=False)
        fd, tmp_path = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp_path, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        original_name = resolve_display_name(document.metadata, fallback="unknown")

        self.index[document.id] = {
            "filename": original_name,
            "original_filename": original_name,
            "display_name": original_name,
            "word_count": document.word_count,
            "section_count": len(document.sections),
            "uploaded_at": datetime.now(UTC).isoformat(),
            "file_type": document.metadata.get("file_type", "unknown"),
        }
        self._save_index()

        return document.id

    def get(self, doc_id: str) -> ParsedDocument | None:
        """
        Retrieve a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            ParsedDocument or None if not found
        """
        doc_path = self._doc_path(doc_id)
        if not doc_path.exists():
            return None

        data = json.loads(doc_path.read_text(encoding="utf-8"))

        return ParsedDocument(
            id=data["id"],
            content=data["content"],
            metadata=data["metadata"],
            sections=[Section(**s) for s in data["sections"]],
        )

    def update_metadata(self, doc_id: str, key: str, value: Any) -> bool:
        """Update a single metadata field for a document in the index."""
        if doc_id not in self.index:
            return False
        self.index[doc_id][key] = value
        self._save_index()
        return True

    def exists(self, doc_id: str) -> bool:
        """Check if a document exists."""
        return doc_id in self.index

    def list_all(self) -> list[str]:
        """Get all document IDs."""
        return list(self.index.keys())

    def list_with_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all documents with their index metadata."""
        return dict(self.index)

    def delete(self, doc_id: str) -> bool:
        """
        Remove a document.

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted, False if not found
        """
        doc_path = self._doc_path(doc_id)
        if not doc_path.exists():
            return False

        doc_path.unlink()
        self.index.pop(doc_id, None)
        self._save_index()
        return True

    def count(self) -> int:
        """Total number of stored documents."""
        return len(self.index)

    def clear(self) -> int:
        """
        Remove all documents.

        Returns:
            Number of documents deleted
        """
        count = len(self.index)
        for doc_id in list(self.index.keys()):
            self.delete(doc_id)
        return count

    def iter_documents(self) -> Iterator[ParsedDocument]:
        """Iterate over all documents."""
        for doc_id in self.list_all():
            doc = self.get(doc_id)
            if doc:
                yield doc

    def search_by_filename(self, filename: str) -> list[str]:
        """Find documents by filename (partial match)."""
        filename_lower = filename.lower()
        return [
            doc_id
            for doc_id, meta in self.index.items()
            if filename_lower in meta.get("filename", "").lower()
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total_words = sum(m.get("word_count", 0) for m in self.index.values())
        total_sections = sum(m.get("section_count", 0) for m in self.index.values())

        return {
            "document_count": len(self.index),
            "total_words": total_words,
            "total_sections": total_sections,
            "storage_path": str(self.path),
        }
