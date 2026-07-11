"""JSON-based storage for parsed documents."""

import contextlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexusrag.ingestion import ParsedDocument
from nexusrag.utils.filenames import resolve_display_name

logger = logging.getLogger(__name__)

# Safe filename pattern - only alphanumeric and limited special chars
SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_ID_LENGTH = 64
# Stem of the on-disk index file; reserved so a document id cannot alias it.
INDEX_STEM = "_index"


class DocumentStore:
    """File-based document storage using JSON."""

    def __init__(self, path: str | Path = "./data/documents"):
        self.path = Path(path).resolve()  # Resolve to absolute path
        self.path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.path / f"{INDEX_STEM}.json"
        self._index: dict[str, dict[str, Any]] | None = None

    def _validate_doc_id(self, doc_id: str) -> str:
        """Validate and sanitize document ID to prevent path traversal."""
        if not doc_id:
            raise ValueError("Document ID cannot be empty")

        if len(doc_id) > MAX_ID_LENGTH:
            raise ValueError(f"Document ID too long (max {MAX_ID_LENGTH} chars)")

        if not SAFE_ID_PATTERN.match(doc_id):
            raise ValueError("Document ID contains invalid characters")

        if doc_id == INDEX_STEM:
            raise ValueError(f"Document ID '{INDEX_STEM}' is reserved for the store index")

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
        """Load index from disk, reconciling it against the doc files.

        The index and per-doc files are written as two separate atomic
        renames, so a crash between them can leave either a doc file with no
        index entry (invisible) or an index entry with no file (a phantom
        that blocks re-ingestion). Both are repaired here.
        """
        index: dict[str, dict[str, Any]] = {}
        if self._index_path.exists():
            index = json.loads(self._index_path.read_text(encoding="utf-8"))

        on_disk = {p.stem for p in self.path.glob("*.json") if p != self._index_path}
        dangling = set(index) - on_disk
        orphaned = on_disk - set(index)

        for doc_id in dangling:
            del index[doc_id]
        for doc_id in orphaned:
            entry = self._index_entry_from_file(doc_id)
            if entry is not None:
                index[doc_id] = entry

        if dangling or orphaned:
            logger.warning(
                "Reconciled document index: dropped %d dangling entr(ies), "
                "recovered %d orphaned doc file(s)",
                len(dangling),
                len(orphaned),
            )
            self._write_index(index)
        return index

    def _index_entry_from_file(self, doc_id: str) -> dict[str, Any] | None:
        """Rebuild an index entry from a doc file (crash recovery)."""
        try:
            data = json.loads((self.path / f"{doc_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        metadata = data.get("metadata", {})
        name = resolve_display_name(metadata, fallback="unknown")
        return {
            "filename": name,
            "original_filename": name,
            "display_name": name,
            "word_count": len(data.get("content", "").split()),
            "section_count": len(data.get("sections", [])),
            "uploaded_at": datetime.now(UTC).isoformat(),
            "file_type": metadata.get("file_type", "unknown"),
        }

    def _save_index(self) -> None:
        self._write_index(self.index)

    def _write_index(self, index: dict[str, dict[str, Any]]) -> None:
        """Persist index to disk atomically via write-to-temp-then-rename."""
        data = json.dumps(index, indent=2, ensure_ascii=False)
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
        safe_id = self._validate_doc_id(doc_id)
        doc_path = (self.path / f"{safe_id}.json").resolve()
        if not doc_path.is_relative_to(self.path):
            raise ValueError("Invalid document path")
        return doc_path

    def add(self, document: ParsedDocument) -> str:
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

    def list_with_metadata(self) -> dict[str, dict[str, Any]]:
        """Get all documents with their index metadata."""
        return dict(self.index)

    def delete(self, doc_id: str) -> bool:
        doc_path = self._doc_path(doc_id)
        if not doc_path.exists():
            return False

        # Index first: a crash after this leaves only a harmless orphan file
        # (swept on next load) instead of a phantom entry blocking re-ingest.
        self.index.pop(doc_id, None)
        self._save_index()
        doc_path.unlink()
        return True

    def count(self) -> int:
        """Total number of stored documents."""
        return len(self.index)

    def clear(self) -> int:
        count = len(self.index)
        for doc_id in list(self.index.keys()):
            self.delete(doc_id)
        return count

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
