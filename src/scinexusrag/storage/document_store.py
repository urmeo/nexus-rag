"""Document storage."""

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from scinexusrag.ingestion.parser import ParsedDocument
from scinexusrag.utils.filenames import resolve_display_name

logger = logging.getLogger(__name__)


SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
MAX_ID_LENGTH = 64

INDEX_STEM = "_index"


class DocumentStoreError(RuntimeError):
    """Document storage."""


class DocumentExistsError(ValueError):
    """Duplicate document."""


class DocumentStore:
    def __init__(self, path: str | Path = "./data/documents") -> None:
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.path / f"{INDEX_STEM}.json"
        self._index: dict[str, dict[str, Any]] | None = None
        self._index_lock = threading.RLock()

    @staticmethod
    def _validate_doc_id(doc_id: str) -> str:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Document ID must be a nonempty string")

        if len(doc_id) > MAX_ID_LENGTH:
            raise ValueError(f"Document ID too long (max {MAX_ID_LENGTH} chars)")

        if not SAFE_ID_PATTERN.fullmatch(doc_id):
            raise ValueError("Document ID contains invalid characters")

        if doc_id.casefold() == INDEX_STEM:
            raise ValueError(f"Document ID '{INDEX_STEM}' is reserved for the store index")

        return doc_id

    @property
    def index(self) -> dict[str, dict[str, Any]]:
        return self.list_with_metadata()

    def _get_index(self) -> dict[str, dict[str, Any]]:
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        if self._index_path.is_symlink():
            raise DocumentStoreError("Document index cannot be a symbolic link")
        if self._index_path.exists():
            index = self._read_json(self._index_path)
            if not isinstance(index, dict):
                raise DocumentStoreError("Document index must be a JSON object")
            for doc_id, entry in index.items():
                try:
                    self._validate_entry(entry)
                except ValueError as exc:
                    raise DocumentStoreError(f"Invalid document index entry: {doc_id!r}") from exc

        on_disk = {
            path.stem
            for path in self.path.glob("*.json")
            if path.name.casefold() != self._index_path.name.casefold()
        }
        folded_ids: set[str] = set()
        for doc_id in set(index) | on_disk:
            try:
                self._validate_doc_id(doc_id)
                if doc_id in on_disk:
                    self._doc_path(doc_id)
            except ValueError as exc:
                raise DocumentStoreError(f"Unsafe stored document ID or path: {doc_id!r}") from exc
            if doc_id.casefold() in folded_ids:
                raise DocumentStoreError("Stored document IDs differ only by letter case")
            folded_ids.add(doc_id.casefold())
        dangling = set(index) - on_disk
        orphaned = on_disk - set(index)

        for doc_id in dangling:
            del index[doc_id]
        for doc_id in orphaned:
            index[doc_id] = self._index_entry_from_file(doc_id)

        if dangling or orphaned:
            logger.warning(
                "Document index reconciled: removed %d stale entries, recovered %d documents",
                len(dangling),
                len(orphaned),
            )
            self._write_json(self._index_path, index)
        return index

    @staticmethod
    def _read_json(path: Path) -> Any:
        def finite_number(value: str) -> float:
            number = float(value)
            if not isfinite(number):
                raise ValueError(f"Nonfinite JSON number: {value}")
            return number

        try:
            return json.loads(
                path.read_text(encoding="utf-8"),
                parse_float=finite_number,
                parse_constant=finite_number,
            )
        except ValueError as exc:
            raise DocumentStoreError(f"Cannot read stored JSON: {path.name}") from exc

    @staticmethod
    def _validate_entry(entry: Any) -> None:
        if not isinstance(entry, dict) or any(not isinstance(key, str) for key in entry):
            raise ValueError("Document metadata must be a JSON object with string keys")
        for key in ("word_count", "section_count", "chunk_count"):
            if key in entry and (
                isinstance(entry[key], bool) or not isinstance(entry[key], int) or entry[key] < 0
            ):
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("filename", "original_filename", "display_name", "uploaded_at", "file_type"):
            if key in entry and not isinstance(entry[key], str):
                raise ValueError(f"{key} must be a string")

    @staticmethod
    def _validate_document(data: Any, doc_id: str) -> None:
        if not isinstance(data, dict) or data.get("id") != doc_id:
            raise ValueError("Document identity must match its filename")
        if not isinstance(data.get("content"), str):
            raise ValueError("Document content must be text")
        if not isinstance(data.get("metadata", {}), dict):
            raise ValueError("Document metadata must be a JSON object")
        sections = data.get("sections", [])
        if not isinstance(sections, list) or any(
            not isinstance(section, dict)
            or not isinstance(section.get("title"), str)
            or not isinstance(section.get("content"), str)
            for section in sections
        ):
            raise ValueError("Document sections must contain titles and text")

    def _index_entry_from_file(self, doc_id: str) -> dict[str, Any]:
        data = self._read_json(self._doc_path(doc_id))
        try:
            self._validate_document(data, doc_id)
        except ValueError as exc:
            raise DocumentStoreError(f"Invalid stored document: {doc_id!r}") from exc
        try:
            return self._metadata_entry(data)
        except ValueError as exc:
            raise DocumentStoreError(f"Invalid stored document metadata: {doc_id!r}") from exc

    def _metadata_entry(self, data: dict[str, Any]) -> dict[str, Any]:
        metadata = data.get("metadata", {})
        name = resolve_display_name(metadata, fallback="unknown")
        entry = {
            "filename": name,
            "original_filename": name,
            "display_name": name,
            "word_count": len(data["content"].split()),
            "section_count": len(data.get("sections", [])),
            "uploaded_at": datetime.now(UTC).isoformat(),
            "file_type": metadata.get("file_type", "unknown"),
        }
        self._validate_entry(entry)
        return entry

    def _save_index(self) -> None:
        self._write_json(self._index_path, self._get_index())

    def _write_json(self, target: Path, value: dict[str, Any], *, overwrite: bool = True) -> None:
        """Atomic JSON."""
        if target.is_symlink():
            raise DocumentStoreError("Stored JSON target cannot be a symbolic link")
        data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
        fd, tmp_path = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            if overwrite:
                os.replace(tmp_path, target)
            else:
                try:
                    os.link(tmp_path, target)
                except FileExistsError as exc:
                    raise DocumentExistsError("Document already exists") from exc
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    def _doc_path(self, doc_id: str) -> Path:
        safe_id = self._validate_doc_id(doc_id)
        doc_path = self.path / f"{safe_id}.json"
        if doc_path.is_symlink():
            raise ValueError("Document path cannot be a symbolic link")
        doc_path = doc_path.resolve()
        if not doc_path.is_relative_to(self.path):
            raise ValueError("Invalid document path")
        return doc_path

    @staticmethod
    def _check_id_collision(doc_id: str, index: dict[str, dict[str, Any]]) -> None:
        if any(
            existing != doc_id and existing.casefold() == doc_id.casefold() for existing in index
        ):
            raise ValueError("Document ID differs from an existing ID only by letter case")

    def add(self, document: ParsedDocument) -> str:
        doc_id = self._validate_doc_id(document.id)
        doc_dict = {
            "id": doc_id,
            "content": document.content,
            "metadata": document.metadata,
            "sections": [asdict(s) for s in document.sections],
        }
        self._validate_document(doc_dict, doc_id)

        with self._index_lock:
            doc_path = self._doc_path(doc_id)
            index = self._get_index()
            self._check_id_collision(doc_id, index)
            if doc_id in index or doc_path.exists():
                self._index = None
                raise DocumentExistsError("Document already exists")
            entry = self._metadata_entry(doc_dict)
            document_written = False
            try:
                self._write_json(doc_path, doc_dict, overwrite=False)
                document_written = True
                index[doc_id] = entry
                self._save_index()
            except BaseException:
                self._index = None
                if document_written:
                    try:
                        doc_path.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Document rollback did not finish", exc_info=True)
                raise

        return doc_id

    def update_metadata(self, doc_id: str, key: str, value: Any) -> bool:
        """Update metadata."""
        self._validate_doc_id(doc_id)
        if not isinstance(key, str) or not key:
            raise ValueError("Metadata key must be a nonempty string")
        with self._index_lock:
            index = self._get_index()
            if doc_id not in index:
                return False
            previous = index[doc_id]
            updated = {**previous, key: deepcopy(value)}
            self._validate_entry(updated)
            index[doc_id] = updated
            try:
                self._save_index()
            except BaseException:
                index[doc_id] = previous
                raise
            return True

    def exists(self, doc_id: str) -> bool:
        self._validate_doc_id(doc_id)
        with self._index_lock:
            return doc_id in self._get_index()

    def list_with_metadata(self) -> dict[str, dict[str, Any]]:
        """Metadata snapshot."""
        with self._index_lock:
            return deepcopy(self._get_index())

    def delete(self, doc_id: str) -> bool:
        with self._index_lock:
            doc_path = self._doc_path(doc_id)
            index = self._get_index()
            self._check_id_collision(doc_id, index)
            if not doc_path.exists() and doc_id not in index:
                return False

            doc_path.unlink(missing_ok=True)
            index.pop(doc_id, None)
            try:
                self._save_index()
            except BaseException:
                self._index = None
                raise
            return True

    def count(self) -> int:
        with self._index_lock:
            return len(self._get_index())

    def clear(self) -> int:
        with self._index_lock:
            index = self._get_index()
            count = len(index)
            for doc_id in list(index):
                self.delete(doc_id)
            return count

    def get_stats(self) -> dict[str, Any]:
        metadata = self.list_with_metadata()
        total_words = sum(entry.get("word_count", 0) for entry in metadata.values())
        total_sections = sum(entry.get("section_count", 0) for entry in metadata.values())

        return {
            "document_count": len(metadata),
            "total_words": total_words,
            "total_sections": total_sections,
            "storage_path": str(self.path),
        }
