"""Checks that an uploaded file is what its name claims before it is parsed."""

from __future__ import annotations

import io
import zipfile
from typing import Protocol, cast

from fastapi import HTTPException, status

from scinexusrag.ingestion.parser import FORMATS, DocumentFormat
from scinexusrag.settings import get_settings

_GENERIC_CONTENT_TYPES = frozenset({"", "application/octet-stream"})


class _MimeDetector(Protocol):
    def from_buffer(self, data: bytes) -> str: ...


def _load_magic() -> _MimeDetector | None:
    try:
        import magic
    except (ImportError, OSError):
        return None
    try:
        return cast(_MimeDetector, magic.Magic(mime=True))
    except (magic.MagicException, OSError):
        return None


_MAGIC = _load_magic()


def validate_upload(ext: str, content: str | bytes, content_type: str | None) -> None:
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    settings = get_settings().api

    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File too large (max {settings.max_upload_mb} MB)",
        )

    ext = ext.lower()
    upload_format = FORMATS.get(ext)
    if upload_format is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {ext}",
        )

    declared = (content_type or "").partition(";")[0].strip().lower()
    if declared not in _GENERIC_CONTENT_TYPES and declared not in upload_format.content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {declared}",
        )

    if upload_format.signatures and not data.startswith(upload_format.signatures):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File contents do not match {ext}",
        )

    _sniff_mime(ext, data, upload_format)

    if upload_format.text:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Text upload is not valid UTF-8",
            ) from None

    if ext == ".docx":
        _validate_docx_size(data, settings.max_uncompressed_mb * 1024 * 1024)


def _sniff_mime(ext: str, data: bytes, upload_format: DocumentFormat) -> None:
    if _MAGIC is None:
        return
    detected = _MAGIC.from_buffer(data).lower()

    if upload_format.text:
        if detected.startswith("text/") or detected == "inode/x-empty":
            return
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Sniffed type {detected} is not text",
        )
    if detected not in upload_format.content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Sniffed type {detected} does not match {ext}",
        )


def _validate_docx_size(data: bytes, max_uncompressed: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            total = sum(member.file_size for member in archive.infolist())
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid DOCX (not a zip archive)",
        ) from None
    if total > max_uncompressed:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="DOCX decompresses to too large a size",
        )
