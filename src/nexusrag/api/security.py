"""API security: key auth, per-IP rate limiting, and upload validation."""

from __future__ import annotations

import hmac
import io
import zipfile

from fastapi import Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from nexusrag.config import get_settings

# Per-IP rate limiter (slowapi). Limits are read from settings per request so
# they stay configurable; registered on the app in api.create_app.
limiter = Limiter(key_func=get_remote_address)


def query_limit() -> str:
    return f"{get_settings().api.query_rate_per_minute}/minute"


def upload_limit() -> str:
    return f"{get_settings().api.upload_rate_per_minute}/minute"


# Lenient content-type allowlist; browsers vary, so magic bytes are authoritative.
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/zip",
    "text/plain",
    "text/markdown",
    "application/octet-stream",
    "",
}

# Required leading bytes per extension; empty means "text, checked via decode".
MAGIC_BYTES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04", b"PK\x05\x06"),
    ".txt": (),
    ".md": (),
}

# Acceptable libmagic-sniffed MIME types per extension (when libmagic is present).
SNIFFED_MIME: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/zip",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    ".txt": {"text/plain", "application/csv", "text/csv", "inode/x-empty"},
    ".md": {"text/plain", "text/markdown", "inode/x-empty"},
}


def _load_magic() -> object | None:
    """Return a libmagic MIME detector, or None if libmagic is unavailable."""
    try:
        import magic

        return magic.Magic(mime=True)
    except Exception:
        return None


_MAGIC = _load_magic()


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Default-deny when a key is configured; open in local (no-key) mode."""
    expected = get_settings().api.api_key
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def validate_upload(ext: str, content: str | bytes, content_type: str | None) -> None:
    """Reject mismatched content type, wrong magic bytes/MIME, and zip bombs."""
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    settings = get_settings().api

    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File too large (max {settings.max_upload_mb} MB)",
        )

    declared = (content_type or "").split(";")[0].strip().lower()
    if declared not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {declared}",
        )

    magics = MAGIC_BYTES.get(ext, ())
    if magics and not any(data.startswith(m) for m in magics):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File contents do not match {ext}",
        )

    _sniff_mime(ext, data)

    if ext in (".txt", ".md"):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Text upload is not valid UTF-8",
            ) from None

    if ext == ".docx":
        _reject_zip_bomb(data, settings.max_uncompressed_mb * 1024 * 1024)


def _sniff_mime(ext: str, data: bytes) -> None:
    """Real MIME sniffing via libmagic when available; a no-op otherwise."""
    if _MAGIC is None:
        return
    detected = str(_MAGIC.from_buffer(data)).lower()  # type: ignore[attr-defined]
    allowed = SNIFFED_MIME.get(ext)
    if allowed and detected not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Sniffed type {detected} does not match {ext}",
        )


def _reject_zip_bomb(data: bytes, max_uncompressed: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(info.file_size for info in zf.infolist())
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
