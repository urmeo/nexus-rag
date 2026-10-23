"""API security."""

from __future__ import annotations

import hmac
import io
import re
import zipfile
from dataclasses import dataclass
from typing import Protocol, cast

from fastapi import Header, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from scinexusrag.settings import get_settings

limiter = Limiter(key_func=get_remote_address)

_LOOPBACK_HOST = re.compile(
    r"(?:localhost|127\.0\.0\.1|\[::1\])(?::(?P<port>[0-9]+))?",
    re.IGNORECASE | re.ASCII,
)
_API_BODY_BYTES = 64 * 1024
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
_SAFE_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})


def is_loopback_host(host: str) -> bool:
    match = _LOOPBACK_HOST.fullmatch(host)
    if match is None:
        return False
    port = match["port"]
    try:
        return port is None or 1 <= int(port) <= 65535
    except ValueError:
        return False


def _declared_body_exceeds_limit(headers: Headers, limit: int) -> bool:
    values = headers.getlist("content-length")
    if not values:
        return False
    if len(values) != 1 or not values[0].isascii() or not values[0].isdecimal():
        raise ValueError("Invalid Content-Length")
    digits = values[0].lstrip("0")
    return len(digits) > len(str(limit)) or int(digits or "0") > limit


class LoopbackHostMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            hosts = Headers(scope=scope).getlist("host")
            if len(hosts) != 1 or not is_loopback_host(hosts[0]):
                await JSONResponse({"detail": "Invalid host header"}, status_code=400)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_upload_bytes: int) -> None:
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    def _body_limit(self, scope: Scope) -> int | None:
        if scope["method"] not in _BODY_METHODS:
            return None
        path = scope["path"]
        root_path = scope.get("root_path", "").rstrip("/")
        if root_path and path.startswith(f"{root_path}/"):
            path = path[len(root_path) :]
        path = path.rstrip("/")
        if not path.startswith("/api/"):
            return None
        return _API_BODY_BYTES + (self.max_upload_bytes if path == "/api/ingest" else 0)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self._body_limit(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return

        response = JSONResponse({"detail": "Request body too large"}, status_code=413)
        try:
            body_exceeds_limit = _declared_body_exceeds_limit(Headers(scope=scope), limit)
        except ValueError:
            await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                scope, receive, send
            )
            return
        if body_exceeds_limit:
            await response(scope, receive, send)
            return

        received_bytes = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received_bytes, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit:
                    exceeded = True
                    raise MultiPartException("Request body too large")
            return message

        async def limited_send(message: Message) -> None:
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except MultiPartException:
            if not exceeded:
                raise
        if exceeded:
            await response(scope, receive, send)


def query_limit() -> str:
    return f"{get_settings().api.query_rate_per_minute}/minute"


def upload_limit() -> str:
    return f"{get_settings().api.upload_rate_per_minute}/minute"


@dataclass(frozen=True, slots=True)
class _UploadFormat:
    content_types: tuple[str, ...]
    signatures: tuple[bytes, ...] = ()


_TEXT_FORMAT = _UploadFormat(("text/plain", "text/markdown"))
_UPLOAD_FORMATS = {
    ".pdf": _UploadFormat(("application/pdf",), (b"%PDF",)),
    ".docx": _UploadFormat(
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        ),
        (b"PK\x03\x04", b"PK\x05\x06"),
    ),
    ".txt": _TEXT_FORMAT,
    ".md": _TEXT_FORMAT,
}
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


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api.api_key
    if not expected:
        return

    if not x_api_key or not hmac.compare_digest(
        x_api_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def require_same_site(request: Request) -> None:
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.lower() not in _SAFE_FETCH_SITES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-site request rejected",
        )

    origin = request.headers.get("origin")
    if origin is None or origin in get_settings().api.cors_origins:
        return
    request_origin = str(request.url.replace(path="", query="", fragment=""))
    if origin != request_origin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-site request rejected",
        )


def validate_upload(ext: str, content: str | bytes, content_type: str | None) -> None:
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    settings = get_settings().api

    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File too large (max {settings.max_upload_mb} MB)",
        )

    ext = ext.lower()
    upload_format = _UPLOAD_FORMATS.get(ext)
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

    if ext in (".txt", ".md"):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Text upload is not valid UTF-8",
            ) from None

    if ext == ".docx":
        _validate_docx_size(data, settings.max_uncompressed_mb * 1024 * 1024)


def _sniff_mime(ext: str, data: bytes, upload_format: _UploadFormat) -> None:
    if _MAGIC is None:
        return
    detected = _MAGIC.from_buffer(data).lower()

    if ext in (".txt", ".md"):
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
