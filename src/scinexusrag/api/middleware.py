"""ASGI middleware that guards the host header and request body size."""

from __future__ import annotations

import re

from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_LOOPBACK_HOST = re.compile(
    r"(?:localhost|127\.0\.0\.1|\[::1\])(?::(?P<port>[0-9]+))?",
    re.IGNORECASE | re.ASCII,
)
_API_BODY_BYTES = 64 * 1024
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})


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
