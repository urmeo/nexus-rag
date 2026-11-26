"""API key, same-site and rate-limit checks for API requests."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from scinexusrag.settings import get_settings

limiter = Limiter(key_func=get_remote_address)

_SAFE_FETCH_SITES = frozenset({"same-origin", "same-site", "none"})


def query_limit() -> str:
    return f"{get_settings().api.query_rate_per_minute}/minute"


def upload_limit() -> str:
    return f"{get_settings().api.upload_rate_per_minute}/minute"


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
