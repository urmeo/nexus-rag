"""API application."""

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from scinexusrag.api.routes import router
from scinexusrag.api.security import (
    LoopbackHostMiddleware,
    RequestBodyLimitMiddleware,
    limiter,
)
from scinexusrag.ingestion.embedder import EmbeddingConfigError
from scinexusrag.settings import get_settings
from scinexusrag.storage.vector_store import VectorStoreError
from scinexusrag.version import __version__

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    logging.getLogger("scinexusrag").setLevel(settings.log_level)

    docs_enabled = settings.api.docs_enabled and not settings.api.api_key
    if not docs_enabled:
        logger.info(
            "API docs disabled: %s",
            "API key configured" if settings.api.api_key else "docs disabled in settings",
        )

    app = FastAPI(
        title="NexusRAG",
        description="Search uploaded documents and ask Ollama for answers with cited passages.",
        version=__version__,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    @app.exception_handler(VectorStoreError)
    async def vector_store_error(_request: Request, exc: VectorStoreError) -> JSONResponse:
        logger.error("Vector index operation failed", exc_info=(type(exc), exc, exc.__traceback__))
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "The vector index is unavailable. "
                    "Check the server logs and storage access, then retry."
                )
            },
        )

    @app.exception_handler(EmbeddingConfigError)
    async def embedding_config_error(_request: Request, _exc: EmbeddingConfigError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": (
                    "Embedding configuration is invalid or incompatible with this index. "
                    "Custom models need a full 40-character EMBEDDING_REVISION commit SHA; "
                    "local model paths are unsupported. "
                    "Restore the original embedding settings, or set new DATA_DIR and "
                    "LANCEDB_PATH directories and re-upload the original documents. "
                    "Clear the old collection only if you intend to delete and rebuild it."
                )
            },
        )

    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_upload_bytes=settings.api.max_upload_mb * 1024 * 1024,
    )
    origins = settings.api.cors_origins
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
    if settings.api.host in {"localhost", "127.0.0.1", "::1"} and not settings.api.api_key:
        app.add_middleware(LoopbackHostMiddleware)

    @app.get("/health")
    async def health_probe() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)

    frontend_dir = Path(
        os.environ.get("NEXUSRAG_FRONTEND_DIR") or Path(__file__).resolve().parents[1] / "web"
    ).resolve()
    if frontend_dir.is_dir():
        frontend = StaticFiles(directory=frontend_dir)
        for name in ("css", "js", "images"):
            directory = frontend_dir / name
            if directory.is_dir():
                app.mount(f"/{name}", StaticFiles(directory=directory), name=name)

        @app.get("/")
        async def serve_frontend(request: Request) -> Response:
            return await frontend.get_response("index.html", request.scope)

        @app.get("/pipeline", include_in_schema=False)
        async def serve_pipeline(request: Request) -> Response:
            return await frontend.get_response("pipeline.html", request.scope)

    return app


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "scinexusrag.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=1,
    )


app = create_app()

__all__ = ["app", "create_app", "main", "router"]
