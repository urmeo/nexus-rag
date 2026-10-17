"""API module for FastAPI endpoints."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from scinexusrag import __version__
from scinexusrag.api.routes import router
from scinexusrag.api.security import limiter
from scinexusrag.config import get_settings
from scinexusrag.ingestion.embedder import EmbeddingConfigError
from scinexusrag.storage import VectorStoreError

FRONTEND_DIR = Path(
    os.environ.get("NEXUSRAG_FRONTEND_DIR") or Path(__file__).resolve().parents[1] / "web"
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    logging.getLogger("scinexusrag").setLevel(settings.log_level)

    docs_on = settings.api.docs_enabled and not settings.api.api_key
    if not docs_on:
        logging.getLogger("scinexusrag").info(
            "Interactive API docs disabled (%s); /docs, /redoc, /openapi.json return 404",
            "API key set" if settings.api.api_key else "docs_enabled=False",
        )

    app = FastAPI(
        title="NexusRAG",
        description="Search uploaded documents and ask Ollama for answers with cited passages.",
        version=__version__,
        docs_url="/docs" if docs_on else None,
        redoc_url="/redoc" if docs_on else None,
        openapi_url="/openapi.json" if docs_on else None,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    @app.exception_handler(VectorStoreError)
    async def vector_store_error(_request: Request, exc: VectorStoreError) -> JSONResponse:
        logging.getLogger(__name__).error(
            "Vector index operation failed", exc_info=(type(exc), exc, exc.__traceback__)
        )
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

    origins = settings.api.cors_origins
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.get("/health")
    async def health_probe() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)

    if FRONTEND_DIR.exists():
        for name in ("css", "js", "images"):
            directory = FRONTEND_DIR / name
            if directory.exists():
                app.mount(f"/{name}", StaticFiles(directory=directory), name=name)

        @app.get("/")
        async def serve_frontend() -> FileResponse:
            """Serve the frontend HTML."""
            return FileResponse(FRONTEND_DIR / "index.html")

        @app.get("/pipeline", include_in_schema=False)
        async def serve_pipeline() -> FileResponse:
            """Show the default document and question pipeline."""
            return FileResponse(FRONTEND_DIR / "pipeline.html")

    return app


app = create_app()

__all__ = ["app", "create_app", "router"]
