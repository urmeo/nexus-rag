"""API module for FastAPI endpoints."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from scinexusrag import __version__
from scinexusrag.api.routes import router
from scinexusrag.api.security import limiter
from scinexusrag.config import get_settings

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
        description="Local hybrid retrieval and faithfulness evaluation for scientific papers",
        version=__version__,
        docs_url="/docs" if docs_on else None,
        redoc_url="/redoc" if docs_on else None,
        openapi_url="/openapi.json" if docs_on else None,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    origins = settings.api.cors_origins
    allow_creds = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_creds,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.get("/health")
    async def health_probe() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)

    if FRONTEND_DIR.exists():
        css_dir = FRONTEND_DIR / "css"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=css_dir), name="css")

        js_dir = FRONTEND_DIR / "js"
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=js_dir), name="js")

        @app.get("/")
        async def serve_frontend() -> FileResponse:
            """Serve the frontend HTML."""
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()

__all__ = ["app", "create_app", "router"]
