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

from nexusrag import __version__
from nexusrag.api.routes import router
from nexusrag.api.security import limiter
from nexusrag.config import get_settings

# Optional static UI. Resolved from the working directory (./frontend) so it
# works both in local dev (run from the repo root) and in the Docker image
# (WORKDIR /app, frontend copied to /app/frontend); override with an env var
# for a non-editable install served from elsewhere.
FRONTEND_DIR = Path(os.environ.get("NEXUSRAG_FRONTEND_DIR") or Path.cwd() / "frontend")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    logging.getLogger("nexusrag").setLevel(settings.log_level)

    # hide docs/openapi in production (api key set) or when disabled
    docs_on = settings.api.docs_enabled and not settings.api.api_key

    app = FastAPI(
        title="NexusRAG",
        description="Local hybrid retrieval and faithfulness evaluation for scientific papers",
        version=__version__,
        docs_url="/docs" if docs_on else None,
        redoc_url="/redoc" if docs_on else None,
        openapi_url="/openapi.json" if docs_on else None,
    )

    # Per-IP rate limiting (slowapi).
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # CORS — never wildcard-with-credentials; disable credentials under wildcard.
    origins = settings.api.cors_origins
    allow_creds = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_creds,
        # Only what the API actually serves and reads.
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # Liveness only (Docker/k8s): must never init the pipeline or block on
    # models. Readiness with the full schema lives at /api/health.
    @app.get("/health")
    async def health_probe() -> dict[str, str]:
        return {"status": "ok"}

    # API routes
    app.include_router(router)

    # Serve static files
    if FRONTEND_DIR.exists():
        # Mount CSS directory
        css_dir = FRONTEND_DIR / "css"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=css_dir), name="css")

        # Mount JS directory
        js_dir = FRONTEND_DIR / "js"
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=js_dir), name="js")

        @app.get("/")
        async def serve_frontend() -> FileResponse:
            """Serve the frontend HTML."""
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


# Create app instance
app = create_app()

__all__ = ["app", "create_app", "router"]
