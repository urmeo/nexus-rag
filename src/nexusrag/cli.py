"""CLI entry point for NexusRAG."""

import uvicorn

from nexusrag.config import get_settings


def main() -> None:
    """Start the NexusRAG server."""
    settings = get_settings()
    uvicorn.run(
        "nexusrag.api:app",
        host=settings.api.host,
        port=settings.api.port,
    )


if __name__ == "__main__":
    main()
