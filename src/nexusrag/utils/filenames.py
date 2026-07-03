"""Filename resolution utilities."""

from typing import Any


def resolve_display_name(metadata: dict[str, Any], fallback: str = "Unknown") -> str:
    """Resolve the user-facing display name from document metadata.

    Prefers original_filename, then filename, then display_name, then
    document_name. Ingestion always overwrites these with the user's real
    filename before storage, so the stored name is authoritative.
    """
    for key in ("original_filename", "filename", "display_name", "document_name"):
        name = metadata.get(key)
        if name:
            return str(name)
    return fallback
