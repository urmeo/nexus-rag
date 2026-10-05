"""Filename resolution utilities."""

from typing import Any


def resolve_display_name(metadata: dict[str, Any], fallback: str = "Unknown") -> str:
    "Resolve the user-facing display name from document metadata."
    for key in ("original_filename", "filename", "display_name", "document_name"):
        name = metadata.get(key)
        if name:
            return str(name)
    return fallback
