"""Filename resolution utilities."""

from typing import Any


def resolve_display_name(metadata: dict[str, Any], fallback: str = "Unknown") -> str:
    """Resolve the user-facing display name from document metadata.

    Prefers original_filename, then filename, then display_name.
    Skips names that look like temp files (start with 'tmp').
    """
    candidates = [
        metadata.get("original_filename"),
        metadata.get("filename"),
        metadata.get("display_name"),
    ]

    for name in candidates:
        if name and not str(name).startswith("tmp"):
            return str(name)

    # All candidates were temp names; try display_name as last resort
    display = metadata.get("display_name")
    if display:
        return str(display)

    return fallback
