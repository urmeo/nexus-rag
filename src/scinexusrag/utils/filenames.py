"""Display names."""

from collections.abc import Mapping


def resolve_display_name(metadata: Mapping[str, object], fallback: str = "Unknown") -> str:
    for key in ("original_filename", "filename", "display_name", "document_name"):
        name = metadata.get(key)
        if isinstance(name, str) and name.strip():
            return name
    return fallback
