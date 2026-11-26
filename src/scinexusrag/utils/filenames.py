"""Display names."""

from collections.abc import Mapping


def resolve_display_name(metadata: Mapping[str, object], fallback: str = "Unknown") -> str:
    name = metadata.get("filename")
    return name if isinstance(name, str) and name.strip() else fallback


def basename(path: str) -> str:
    """Last component of a POSIX or Windows path."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]
