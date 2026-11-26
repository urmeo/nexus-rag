"""Citation parsing."""

from __future__ import annotations

import re
from unicodedata import decimal

_CITATION = re.compile(r"\[(\d+)\][ \t]*")
_CLOSING = frozenset(".,;:!?)]}\r\n")


def _citation_number(digits: str) -> int | None:
    normalized = "".join(str(decimal(digit)) for digit in digits).lstrip("0") or "0"
    try:
        return int(normalized)
    except ValueError:
        return None


def extract_citations(text: str) -> list[int]:
    return [
        number
        for digits in _CITATION.findall(text)
        if (number := _citation_number(digits)) is not None
    ]


def count_unparseable_citations(text: str) -> int:
    return sum(_citation_number(digits) is None for digits in _CITATION.findall(text))


def _trim_trailing_spaces(parts: list[str]) -> None:
    while parts:
        trimmed = parts[-1].rstrip(" \t")
        if trimmed:
            parts[-1] = trimmed
            return
        parts.pop()


def strip_citations(text: str, keep: set[int]) -> str:
    """Remove citations not in ``keep``, changing only the spacing around removed ones."""
    parts: list[str] = []
    position = 0
    for match in _CITATION.finditer(text):
        parts.append(text[position : match.start()])
        position = match.end()
        if _citation_number(match.group(1)) in keep:
            parts.append(match.group(0))
            continue
        padding = match.group(0)[len(match.group(1)) + 2 :]
        following = text[position : position + 1]
        if padding.endswith("  ") and following in {"\r", "\n"}:
            parts.append(padding)
        elif not following or following in _CLOSING:
            _trim_trailing_spaces(parts)
        elif match.start() > 0 and not text[match.start() - 1].isspace():
            parts.append(padding)
    parts.append(text[position:])
    return "".join(parts)
