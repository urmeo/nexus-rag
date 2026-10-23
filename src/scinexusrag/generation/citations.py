"""Citation parsing."""

from __future__ import annotations

import re
from unicodedata import decimal

_CITATION = re.compile(r"\[(\d+)\][ \t]*")


def _citation_number(digits: str) -> int | None:
    normalized = "".join(str(decimal(digit)) for digit in digits).lstrip("0") or "0"
    try:
        return int(normalized)
    except ValueError:
        return None


def extract_citations(text: str) -> list[int]:
    """Citation numbers."""
    return [
        number
        for digits in _CITATION.findall(text)
        if (number := _citation_number(digits)) is not None
    ]


def count_unparseable_citations(text: str) -> int:
    return sum(_citation_number(digits) is None for digits in _CITATION.findall(text))


def strip_citations(text: str, keep: set[int]) -> str:
    """Preserve layout."""

    def replace(match: re.Match[str]) -> str:
        if _citation_number(match.group(1)) in keep:
            return match.group(0)
        padding = match.group(0)[len(match.group(1)) + 2 :]
        if padding.endswith("  ") and text[match.end() : match.end() + 1] in {"\r", "\n"}:
            return padding
        if match.start() == 0 or text[match.start() - 1].isspace():
            return ""
        return padding

    return _CITATION.sub(replace, text)
