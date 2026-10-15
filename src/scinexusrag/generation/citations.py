"""Shared citation parsing and cleanup."""

from __future__ import annotations

import re
from unicodedata import decimal

_CITATION = re.compile(r"\[(\d+)\]")


def _citation_number(digits: str) -> int | None:
    normalized = "".join(str(decimal(digit)) for digit in digits).lstrip("0") or "0"
    try:
        return int(normalized)
    except ValueError:
        return None


def extract_citations(text: str) -> list[int]:
    """Return parseable [n] citation numbers in order."""
    return [
        number
        for digits in _CITATION.findall(text)
        if (number := _citation_number(digits)) is not None
    ]


def count_unparseable_citations(text: str) -> int:
    return sum(_citation_number(digits) is None for digits in _CITATION.findall(text))


def strip_citations(text: str, keep: set[int]) -> str:
    """Drop any [n] not in ``keep`` and tidy spaces, preserving line structure."""
    cleaned = _CITATION.sub(
        lambda m: m.group(0) if _citation_number(m.group(1)) in keep else "", text
    )
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
