"""Shared citation parsing and cleanup."""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[(\d+)\]")


def extract_citations(text: str) -> list[int]:
    """All [n] citation numbers, in order."""
    return [int(m) for m in _CITATION.findall(text)]


def strip_citations(text: str, keep: set[int]) -> str:
    """Drop any [n] not in ``keep`` and tidy spaces, preserving line structure."""
    cleaned = _CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in keep else "", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
