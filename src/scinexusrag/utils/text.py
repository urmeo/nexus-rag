"""Shared text utilities."""

import re

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping empty/near-empty fragments."""
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in SENTENCE_BOUNDARY.split(text) if len(p.strip()) > 2]
