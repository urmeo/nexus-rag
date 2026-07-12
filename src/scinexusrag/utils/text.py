"""Shared text utilities."""

import re

# One sentence-boundary rule for the whole codebase: split after .!? when the
# next sentence starts with a capital, digit, or opening paren.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping empty/near-empty fragments."""
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in SENTENCE_BOUNDARY.split(text) if len(p.strip()) > 2]
