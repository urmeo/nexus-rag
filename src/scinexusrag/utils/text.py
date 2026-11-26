"""Sentence splitting."""

import re

SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9(])"
    r"""|(?<=[.!?]["'”’])\s+(?=[A-Z0-9("'“‘])"""
)


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY.split(text)
        if any(character.isalnum() for character in sentence)
    ]
