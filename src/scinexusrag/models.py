"""Documents and passages shared by ingestion, storage, retrieval and answers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Section:
    """A titled span of a document; ``heading_only`` marks a heading without body text."""

    title: str
    content: str
    level: int = 0
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    heading_only: bool = False


@dataclass
class ParsedDocument:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class Chunk:
    """A passage of a document with its location and the text around it."""

    id: str
    content: str
    document_id: str
    metadata: dict[str, object] = field(default_factory=dict)

    document_name: str = ""
    section_title: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    context_before: str = ""
    context_after: str = ""
    start_offset: int | None = None
    end_offset: int | None = None

    @property
    def full_context(self) -> str:
        parts = []
        if self.context_before:
            parts.append(f"[...]{self.context_before}")
        parts.append(self.content)
        if self.context_after:
            parts.append(f"{self.context_after}[...]")
        return " ".join(parts)
