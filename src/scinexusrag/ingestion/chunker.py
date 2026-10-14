"""Hierarchical, structure-aware chunking."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from scinexusrag.ingestion.parser import ParsedDocument
from scinexusrag.utils.filenames import resolve_display_name
from scinexusrag.utils.text import SENTENCE_BOUNDARY

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata for retrieval and citation."""

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

    @property
    def full_context(self) -> str:
        parts = []
        if self.context_before:
            parts.append(f"[...]{self.context_before}")
        parts.append(self.content)
        if self.context_after:
            parts.append(f"{self.context_after}[...]")
        return " ".join(parts)


class HierarchicalChunker:
    "Splits documents by section, then paragraph, then sentence."

    DEFAULT_TARGET_SIZE = 1200
    DEFAULT_MAX_SIZE = 1500
    DEFAULT_MIN_SIZE = 400
    DEFAULT_OVERLAP = 300
    DEFAULT_CONTEXT_CHARS = 150

    def __init__(
        self,
        min_chunk_size: int = DEFAULT_MIN_SIZE,
        target_chunk_size: int = DEFAULT_TARGET_SIZE,
        max_chunk_size: int = DEFAULT_MAX_SIZE,
        overlap_size: int = DEFAULT_OVERLAP,
        include_context: bool = True,
        context_chars: int = DEFAULT_CONTEXT_CHARS,
    ):
        if min_chunk_size <= 0 or target_chunk_size <= 0 or max_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        if min_chunk_size > min(target_chunk_size, max_chunk_size):
            raise ValueError("min_chunk_size must not exceed target or max chunk size")
        if overlap_size < 0 or overlap_size >= target_chunk_size:
            raise ValueError("overlap_size must be nonnegative and smaller than target_chunk_size")
        if context_chars < 0:
            raise ValueError("context_chars must be nonnegative")

        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.include_context = include_context
        self.context_chars = context_chars

        logger.info(
            "Initialized HierarchicalChunker: target=%d, max=%d, overlap=%d",
            target_chunk_size,
            max_chunk_size,
            overlap_size,
        )

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        doc_name = resolve_display_name(document.metadata, fallback="Document")
        if document.sections:
            chunks = self._chunk_by_sections(document, doc_name)
        else:
            chunks = self._chunk_by_paragraphs(document, doc_name)
        if self.include_context:
            self._add_context_windows(chunks, document.content)
        logger.info("Chunked %s into %d chunks", doc_name, len(chunks))
        return chunks

    def _chunk_by_sections(self, document: ParsedDocument, doc_name: str) -> list[Chunk]:
        """Create chunks from document sections with headers preserved."""
        chunks: list[Chunk] = []
        chunk_index = 0

        for section in document.sections:
            section_text = section.content.strip()
            if not section_text:
                continue

            section_title = section.title or ""
            header_prefix = f"[{section_title}]\n\n" if section_title else ""
            for text in self._pack_paragraphs(
                self._split_into_paragraphs(section_text), header_prefix
            ):
                chunks.append(
                    self._create_chunk(
                        document, chunk_index, text, doc_name, section_title, section.page_number
                    )
                )
                chunk_index += 1

        return chunks

    def _chunk_by_paragraphs(self, document: ParsedDocument, doc_name: str) -> list[Chunk]:
        """Fallback: chunk by paragraphs when no sections detected."""
        texts = self._pack_paragraphs(self._split_into_paragraphs(document.content), "")
        return [
            self._create_chunk(document, index, text, doc_name, "", None)
            for index, text in enumerate(texts)
        ]

    def _pack_paragraphs(self, paragraphs: list[str], header_prefix: str) -> list[str]:
        "Pack paragraphs into chunk texts near the target size."
        if len(header_prefix) >= self.max_chunk_size:
            raise ValueError("section header leaves no room within max_chunk_size")
        texts: list[str] = []
        current: list[str] = []
        current_length = len(header_prefix)
        overlap_count = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para)

            if para_len > self.max_chunk_size - len(header_prefix):
                if current:
                    texts.append(header_prefix + "\n\n".join(current))
                    current, current_length = [], len(header_prefix)
                    overlap_count = 0
                for sent_chunk in self._split_by_sentences(para, header_prefix):
                    body = sent_chunk.removeprefix(header_prefix).strip()
                    if (
                        len(sent_chunk.strip()) >= self.min_chunk_size
                        or not texts
                        or len(texts[-1]) + 2 + len(body) > self.max_chunk_size
                    ):
                        texts.append(sent_chunk)
                    else:
                        texts[-1] += "\n\n" + body
                continue

            candidate_length = len(header_prefix) + len("\n\n".join([*current, para]))
            exceeds_max = candidate_length > self.max_chunk_size
            exceeds_target = current_length + para_len + 2 > self.target_chunk_size
            if (
                current
                and (exceeds_target or exceeds_max)
                and (current_length >= self.min_chunk_size or exceeds_max)
            ):
                texts.append(header_prefix + "\n\n".join(current))
                overlap = self._get_overlap_content(current)
                if len(header_prefix) + len("\n\n".join([*overlap, para])) > self.max_chunk_size:
                    overlap = []
                current = overlap + [para]
                overlap_count = len(overlap)
                current_length = len(header_prefix) + sum(len(c) + 2 for c in current)
            else:
                current.append(para)
                current_length += para_len + 2

        if not current:
            return texts
        body = "\n\n".join(current)
        tail = header_prefix + body
        new_body = "\n\n".join(current[overlap_count:])
        if (
            len(tail) >= self.min_chunk_size
            or not texts
            or len(texts[-1]) + 2 + len(new_body) > self.max_chunk_size
        ):
            texts.append(tail)
        else:
            texts[-1] += "\n\n" + new_body
        return texts

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines and discard empty paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str, prefix: str) -> list[str]:
        """Pack sentences into chunks, wrapping oversized sentences at whitespace."""
        budget = self.max_chunk_size - len(prefix)
        if budget <= 0:
            raise ValueError("section header leaves no room within max_chunk_size")
        sentences: list[str] = []
        for raw in SENTENCE_BOUNDARY.split(text):
            raw = raw.strip()
            if not raw:
                continue
            sentences.extend([raw] if len(raw) <= budget else self._wrap_on_whitespace(raw, budget))

        chunks = []
        current = prefix

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current) + len(sentence) + 1 <= self.max_chunk_size:
                if current.strip() == prefix.strip():
                    current = prefix + sentence
                else:
                    current = current + " " + sentence
            else:
                if current.strip() and current.strip() != prefix.strip():
                    chunks.append(current.strip())
                current = prefix + sentence

        if current.strip() and current.strip() != prefix.strip():
            chunks.append(current.strip())

        return chunks

    @staticmethod
    def _wrap_on_whitespace(text: str, width: int) -> list[str]:
        """Wrap at whitespace, rejecting words that cannot fit without splitting."""
        windows: list[str] = []
        current = ""
        for word in text.split():
            if len(word) > width:
                raise ValueError("word length exceeds available chunk size")
            if current and len(current) + 1 + len(word) > width:
                windows.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            windows.append(current)
        return windows

    def _get_overlap_content(self, content: list[str]) -> list[str]:
        """Get content for overlap based on character count."""
        if not content:
            return []

        overlap_content: list[str] = []
        total_length = 0

        for item in reversed(content):
            if total_length + len(item) <= self.overlap_size:
                overlap_content.insert(0, item)
                total_length += len(item) + 2
            else:
                break

        return overlap_content

    def _add_context_windows(self, chunks: list[Chunk], full_text: str) -> None:
        search_start = 0
        previous_body = ""
        for chunk in chunks:
            body = chunk.content
            if chunk.section_title:
                body = body.removeprefix(f"[{chunk.section_title}]\n\n")
            if not body:
                continue
            lookup_start = search_start + 1 if previous_body.startswith(body) else search_start
            start_idx = full_text.find(body, lookup_start)
            if start_idx == -1:
                continue
            search_start = start_idx
            previous_body = body
            end_idx = start_idx + len(body)
            if start_idx > 0:
                chunk.context_before = full_text[
                    max(0, start_idx - self.context_chars) : start_idx
                ].strip()
            if end_idx < len(full_text):
                chunk.context_after = full_text[end_idx : end_idx + self.context_chars].strip()

    def _create_chunk(
        self,
        document: ParsedDocument,
        index: int,
        content: str,
        doc_name: str,
        section_title: str,
        page_number: int | None,
    ) -> Chunk:
        """Create a chunk with full metadata."""
        chunk_id = self._generate_chunk_id(document.id, index, content)

        return Chunk(
            id=chunk_id,
            content=content,
            document_id=document.id,
            document_name=doc_name,
            section_title=section_title,
            page_number=page_number,
            chunk_index=index,
            metadata={
                **document.metadata,
                "chunk_index": index,
                "section_title": section_title,
                "page_number": page_number,
                "document_name": doc_name,
                "original_filename": doc_name,
            },
        )

    def _generate_chunk_id(self, document_id: str, index: int, content: str) -> str:
        """Generate deterministic chunk ID."""
        hash_input = f"{document_id}:{index}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


class FixedSizeChunker:
    """Group whole words within a character or word budget, with optional overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: str = "chars",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be nonnegative and smaller than chunk_size")
        if length_function not in {"chars", "words"}:
            raise ValueError("length_function must be 'chars' or 'words'")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        doc_name = resolve_display_name(document.metadata, fallback="Unknown Document")

        chunks: list[Chunk] = []
        for index, content in enumerate(self._split_text(text)):
            chunk_id = hashlib.sha256(
                f"{document.id}:{index}:{content[:100]}".encode()
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=content.strip(),
                    document_id=document.id,
                    document_name=doc_name,
                    chunk_index=index,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                        "filename": doc_name,
                    },
                )
            )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        if self.length_function == "words":
            words = text.split()
            step = self.chunk_size - self.chunk_overlap
            windows = []
            for start in range(0, len(words), step):
                windows.append(" ".join(words[start : start + self.chunk_size]))
                if start + self.chunk_size >= len(words):
                    break
            return windows

        texts: list[str] = []
        current: list[str] = []
        for word in text.split():
            if len(word) > self.chunk_size:
                raise ValueError("word length exceeds available chunk size")
            if current and len(" ".join([*current, word])) > self.chunk_size:
                texts.append(" ".join(current))
                overlap: list[str] = []
                for previous_word in reversed(current):
                    candidate = " ".join([previous_word, *overlap])
                    if (
                        len(candidate) > self.chunk_overlap
                        or len(candidate) + 1 + len(word) > self.chunk_size
                    ):
                        break
                    overlap.insert(0, previous_word)
                current = overlap
            current.append(word)

        if current:
            texts.append(" ".join(current))
        return texts


SemanticChunker = HierarchicalChunker


def get_chunker(
    strategy: Literal["hierarchical", "semantic", "fixed"] = "hierarchical",
    **kwargs: Any,
) -> HierarchicalChunker | FixedSizeChunker:
    """Factory function to create a chunker."""
    if strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    return HierarchicalChunker(**kwargs)
