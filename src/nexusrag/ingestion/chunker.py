"""Hierarchical, structure-aware chunking."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from nexusrag.ingestion.parser import ParsedDocument
from nexusrag.utils.filenames import resolve_display_name

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
    """Splits documents by section, then paragraph, then sentence.

    Keeps chunks near a target size without breaking sentences, prepends
    the section header, and carries page numbers for citation.
    """

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
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.include_context = include_context
        self.context_chars = context_chars

        logger.info(
            f"Initialized HierarchicalChunker: target={target_chunk_size}, "
            f"max={max_chunk_size}, overlap={overlap_size}"
        )

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        doc_name = resolve_display_name(document.metadata, fallback="Document")
        if document.sections:
            chunks = self._chunk_by_sections(document, doc_name)
        else:
            chunks = self._chunk_by_paragraphs(document, doc_name)
        if self.include_context:
            self._add_context_windows(chunks, document.content)
        logger.info(f"Chunked {doc_name} into {len(chunks)} chunks")
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
        """Pack paragraphs into chunk texts near the target size.

        Pure accumulator shared by both strategies: paragraphs fill a chunk up
        to the target, oversized paragraphs are split by sentence, and
        consecutive chunks share ``overlap_size`` characters of tail context. A
        trailing chunk below ``min_chunk_size`` is merged into the previous
        chunk of THIS batch (same section) so citation metadata stays correct;
        if there is nothing to merge into, it is emitted on its own rather than
        dropped, so a short document is never lost.
        """
        texts: list[str] = []
        current: list[str] = []
        current_length = len(header_prefix)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para)

            # A paragraph that alone exceeds max: flush, then sentence-split.
            if para_len > self.max_chunk_size - len(header_prefix):
                if current:
                    texts.append(header_prefix + "\n\n".join(current))
                    current, current_length = [], len(header_prefix)
                for sent_chunk in self._split_by_sentences(para, header_prefix):
                    if len(sent_chunk.strip()) >= self.min_chunk_size or not texts:
                        texts.append(sent_chunk)
                    else:
                        # Merge a short trailing sentence into the previous
                        # chunk (same section) instead of dropping it; strip
                        # its duplicate header prefix first.
                        texts[-1] += "\n\n" + sent_chunk.removeprefix(header_prefix).strip()
                continue

            # Past the target with enough content: emit and carry overlap.
            if current_length + para_len + 2 > self.target_chunk_size:
                if current and current_length >= self.min_chunk_size:
                    texts.append(header_prefix + "\n\n".join(current))
                    current = self._get_overlap_content(current) + [para]
                    current_length = len(header_prefix) + sum(len(c) + 2 for c in current)
                else:
                    current.append(para)
                    current_length += para_len + 2
            else:
                current.append(para)
                current_length += para_len + 2

        if not current:
            return texts
        tail = header_prefix + "\n\n".join(current)
        if len(tail) >= self.min_chunk_size or not texts:
            texts.append(tail)
        else:
            # Merge the short tail into the previous chunk of this same section.
            texts[-1] += "\n\n" + "\n\n".join(current)
        return texts

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into semantic paragraphs, preserving special blocks."""
        # Split on double newlines
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str, prefix: str) -> list[str]:
        """Split text into sentence-based chunks, never breaking mid-sentence.

        A single sentence with no usable boundary that still exceeds the size
        budget is hard-wrapped on whitespace, so no emitted chunk can exceed
        ``max_chunk_size``.
        """
        budget = max(1, self.max_chunk_size - len(prefix))
        sentences: list[str] = []
        for raw in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
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
        """Wrap text into <=width windows, breaking on spaces where possible."""
        windows: list[str] = []
        current = ""
        for word in text.split():
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
        for chunk in chunks:
            body = chunk.content
            header = re.match(r"^\[[^\]]*\]\n\n", body)  # synthetic section prefix
            if header:
                body = body[header.end() :]
            needle = body[:80]
            if not needle:
                continue
            start_idx = full_text.find(needle)
            if start_idx == -1:
                continue
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
    """Simple fixed-size chunking strategy with configurable overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        length_function: str = "chars",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def _measure(self, text: str) -> int:
        if self.length_function == "words":
            return len(text.split())
        return len(text)

    def _split_text(self, text: str) -> list[str]:
        if self.length_function == "words":
            return text.split()
        return list(text)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        text = document.content.strip()
        if not text:
            return []

        doc_name = (
            document.metadata.get("original_filename")
            or document.metadata.get("filename")
            or "Unknown Document"
        )

        chunks: list[Chunk] = []

        if self.length_function == "words":
            words = text.split()
            start = 0
            idx = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                content = " ".join(words[start:end])
                chunk_id = hashlib.sha256(
                    f"{document.id}:{idx}:{content[:100]}".encode()
                ).hexdigest()[:12]
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        content=content,
                        document_id=document.id,
                        document_name=doc_name,
                        chunk_index=idx,
                        metadata={
                            **document.metadata,
                            "chunk_index": idx,
                            "filename": doc_name,
                        },
                    )
                )
                idx += 1
                step = self.chunk_size - self.chunk_overlap
                start += max(step, 1)
        else:
            # Char-based: split on sentence boundaries when possible
            sentences = re.split(r"(?<=[.!?])\s+", text)
            current = ""
            idx = 0

            for sentence in sentences:
                if not sentence.strip():
                    continue
                if current and len(current) + len(sentence) + 1 > self.chunk_size:
                    chunk_id = hashlib.sha256(
                        f"{document.id}:{idx}:{current[:100]}".encode()
                    ).hexdigest()[:12]
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            content=current.strip(),
                            document_id=document.id,
                            document_name=doc_name,
                            chunk_index=idx,
                            metadata={
                                **document.metadata,
                                "chunk_index": idx,
                                "filename": doc_name,
                            },
                        )
                    )
                    idx += 1
                    # Overlap: keep tail of current chunk
                    if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                        current = current[-self.chunk_overlap :] + " " + sentence
                    else:
                        current = sentence
                else:
                    current = (current + " " + sentence).strip() if current else sentence

            if current.strip():
                chunk_id = hashlib.sha256(
                    f"{document.id}:{idx}:{current[:100]}".encode()
                ).hexdigest()[:12]
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        content=current.strip(),
                        document_id=document.id,
                        document_name=doc_name,
                        chunk_index=idx,
                        metadata={
                            **document.metadata,
                            "chunk_index": idx,
                            "filename": doc_name,
                        },
                    )
                )

        return chunks


# Pipeline-facing name for the hierarchical chunker.
SemanticChunker = HierarchicalChunker


def get_chunker(
    strategy: Literal["hierarchical", "semantic", "fixed"] = "hierarchical",
    **kwargs: Any,
) -> HierarchicalChunker | FixedSizeChunker:
    """Factory function to create a chunker."""
    if strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    return HierarchicalChunker(**kwargs)
