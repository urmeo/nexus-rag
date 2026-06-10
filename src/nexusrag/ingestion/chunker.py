"""Production-grade hierarchical chunking for RAG systems."""

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
    """A text chunk with rich metadata for high-quality retrieval."""

    id: str
    content: str
    document_id: str
    metadata: dict[str, object] = field(default_factory=dict)

    # Rich metadata for better retrieval and citation
    document_name: str = ""
    section_title: str = ""
    page_number: int | None = None
    chunk_index: int = 0
    context_before: str = ""  # Previous ~100 chars for context
    context_after: str = ""  # Next ~100 chars for context

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (words * 1.3)."""
        return int(len(self.content.split()) * 1.3)

    @property
    def full_context(self) -> str:
        """Content with surrounding context for better understanding."""
        parts = []
        if self.context_before:
            parts.append(f"[...]{self.context_before}")
        parts.append(self.content)
        if self.context_after:
            parts.append(f"{self.context_after}[...]")
        return " ".join(parts)

    @property
    def header(self) -> str:
        """Generate a header for this chunk."""
        parts = []
        if self.document_name:
            parts.append(self.document_name)
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        return " | ".join(parts) if parts else ""


class HierarchicalChunker:
    """
    Production-grade hierarchical chunking strategy.

    Hierarchy:
    1. Document → Sections (by headings)
    2. Sections → Paragraphs
    3. Paragraphs → Sentences (only if too long)

    Features:
    - Preserves document structure
    - Never splits mid-sentence
    - Includes context windows
    - Prepends section headers
    - Maintains page numbers

    Default configuration optimized for RAG quality:
    - Target: 1200 chars (larger chunks for more context)
    - Max: 1500 chars
    - Min: 400 chars
    - Overlap: 300 chars (prevents missing info at boundaries)
    """

    # Default configuration values
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
        """
        Split document into chunks using hierarchical strategy.

        Args:
            document: Parsed document with sections

        Returns:
            List of chunks with rich metadata
        """
        doc_name = resolve_display_name(document.metadata, fallback="Document")

        logger.info(f"Chunking document: {doc_name}")

        if document.sections:
            chunks = self._chunk_by_sections(document, doc_name)
        else:
            chunks = self._chunk_by_paragraphs(document, doc_name)

        # Add context windows
        if self.include_context:
            self._add_context_windows(chunks, document.content)

        logger.info(f"Created {len(chunks)} chunks from document")
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
            page_num = section.page_number

            # Create header prefix for context
            header_prefix = ""
            if section_title:
                header_prefix = f"[{section_title}]\n\n"

            # Split section into paragraphs
            paragraphs = self._split_into_paragraphs(section_text)

            current_content: list[str] = []
            current_length = len(header_prefix)

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                para_len = len(para)

                # If paragraph alone exceeds max, split by sentences
                if para_len > self.max_chunk_size - len(header_prefix):
                    # Flush current content first
                    if current_content:
                        chunk_text = header_prefix + "\n\n".join(current_content)
                        chunks.append(
                            self._create_chunk(
                                document, chunk_index, chunk_text, doc_name, section_title, page_num
                            )
                        )
                        chunk_index += 1
                        current_content = []
                        current_length = len(header_prefix)

                    # Split large paragraph by sentences
                    sentence_chunks = self._split_by_sentences(para, header_prefix)
                    for sent_chunk in sentence_chunks:
                        if len(sent_chunk.strip()) >= self.min_chunk_size:
                            chunks.append(
                                self._create_chunk(
                                    document,
                                    chunk_index,
                                    sent_chunk,
                                    doc_name,
                                    section_title,
                                    page_num,
                                )
                            )
                            chunk_index += 1
                    continue

                # Check if adding paragraph exceeds target (but not max)
                if current_length + para_len + 2 > self.target_chunk_size:
                    # Check if we have enough content
                    if current_content and current_length >= self.min_chunk_size:
                        chunk_text = header_prefix + "\n\n".join(current_content)
                        chunks.append(
                            self._create_chunk(
                                document, chunk_index, chunk_text, doc_name, section_title, page_num
                            )
                        )
                        chunk_index += 1

                        # Start new chunk with overlap
                        overlap_content = self._get_overlap_content(current_content)
                        current_content = overlap_content + [para]
                        current_length = len(header_prefix) + sum(
                            len(c) + 2 for c in current_content
                        )
                    else:
                        # Not enough content yet, keep adding
                        current_content.append(para)
                        current_length += para_len + 2
                else:
                    current_content.append(para)
                    current_length += para_len + 2

            # Flush remaining content
            if current_content:
                chunk_text = header_prefix + "\n\n".join(current_content)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            document, chunk_index, chunk_text, doc_name, section_title, page_num
                        )
                    )
                    chunk_index += 1
                elif chunks:
                    # Merge small remainder with previous chunk
                    chunks[-1].content += "\n\n" + "\n\n".join(current_content)

        return chunks

    def _chunk_by_paragraphs(self, document: ParsedDocument, doc_name: str) -> list[Chunk]:
        """Fallback: chunk by paragraphs when no sections detected."""
        paragraphs = self._split_into_paragraphs(document.content)
        chunks: list[Chunk] = []

        current_content: list[str] = []
        current_length = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # Large paragraph - split by sentences
            if para_len > self.max_chunk_size:
                if current_content:
                    chunk_text = "\n\n".join(current_content)
                    chunks.append(
                        self._create_chunk(document, chunk_index, chunk_text, doc_name, "", None)
                    )
                    chunk_index += 1
                    current_content = []
                    current_length = 0

                sentence_chunks = self._split_by_sentences(para, "")
                for sent_chunk in sentence_chunks:
                    if len(sent_chunk.strip()) >= self.min_chunk_size:
                        chunks.append(
                            self._create_chunk(
                                document, chunk_index, sent_chunk, doc_name, "", None
                            )
                        )
                        chunk_index += 1
                continue

            # Check if exceeds target
            if current_length + para_len + 2 > self.target_chunk_size:
                if current_content and current_length >= self.min_chunk_size:
                    chunk_text = "\n\n".join(current_content)
                    chunks.append(
                        self._create_chunk(document, chunk_index, chunk_text, doc_name, "", None)
                    )
                    chunk_index += 1

                    overlap_content = self._get_overlap_content(current_content)
                    current_content = overlap_content + [para]
                    current_length = sum(len(c) + 2 for c in current_content)
                else:
                    current_content.append(para)
                    current_length += para_len + 2
            else:
                current_content.append(para)
                current_length += para_len + 2

        # Flush remaining
        if current_content:
            chunk_text = "\n\n".join(current_content)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    self._create_chunk(document, chunk_index, chunk_text, doc_name, "", None)
                )
            elif chunks:
                chunks[-1].content += "\n\n" + chunk_text

        return chunks

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into semantic paragraphs, preserving special blocks."""
        # Split on double newlines
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str, prefix: str) -> list[str]:
        """Split text into sentence-based chunks, never breaking mid-sentence."""
        # Sentence boundary pattern
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
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
        """Add context_before and context_after to each chunk."""
        for chunk in chunks:
            # Find chunk position in full text
            start_idx = full_text.find(chunk.content[:100])
            if start_idx == -1:
                continue

            end_idx = start_idx + len(chunk.content)

            # Context before
            if start_idx > 0:
                before_start = max(0, start_idx - self.context_chars)
                chunk.context_before = full_text[before_start:start_idx].strip()

            # Context after
            if end_idx < len(full_text):
                after_end = min(len(full_text), end_idx + self.context_chars)
                chunk.context_after = full_text[end_idx:after_end].strip()

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


# Backward compatibility alias
SemanticChunker = HierarchicalChunker


def get_chunker(
    strategy: Literal["hierarchical", "semantic", "fixed"] = "hierarchical",
    **kwargs: Any,
) -> HierarchicalChunker | FixedSizeChunker:
    """Factory function to create a chunker."""
    if strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    return HierarchicalChunker(**kwargs)
