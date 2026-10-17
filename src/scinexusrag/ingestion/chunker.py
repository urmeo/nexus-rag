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
    """A text chunk with retrieval metadata; source offsets use an exclusive end."""

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


@dataclass(frozen=True)
class _TextSpan:
    text: str
    start: int | None = None
    end: int | None = None

    def slice(self, start: int, end: int) -> "_TextSpan":
        text = self.text[start:end]
        left = len(text) - len(text.lstrip())
        right = len(text.rstrip())
        return _TextSpan(
            text.strip(),
            None if self.start is None else self.start + start + left,
            None if self.start is None else self.start + start + right,
        )


def _generate_chunk_id(document_id: str, index: int, content: str) -> str:
    hash_input = f"{document_id}:{index}:{content[:100]}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


def _join_spans(spans: list[_TextSpan], separator: str, prefix: str = "") -> _TextSpan:
    return _TextSpan(
        prefix + separator.join(span.text for span in spans), spans[0].start, spans[-1].end
    )


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
        chunks = self._chunk_by_sections(document, doc_name) if document.sections else []
        if not chunks:
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

            start, end = section.start_offset, section.end_offset
            if start is None and end is None:
                found = document.content.find(section_text)
                unique = found >= 0 and document.content.find(section_text, found + 1) < 0
                start = found if unique else None
                end = start + len(section_text) if start is not None else None
                source = _TextSpan(section_text, start, end)
            elif (
                start is not None
                and end is not None
                and 0 <= start <= end <= len(document.content)
                and document.content[start:end] == section.content
            ):
                source = _TextSpan(section.content, start, end).slice(0, len(section.content))
            else:
                source = _TextSpan(section_text)
            section_title = section.title or ""
            header_prefix = (
                f"[{section_title}]\n\n" if section_title and not section.heading_only else ""
            )
            for span in self._pack_spans(self._paragraph_spans(source), header_prefix):
                chunks.append(
                    self._create_chunk(
                        document,
                        chunk_index,
                        span.text,
                        doc_name,
                        section_title,
                        section.page_number,
                        span.start,
                        span.end,
                    )
                )
                chunk_index += 1

        return chunks

    def _chunk_by_paragraphs(self, document: ParsedDocument, doc_name: str) -> list[Chunk]:
        """Fallback: chunk by paragraphs when no sections detected."""
        source = _TextSpan(document.content, 0, len(document.content))
        spans = self._pack_spans(self._paragraph_spans(source), "")
        return [
            self._create_chunk(document, index, span.text, doc_name, "", None, span.start, span.end)
            for index, span in enumerate(spans)
        ]

    def _pack_spans(self, paragraphs: list[_TextSpan], header_prefix: str) -> list[_TextSpan]:
        "Pack paragraphs into chunk texts near the target size."
        if len(header_prefix) >= self.max_chunk_size:
            raise ValueError("section header leaves no room within max_chunk_size")
        texts: list[_TextSpan] = []
        current: list[_TextSpan] = []
        current_length = len(header_prefix)
        overlap_count = 0

        for para in paragraphs:
            para = para.slice(0, len(para.text))
            if not para.text:
                continue
            para_len = len(para.text)

            if para_len > self.max_chunk_size - len(header_prefix):
                if current:
                    texts.append(_join_spans(current, "\n\n", header_prefix))
                    current, current_length = [], len(header_prefix)
                    overlap_count = 0
                for sent_chunk in self._sentence_spans(para, header_prefix):
                    body = sent_chunk.text.removeprefix(header_prefix).strip()
                    if (
                        len(sent_chunk.text.strip()) >= self.min_chunk_size
                        or not texts
                        or len(texts[-1].text) + 2 + len(body) > self.max_chunk_size
                    ):
                        texts.append(sent_chunk)
                    else:
                        previous = texts[-1]
                        texts[-1] = _TextSpan(
                            previous.text + "\n\n" + body, previous.start, sent_chunk.end
                        )
                continue

            candidate_length = len(header_prefix) + len(
                "\n\n".join(span.text for span in [*current, para])
            )
            exceeds_max = candidate_length > self.max_chunk_size
            exceeds_target = current_length + para_len + 2 > self.target_chunk_size
            if (
                current
                and (exceeds_target or exceeds_max)
                and (current_length >= self.min_chunk_size or exceeds_max)
            ):
                texts.append(_join_spans(current, "\n\n", header_prefix))
                overlap = self._overlap_spans(current)
                if (
                    len(header_prefix) + len("\n\n".join(span.text for span in [*overlap, para]))
                    > self.max_chunk_size
                ):
                    overlap = []
                current = overlap + [para]
                overlap_count = len(overlap)
                current_length = len(header_prefix) + sum(len(span.text) + 2 for span in current)
            else:
                current.append(para)
                current_length += para_len + 2

        if not current:
            return texts
        tail = _join_spans(current, "\n\n", header_prefix)
        new_body = "\n\n".join(span.text for span in current[overlap_count:])
        if (
            len(tail.text) >= self.min_chunk_size
            or not texts
            or len(texts[-1].text) + 2 + len(new_body) > self.max_chunk_size
        ):
            texts.append(tail)
        else:
            previous = texts[-1]
            texts[-1] = _TextSpan(
                previous.text + "\n\n" + new_body, previous.start, current[-1].end
            )
        return texts

    @staticmethod
    def _split_spans(source: _TextSpan, separator: re.Pattern[str]) -> list[_TextSpan]:
        spans: list[_TextSpan] = []
        start = 0
        for match in separator.finditer(source.text):
            span = source.slice(start, match.start())
            if span.text:
                spans.append(span)
            start = match.end()
        tail = source.slice(start, len(source.text))
        if tail.text:
            spans.append(tail)
        return spans

    def _paragraph_spans(self, source: _TextSpan) -> list[_TextSpan]:
        return self._split_spans(source, re.compile(r"\n\s*\n"))

    def _sentence_spans(self, source: _TextSpan, prefix: str) -> list[_TextSpan]:
        """Pack sentences into chunks, wrapping oversized sentences at whitespace."""
        budget = self.max_chunk_size - len(prefix)
        if budget <= 0:
            raise ValueError("section header leaves no room within max_chunk_size")
        sentences: list[_TextSpan] = []
        for span in self._split_spans(source, SENTENCE_BOUNDARY):
            sentences.extend(
                [span] if len(span.text) <= budget else self._word_windows(span, budget)
            )

        chunks: list[_TextSpan] = []
        current = _TextSpan(prefix)

        for sentence in sentences:
            if len(current.text) + len(sentence.text) + 1 <= self.max_chunk_size:
                if current.text.strip() == prefix.strip():
                    current = _TextSpan(prefix + sentence.text, sentence.start, sentence.end)
                else:
                    current = _TextSpan(
                        current.text + " " + sentence.text, current.start, sentence.end
                    )
            else:
                if current.text.strip() and current.text.strip() != prefix.strip():
                    chunks.append(_TextSpan(current.text.strip(), current.start, current.end))
                current = _TextSpan(prefix + sentence.text, sentence.start, sentence.end)

        if current.text.strip() and current.text.strip() != prefix.strip():
            chunks.append(_TextSpan(current.text.strip(), current.start, current.end))

        return chunks

    @staticmethod
    def _word_windows(source: _TextSpan, width: int) -> list[_TextSpan]:
        """Wrap at whitespace, rejecting words that cannot fit without splitting."""
        windows: list[_TextSpan] = []
        current: list[_TextSpan] = []
        current_length = 0
        for match in re.finditer(r"\S+", source.text):
            word = source.slice(match.start(), match.end())
            if len(word.text) > width:
                raise ValueError("word length exceeds available chunk size")
            if current and current_length + 1 + len(word.text) > width:
                windows.append(_join_spans(current, " "))
                current = []
                current_length = 0
            current_length += len(word.text) + (1 if current else 0)
            current.append(word)
        if current:
            windows.append(_join_spans(current, " "))
        return windows

    def _overlap_spans(self, content: list[_TextSpan]) -> list[_TextSpan]:
        """Keep trailing paragraphs within the overlap budget."""
        if not content:
            return []

        overlap_content: list[_TextSpan] = []
        total_length = 0

        for item in reversed(content):
            if total_length + len(item.text) <= self.overlap_size:
                overlap_content.insert(0, item)
                total_length += len(item.text) + 2
            else:
                break

        return overlap_content

    def _add_context_windows(self, chunks: list[Chunk], full_text: str) -> None:
        for chunk in chunks:
            start, end = chunk.start_offset, chunk.end_offset
            if start is None or end is None:
                continue
            chunk.context_before = full_text[max(0, start - self.context_chars) : start].strip()
            chunk.context_after = full_text[end : end + self.context_chars].strip()

    def _create_chunk(
        self,
        document: ParsedDocument,
        index: int,
        content: str,
        doc_name: str,
        section_title: str,
        page_number: int | None,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ) -> Chunk:
        return Chunk(
            id=_generate_chunk_id(document.id, index, content),
            content=content,
            document_id=document.id,
            document_name=doc_name,
            section_title=section_title,
            page_number=page_number,
            chunk_index=index,
            start_offset=start_offset,
            end_offset=end_offset,
            metadata={
                **document.metadata,
                "chunk_index": index,
                "section_title": section_title,
                "page_number": page_number,
                "document_name": doc_name,
                "original_filename": doc_name,
                "start_offset": start_offset,
                "end_offset": end_offset,
            },
        )


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
            chunks.append(
                Chunk(
                    id=_generate_chunk_id(document.id, index, content),
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
    if strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    return HierarchicalChunker(**kwargs)
