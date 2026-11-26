"""Ingestion checks."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Barrier, Event
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pyarrow as pa
import pytest
from docx import Document as DocxDocument
from support import make_config

from scinexusrag.ingestion.chunker import HierarchicalChunker
from scinexusrag.ingestion.embedder import Embedder, EmbeddingConfig, EmbeddingConfigError
from scinexusrag.ingestion.parser import DocumentParseError, DocumentParser
from scinexusrag.models import Chunk, ParsedDocument, Section
from scinexusrag.settings import HF_REVISIONS
from scinexusrag.storage.vector_store import VectorStoreError


class TestHierarchicalChunker:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"min_chunk_size": 0},
            {"target_chunk_size": 0},
            {"max_chunk_size": 0},
            {"min_chunk_size": -1},
            {"target_chunk_size": -1},
            {"max_chunk_size": -1},
            {"min_chunk_size": 101},
            {"max_chunk_size": 19},
            {"overlap_size": -1},
            {"overlap_size": 100},
            {"context_chars": -1},
        ],
    )
    def test_invalid_sizes_fail_before_chunking(self, overrides):
        kwargs = {
            "min_chunk_size": 20,
            "target_chunk_size": 100,
            "max_chunk_size": 150,
            "overlap_size": 0,
            **overrides,
        }
        with pytest.raises(ValueError):
            HierarchicalChunker(**kwargs)

    def test_zero_overlap_and_context_are_valid(self):
        chunker = HierarchicalChunker(overlap_size=0, context_chars=0)
        chunks = chunker.chunk(ParsedDocument(id="short", content="Short content."))

        assert len(chunks) == 1
        assert chunks[0].context_before == ""
        assert chunks[0].context_after == ""

    def test_semantic_chunker_basic(self, parsed_document):
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=500)
        chunks = chunker.chunk(parsed_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_respects_sections(self):
        sections = [
            Section(title="Introduction", content="Intro content here.", level=1),
            Section(title="Methods", content="Methods description.", level=1),
            Section(title="Results", content="Results summary.", level=1),
        ]
        doc = ParsedDocument(
            id="sectioned",
            content="Full document content",
            metadata={"filename": "test.txt"},
            sections=sections,
        )

        chunker = HierarchicalChunker(min_chunk_size=10, max_chunk_size=500)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= len(sections)

        titles_found = sum(1 for c in chunks if any(s.title in c.content for s in sections))
        assert titles_found >= 2

    def test_section_metadata_preserved(self):
        sections = [
            Section(title="Methods", content="Detailed methods.", level=2, page_number=5),
        ]
        doc = ParsedDocument(id="test", content="Content", metadata={}, sections=sections)

        chunker = HierarchicalChunker(min_chunk_size=10, max_chunk_size=500)
        chunks = chunker.chunk(doc)

        assert any(chunk.section_title == "Methods" for chunk in chunks)

    def test_large_section_split(self):
        large_content = "This is a sentence. " * 100
        sections = [Section(title="Large", content=large_content, level=1)]
        doc = ParsedDocument(id="large", content=large_content, metadata={}, sections=sections)

        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=200)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1

    def test_paragraph_based_chunking(self):
        content = """First paragraph with some content.

Second paragraph here.

Third paragraph follows."""

        doc = ParsedDocument(id="paras", content=content, metadata={}, sections=[])
        chunker = HierarchicalChunker(min_chunk_size=10, max_chunk_size=500)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1

    def test_small_chunks_merged(self):
        content = "A.\n\nB.\n\nC.\n\nD.\n\nE."
        doc = ParsedDocument(id="small", content=content, metadata={}, sections=[])

        chunker = HierarchicalChunker(min_chunk_size=20, max_chunk_size=500)
        chunks = chunker.chunk(doc)

        assert [chunk.content for chunk in chunks] == [content]

    def test_empty_document(self):
        doc = ParsedDocument(id="empty", content="", metadata={}, sections=[])
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(doc)

        assert chunks == []


class TestParagraphChunking:
    def _texts(self, paragraphs, section_title="", **overrides):
        params = {
            "min_chunk_size": 20,
            "target_chunk_size": 80,
            "max_chunk_size": 120,
            "overlap_size": 60,
            "include_context": False,
        }
        params.update(overrides)
        content = "\n\n".join(paragraphs)
        document = ParsedDocument(id="paragraphs", content=content)
        if section_title:
            document.sections = [Section(title=section_title, content=content)]
        return [chunk.content for chunk in HierarchicalChunker(**params).chunk(document)]

    def test_small_paragraphs_pack_into_one_chunk(self):
        assert self._texts(["one two", "three four"], min_chunk_size=5) == ["one two\n\nthree four"]

    def test_empty_paragraphs_skipped(self):
        assert self._texts(["", "  ", "real content here"], min_chunk_size=5) == [
            "real content here"
        ]

    def test_header_prefix_on_every_emitted_chunk(self):
        paras = ["a" * 50, "b" * 50, "c" * 50]
        texts = self._texts(paras, section_title="Methods", overlap_size=10)
        assert len(texts) > 1
        assert all(t.startswith("[Methods]\n\n") for t in texts)

    def test_target_overflow_starts_new_chunk_with_overlap(self):
        paras = ["a" * 50, "b" * 50, "c" * 50]
        texts = self._texts(paras)

        assert texts[0] == "a" * 50
        assert texts[1].startswith("a" * 50)
        assert "b" * 50 in texts[1]

    def test_oversized_paragraph_split_by_sentences(self):
        big = " ".join(f"Sentence number {i} is here." for i in range(20))
        assert len(big) > 120
        texts = self._texts([big])
        assert len(texts) > 1
        assert all(len(t) <= 120 for t in texts)
        assert all(len(t) >= 20 for t in texts)

    def test_boundaryless_oversized_paragraph_bounded_by_max(self):

        giant = "word " * 200
        texts = self._texts([giant.strip()])
        assert texts and all(len(t) <= 120 for t in texts)

    def test_short_tail_merges_into_same_batch_chunk(self):

        texts = self._texts(["a" * 100, "zz"])
        assert len(texts) == 1
        assert texts[0].startswith("a" * 100)
        assert texts[0].endswith("zz")

    def test_sole_short_content_still_emitted(self):

        assert self._texts(["tiny note"]) == ["tiny note"]

    def test_oversized_paragraph_flushes_pending_content_first(self):
        big = " ".join(f"Sentence number {i} is here." for i in range(20))
        texts = self._texts(["preamble text first", big])
        assert texts[0] == "preamble text first"

    def test_oversized_paragraph_short_trailing_sentence_not_dropped(self):

        big = "This is a long lead sentence that fills the budget nicely here. " * 3
        texts = self._texts(
            [big + "Short unique CONTENTX."], min_chunk_size=60, target_chunk_size=90
        )
        assert any("CONTENTX" in t for t in texts)


class TestChunkerNoDataLoss:
    def test_short_document_yields_a_chunk(self):
        doc = ParsedDocument(id="d", content="A short valid note.", metadata={})
        doc.sections = [Section(title="", content="A short valid note.", level=0)]
        chunks = HierarchicalChunker(include_context=False).chunk(doc)
        assert len(chunks) == 1
        assert "short valid note" in chunks[0].content

    def test_under_min_section_keeps_its_own_metadata(self):
        doc = ParsedDocument(id="d", content="x", metadata={})
        doc.sections = [
            Section(title="Alpha", content="alpha " * 300, level=0, page_number=1),
            Section(title="Beta", content="short beta note.", level=0, page_number=2),
        ]
        chunks = HierarchicalChunker(include_context=False).chunk(doc)
        beta = [c for c in chunks if c.section_title == "Beta"]
        assert beta and beta[0].page_number == 2

        assert not any(c.section_title == "Alpha" and "beta" in c.content.lower() for c in chunks)


@pytest.mark.parametrize("extension", [".txt", ".md", ".docx", ".pdf"])
def test_heading_only_documents_produce_chunks_with_exact_source_spans(extension, monkeypatch):
    headings = ["CALIBRATION PASSED.", "THE CURRENT IS 42 MA.", "CALIBRATION PASSED."]
    if extension == ".docx":
        source = DocxDocument()
        for heading in headings:
            source.add_heading(heading, level=1)
        buffer = BytesIO()
        source.save(buffer)
        data = buffer.getvalue()
        expected = headings
    elif extension == ".pdf":
        pages = [SimpleNamespace(extract_text=lambda text=text: text) for text in headings]
        reader = SimpleNamespace(is_encrypted=False, pages=pages)
        monkeypatch.setattr("scinexusrag.ingestion.parser.PdfReader", lambda path: reader)
        data = b"%PDF-1.4 fixture"
        expected = headings
    else:
        expected = [f"# {heading}" for heading in headings] if extension == ".md" else headings
        data = "\n\n".join(expected).encode()
    document = DocumentParser().parse_bytes(data, "readings" + extension, extension)

    chunks = HierarchicalChunker(context_chars=12).chunk(document)

    assert [chunk.content for chunk in chunks] == expected
    assert [chunk.section_title for chunk in chunks] == headings
    for chunk, section in zip(chunks, document.sections, strict=True):
        assert section.heading_only
        assert document.content[chunk.start_offset : chunk.end_offset] == chunk.content
        _assert_context_span(chunk, document.content, section.start_offset, section.end_offset, 12)
    assert chunks[0].start_offset < chunks[2].start_offset
    if extension == ".pdf":
        assert [chunk.page_number for chunk in chunks] == [1, 2, 3]


def test_body_and_empty_final_heading_are_chunked_once_with_their_own_metadata():
    source = "# Methods\nMeasured current: 42 mA.\n\n## CALIBRATION PASSED."
    document = DocumentParser().parse_bytes(source.encode(), "readings.md", ".md")

    chunks = HierarchicalChunker(context_chars=16).chunk(document)

    assert [chunk.content for chunk in chunks] == [
        "[Methods]\n\nMeasured current: 42 mA.",
        "## CALIBRATION PASSED.",
    ]
    assert [chunk.section_title for chunk in chunks] == ["Methods", "CALIBRATION PASSED."]
    for chunk, section in zip(chunks, document.sections, strict=True):
        _assert_context_span(chunk, document.content, section.start_offset, section.end_offset, 16)


@pytest.mark.parametrize("section_title", ["", "Legacy heading"])
def test_empty_sections_fall_back_to_original_text_with_exact_source_offsets(section_title):
    content = "  THE CURRENT IS 42 MA.\n\nCALIBRATION PASSED.  "
    document = ParsedDocument(
        id="empty-sections",
        content=content,
        sections=[Section(title=section_title, content=""), Section(title="", content="  ")],
    )

    chunks = HierarchicalChunker(context_chars=10).chunk(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content == content.strip()
    assert chunk.section_title == ""
    _assert_context_span(chunk, content, 2, len(content) - 2, 10)


@pytest.mark.parametrize("sectioned", [False, True])
@pytest.mark.parametrize("content", ["word " * 330, "long " * 299 + "\n\nTAILWORD"])
def test_short_tail_does_not_overflow_full_chunk(content, sectioned):
    prefix = "[Methods]\n\n" if sectioned else ""
    document = ParsedDocument(id="tail", content=content)
    if sectioned:
        document.sections = [Section(title="Methods", content=content, level=1, page_number=4)]
    chunker = HierarchicalChunker(include_context=False, overlap_size=0)

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    recovered = " ".join(chunk.content.removeprefix(prefix) for chunk in chunks)
    assert recovered.split() == content.split()
    if sectioned:
        assert all(chunk.section_title == "Methods" and chunk.page_number == 4 for chunk in chunks)


@pytest.mark.parametrize("sectioned", [False, True])
def test_short_tail_merge_does_not_duplicate_overlap(sectioned):
    content = "\n\n".join(
        [("alpha " * 185).strip(), ("beta " * 10).strip(), ("gamma " * 10).strip()]
    )
    document = ParsedDocument(id="overlap-tail", content=content)
    if sectioned:
        document.sections = [Section(title="Methods", content=content, page_number=4)]
    chunker = HierarchicalChunker(min_chunk_size=200, include_context=False)

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content.removeprefix("[Methods]\n\n") == content
    assert len(chunks[0].content) <= chunker.max_chunk_size
    if sectioned:
        assert chunks[0].section_title == "Methods"
        assert chunks[0].page_number == 4


def test_context_matches_full_passage_when_prefixes_repeat():
    prefix = "The shared introduction explains the study design and measurement procedure. "
    prefix += "Both groups followed this procedure. "
    first = prefix + "The first group used the original instrument."
    second = prefix + "The second group used the replacement instrument."
    content = f"First\n\n{first}\n\nSecond\n\n{second}\n\nEnd of the document."
    document = ParsedDocument(
        id="shared-prefix",
        content=content,
        sections=[Section(title="First", content=first), Section(title="Second", content=second)],
    )
    chunker = HierarchicalChunker(context_chars=20)

    chunks = chunker.chunk(document)

    second_start = content.index(second)
    second_end = second_start + len(second)
    assert chunks[1].context_before == content[second_start - 20 : second_start].strip()
    assert chunks[1].context_after == content[second_end : second_end + 20].strip()


def test_identical_passages_get_context_from_their_document_order():
    passage = "The instrument was calibrated before collecting the measurements."
    content = f"First marker.\n\n{passage}\n\nMiddle marker.\n\n{passage}\n\nLast marker."
    first_start = content.index(passage)
    second_start = content.rindex(passage)
    document = ParsedDocument(
        id="repeated-passage",
        content=content,
        sections=[
            Section(
                title="First",
                content=passage,
                start_offset=first_start,
                end_offset=first_start + len(passage),
            ),
            Section(
                title="Second",
                content=passage,
                start_offset=second_start,
                end_offset=second_start + len(passage),
            ),
        ],
    )

    chunks = HierarchicalChunker(context_chars=16).chunk(document)

    assert chunks[0].context_before == "First marker."
    assert chunks[1].context_before == "Middle marker."
    assert chunks[1].context_after == "Last marker."


def test_full_overlap_chunks_keep_context_from_the_same_source_position():
    first = ("alpha " * 40).strip()
    second = ("beta " * 210).strip()
    methods = first + "\n\n" + second
    results = "Following measurements were performed after calibration."
    content = f"Introduction marker.\n\n{methods}\n\nResults marker.\n\n{results}"
    document = ParsedDocument(
        id="full-overlap-context",
        content=content,
        sections=[
            Section(title="Methods", content=methods),
            Section(title="Results", content=results),
        ],
    )
    chunker = HierarchicalChunker(min_chunk_size=200, context_chars=40)

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    assert chunks[1].content == "[Methods]\n\n" + methods
    assert chunks[1].context_before == "Introduction marker."
    methods_end = content.index(methods) + len(methods)
    assert chunks[1].context_after == content[methods_end : methods_end + 40].strip()


def test_shorter_repeated_prefix_gets_context_from_its_later_occurrence():
    passage = "The instrument was calibrated before collecting the measurements."
    first = passage + " The first group used the original instrument."
    content = f"First marker.\n\n{first}\n\nMiddle marker.\n\n{passage}\n\nLast marker."
    second_start = content.rindex(passage)
    document = ParsedDocument(
        id="shorter-prefix",
        content=content,
        sections=[
            Section(title="First", content=first),
            Section(
                title="Second",
                content=passage,
                start_offset=second_start,
                end_offset=second_start + len(passage),
            ),
        ],
    )

    chunks = HierarchicalChunker(context_chars=16).chunk(document)

    assert chunks[1].context_before == "Middle marker."
    assert chunks[1].context_after == "Last marker."


def test_unmatched_passage_does_not_borrow_context_from_a_shared_prefix():
    prefix = "This shared passage prefix describes the methods used to collect the measurements. "
    document = ParsedDocument(
        id="unmatched-passage",
        content="Beginning.\n\n" + prefix + "The original result.\n\nEnding.",
        sections=[Section(title="Methods", content=prefix + "A different result.")],
    )

    chunk = HierarchicalChunker().chunk(document)[0]

    assert chunk.context_before == ""
    assert chunk.context_after == ""


def test_literal_bracketed_text_is_not_treated_as_an_added_section_header():
    content = "[Literal source label]\n\nA short document with a label in its source text."

    chunk = HierarchicalChunker().chunk(ParsedDocument(id="literal-label", content=content))[0]

    assert chunk.full_context == content
    assert chunk.context_before == ""
    assert chunk.context_after == ""


def _assert_context_span(chunk, content, start, end, context_chars):
    assert chunk.start_offset == start
    assert chunk.end_offset == end
    assert chunk.context_before == content[max(0, start - context_chars) : start].strip()
    assert chunk.context_after == content[end : end + context_chars].strip()


def test_repeated_wrapped_words_keep_their_original_whitespace_positions():
    body = "alpha\tbeta   alpha\nbeta alpha\tbeta"
    content = "Before marker.\n\n" + body + "\n\nAfter marker."
    start = len("Before marker.\n\n")
    document = ParsedDocument(
        id="wrapped-positions",
        content=content,
        sections=[
            Section(
                title="",
                content=body,
                start_offset=start,
                end_offset=start + len(body),
            )
        ],
    )
    chunker = HierarchicalChunker(
        min_chunk_size=1,
        target_chunk_size=12,
        max_chunk_size=12,
        overlap_size=0,
        context_chars=16,
    )

    chunks = chunker.chunk(document)

    assert [chunk.content for chunk in chunks] == ["alpha beta"] * 3
    for chunk, relative_start in zip(chunks, [0, 13, 24], strict=True):
        _assert_context_span(
            chunk, content, start + relative_start, start + relative_start + 10, 16
        )


def test_sentence_packing_keeps_source_offsets_when_separator_whitespace_changes():
    first = "First sentence."
    second = "Second sentence."
    third = "Third sentence."
    content = first + "\t  " + second + "\n " + third
    chunker = HierarchicalChunker(
        min_chunk_size=1,
        target_chunk_size=35,
        max_chunk_size=35,
        overlap_size=0,
        context_chars=18,
    )

    chunks = chunker.chunk(ParsedDocument(id="sentence-positions", content=content))

    assert [chunk.content for chunk in chunks] == [first + " " + second, third]
    second_end = len(first) + 3 + len(second)
    _assert_context_span(chunks[0], content, 0, second_end, 18)
    _assert_context_span(chunks[1], content, second_end + 2, len(content), 18)


def test_paragraph_overlap_keeps_source_offsets_with_headers_and_irregular_blank_lines():
    first = "alpha beta gamma."
    second = "delta epsilon zeta."
    third = "eta theta iota."
    body = first + "\n \n" + second + "\n\t\n" + third
    content = "Introduction.\n\n" + body + "\n\nConclusion."
    start = len("Introduction.\n\n")
    document = ParsedDocument(
        id="overlap-positions",
        content=content,
        sections=[
            Section(
                title="Methods",
                content=body,
                start_offset=start,
                end_offset=start + len(body),
            )
        ],
    )
    chunker = HierarchicalChunker(
        min_chunk_size=1,
        target_chunk_size=35,
        max_chunk_size=70,
        overlap_size=20,
        context_chars=16,
    )

    chunks = chunker.chunk(document)

    prefix = "[Methods]\n\n"
    assert [chunk.content for chunk in chunks] == [
        prefix + first,
        prefix + first + "\n\n" + second,
        prefix + second + "\n\n" + third,
    ]
    second_start = start + len(first) + 3
    _assert_context_span(chunks[0], content, start, start + len(first), 16)
    _assert_context_span(chunks[1], content, start, second_start + len(second), 16)
    _assert_context_span(chunks[2], content, second_start, start + len(body), 16)


def test_explicit_section_offsets_distinguish_identical_passages():
    passage = "The instrument was calibrated."
    content = passage + "\n\nMiddle marker.\n\n" + passage + "\n\nEnd marker."
    start = len(passage + "\n\nMiddle marker.\n\n")
    document = ParsedDocument(
        id="explicit-section-position",
        content=content,
        sections=[
            Section(
                title="Methods",
                content=passage,
                start_offset=start,
                end_offset=start + len(passage),
            )
        ],
    )

    chunk = HierarchicalChunker(context_chars=16).chunk(document)[0]

    assert chunk.content == "[Methods]\n\n" + passage
    _assert_context_span(chunk, content, start, start + len(passage), 16)


@pytest.mark.parametrize(
    "extension,headers", [(".txt", ["FIRST", "SECOND"]), (".md", ["# First", "# Second"])]
)
def test_parsed_repeated_sections_keep_context_after_whitespace_cleaning(extension, headers):
    body = "The instrument\twas calibrated.\n\n  Measurements were recorded."
    source = headers[0] + "\n" + body + "\n\n" + headers[1] + "\n" + body
    document = DocumentParser().parse_bytes(source.encode(), "study" + extension, extension)
    chunker = HierarchicalChunker(context_chars=20)

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    for chunk, section in zip(chunks, document.sections, strict=True):
        _assert_context_span(chunk, document.content, section.start_offset, section.end_offset, 20)
    assert chunks[0].start_offset < chunks[1].start_offset


def test_invalid_section_offsets_do_not_borrow_context_from_matching_text_elsewhere():
    passage = "The instrument was calibrated."
    content = "Before marker.\n\n" + passage + "\n\nAfter marker."
    document = ParsedDocument(
        id="invalid-section-position",
        content=content,
        sections=[
            Section(title="Methods", content=passage, start_offset=0, end_offset=len(passage))
        ],
    )

    chunk = HierarchicalChunker().chunk(document)[0]

    assert chunk.start_offset is None
    assert chunk.end_offset is None
    assert chunk.context_before == ""
    assert chunk.context_after == ""


def test_legacy_section_with_repeated_body_has_no_guessed_context():
    passage = "The instrument was calibrated."
    content = "First marker.\n\n" + passage + "\n\nSecond marker.\n\n" + passage
    document = ParsedDocument(
        id="ambiguous-legacy-section",
        content=content,
        sections=[Section(title="Methods", content=passage)],
    )

    chunk = HierarchicalChunker().chunk(document)[0]

    assert chunk.start_offset is None
    assert chunk.end_offset is None
    assert chunk.context_before == ""
    assert chunk.context_after == ""


def test_short_tail_merge_extends_source_end_without_repeating_overlap():
    first = ("alpha " * 185).strip()
    second = ("beta " * 10).strip()
    third = ("gamma " * 10).strip()
    content = " \t" + first + "\n \n" + second + "\n\t\n" + third + " \n"
    chunker = HierarchicalChunker(min_chunk_size=200, context_chars=20)

    chunks = chunker.chunk(ParsedDocument(id="merged-tail-position", content=content))

    assert len(chunks) == 1
    assert chunks[0].content == first + "\n\n" + second + "\n\n" + third
    _assert_context_span(chunks[0], content, 2, len(content) - 2, 20)


@pytest.mark.parametrize(
    "option",
    ["min_chunk_size", "target_chunk_size", "max_chunk_size", "overlap_size", "context_chars"],
)
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf"), "100"])
def test_hierarchical_sizes_reject_nonintegers_at_construction(option, value):
    with pytest.raises(ValueError, match=rf"{option} must be an integer"):
        HierarchicalChunker(**{option: value})


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_context_switch_requires_a_boolean(value):
    with pytest.raises(ValueError, match="include_context must be a boolean"):
        HierarchicalChunker(include_context=value)


@pytest.mark.parametrize("sectioned", [False, True])
@pytest.mark.parametrize("paragraphs", [["alpha", "beta"], ["alpha", "beta", "gamma"]])
def test_paragraphs_that_exactly_fill_the_target_stay_together(sectioned, paragraphs):
    body = "\n\n".join(paragraphs)
    prefix = "[Methods]\n\n" if sectioned else ""
    document = ParsedDocument(id="exact-target", content=body)
    if sectioned:
        document.sections = [Section(title="Methods", content=body)]
    size = len(prefix) + len(body)
    chunker = HierarchicalChunker(
        min_chunk_size=1,
        target_chunk_size=size,
        max_chunk_size=size + 10,
        overlap_size=0,
    )

    chunks = chunker.chunk(document)

    assert [chunk.content for chunk in chunks] == [prefix + body]
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(body)


@pytest.mark.parametrize("error_type", [EmbeddingConfigError, VectorStoreError])
def test_index_validation_errors_propagate_without_encoding(retriever, error_type):
    failure = error_type("index unavailable")
    retriever.vector_store.validate_embedding_config.side_effect = failure

    with pytest.raises(error_type) as caught:
        retriever.retrieve("evidence")

    assert caught.value is failure
    retriever.embedder.embed_query.assert_not_called()
    retriever.vector_store.search.assert_not_called()


class TestEmbedder:
    @pytest.fixture(autouse=True)
    def sentence_transformers_module(self, monkeypatch):
        module = ModuleType("sentence_transformers")
        module.SentenceTransformer = MagicMock()
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_model = MagicMock()
            mock_model.get_embedding_dimension.return_value = 384
            mock_model.get_sentence_embedding_dimension.return_value = 384
            rng = np.random.default_rng(0)

            def mock_encode(texts, **kwargs):
                shape = (384,) if isinstance(texts, str) else (len(texts), 384)
                return rng.random(shape, dtype=np.float32)

            mock_model.encode.side_effect = mock_encode
            mock_cls.return_value = mock_model
            yield mock_model

    def test_embed_single_text(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = ["This is a test sentence."]

        embeddings = embedder.embed(texts)

        assert embeddings.shape == (1, 384)
        assert embeddings.dtype == np.float32

    def test_embed_batch(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]

        embeddings = embedder.embed(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
        mock_sentence_transformer.encode.assert_called_once()

    def test_embed_query(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        query = "What is the meaning of life?"

        embedding = embedder.embed_query(query)

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embedding_dimension(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")

        assert embedder.dimension == 384
        mock_sentence_transformer.get_embedding_dimension.assert_called()

    def test_empty_list(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")

        embeddings = embedder.embed([])

        assert embeddings.shape == (0, 384)

    def test_batch_size_passed(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = ["text"] * 10

        embedder.embed(texts, batch_size=16)

        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["batch_size"] == 16

    @pytest.mark.parametrize("normalize", [True, False])
    def test_normalization_option_reaches_encoder(self, mock_sentence_transformer, normalize):
        Embedder(model_name="test-model", normalize=normalize).embed(["Test text"])
        assert (
            mock_sentence_transformer.encode.call_args.kwargs["normalize_embeddings"] is normalize
        )

    def test_lazy_model_loading(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            embedder = Embedder(model_name="test-model")

            mock_cls.assert_not_called()

            _ = embedder.model

            mock_cls.assert_called_once_with("test-model", device=None, revision=None)

    def test_device_configuration(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = MagicMock()
            embedder = Embedder(model_name="test-model", device="cuda")

            _ = embedder.model

            mock_cls.assert_called_once_with("test-model", device="cuda", revision=None)

    def test_concurrent_model_first_access_constructs_once(self, observed_lock):
        embedder = Embedder(model_name="test-model", revision="a" * 40)
        lock = observed_lock
        embedder._model_lock = lock
        start = Barrier(4)
        constructing = Event()
        release = Event()
        model = object()

        def construct(*_args, **_kwargs):
            constructing.set()
            assert release.wait(timeout=5)
            return model

        def load():
            start.wait(timeout=5)
            return embedder.model

        with (
            patch("sentence_transformers.SentenceTransformer", side_effect=construct) as factory,
            ThreadPoolExecutor(max_workers=4) as workers,
        ):
            futures = [workers.submit(load) for _ in range(4)]
            try:
                assert constructing.wait(timeout=5)
                assert len(set(lock.wait_for_attempts(4))) == 4
            finally:
                release.set()
            assert all(future.result(timeout=5) is model for future in futures)

        factory.assert_called_once_with("test-model", device=None, revision="a" * 40)
        assert embedder._model is model
        assert embedder._loaded_model_source == ("test-model", "a" * 40)

    def test_failed_constructor_can_retry_without_deadlock(self, observed_lock):
        embedder = Embedder(model_name="test-model", revision="a" * 40)
        embedder._model_lock = observed_lock
        model = object()
        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=[RuntimeError("Constructor failed"), model],
        ) as factory:
            with pytest.raises(RuntimeError, match="Constructor failed"):
                _ = embedder.model
            assert embedder._model is None
            assert embedder._loaded_model_source is None

            with ThreadPoolExecutor(max_workers=1) as workers:
                assert workers.submit(lambda: embedder.model).result(timeout=5) is model
            assert embedder.model is model

        assert factory.call_count == 2
        assert embedder._loaded_model_source == ("test-model", "a" * 40)

    def test_show_progress_flag(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = ["text"] * 5

        embedder.embed(texts, show_progress=True)

        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["show_progress_bar"] is True

    def test_embedding_configuration_uses_effective_defaults(self, mock_sentence_transformer):
        embedder = Embedder()

        assert embedder.embedding_config == EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
            revision=HF_REVISIONS["BAAI/bge-small-en-v1.5"],
            query_prefix="Represent this sentence for searching relevant passages: ",
            doc_prefix="",
            normalize=True,
            dimension=384,
        )
        mock_sentence_transformer.encode.assert_not_called()

    def test_embedding_configuration_preserves_exact_overrides(self, mock_sentence_transformer):
        embedder = Embedder(
            model_name="test-model",
            revision="a" * 40,
            query_prefix="  query: ",
            doc_prefix="passage:  ",
            normalize=False,
        )

        assert embedder.embedding_config == EmbeddingConfig(
            "test-model", "a" * 40, "  query: ", "passage:  ", False, 384
        )

    @pytest.mark.parametrize("revision", [None, "main", "v1.0", "a" * 39, "g" * 40])
    def test_unknown_revision_rejects_before_model_loading(self, revision):
        with patch("sentence_transformers.SentenceTransformer") as model:
            embedder = Embedder(model_name="test-model", revision=revision)

            with pytest.raises(EmbeddingConfigError):
                _ = embedder.embedding_config

            model.assert_not_called()

    def test_local_model_path_rejects_even_with_commit_shaped_revision(self, tmp_path):
        with patch("sentence_transformers.SentenceTransformer") as model:
            embedder = Embedder(model_name=str(tmp_path), revision="a" * 40)

            with pytest.raises(EmbeddingConfigError):
                _ = embedder.embedding_config

            model.assert_not_called()

    @pytest.mark.parametrize("changed_field", ["model_name", "revision"])
    def test_loaded_weights_cannot_be_relabelled(self, mock_sentence_transformer, changed_field):
        embedder = Embedder(model_name="test-model", revision="a" * 40)
        _ = embedder.embedding_config
        changed = "another-model" if changed_field == "model_name" else "b" * 40
        setattr(embedder, changed_field, changed)

        with pytest.raises(EmbeddingConfigError):
            _ = embedder.embedding_config
        mock_sentence_transformer.encode.assert_not_called()


@pytest.fixture
def model_factory(monkeypatch):
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = MagicMock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module.SentenceTransformer


@pytest.mark.parametrize(
    "options",
    [
        {"model_name": ""},
        {"model_name": "  "},
        {"model_name": None},
        {"normalize": 1},
        {"query_prefix": 1},
        {"doc_prefix": False},
        {"revision": ""},
        {"revision": "  "},
        {"revision": 123},
    ],
)
def test_invalid_settings_fail_without_loading(model_factory, options):
    with pytest.raises(EmbeddingConfigError):
        Embedder(**options)

    model_factory.assert_not_called()


def test_invalid_device_fails_without_loading(model_factory):
    with pytest.raises(ValueError, match="device"):
        Embedder(device="automatic")

    model_factory.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_name", ""), ("query_prefix", 1), ("doc_prefix", False), ("normalize", 1)],
)
def test_changed_invalid_settings_fail_before_index_model_load(model_factory, field, value):
    embedder = Embedder()
    setattr(embedder, field, value)

    with pytest.raises(EmbeddingConfigError):
        _ = embedder.embedding_config

    model_factory.assert_not_called()


def test_explicit_empty_prefixes_override_model_defaults(model_factory):
    embedder = Embedder(query_prefix="", doc_prefix="")

    assert embedder.query_prefix == ""
    assert embedder.doc_prefix == ""
    model_factory.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"texts": "sentence"},
        {"texts": ["sentence", None]},
        {"texts": None},
        {"texts": ["sentence"], "batch_size": 0},
        {"texts": [], "batch_size": -1},
        {"texts": ["sentence"], "batch_size": True},
        {"texts": ["sentence"], "batch_size": 1.5},
        {"texts": ["sentence"], "show_progress": 1},
    ],
)
def test_invalid_batch_inputs_fail_without_model_load(model_factory, options):
    with pytest.raises(ValueError):
        Embedder().embed(**options)

    model_factory.assert_not_called()


@pytest.mark.parametrize("query", [None, 3, ["sentence"]])
def test_invalid_query_fails_without_model_load(model_factory, query):
    with pytest.raises(ValueError, match="query"):
        Embedder().embed_query(query)

    model_factory.assert_not_called()


@pytest.mark.parametrize("dimension", [None, False, 0, -1, 3.5, "3", np.nan])
def test_invalid_model_dimension_never_reaches_encoder(model_factory, dimension):
    model = model_factory.return_value
    model.get_embedding_dimension.return_value = dimension

    with pytest.raises(EmbeddingConfigError, match="dimension"):
        Embedder().embed(["sentence"])

    model.encode.assert_not_called()


def test_sentence_dimension_api_is_supported(model_factory):
    model = model_factory.return_value
    model.get_embedding_dimension = None
    model.get_sentence_embedding_dimension.return_value = np.int64(3)

    assert Embedder().dimension == 3


def test_missing_dimension_api_has_clear_error(model_factory):
    model = model_factory.return_value
    model.get_embedding_dimension = None
    model.get_sentence_embedding_dimension = None

    with pytest.raises(EmbeddingConfigError, match="dimension"):
        _ = Embedder().dimension


@pytest.mark.parametrize(
    "values",
    [
        np.ones(3),
        np.ones((1, 2)),
        np.ones((2, 3)),
        np.array([[1, np.nan, 0]]),
        np.array([[1, np.inf, 0]]),
        np.array([[1, 1e100, 0]]),
        np.array([["1", "2", "3"]]),
        np.ones((1, 3), dtype=complex),
    ],
)
def test_invalid_batch_result_rejected(model_factory, values):
    model = model_factory.return_value
    model.get_embedding_dimension.return_value = 3
    model.encode.return_value = values

    with pytest.raises(ValueError, match="Embedding model"):
        Embedder().embed(["sentence"])


@pytest.mark.parametrize("values", [np.ones((1, 3)), np.ones(2), [1, np.nan, 0]])
def test_invalid_query_result_rejected(model_factory, values):
    model = model_factory.return_value
    model.get_embedding_dimension.return_value = 3
    model.encode.return_value = values

    with pytest.raises(ValueError, match="Embedding model"):
        Embedder().embed_query("question")


def test_valid_vectors_are_preserved_as_float32(model_factory):
    model = model_factory.return_value
    model.get_embedding_dimension.return_value = 3
    values = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
    model.encode.return_value = values
    embedder = Embedder(query_prefix="q: ", doc_prefix="d: ", normalize=False)

    result = embedder.embed(["sentence"], batch_size=2)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, values.astype(np.float32))
    model.encode.assert_called_once_with(
        ["d: sentence"],
        batch_size=2,
        show_progress_bar=False,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )
    model.encode.reset_mock()
    model.encode.return_value = values[0]

    query = embedder.embed_query("question")

    np.testing.assert_array_equal(query, values[0].astype(np.float32))
    model.encode.assert_called_once_with(
        "q: question",
        normalize_embeddings=False,
        convert_to_numpy=True,
    )


@pytest.mark.parametrize("invalid_name", [" \t", 42])
def test_chunks_use_a_generic_name_when_the_filename_is_invalid(invalid_name):
    document = ParsedDocument("paper", "The measured voltage is 3.3 V.", {"filename": invalid_name})

    chunks = HierarchicalChunker().chunk(document)

    assert chunks
    assert all(chunk.document_name == "Document" for chunk in chunks)


def _assert_section_spans(document):
    previous_end = 0
    for section in document.sections:
        assert section.start_offset is not None
        assert section.end_offset is not None
        assert previous_end <= section.start_offset <= section.end_offset <= len(document.content)
        assert document.content[section.start_offset : section.end_offset] == section.content
        previous_end = section.end_offset


def _add_table(container, rows):
    table = container.add_table(rows=len(rows), cols=len(rows[0]))
    for row, values in zip(table.rows, rows, strict=True):
        for cell, value in zip(row.cells, values, strict=True):
            cell.text = value
    return table


class TestDocumentParser:
    @pytest.fixture
    def parser(self):
        return DocumentParser()

    def test_parse_txt_file(self, parser, sample_text_file):
        doc = parser.parse(sample_text_file)

        assert isinstance(doc, ParsedDocument)
        assert doc.id is not None
        assert len(doc.id) == 16
        assert "introduction" in doc.content.lower()
        assert "methodology" in doc.content.lower()
        assert "results" in doc.content.lower()
        assert doc.metadata["filename"] == "sample.txt"
        assert doc.metadata["extension"] == ".txt"

    def test_parse_md_file(self, parser, sample_markdown_file):
        doc = parser.parse(sample_markdown_file)

        assert isinstance(doc, ParsedDocument)
        assert "research paper title" in doc.content.lower()
        assert doc.metadata["extension"] == ".md"

        assert len(doc.sections) > 0
        section_titles = [s.title.lower() for s in doc.sections]
        assert any("abstract" in t for t in section_titles)
        assert any("introduction" in t for t in section_titles)

    @pytest.mark.parametrize("name", ["nonexistent.txt", "folder.txt"])
    def test_parse_missing_file(self, parser, tmp_path, name):
        (tmp_path / "folder.txt").mkdir()

        with pytest.raises(DocumentParseError, match=f"File not found: {name}"):
            parser.parse(tmp_path / name)

    def test_parse_unsupported_format(self, parser, tmp_path):
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("content")

        with pytest.raises(DocumentParseError, match="Unsupported format"):
            parser.parse(unsupported)

    @pytest.mark.parametrize("extension,content", [(".txt", "café résumé"), (".md", "# café")])
    def test_non_utf8_files_raise_parse_error(self, parser, tmp_path, extension, content):
        path = tmp_path / ("latin1" + extension)
        path.write_bytes(content.encode("latin-1"))
        with pytest.raises(DocumentParseError, match="UTF-8"):
            parser.parse(path)

    def test_documents_sharing_long_prefix_are_not_duplicates(self, parser):

        prefix = "alpha " * 200
        a = parser.parse_bytes((prefix + "ending one").encode(), "paper.txt", ".txt")
        b = parser.parse_bytes((prefix + "ending two").encode(), "paper.txt", ".txt")
        assert a.id != b.id

    def test_generate_document_id(self, parser, sample_text_file):
        doc1 = parser.parse(sample_text_file)
        doc2 = parser.parse(sample_text_file)

        assert doc1.id == doc2.id
        assert len(doc1.id) == 16
        assert doc1.id.isalnum()

    def test_different_files_different_ids(self, parser, sample_text_file, sample_markdown_file):
        doc1 = parser.parse(sample_text_file)
        doc2 = parser.parse(sample_markdown_file)

        assert doc1.id != doc2.id

    def test_section_extraction_txt(self, parser, sample_text_file):
        doc = parser.parse(sample_text_file)

        assert len(doc.sections) >= 1

    def test_section_extraction_md(self, parser, sample_markdown_file):
        doc = parser.parse(sample_markdown_file)

        assert len(doc.sections) >= 4

        levels = [s.level for s in doc.sections]
        assert 1 in levels
        assert 2 in levels

    def test_section_content_preserved(self, parser, sample_markdown_file):
        doc = parser.parse(sample_markdown_file)

        abstract_sections = [s for s in doc.sections if "abstract" in s.title.lower()]
        assert len(abstract_sections) == 1
        assert "two sensor readings per timestamp" in abstract_sections[0].content.lower()

    def test_metadata_extraction(self, parser, sample_text_file):
        doc = parser.parse(sample_text_file)

        assert "filename" in doc.metadata
        assert "extension" in doc.metadata
        assert "size_bytes" in doc.metadata
        assert "created_at" in doc.metadata
        assert "modified_at" in doc.metadata
        assert doc.metadata["size_bytes"] > 0

    def test_word_count_property(self, parser, sample_text_file):
        doc = parser.parse(sample_text_file)

        assert doc.word_count > 0
        assert doc.word_count == len(doc.content.split())

    def test_empty_file(self, parser, empty_file):
        doc = parser.parse(empty_file)

        assert doc.content == ""
        assert doc.word_count == 0

    def test_parse_bytes(self, parser, sample_text_file):
        content = sample_text_file.read_bytes()
        doc = parser.parse_bytes(content, "uploaded.txt", ".txt")

        assert doc.metadata["filename"] == "uploaded.txt"
        assert "introduction" in doc.content.lower()

    def test_text_cleaning(self, parser, tmp_path):
        messy_file = tmp_path / "messy.txt"
        messy_file.write_text("Multiple   spaces\n\n\n\nToo many newlines")

        doc = parser.parse(messy_file)

        assert "   " not in doc.content
        assert "\n\n\n" not in doc.content

    @pytest.mark.parametrize("extension", [".txt", ".md"])
    def test_repeated_section_bodies_keep_their_own_normalized_source_spans(
        self, parser, tmp_path, extension
    ):
        first_heading = "METHODS" if extension == ".txt" else "## Methods"
        second_heading = "RESULTS" if extension == ".txt" else "## Results"
        passage = "Shared reading.\nSecond line.\n\nAnother paragraph."
        source = (
            "  Preamble\twith   spacing.\n\n\n"
            f"{first_heading}\n\n{passage}\n\n\n"
            f"{second_heading}\n\n{passage}\n\n"
        )
        path = tmp_path / f"repeated{extension}"
        path.write_text(source)

        document = parser.parse(path)

        _assert_section_spans(document)
        assert [section.content for section in document.sections] == [
            "Preamble with spacing.",
            passage,
            passage,
        ]
        assert document.sections[1].start_offset == document.content.index(first_heading) + len(
            first_heading + "\n\n"
        )
        assert document.sections[2].start_offset == document.content.index(second_heading) + len(
            second_heading + "\n\n"
        )
        assert document.sections[1].start_offset != document.sections[2].start_offset

    @pytest.mark.parametrize("extension", [".txt", ".md"])
    def test_headerless_text_has_one_exact_span(self, parser, tmp_path, extension):
        path = tmp_path / f"plain{extension}"
        path.write_text("\n\n First\t line.\n\n\n Second   paragraph. \n\n")

        document = parser.parse(path)

        _assert_section_spans(document)
        assert len(document.sections) == 1
        assert document.sections[0].title == ""
        assert document.sections[0].content == "First line.\n\n Second paragraph."
        assert document.sections[0].start_offset == 0
        assert document.sections[0].end_offset == len(document.content)

    def test_markdown_empty_heading_bodies_preserve_the_heading_spans(self, parser, tmp_path):
        path = tmp_path / "headings.md"
        path.write_text("# Empty\n\n## Methods\n\nA measurement.\n\n### Empty tail\n")

        document = parser.parse(path)

        _assert_section_spans(document)
        assert [section.level for section in document.sections] == [1, 2, 3]
        assert [section.content for section in document.sections] == [
            "# Empty",
            "A measurement.",
            "### Empty tail",
        ]
        assert [section.heading_only for section in document.sections] == [True, False, True]

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("THE CURRENT IS 42 MA.", ["THE CURRENT IS 42 MA."]),
            (
                "  CALIBRATION PASSED.\n\nTHE CURRENT IS 42 MA.\n\nCALIBRATION PASSED.\n",
                ["CALIBRATION PASSED.", "THE CURRENT IS 42 MA.", "CALIBRATION PASSED."],
            ),
            (
                "RESULTS\nA measurement.\n\nCALIBRATION PASSED.",
                ["A measurement.", "CALIBRATION PASSED."],
            ),
        ],
    )
    def test_uppercase_sentences_keep_nonempty_source_spans(self, parser, source, expected):
        document = parser.parse_bytes(source.encode(), "readings.txt", ".txt")

        _assert_section_spans(document)
        assert [section.content for section in document.sections] == expected
        assert all(section.start_offset < section.end_offset for section in document.sections)
        assert document.sections[-1].heading_only

    def test_docx_sections_preserve_paragraph_joins_and_repeated_source_positions(
        self, parser, tmp_path
    ):
        source = DocxDocument()
        source.add_paragraph("  First\t preamble.  ")
        source.add_paragraph("Second preamble.")
        source.add_heading("Methods", level=1)
        source.add_paragraph("  Shared   passage.  ")
        source.add_paragraph("Line one.\nLine two.")
        source.add_heading("Results", level=2)
        source.add_paragraph("Shared\tpassage.")
        source.add_paragraph("Line one.\nLine two.")
        path = tmp_path / "repeated.docx"
        source.save(path)

        document = parser.parse(path)

        assert document.content == (
            "First preamble.\n\nSecond preamble.\n\nMethods\n\n"
            "Shared passage.\n\nLine one.\nLine two.\n\nResults\n\n"
            "Shared passage.\n\nLine one.\nLine two."
        )
        _assert_section_spans(document)
        assert [section.content for section in document.sections] == [
            "First preamble.\n\nSecond preamble.",
            "Shared passage.\n\nLine one.\nLine two.",
            "Shared passage.\n\nLine one.\nLine two.",
        ]
        assert [section.level for section in document.sections] == [0, 1, 2]
        assert document.sections[1].start_offset == document.content.index("Methods\n\n") + len(
            "Methods\n\n"
        )
        assert document.sections[2].start_offset == document.content.index("Results\n\n") + len(
            "Results\n\n"
        )

    def test_docx_tables_follow_paragraph_and_heading_order(self, parser, tmp_path):
        source = DocxDocument()
        source.add_paragraph("Preamble.")
        _add_table(source, [["Sensor", "Offset"], ["A", "-0.25 °C"]])
        source.add_heading("Results", level=1)
        source.add_paragraph("Recorded readings.")
        _add_table(source, [["Sensor", "Value"], ["B", "12.50%"]])
        source.add_paragraph("After table.")
        path = tmp_path / "ordered.docx"
        source.save(path)

        document = parser.parse(path)

        assert document.content == (
            "Preamble.\n\nSensor | Offset\nA | -0.25 °C\n\nResults\n\n"
            "Recorded readings.\n\nSensor | Value\nB | 12.50%\n\nAfter table."
        )
        assert [section.title for section in document.sections] == ["", "Results"]
        assert document.sections[0].content == "Preamble.\n\nSensor | Offset\nA | -0.25 °C"
        assert document.sections[1].content == (
            "Recorded readings.\n\nSensor | Value\nB | 12.50%\n\nAfter table."
        )
        _assert_section_spans(document)

    def test_table_only_docx_preserves_numeric_text_empty_columns_and_upload_metadata(
        self, parser, tmp_path
    ):
        rows = [
            ["-12.50 °C", "0.0001 mg/L", "98.6%", ""],
            ["", "+0.25 m/s", "−3.2 kPa", "1.0e-06 mol"],
        ]
        source = DocxDocument()
        source.add_table(rows=1, cols=4)
        _add_table(source, rows)
        source.add_table(rows=1, cols=4)
        path = tmp_path / "readings.docx"
        source.save(path)

        document = parser.parse_bytes(path.read_bytes(), "readings.docx", ".docx")

        extracted_rows = [
            [value.strip() for value in line.split("|")]
            for line in document.content.splitlines()
            if line.strip()
        ]
        assert extracted_rows == rows
        assert [section.title for section in document.sections] == [""]
        assert document.sections[0].content == document.content
        _assert_section_spans(document)
        assert document.metadata["filename"] == "readings.docx"
        assert document.metadata["extension"] == ".docx"
        assert document.metadata["file_type"] == "docx"

    def test_repeated_docx_tables_have_distinct_section_offsets(self, parser, tmp_path):
        source = DocxDocument()
        for title in ("Methods", "Results"):
            source.add_heading(title, level=1)
            _add_table(source, [["Sensor A", "-0.25 °C"]])
        path = tmp_path / "repeated-tables.docx"
        source.save(path)

        document = parser.parse(path)

        table_text = "Sensor A | -0.25 °C"
        assert [section.content for section in document.sections] == [table_text, table_text]
        assert document.sections[0].start_offset == document.content.index(table_text)
        assert document.sections[1].start_offset == document.content.rindex(table_text)
        assert document.sections[0].start_offset != document.sections[1].start_offset
        _assert_section_spans(document)

    @pytest.mark.parametrize("direction", ["horizontal", "vertical"])
    def test_merged_docx_cells_do_not_duplicate_evidence(self, parser, tmp_path, direction):
        source = DocxDocument()
        table = source.add_table(rows=2, cols=3)
        table.cell(0, 0).text = "Shared result."
        table.cell(0, 0).merge(table.cell(0, 1) if direction == "horizontal" else table.cell(1, 0))
        table.cell(0, 2).text = "First row."
        table.cell(1, 2).text = "Second row."
        path = tmp_path / "merged.docx"
        source.save(path)

        document = parser.parse(path)

        assert document.content.count("Shared result.") == 1
        assert [
            [value.strip() for value in line.split("|")]
            for line in document.content.splitlines()
            if line.strip()
        ] == [["Shared result.", "", "First row."], ["", "", "Second row."]]
        _assert_section_spans(document)

    def test_docx_omitted_edge_cells_keep_their_column_positions(self, parser, tmp_path):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        source = DocxDocument()
        table = _add_table(source, [["", "-0.25 °C", ""], ["Left", "Center", "Right"]])
        row = table.rows[0]._tr
        first, _, last = row.tc_lst
        row.remove(first)
        row.remove(last)
        for name in ("gridBefore", "gridAfter"):
            omitted = OxmlElement("w:" + name)
            omitted.set(qn("w:val"), "1")
            row.get_or_add_trPr().append(omitted)
        path = tmp_path / "omitted.docx"
        source.save(path)

        document = parser.parse(path)

        assert [
            [value.strip() for value in line.split("|")]
            for line in document.content.splitlines()
            if line.strip()
        ] == [["", "-0.25 °C", ""], ["Left", "Center", "Right"]]
        _assert_section_spans(document)

    def test_nested_docx_tables_keep_cell_block_order(self, parser, tmp_path):
        source = DocxDocument()
        table = source.add_table(rows=1, cols=2)
        cell = table.cell(0, 0)
        cell.text = "Before nested."
        _add_table(cell, [["Nested A", "-0.5 °C"], ["Nested B", "2.0%"]])
        cell.add_paragraph("After nested.")
        table.cell(0, 1).text = "Adjacent cell."
        path = tmp_path / "nested.docx"
        source.save(path)

        document = parser.parse(path)

        assert document.content == (
            "Before nested.; [Nested A | -0.5 °C; Nested B | 2.0%]; After nested. | Adjacent cell."
        )
        _assert_section_spans(document)


class TestPdfEdgeCases:
    def test_page_offsets_follow_cleaned_joins_and_retain_page_numbers(self, tmp_path, monkeypatch):
        page_texts = [
            "\n  METHODS\n\nShared   passage.\n\n\nFirst page tail.  \n",
            " \n\t ",
            "\nRESULTS\n\nShared\tpassage.\n\nLast page tail.\n",
            "  Shared passage.\n\nA fourth-page note. \n",
        ]
        reader = SimpleNamespace(
            is_encrypted=False,
            pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in page_texts],
        )
        monkeypatch.setattr("scinexusrag.ingestion.parser.PdfReader", lambda path: reader)
        path = tmp_path / "repeated.pdf"
        path.write_bytes(b"%PDF-1.4 fake")

        document = DocumentParser().parse(path)

        _assert_section_spans(document)
        assert [section.page_number for section in document.sections] == [1, 3, 4]
        assert [section.title for section in document.sections] == ["METHODS", "RESULTS", ""]
        assert [section.content for section in document.sections] == [
            "Shared passage.\n\nFirst page tail.",
            "Shared passage.\n\nLast page tail.",
            "Shared passage.\n\nA fourth-page note.",
        ]
        first_page = "METHODS\n\nShared passage.\n\nFirst page tail."
        third_page = "RESULTS\n\nShared passage.\n\nLast page tail."
        assert document.sections[1].start_offset == len(first_page + "\n\nRESULTS\n\n")
        assert document.sections[2].start_offset == len(first_page + "\n\n" + third_page + "\n\n")

    def test_headerless_pages_reach_chunks(self, tmp_path, monkeypatch):

        pages = [
            SimpleNamespace(extract_text=lambda: "METHODS\nThe sample was measured."),
            SimpleNamespace(
                extract_text=lambda: "Critical measurements on a page without a heading."
            ),
        ]
        reader = SimpleNamespace(pages=pages, is_encrypted=False)
        monkeypatch.setattr("scinexusrag.ingestion.parser.PdfReader", lambda path: reader)
        pdf_path = tmp_path / "mixed.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        chunks = HierarchicalChunker().chunk(DocumentParser().parse(pdf_path))

        assert any("Critical measurements" in chunk.content for chunk in chunks)
        assert (
            next(chunk for chunk in chunks if "Critical measurements" in chunk.content).page_number
            == 2
        )

    def test_encrypted_pdf_actionable_error(self, tmp_path):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret")
        pdf_path = tmp_path / "locked.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        with pytest.raises(DocumentParseError, match="password-protected"):
            DocumentParser().parse(pdf_path)

    def test_scanned_pdf_actionable_error(self, tmp_path, monkeypatch):
        page = SimpleNamespace(images=[object()], extract_text=lambda: "")
        reader = SimpleNamespace(pages=[page], is_encrypted=False)
        monkeypatch.setattr("scinexusrag.ingestion.parser.PdfReader", lambda path: reader)
        pdf_path = tmp_path / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with pytest.raises(DocumentParseError, match="scanned"):
            DocumentParser().parse(pdf_path)


class TestDuplicateNormalization:
    def test_whitespace_only_edits_keep_same_id(self, tmp_path):
        parser = DocumentParser()
        doc = tmp_path / "doc.txt"
        doc.write_text("Alpha beta. Gamma delta.")
        original_id = parser.parse(doc).id

        doc.write_text("Alpha beta.\n\nGamma  delta. ")

        assert parser.parse(doc).id == original_id

    def test_content_edits_change_id(self, tmp_path):
        parser = DocumentParser()
        doc = tmp_path / "doc.txt"
        doc.write_text("Alpha beta. Gamma delta.")
        original_id = parser.parse(doc).id

        doc.write_text("Alpha beta. Gamma delta. Epsilon.")

        assert parser.parse(doc).id != original_id


class TestPreamblePreserved:
    def test_text_preamble_before_first_heading_reaches_chunks(self, tmp_path):

        txt = tmp_path / "doc.txt"
        txt.write_text("Critical preamble sentence.\n\nMETHODS\nBody under heading.\n")
        chunks = HierarchicalChunker().chunk(DocumentParser().parse(txt))

        assert any("Critical preamble sentence." in chunk.content for chunk in chunks)
        assert any("Body under heading." in chunk.content for chunk in chunks)

    def test_markdown_preamble_before_first_heading_kept(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("Critical preamble sentence.\n\n# Heading One\n\nBody under heading.\n")
        doc = DocumentParser().parse(md)
        assert "critical preamble" in doc.content.lower()
        assert any("critical preamble" in s.content.lower() for s in doc.sections)

    def test_docx_preamble_before_first_heading_kept(self, tmp_path):
        d = DocxDocument()
        d.add_paragraph("Critical preamble sentence.")
        d.add_heading("Heading One", level=1)
        d.add_paragraph("Body under heading.")
        path = tmp_path / "doc.docx"
        d.save(path)

        doc = DocumentParser().parse(path)
        assert any("critical preamble" in s.content.lower() for s in doc.sections)


@pytest.mark.parametrize(
    "extension,data",
    [(".pdf", b"%PDF-1.4\ntruncated"), (".docx", b"not a Word document")],
)
def test_damaged_documents_have_actionable_parse_errors(extension, data):
    with pytest.raises(DocumentParseError, match="Cannot read") as error:
        DocumentParser().parse_bytes(data, "damaged" + extension, extension)

    assert error.value.__cause__ is not None


@pytest.mark.parametrize("content_types", [None, b"<broken>"])
def test_docx_missing_or_malformed_package_entries_are_parse_errors(content_types):
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"<unused />")
        if content_types is not None:
            archive.writestr("[Content_Types].xml", content_types)

    with pytest.raises(DocumentParseError, match="Cannot read DOCX"):
        DocumentParser().parse_bytes(stream.getvalue(), "damaged.docx", ".docx")


@pytest.mark.parametrize("root", ["document", "invalid"])
def test_docx_requires_a_document_root_and_body(root):
    source = BytesIO()
    DocxDocument().save(source)
    source.seek(0)
    damaged = BytesIO()
    replacement = (
        f'<w:{root} xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" />'
    ).encode()
    with ZipFile(source) as original, ZipFile(damaged, "w") as rewritten:
        for entry in original.infolist():
            content = replacement if entry.filename == "word/document.xml" else original.read(entry)
            rewritten.writestr(entry, content)

    with pytest.raises(DocumentParseError, match="document body"):
        DocumentParser().parse_bytes(damaged.getvalue(), "damaged.docx", ".docx")


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_parser_preserves_non_document_failures(tmp_path, monkeypatch, error_type):
    path = tmp_path / "paper.docx"
    path.write_bytes(b"placeholder")
    failure = error_type("unrelated failure")

    def fail(_path):
        raise failure

    monkeypatch.setattr("scinexusrag.ingestion.parser.DocxDocument", fail)

    with pytest.raises(error_type) as error:
        DocumentParser().parse(path)

    assert error.value is failure


@pytest.mark.parametrize("extension", [".unknown", ".bad/format", ""])
def test_upload_rejects_unsupported_extensions_before_creating_files(
    tmp_path, monkeypatch, extension
):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    with pytest.raises(DocumentParseError, match="Unsupported format"):
        DocumentParser().parse_bytes(b"contents", "paper", extension)

    assert not list(tmp_path.iterdir())


def test_upload_cleans_temporary_files_after_a_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    def fail_write(_path, _data):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", fail_write)

    with pytest.raises(OSError, match="disk full"):
        DocumentParser().parse_bytes(b"contents", "paper.txt", ".txt")

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("data", [b"a reading", b"\xff"])
def test_upload_cleans_temporary_files_after_parsing(tmp_path, monkeypatch, data):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    if data == b"\xff":
        with pytest.raises(DocumentParseError, match="UTF-8"):
            DocumentParser().parse_bytes(data, "paper.txt", ".txt")
    else:
        assert DocumentParser().parse_bytes(data, "paper.txt", ".txt").content == "a reading"

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("marker", ["```", "~~~", " ````", "   ~~~~"])
def test_markdown_fenced_comments_remain_in_their_section(marker):
    source = (
        "# Methods\n\n"
        f"{marker}python\n# calibration comment\n\n\nprint(42)\n{marker}\n\n"
        "## Results\n\nValue   42."
    )

    document = DocumentParser().parse_bytes(source.encode(), "paper.md", ".md")

    assert [section.title for section in document.sections] == ["Methods", "Results"]
    assert "# calibration comment" in document.sections[0].content
    assert "print(42)" in document.sections[0].content
    assert document.sections[1].content == "Value 42."
    for section in document.sections:
        assert document.content[section.start_offset : section.end_offset] == section.content


@pytest.mark.parametrize("false_close", ["```", "~~~~", "```` trailing text"])
def test_markdown_fences_require_a_matching_close(false_close):
    source = f"# Methods\n\n````python\n{false_close}\n# inside code\n`````\n\n## Results\n\n42."

    document = DocumentParser().parse_bytes(source.encode(), "paper.md", ".md")

    assert [section.title for section in document.sections] == ["Methods", "Results"]
    assert "# inside code" in document.sections[0].content


def test_unclosed_markdown_fences_keep_later_hash_lines_as_code():
    source = "# Methods\n\n```python\n# calibration comment\n\n## still code\nprint(42)"

    document = DocumentParser().parse_bytes(source.encode(), "paper.md", ".md")

    assert [section.title for section in document.sections] == ["Methods"]
    assert "## still code" in document.sections[0].content


@pytest.mark.parametrize("literal_fence", ["    ```", "```not`a`fence", "\t~~~"])
def test_literal_fence_text_does_not_hide_markdown_headings(literal_fence):
    source = f"Preamble.\n\n{literal_fence}\n\n## Results\n\n42."

    document = DocumentParser().parse_bytes(source.encode(), "paper.md", ".md")

    assert [section.title for section in document.sections] == ["", "Results"]
    assert document.sections[1].content == "42."


def test_indented_markdown_heading_has_an_exact_normalized_span():
    source = "Preamble.\n \n\n   ## Methods\n\nMeasured   42.\n\n  ## Results\n\n42."

    document = DocumentParser().parse_bytes(source.encode(), "paper.md", ".md")

    assert [section.title for section in document.sections] == ["", "Methods", "Results"]
    assert document.sections[1].content == "Measured 42."
    for section in document.sections:
        assert document.content[section.start_offset : section.end_offset] == section.content


@pytest.mark.parametrize("extension,heading", [(".md", "# Methods"), (".txt", "METHODS")])
def test_utf8_bom_does_not_change_content_sections_or_document_identity(extension, heading):
    source = f"{heading}\n\nMeasured 42 mA."
    parser = DocumentParser()
    plain = parser.parse_bytes(source.encode(), "paper" + extension, extension)
    with_bom = parser.parse_bytes(source.encode("utf-8-sig"), "paper" + extension, extension)

    assert with_bom.content == plain.content
    assert with_bom.sections == plain.sections
    assert with_bom.id == plain.id
    assert with_bom.sections[0].title.lower() == "methods"
    assert with_bom.sections[0].content == "Measured 42 mA."


def test_utf8_bom_only_file_is_empty():
    document = DocumentParser().parse_bytes(b"\xef\xbb\xbf", "empty.txt", ".txt")

    assert document.content == ""
    assert document.sections == []
    assert document.word_count == 0


def test_interior_byte_order_mark_is_preserved():
    source = "Measured \ufeff42 mA."

    document = DocumentParser().parse_bytes(source.encode(), "paper.txt", ".txt")

    assert document.content == source


def _document(content, sectioned):
    document = ParsedDocument(id="limits", content=content, metadata={"filename": "limits.txt"})
    if sectioned:
        document.sections = [Section(title="Methods", content=content, level=1, page_number=4)]
    return document


def _assert_source(chunks, content, sectioned):
    prefix = "[Methods]\n\n" if sectioned else ""
    words = " ".join(chunk.content.removeprefix(prefix) for chunk in chunks).split()
    assert words == content.split()
    if sectioned:
        assert all(chunk.section_title == "Methods" and chunk.page_number == 4 for chunk in chunks)
        assert all(chunk.content.startswith(prefix) for chunk in chunks)
    assert all(chunk.document_name == "limits.txt" for chunk in chunks)


@pytest.mark.parametrize("sectioned", [False, True])
@pytest.mark.parametrize("target", [1200, 2000])
def test_small_leading_paragraph_cannot_merge_past_maximum(sectioned, target):
    content = "word " * 70 + "\n\n" + "long " * 295
    chunker = HierarchicalChunker(target_chunk_size=target, overlap_size=0, include_context=False)
    chunks = chunker.chunk(_document(content, sectioned))
    assert len(chunks) >= 2
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    _assert_source(chunks, content, sectioned)


@pytest.mark.parametrize("sectioned", [False, True])
@pytest.mark.parametrize("leading_words", [400, 500])
def test_default_overlap_is_dropped_when_next_paragraph_needs_the_budget(sectioned, leading_words):
    content = "a " * leading_words + "\n\n" + "b " * 100 + "\n\n" + "c " * 700
    chunker = HierarchicalChunker(include_context=False)
    chunks = chunker.chunk(_document(content, sectioned))
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    _assert_source(chunks, content, sectioned)


def test_overlap_is_retained_when_header_and_new_paragraph_fit():
    content = "alpha " * 10 + "\n\n" + "beta " * 10
    document = _document(content, True)
    chunker = HierarchicalChunker(
        min_chunk_size=20,
        target_chunk_size=80,
        max_chunk_size=150,
        overlap_size=60,
        include_context=False,
    )
    chunks = chunker.chunk(document)
    assert len(chunks) == 2
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    assert (
        chunks[1].content
        == "[Methods]\n\n" + ("alpha " * 10).strip() + "\n\n" + ("beta " * 10).strip()
    )


@pytest.mark.parametrize("extra", [0, 1])
def test_exact_joined_paragraph_length_includes_header_and_separator(extra):
    content = "a" * 60 + "\n\n" + "b" * (77 + extra)
    chunker = HierarchicalChunker(
        min_chunk_size=20,
        target_chunk_size=200,
        max_chunk_size=150,
        overlap_size=0,
        include_context=False,
    )
    chunks = chunker.chunk(_document(content, True))
    assert len(chunks) == 1 + extra
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    _assert_source(chunks, content, True)


@pytest.mark.parametrize("title", ["H" * 146, "H" * 147])
def test_section_header_without_any_content_budget_is_rejected(title):
    document = ParsedDocument(
        id="header", content="word", sections=[Section(title=title, content="word", level=1)]
    )
    chunker = HierarchicalChunker(
        min_chunk_size=20, target_chunk_size=100, max_chunk_size=150, overlap_size=0
    )
    with pytest.raises(ValueError, match="section header.*max_chunk_size"):
        chunker.chunk(document)


def test_header_and_word_can_use_exactly_the_maximum_budget():
    title = "H" * 145
    document = ParsedDocument(
        id="exact", content="x", sections=[Section(title=title, content="x", level=1)]
    )
    chunker = HierarchicalChunker(
        min_chunk_size=20, target_chunk_size=100, max_chunk_size=150, overlap_size=0
    )
    chunks = chunker.chunk(document)
    assert len(chunks) == 1
    assert len(chunks[0].content) == 150
    assert chunks[0].content == "[" + title + "]\n\nx"


@pytest.mark.parametrize("sectioned", [False, True])
def test_oversized_word_is_split_within_the_chunk_budget(sectioned):
    content = "Valid first paragraph.\n\n" + "x" * 3001 + " tail"
    chunker = HierarchicalChunker(include_context=False)
    chunks = chunker.chunk(_document(content, sectioned))
    prefix = "[Methods]\n\n" if sectioned else ""
    bodies = [chunk.content.removeprefix(prefix) for chunk in chunks]

    assert bodies[0] == "Valid first paragraph."
    assert "".join(bodies[1:]) == "x" * 3001 + " tail"
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    assert all(
        content[chunk.start_offset : chunk.end_offset] == body
        for chunk, body in zip(chunks, bodies, strict=True)
    )


def test_header_reduces_the_available_word_budget():
    document = _document("x" * 141, True)
    chunker = HierarchicalChunker(
        min_chunk_size=20, target_chunk_size=100, max_chunk_size=150, overlap_size=0
    )

    assert [chunk.content for chunk in chunker.chunk(document)] == [
        "[Methods]\n\n" + "x" * 139,
        "[Methods]\n\nxx",
    ]


def test_quoted_sentence_chunking_keeps_exact_source_offsets():
    first, second = '"First complete claim."', '"Second complete claim."'
    content = first + "  \n\t" + second
    chunker = HierarchicalChunker(
        target_chunk_size=32,
        max_chunk_size=32,
        min_chunk_size=1,
        overlap_size=0,
        context_chars=100,
    )

    chunks = chunker.chunk(ParsedDocument(id="quoted-claims", content=content))

    assert [chunk.content for chunk in chunks] == [first, second]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, len(first)),
        (content.index(second), len(content)),
    ]
    assert all(content[chunk.start_offset : chunk.end_offset] == chunk.content for chunk in chunks)
    assert chunks[0].context_before == ""
    assert chunks[0].context_after == second
    assert chunks[1].context_before == first
    assert chunks[1].context_after == ""


def test_vector_schema_dimension_must_match_verified_metadata(vector_store) -> None:
    schema = vector_store.table.schema
    vector_store.clear()
    incompatible = schema.set(
        schema.get_field_index("vector"), pa.field("vector", pa.list_(pa.float32(), 2))
    )
    vector_store.db.create_table("chunks", schema=incompatible)

    with pytest.raises(EmbeddingConfigError):
        vector_store.count()

    assert vector_store._table is None


def test_active_embedder_configuration_must_match_store(vector_store) -> None:
    vector_store.validate_embedding_config(make_config())

    with pytest.raises(EmbeddingConfigError):
        vector_store.validate_embedding_config(make_config(query_prefix="different: "))

    assert vector_store.count() == 0


@pytest.mark.parametrize("count,accepted", [(2, True), (3, False)])
def test_pdf_page_limit_precedes_extraction(tmp_path, monkeypatch, count, accepted):
    parser = DocumentParser()
    monkeypatch.setattr(parser, "MAX_PDF_PAGES", 2)
    extract = Mock(return_value="Evidence.")
    reader = SimpleNamespace(
        is_encrypted=False, pages=[SimpleNamespace(extract_text=extract)] * count
    )
    monkeypatch.setattr("scinexusrag.ingestion.parser.PdfReader", lambda path: reader)
    path = tmp_path / "pages.pdf"
    path.write_bytes(b"%PDF-1.4 fixture")
    if accepted:
        assert parser.parse(path).content == "Evidence.\n\nEvidence."
        assert extract.call_count == count
    else:
        with pytest.raises(DocumentParseError, match="page limit"):
            parser.parse(path)
        extract.assert_not_called()


@pytest.mark.parametrize(
    "texts,accepted,calls",
    [
        (["abc", "def", "g"], False, 2),
        (["      "], False, 1),
        (["a", "bc"], True, 2),
        (["ab", "cd"], False, 2),
    ],
)
def test_pdf_text_budget_counts_raw_text_and_join_separators(
    tmp_path, monkeypatch, texts, accepted, calls
):
    parser = DocumentParser()
    monkeypatch.setattr(parser, "MAX_EXTRACTED_CHARS", 5)
    extracts = [Mock(return_value=text) for text in texts]
    reader = SimpleNamespace(
        is_encrypted=False, pages=[SimpleNamespace(extract_text=extract) for extract in extracts]
    )
    monkeypatch.setattr("scinexusrag.ingestion.parser.PdfReader", lambda path: reader)
    path = tmp_path / "text.pdf"
    path.write_bytes(b"%PDF-1.4 fixture")
    if accepted:
        document = parser.parse(path)
        assert document.content == "a\n\nbc"
        _assert_section_spans(document)
    else:
        with pytest.raises(DocumentParseError, match="extracted text limit"):
            parser.parse(path)
    assert sum(extract.call_count for extract in extracts) == calls


@pytest.mark.parametrize("extension", [".txt", ".md", ".docx"])
@pytest.mark.parametrize("length", [5, 6])
def test_document_text_limit_applies_to_each_format(tmp_path, monkeypatch, extension, length):
    parser = DocumentParser()
    monkeypatch.setattr(parser, "MAX_EXTRACTED_CHARS", 5)
    path = tmp_path / ("text" + extension)
    if extension == ".docx":
        document = DocxDocument()
        document.add_paragraph("a" * length)
        document.save(path)
    else:
        path.write_text("a" * length, encoding="utf-8")
    if length == 5:
        assert parser.parse(path).content == "a" * length
    else:
        with pytest.raises(DocumentParseError, match="extracted text limit"):
            parser.parse(path)
