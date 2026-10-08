"""Enforce semantic chunk limits without dropping source words or section metadata."""

import pytest

from scinexusrag.ingestion import ParsedDocument, Section, SemanticChunker


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
    chunker = SemanticChunker(target_chunk_size=target, overlap_size=0, include_context=False)
    chunks = chunker.chunk(_document(content, sectioned))
    assert len(chunks) >= 2
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    _assert_source(chunks, content, sectioned)


@pytest.mark.parametrize("sectioned", [False, True])
@pytest.mark.parametrize("leading_words", [400, 500])
def test_default_overlap_is_dropped_when_next_paragraph_needs_the_budget(sectioned, leading_words):
    content = "a " * leading_words + "\n\n" + "b " * 100 + "\n\n" + "c " * 700
    chunker = SemanticChunker(include_context=False)
    chunks = chunker.chunk(_document(content, sectioned))
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    _assert_source(chunks, content, sectioned)


def test_overlap_is_retained_when_header_and_new_paragraph_fit():
    content = "alpha " * 10 + "\n\n" + "beta " * 10
    document = _document(content, True)
    chunker = SemanticChunker(
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
    chunker = SemanticChunker(
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
    chunker = SemanticChunker(
        min_chunk_size=20, target_chunk_size=100, max_chunk_size=150, overlap_size=0
    )
    with pytest.raises(ValueError, match="section header.*max_chunk_size"):
        chunker.chunk(document)


def test_header_and_word_can_use_exactly_the_maximum_budget():
    title = "H" * 145
    document = ParsedDocument(
        id="exact", content="x", sections=[Section(title=title, content="x", level=1)]
    )
    chunker = SemanticChunker(
        min_chunk_size=20, target_chunk_size=100, max_chunk_size=150, overlap_size=0
    )
    chunks = chunker.chunk(document)
    assert len(chunks) == 1
    assert len(chunks[0].content) == 150
    assert chunks[0].content == "[" + title + "]\n\nx"


@pytest.mark.parametrize("sectioned", [False, True])
def test_indivisible_oversized_word_is_rejected_without_returning_partial_chunks(sectioned):
    content = "Valid first paragraph.\n\n" + "x" * 1501
    chunker = SemanticChunker(include_context=False)
    with pytest.raises(ValueError, match="word length exceeds available chunk size"):
        chunker.chunk(_document(content, sectioned))


def test_header_reduces_the_available_word_budget():
    document = _document("x" * 141, True)
    chunker = SemanticChunker(
        min_chunk_size=20, target_chunk_size=100, max_chunk_size=150, overlap_size=0
    )
    with pytest.raises(ValueError, match="word length exceeds available chunk size"):
        chunker.chunk(document)
