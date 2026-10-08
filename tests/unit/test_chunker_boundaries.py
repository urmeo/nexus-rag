"""Keep all input text while respecting split and overlap boundaries."""

import pytest

from scinexusrag.ingestion import FixedSizeChunker, ParsedDocument, Section, SemanticChunker


@pytest.mark.parametrize("sectioned", [False, True])
@pytest.mark.parametrize("content", ["word " * 330, "long " * 299 + "\n\nTAILWORD"])
def test_short_tail_does_not_overflow_full_chunk(content, sectioned):
    prefix = "[Methods]\n\n" if sectioned else ""
    document = ParsedDocument(id="tail", content=content)
    if sectioned:
        document.sections = [Section(title="Methods", content=content, level=1, page_number=4)]
    chunker = SemanticChunker(include_context=False, overlap_size=0)

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2
    assert all(len(chunk.content) <= chunker.max_chunk_size for chunk in chunks)
    recovered = " ".join(chunk.content.removeprefix(prefix) for chunk in chunks)
    assert recovered.split() == content.split()
    if sectioned:
        assert all(chunk.section_title == "Methods" and chunk.page_number == 4 for chunk in chunks)


@pytest.mark.parametrize("length_function", ["chars", "words"])
def test_negative_overlap_fails_before_it_can_skip_content(length_function):
    with pytest.raises(ValueError, match="nonnegative"):
        FixedSizeChunker(chunk_size=2, chunk_overlap=-1, length_function=length_function)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_nonpositive_chunk_size_rejected(chunk_size):
    with pytest.raises(ValueError, match="positive"):
        FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=-2)
