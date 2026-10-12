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


@pytest.mark.parametrize(
    "content",
    [
        " ".join(f"word{i}" for i in range(60)),
        "A sentence fills this chunk. Another sentence also has enough words to overflow.",
    ],
)
def test_fixed_character_budget_preserves_every_word_without_overlap(content):
    chunker = FixedSizeChunker(chunk_size=30, chunk_overlap=0)
    chunks = chunker.chunk(ParsedDocument(id="fixed", content=content))

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 30 for chunk in chunks)
    assert " ".join(chunk.content for chunk in chunks).split() == content.split()


def test_fixed_character_overlap_keeps_whole_words_and_respects_the_budget():
    content = "alpha beta gamma delta epsilon zeta"
    chunks = FixedSizeChunker(chunk_size=22, chunk_overlap=10).chunk(
        ParsedDocument(id="overlap", content=content)
    )

    assert [chunk.content for chunk in chunks] == [
        "alpha beta gamma delta",
        "delta epsilon zeta",
    ]


def test_fixed_character_overlap_shrinks_to_fit_the_next_word():
    chunks = FixedSizeChunker(chunk_size=10, chunk_overlap=8).chunk(
        ParsedDocument(id="budget", content="one two abcdefghi end")
    )

    assert [chunk.content for chunk in chunks] == ["one two", "abcdefghi", "end"]


def test_fixed_character_chunker_rejects_an_indivisible_oversized_word():
    with pytest.raises(ValueError, match="word length exceeds available chunk size"):
        FixedSizeChunker(chunk_size=10, chunk_overlap=0).chunk(
            ParsedDocument(id="word", content="valid " + "x" * 11)
        )


def test_fixed_chunker_rejects_an_unknown_length_unit():
    with pytest.raises(ValueError, match="length_function"):
        FixedSizeChunker(length_function="tokens")


@pytest.mark.parametrize("word_count,expected_chunks", [(20, 1), (35, 2), (36, 3)])
def test_fixed_word_chunks_do_not_emit_a_tail_made_only_of_overlap(word_count, expected_chunks):
    content = " ".join(f"word{i}" for i in range(word_count))
    chunks = FixedSizeChunker(chunk_size=20, chunk_overlap=5, length_function="words").chunk(
        ParsedDocument(id="word-tail", content=content)
    )

    assert len(chunks) == expected_chunks
    assert chunks[-1].content.split()[-1] == f"word{word_count - 1}"
