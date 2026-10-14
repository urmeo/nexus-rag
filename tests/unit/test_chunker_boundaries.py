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


@pytest.mark.parametrize("sectioned", [False, True])
def test_short_tail_merge_does_not_duplicate_overlap(sectioned):
    content = "\n\n".join(
        [("alpha " * 185).strip(), ("beta " * 10).strip(), ("gamma " * 10).strip()]
    )
    document = ParsedDocument(id="overlap-tail", content=content)
    if sectioned:
        document.sections = [Section(title="Methods", content=content, page_number=4)]
    chunker = SemanticChunker(min_chunk_size=200, include_context=False)

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
    chunker = SemanticChunker(context_chars=20)

    chunks = chunker.chunk(document)

    second_start = content.index(second)
    second_end = second_start + len(second)
    assert chunks[1].context_before == content[second_start - 20 : second_start].strip()
    assert chunks[1].context_after == content[second_end : second_end + 20].strip()


def test_identical_passages_get_context_from_their_document_order():
    passage = "The instrument was calibrated before collecting the measurements."
    content = f"First marker.\n\n{passage}\n\nMiddle marker.\n\n{passage}\n\nLast marker."
    document = ParsedDocument(
        id="repeated-passage",
        content=content,
        sections=[
            Section(title="First", content=passage),
            Section(title="Second", content=passage),
        ],
    )

    chunks = SemanticChunker(context_chars=16).chunk(document)

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
    chunker = SemanticChunker(min_chunk_size=200, context_chars=40)

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
    document = ParsedDocument(
        id="shorter-prefix",
        content=content,
        sections=[Section(title="First", content=first), Section(title="Second", content=passage)],
    )

    chunks = SemanticChunker(context_chars=16).chunk(document)

    assert chunks[1].context_before == "Middle marker."
    assert chunks[1].context_after == "Last marker."


def test_unmatched_passage_does_not_borrow_context_from_a_shared_prefix():
    prefix = "This shared passage prefix describes the methods used to collect the measurements. "
    document = ParsedDocument(
        id="unmatched-passage",
        content="Beginning.\n\n" + prefix + "The original result.\n\nEnding.",
        sections=[Section(title="Methods", content=prefix + "A different result.")],
    )

    chunk = SemanticChunker().chunk(document)[0]

    assert chunk.context_before == ""
    assert chunk.context_after == ""


def test_literal_bracketed_text_is_not_treated_as_an_added_section_header():
    content = "[Literal source label]\n\nA short document with a label in its source text."

    chunk = SemanticChunker().chunk(ParsedDocument(id="literal-label", content=content))[0]

    assert chunk.full_context == content
    assert chunk.context_before == ""
    assert chunk.context_after == ""


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
