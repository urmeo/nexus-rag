"""Utility checks."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from scinexusrag.utils.filenames import resolve_display_name
from scinexusrag.utils.text import SENTENCE_BOUNDARY, split_sentences


@pytest.mark.parametrize(
    "invalid_name",
    [
        None,
        False,
        True,
        0,
        42,
        b"paper.pdf",
        [],
        ["paper.pdf"],
        {},
        {"name": "paper.pdf"},
        "",
        " \t\n\u2003",
    ],
)
def test_invalid_filenames_use_the_fallback(invalid_name):
    assert resolve_display_name({"filename": invalid_name}, fallback="Document") == "Document"


def test_empty_metadata_uses_the_default_name():
    assert resolve_display_name({}) == "Unknown"


@pytest.mark.parametrize(
    "name",
    [
        " \treport.pdf ",
        "résumé 数据 α.md",
        "folder/report.pdf",
        r"C:\papers\report.pdf",
        "file  name.pdf",
    ],
)
def test_valid_names_keep_their_text_and_whitespace(name):
    assert resolve_display_name({"filename": name}) == name


def test_read_only_metadata_is_supported():
    assert resolve_display_name(MappingProxyType({"filename": "paper.pdf"})) == "paper.pdf"


def test_non_string_candidates_do_not_invoke_conversion_or_truthiness():
    class InvalidName:
        def __bool__(self):
            raise AssertionError("metadata value must not be interpreted as a Boolean")

        def __str__(self):
            raise AssertionError("metadata value must not become a fabricated filename")

    assert resolve_display_name({"filename": InvalidName()}) == "Unknown"


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "First claim. Second one! Third? Yes.",
            ["First claim.", "Second one!", "Third?", "Yes."],
        ),
        (
            "We ran the test. 3 trials passed.",
            ["We ran the test.", "3 trials passed."],
        ),
    ],
)
def test_sentence_boundaries_preserve_the_complete_text(text, expected):
    assert split_sentences(text) == expected


def test_grounding_and_chunker_share_sentence_boundaries():
    from scinexusrag.generation import grounding
    from scinexusrag.ingestion import chunker

    assert grounding.split_sentences is split_sentences
    assert chunker.SENTENCE_BOUNDARY is SENTENCE_BOUNDARY


@pytest.mark.parametrize("text", ["42", "5.", "-5", "no", "是", "A.", "نعم"])
def test_short_claims_are_preserved(text):
    assert split_sentences(f"  {text}\n") == [text]


@pytest.mark.parametrize("text", ["", " \n\t", "...", "?!", "—", "()"])
def test_empty_and_punctuation_only_text_has_no_sentences(text):
    assert split_sentences(text) == []


@pytest.mark.parametrize("text", [None, 42, ["claim"]])
def test_nontext_input_has_clear_error(text):
    with pytest.raises(ValueError, match="text must be a string"):
        split_sentences(text)


def test_citations_remain_with_their_claims():
    assert split_sentences("First claim [1].\n\nNext claim [2].") == [
        "First claim [1].",
        "Next claim [2].",
    ]
    assert split_sentences("First claim. [1] Next claim. [2]") == [
        "First claim. [1] Next claim. [2]"
    ]


@pytest.mark.parametrize(
    "first,second",
    [
        ('"First claim."', "Next claim."),
        ("“First claim.”", "Next claim."),
        ("‘First claim!’", "Next claim?"),
        ("'First claim?'", "3 trials passed."),
        ("“First claim.”", "“Second claim.”"),
        ('"First claim."', '"Second claim."'),
    ],
)
def test_quoted_sentences_keep_their_quotes(first, second):
    text = first + " \n\t " + second

    assert split_sentences(text) == [first, second]
    assert all(match.group().isspace() for match in SENTENCE_BOUNDARY.finditer(text))


@pytest.mark.parametrize(
    "text",
    [
        "Smith et al. found an effect.",
        "Several methods, e.g. a baseline, were tested.",
        'Several methods, e.g. "a baseline", were tested.',
        "The measured value was 3.14 mg.",
        '"First claim." [1] Next claim. [2]',
    ],
)
def test_existing_abbreviation_decimal_and_citation_grouping_is_preserved(text):
    assert split_sentences(text) == [text]
