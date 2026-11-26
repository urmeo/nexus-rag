"""Answer generation checks."""

from __future__ import annotations

import gc
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import httpx
import numpy as np
import pytest

from scinexusrag.generation.citations import extract_citations, strip_citations
from scinexusrag.generation.grounding import GroundingReport, GroundingVerifier
from scinexusrag.generation.llm import LLMClient, LLMError
from scinexusrag.generation.orchestrator import Orchestrator, RAGResponse
from scinexusrag.generation.query_analyzer import QueryAnalyzer, QueryType
from scinexusrag.generation.synthesizer import Source, Synthesizer
from scinexusrag.generation.verifier import AnswerVerifier
from scinexusrag.models import Chunk
from scinexusrag.retrieval.base import RetrievalResult
from scinexusrag.retrieval.corrective import CorrectiveRetriever


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_grounding_rejects_invalid_threshold_before_loading(threshold):
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        GroundingVerifier(threshold=threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_grounding_accepts_threshold_boundaries(threshold):
    verifier = GroundingVerifier(threshold=threshold)
    assert verifier.threshold == threshold
    assert verifier._model is None


def test_failed_label_resolution_does_not_leak_into_retry(monkeypatch):
    bad = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(id2label={2: "entailment", "invalid": "contradiction"})
        )
    )
    good = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(id2label={})))
    factory = Mock(side_effect=[bad, good])
    module = ModuleType("sentence_transformers")
    module.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    verifier = GroundingVerifier(model_name="fixture")

    with pytest.raises(ValueError):
        _ = verifier.model

    assert verifier._model is None
    assert (verifier.entail_idx, verifier.contra_idx) == (1, 0)
    assert verifier.model is good
    assert (verifier.entail_idx, verifier.contra_idx) == (1, 0)


@pytest.mark.parametrize("index", [-1, 3])
def test_entailment_cannot_fall_back_to_another_class(index):
    verifier = GroundingVerifier()
    verifier._model = SimpleNamespace(predict=Mock(return_value=np.array([[0.0, 1.0, 2.0]])))
    verifier.entail_idx = index

    with pytest.raises(ValueError, match="entailment label is outside"):
        verifier.entailment_scores([("evidence", "claim")])


def test_unnamed_model_preserves_explicit_class_indices(monkeypatch):
    model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(id2label={})))
    module = ModuleType("sentence_transformers")
    module.CrossEncoder = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    verifier = GroundingVerifier(model_name="fixture")
    verifier.entail_idx = 2

    assert verifier.model is model
    assert verifier.entail_idx == 2


def test_binary_probabilities_apply_the_sigmoid_once():
    logits = np.array([-2.0, 2.0])

    def predict(pairs, *, activation_fn=None, **kwargs):
        return activation_fn(logits) if activation_fn else 1 / (1 + np.exp(-logits))

    model = SimpleNamespace(predict=Mock(side_effect=predict))
    verifier = GroundingVerifier()
    verifier._model = model

    probabilities = verifier.class_probs([("first", "claim"), ("second", "claim")])

    expected = 1 / (1 + np.exp(-logits))
    np.testing.assert_allclose(probabilities, np.column_stack([1 - expected, expected]))
    assert model.predict.call_args.kwargs["apply_softmax"] is False


def test_multiclass_probabilities_preserve_raw_logit_softmax():
    logits = np.array([[1.2, -0.3, 0.6], [-5.1, 2.0, 1.5]])
    verifier = GroundingVerifier()
    verifier._model = SimpleNamespace(predict=Mock(return_value=logits))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))

    actual = verifier.class_probs([("first", "claim"), ("second", "claim")])

    np.testing.assert_array_equal(actual, exp / exp.sum(axis=1, keepdims=True))


@pytest.mark.parametrize(
    "prediction,pair_count",
    [
        (np.array([0.1, 0.2, 0.3]), 1),
        (np.ones((1, 3)), 2),
        (np.ones((2, 3)), 1),
        (np.empty((1, 0)), 1),
        (np.array(0.5), 1),
        (np.ones((1, 3, 1)), 1),
    ],
)
def test_grounding_rejects_misaligned_predictions(prediction, pair_count):
    verifier = GroundingVerifier()
    verifier._model = SimpleNamespace(predict=Mock(return_value=prediction))

    with pytest.raises(ValueError, match="one row per pair"):
        verifier.class_probs([("evidence", "claim")] * pair_count)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_grounding_rejects_nonfinite_predictions(number):
    verifier = GroundingVerifier()
    verifier._model = SimpleNamespace(predict=Mock(return_value=np.array([[0.0, number, 1.0]])))

    with pytest.raises(ValueError, match="NLI predictions must be finite"):
        verifier.class_probs([("evidence", "claim")])


def test_extreme_binary_logits_remain_finite_without_overflow():
    verifier = GroundingVerifier()
    verifier._model = SimpleNamespace(predict=Mock(return_value=np.array([-10000.0, 0.0, 10000.0])))

    with np.errstate(over="raise"):
        probabilities = verifier.class_probs([("evidence", "claim")] * 3)

    np.testing.assert_array_equal(probabilities, [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])


def test_empty_nli_batch_does_not_load_a_model():
    verifier = GroundingVerifier()

    assert verifier.class_probs([]).shape == (0, 3)
    assert verifier._model is None


def test_blank_sources_cannot_ground_an_answer_or_load_a_model():
    verifier = GroundingVerifier()

    report = verifier.verify("The measurement is valid.", ["", None, " \n\t"])

    assert report.faithfulness == 0.0
    assert report.unsupported == ["The measurement is valid."]
    assert verifier._model is None


def test_grounding_preserves_original_source_positions_after_skipping_blanks():
    predict = Mock(return_value=np.array([[0.0, 4.0, -1.0]]))
    verifier = GroundingVerifier()
    verifier._model = SimpleNamespace(predict=predict)

    report = verifier.verify(
        "The measurement is valid.",
        ["", None, " \n\t", SimpleNamespace(content="Measured evidence.")],
    )

    assert report.faithfulness == 1.0
    assert report.sentences[0].source_index == 3
    assert predict.call_args.args == ([("Measured evidence.", "The measurement is valid.")],)


def test_invalid_source_content_fails_before_model_loading():
    verifier = GroundingVerifier()

    with pytest.raises(ValueError, match="source text must be a string"):
        verifier.verify("The measurement is valid.", [SimpleNamespace(content=123)])

    assert verifier._model is None


@pytest.mark.parametrize(
    "answer",
    [
        "  Claim [1].  \n\n\n    indented  evidence\n",
        "Claim [1].\n\n```python\n    values =  [1,  2]\n    print(values)\n```\n",
        "Claim [1].\n\n| Measure  | Value |\n| -------- | ----- |\n| pH       | 7.4   |\n",
        "Main finding [1].\n\nDetails:\n1. First [2]\n2. Second [3]",
    ],
)
def test_valid_citations_preserve_answer_layout_exactly(answer):
    assert strip_citations(answer, {1, 2, 3}) == answer


def test_invalid_citation_removal_does_not_reformat_other_lines():
    answer = "Claim [99] and evidence [1].\n\n```python\n    value =  42\n```\n"

    cleaned = strip_citations(answer, {1})

    assert cleaned == "Claim and evidence [1].\n\n```python\n    value =  42\n```\n"
    assert extract_citations(cleaned) == [1]


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("a [9] b", "a b"),
        ("a[9] b", "a b"),
        ("[9] Claim", "Claim"),
        ("    [9] item", "    item"),
        ("a[9]b", "ab"),
    ],
)
def test_citation_removal_cleans_only_its_local_padding(answer, expected):
    assert strip_citations(answer, set()) == expected


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("invalid [9].", "invalid."),
        ("invalid [9] .", "invalid."),
        ("see [1] [9], then", "see [1], then"),
        ("(see [9])", "(see)"),
        ("Claim [9]\nNext", "Claim\nNext"),
        ("Claim [9]", "Claim"),
    ],
)
def test_citation_removal_leaves_no_space_before_punctuation_or_line_ends(answer, expected):
    assert strip_citations(answer, {1}) == expected


def test_citation_cleanup_retains_unicode_numbers_and_original_spelling():
    answer = "Evidence [١] [００２] and invalid [9]."

    cleaned = strip_citations(answer, {1, 2})

    assert cleaned == "Evidence [١] [００２] and invalid."
    assert extract_citations(cleaned) == [1, 2]


@pytest.mark.parametrize("index", [0, -1, True, 1.0, "1"])
def test_citation_verification_rejects_invalid_source_indices(index):
    source = Source(index=index, chunk_id="chunk", document_id="document", content="evidence")

    with pytest.raises(ValueError, match="unique positive integers"):
        AnswerVerifier().verify("Claim [1].", [source])


def test_citation_verification_rejects_ambiguous_source_indices():
    sources = [
        Source(index=1, chunk_id="first", document_id="first", content="alpha"),
        Source(index=1, chunk_id="second", document_id="second", content="beta"),
    ]

    with pytest.raises(ValueError, match="unique positive integers"):
        AnswerVerifier().verify("Claim [1].", sources)


def test_answer_verification_preserves_valid_noncontiguous_sources_and_layout():
    sources = [
        Source(index=2, chunk_id="second", document_id="second", content="alpha"),
        Source(index=4, chunk_id="fourth", document_id="fourth", content="beta"),
    ]
    answer = "  First [2].\n\n\n    Second [4].\n"

    result = AnswerVerifier().verify(answer, sources)

    assert result.verified_answer == answer
    assert result.citations_valid == [2, 4]
    assert result.warnings == []


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_invalid_citation_removal_preserves_markdown_hard_breaks(newline):
    answer = f"Claim [99]  {newline}Next claim [1]."

    cleaned = strip_citations(answer, {1})

    assert cleaned == f"Claim   {newline}Next claim [1]."
    assert extract_citations(cleaned) == [1]


@pytest.mark.parametrize("stage", ["construction", "labels"])
def test_concurrent_first_requests_share_model_and_resolved_labels(
    monkeypatch, stage, observed_lock
):
    verifier = GroundingVerifier(model_name="test-nli", revision="a" * 40)
    lock = observed_lock
    verifier._model_lock = lock
    start = Barrier(4)
    initializing = Event()
    release = Event()

    def pause():
        initializing.set()
        assert release.wait(timeout=5)

    class Config:
        @property
        def id2label(self):
            if stage == "labels":
                pause()
            return {0: "neutral", 1: "contradiction", 2: "entailment"}

    model = SimpleNamespace(model=SimpleNamespace(config=Config()))

    def construct(*_args, **_kwargs):
        if stage == "construction":
            pause()
        return model

    factory = Mock(side_effect=construct)
    module = ModuleType("sentence_transformers")
    module.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    def load():
        start.wait(timeout=5)
        loaded = verifier.model
        return loaded, verifier.entail_idx, verifier.contra_idx

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(load) for _ in range(4)]
        try:
            assert initializing.wait(timeout=5)
            assert len({lock.attempts.get(timeout=5) for _ in range(4)}) == 4
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        assert all(future.result(timeout=5) == (model, 2, 1) for future in futures)

    factory.assert_called_once_with("test-nli", device=None, revision="a" * 40)


def test_failed_grounding_model_load_can_retry(monkeypatch):
    verifier = GroundingVerifier(model_name="test-nli")
    model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(id2label={})))
    factory = Mock(side_effect=[RuntimeError("Load failed"), model])
    module = ModuleType("sentence_transformers")
    module.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(RuntimeError, match="Load failed"):
        _ = verifier.model
    assert verifier._model is None

    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(lambda: verifier.model).result(timeout=5) is model
    assert verifier.model is model
    assert factory.call_count == 2


def test_concurrent_first_requests_share_http_client(monkeypatch, observed_lock):
    llm = LLMClient()
    lock = observed_lock
    llm._client_lock = lock
    start = Barrier(4)
    constructing = Event()
    release = Event()
    client = Mock()

    def construct(**_kwargs):
        constructing.set()
        assert release.wait(timeout=5)
        return client

    factory = Mock(side_effect=construct)
    monkeypatch.setattr("scinexusrag.generation.llm.httpx.Client", factory)

    def load():
        start.wait(timeout=5)
        return llm.client

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(load) for _ in range(4)]
        try:
            assert constructing.wait(timeout=5)
            assert len({lock.attempts.get(timeout=5) for _ in range(4)}) == 4
        finally:
            release.set()
        assert all(future.result(timeout=5) is client for future in futures)

    assert factory.call_count == 1
    assert factory.call_args.kwargs["base_url"] == llm.base_url
    assert factory.call_args.kwargs["timeout"].read == llm.timeout
    llm.close()
    client.close.assert_called_once()


def test_failed_http_client_construction_can_retry(monkeypatch):
    llm = LLMClient()
    client = Mock()
    factory = Mock(side_effect=[RuntimeError("Client construction failed"), client])
    monkeypatch.setattr("scinexusrag.generation.llm.httpx.Client", factory)

    with pytest.raises(RuntimeError, match="Client construction failed"):
        _ = llm.client
    assert llm._client is None

    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(lambda: llm.client).result(timeout=5) is client
    assert llm.client is client
    assert factory.call_count == 2
    llm.close()


def test_close_waits_for_http_client_construction(monkeypatch, observed_lock):
    llm = LLMClient()
    lock = observed_lock
    llm._client_lock = lock
    constructing = Event()
    release = Event()
    original, fresh = Mock(), Mock()

    def construct(**_kwargs):
        constructing.set()
        assert release.wait(timeout=5)
        return original

    factory = Mock(side_effect=construct)
    monkeypatch.setattr("scinexusrag.generation.llm.httpx.Client", factory)

    with ThreadPoolExecutor(max_workers=2) as workers:
        loading = workers.submit(lambda: llm.client)
        try:
            assert constructing.wait(timeout=5)
            lock.attempts.get(timeout=5)
            closing = workers.submit(llm.close)
            lock.attempts.get(timeout=5)
            assert not closing.done()
        finally:
            release.set()
        assert loading.result(timeout=5) is original
        closing.result(timeout=5)

    original.close.assert_called_once()
    assert llm._client is None
    factory.side_effect = None
    factory.return_value = fresh
    assert llm.client is fresh
    llm.close()
    fresh.close.assert_called_once()


@pytest.mark.parametrize(
    "query", ["What is CVS?", "What is VSV infection?", "What is reasonable evidence?"]
)
def test_query_words_do_not_trigger_substring_instructions(query):
    assert QueryAnalyzer().classify(query) is QueryType.FACTUAL


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Compare the two methods.", QueryType.COMPARISON),
        ("A vs. B", QueryType.COMPARISON),
        ("A versus B", QueryType.COMPARISON),
        ("Why did the study stop?", QueryType.CAUSAL),
        ("Explain the reason for the change.", QueryType.CAUSAL),
        ("Explain the reasons for the change.", QueryType.CAUSAL),
    ],
)
def test_comparison_and_causal_words_keep_their_meaning(query, expected):
    assert QueryAnalyzer().classify(query) is expected


@pytest.mark.parametrize(
    "query", ["  TELL ME\nABOUT\tTHIS PAPER?  ", "tell me   about this paper."]
)
def test_vague_rewriting_normalizes_whitespace(query):
    analyzer = QueryAnalyzer()
    assert analyzer.rewrite_vague_query(query) == analyzer.rewrite_vague_query(
        "tell me about this paper"
    )


def test_specific_query_is_not_rewritten():
    query = "  What is the measured threshold?\n"
    assert QueryAnalyzer().rewrite_vague_query(query) == query


@pytest.mark.parametrize(
    "chunk_name,expected",
    [("chunk.pdf", "chunk.pdf"), (" \t", "Document 1"), ("", "Document 1")],
)
def test_prompt_source_names_fall_back_to_the_source_number(passage, chunk_name, expected):
    passage.chunk.document_name = chunk_name
    prepared = Synthesizer(Mock(spec=LLMClient)).prepare("What was measured?", [passage])
    assert prepared.error is None
    assert prepared.sources[0].document_name == expected
    assert f"[1] Source: {expected}" in prepared.prompt


def test_source_location_appears_in_the_prompt(passage):
    passage.chunk.page_number = 0
    passage.chunk.section_title = "Results"
    prepared = Synthesizer(Mock(spec=LLMClient)).prepare("What was measured?", [passage])
    assert prepared.sources[0].page_number == 0
    assert prepared.sources[0].section_title == "Results"
    assert "Section: Results, Page 0" in prepared.prompt


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("max_sources", -1, ValueError),
        ("max_sources", True, TypeError),
        ("max_sources", 1.5, TypeError),
        ("max_context_length", -1, ValueError),
        ("max_context_length", False, TypeError),
        ("max_tokens", 0, ValueError),
        ("max_tokens", 1.5, TypeError),
        ("temperature", True, TypeError),
        ("temperature", "0.1", TypeError),
        ("temperature", -0.1, ValueError),
        ("temperature", float("nan"), ValueError),
        ("temperature", float("inf"), ValueError),
        ("temperature", 10**1000, ValueError),
    ],
)
def test_invalid_synthesis_limits_fail_before_generation(name, value, error):
    llm = Mock(spec=LLMClient)
    with pytest.raises(error, match=name):
        Synthesizer(llm, **{name: value})
    llm.generate.assert_not_called()


@pytest.mark.parametrize("name", ["max_sources", "max_context_length"])
def test_zero_source_budget_keeps_no_fit_response(passage, name):
    llm = Mock(spec=LLMClient)
    result = Synthesizer(llm, **{name: 0}).synthesize("What was measured?", [passage])
    assert result.error == "No source passages fit the context limit."
    assert result.sources == []
    llm.generate.assert_not_called()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_source_scores_fail_before_generation(passage, score):
    llm = Mock(spec=LLMClient)
    passage.score = score
    with pytest.raises(ValueError, match="scores must be finite"):
        Synthesizer(llm).synthesize("What was measured?", [passage])
    llm.generate.assert_not_called()


@pytest.mark.parametrize("score", [sys.float_info.max, -sys.float_info.max, -0.25])
def test_finite_source_scores_are_preserved_without_overflow(passage, score):
    passage.score = score
    synth = Synthesizer(Mock(spec=LLMClient), max_context_length=10000)
    prepared = synth.prepare("What was measured?", [passage])
    assert prepared.error is None
    assert prepared.sources[0].score == score
    assert "relevance:" in prepared.prompt
    assert "inf%" not in prepared.prompt


def test_default_prompt_and_generation_options_are_preserved(passage):
    llm = Mock(spec=LLMClient)
    llm.generate.return_value = " Answer [1].\n"
    passage.chunk.document_name = "paper.pdf"
    passage.chunk.section_title = "Results"
    passage.chunk.page_number = 7
    response = Synthesizer(llm).synthesize("What was measured?", [passage])

    assert response.answer == " Answer [1].\n"
    assert llm.generate.call_args.args == (
        "QUESTION: What was measured?\n\n"
        "SOURCES:\n"
        "====================\n"
        "[1] Source: paper.pdf, Section: Results, Page 7 (relevance: 80%)\n"
        '"Measured evidence."\n'
        "====================\n\n"
        "Answer using only the sources above. Cite with [1], [2], etc.",
    )
    options = llm.generate.call_args.kwargs
    assert options["temperature"] == 0.1
    assert options["max_tokens"] == 356
    assert 'say: "This information is not found in the provided documents."' in options["system"]
    assert "Use ONLY information from the sources" in options["system"]


def _generation_result(idx: int, content: str, score: float = 0.9) -> RetrievalResult:
    chunk = Chunk(id=f"c{idx}", content=content, document_id=f"d{idx}")
    return RetrievalResult(chunk=chunk, score=score)


class StubLLM:
    """Fixed output."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        return self.answer


class StubRetriever:
    """Fixed retrieval."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.results


class StubGrounding:
    """Fixed grounding."""

    def __init__(self, faithfulness: float, unsupported: list[str]) -> None:
        self.report = GroundingReport(
            faithfulness=faithfulness, sentences=[], unsupported=unsupported
        )

    def verify(self, answer: str, sources: list[Any]) -> GroundingReport:
        return self.report


@pytest.fixture
def integer_string_limit():
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(previous)


class TestCitationValidation:
    def test_out_of_range_stripped_and_flagged(self) -> None:
        sources = [Source(index=1, chunk_id="c1", document_id="d1", content="a")]
        result = AnswerVerifier().verify("Claim one [1] and bogus [5].", sources)

        assert result.citations_valid == [1]
        assert "[5]" not in result.verified_answer
        assert "[1]" in result.verified_answer
        assert any("out-of-range" in w for w in result.warnings)

    def test_no_citations_warns(self) -> None:
        result = AnswerVerifier().verify("Plain answer.", [])
        assert result.citations_valid == []
        assert any("no sources" in w for w in result.warnings)

    def test_citations_match_actual_source_indices(self) -> None:
        sources = [
            Source(index=2, chunk_id="c2", document_id="d2", content="alpha"),
            Source(index=4, chunk_id="c4", document_id="d4", content="beta"),
        ]

        result = AnswerVerifier().verify("Claim [1] [2] [4] [5].", sources)

        assert result.citations_valid == [2, 4]
        assert "removed 2 out-of-range citation(s): [1, 5]" in result.warnings
        assert "[2]" in result.verified_answer and "[4]" in result.verified_answer
        assert "[1]" not in result.verified_answer and "[5]" not in result.verified_answer

    @pytest.mark.usefixtures("integer_string_limit")
    def test_unparseable_citations_are_removed_without_inventing_indices(self) -> None:
        citation = "[" + "9" * 5000 + "]"
        sources = [Source(index=1, chunk_id="c1", document_id="d1", content="evidence")]

        result = AnswerVerifier().verify(
            f"Claim {citation} and {citation}. Known [1], invalid [9].", sources
        )

        assert result.verified_answer == "Claim and. Known [1], invalid."
        assert result.citations_valid == [1]
        assert "removed 2 unparseable numeric citation(s)" in result.warnings
        assert "removed 1 out-of-range citation(s): [9]" in result.warnings


class TestSynthesizerFormatting:
    def test_format_sources_numbered(self) -> None:
        synth = Synthesizer(StubLLM("unused"))  # type: ignore[arg-type]
        sources = [
            Source(index=1, chunk_id="c1", document_id="d1", content="alpha", score=0.8),
            Source(index=2, chunk_id="c2", document_id="d2", content="beta", score=0.6),
        ]
        out = synth._format_sources_for_llm(sources)

        assert "[1]" in out and "[2]" in out
        assert "alpha" in out and "beta" in out
        assert "relevance: 80%" in out

    def test_synthesize_uses_stub_llm(self) -> None:
        stub = StubLLM("Answer with citation [1].")
        synth = Synthesizer(stub)  # type: ignore[arg-type]
        out = synth.synthesize("q?", [_generation_result(1, "evidence")])

        assert stub.calls == 1
        assert out.answer == "Answer with citation [1]."
        assert len(out.sources) == 1
        assert out.sources[0].index == 1

    def test_token_budget_capped_by_config(self) -> None:
        five = [_generation_result(i, f"s{i}") for i in range(1, 6)]
        sources = Synthesizer(StubLLM("x"))._build_sources(five)  # type: ignore[arg-type]

        assert Synthesizer(StubLLM("x"), max_tokens=2048)._token_budget(sources) == 756  # type: ignore[arg-type]

        assert Synthesizer(StubLLM("x"), max_tokens=300)._token_budget(sources) == 300  # type: ignore[arg-type]

    @pytest.mark.parametrize("shortfall, expected_count", [(0, 2), (1, 1)])
    def test_context_budget_keeps_only_complete_blocks(
        self, shortfall: int, expected_count: int
    ) -> None:
        synth = Synthesizer(Mock(spec=LLMClient))
        results = [_generation_result(i, f"Passage {i}.") for i in range(1, 4)]
        results[1].chunk.context_before = "Nearby evidence."
        sources = synth._build_sources(results)
        synth.max_context_length = len(synth._format_sources_for_llm(sources[:2])) - shortfall

        prepared = synth.prepare("What did the study find?", results)

        assert prepared.error is None
        assert prepared.sources == sources[:expected_count]
        for source in prepared.sources:
            assert synth._format_sources_for_llm([source]) in prepared.prompt
        assert results[expected_count].chunk.content not in prepared.prompt
        assert "[truncated]" not in prepared.prompt

    def test_source_limit_applies_after_deduplication(self) -> None:
        first = _generation_result(1, "First passage.")
        synth = Synthesizer(Mock(spec=LLMClient), max_sources=2)

        prepared = synth.prepare(
            "What did the study find?",
            [
                first,
                first,
                _generation_result(2, "Second passage."),
                _generation_result(3, "Omitted passage."),
            ],
        )

        assert [source.chunk_id for source in prepared.sources] == ["c1", "c2"]
        assert [source.index for source in prepared.sources] == [1, 2]
        assert prepared.prompt.count("First passage.") == 1
        assert "Omitted passage." not in prepared.prompt

    def test_synthesis_keeps_sources_that_fit_and_scales_the_token_budget(self) -> None:
        results = [_generation_result(i, f"Passage {i}: " + "evidence " * 145) for i in range(1, 6)]
        llm = Mock(spec=LLMClient)
        llm.generate.return_value = "Answer [2]."
        synth = Synthesizer(llm)

        response = synth.synthesize("What did the study find?", results)

        assert response.answer == "Answer [2]."
        assert [source.chunk_id for source in response.sources] == ["c1", "c2", "c3", "c4"]
        assert llm.generate.call_args.kwargs["max_tokens"] == 656
        prompt = llm.generate.call_args.args[0]
        assert results[4].chunk.content not in prompt
        assert all(source.content in prompt for source in response.sources)


class TestOrchestratorWiring:
    @pytest.mark.usefixtures("integer_string_limit")
    @pytest.mark.parametrize(
        ("digits", "valid", "warning"),
        [
            ("9" * 5000, False, "removed 1 unparseable numeric citation(s)"),
            ("0" * 5000 + "1", True, None),
            ("٠" * 5000 + "١", True, None),
            ("0" * 5000, False, "removed 1 out-of-range citation(s): [0]"),
            ("0" * 5000 + "99", False, "removed 1 out-of-range citation(s): [99]"),
        ],
        ids=["unparseable", "leading-zeros", "unicode-leading-zeros", "zero", "invalid-index"],
    )
    def test_long_citations_do_not_interrupt_the_final_answer(
        self, digits: str, valid: bool, warning: str | None
    ) -> None:
        citation = f"[{digits}]"
        raw_answer = f"Claim {citation}. Known [1]."
        expected = raw_answer if valid else "Claim. Known [1]."

        def reply(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": raw_answer})

        with LLMClient() as llm:
            llm._client = httpx.Client(base_url=llm.base_url, transport=httpx.MockTransport(reply))
            orch = Orchestrator(StubRetriever([_generation_result(1, "evidence")]), llm)  # type: ignore[arg-type]
            response = orch.query("What was measured?")

        assert [source.index for source in response.sources] == [1]
        assert response.answer == expected
        assert response.warnings == ([] if warning is None else [warning])

    def test_returns_rag_response(self) -> None:
        retriever = StubRetriever([_generation_result(1, "the sky is blue")])
        llm = StubLLM("The sky is blue [1].")
        orch = Orchestrator(retriever, llm)  # type: ignore[arg-type]

        resp = orch.query("why is the sky blue?")

        assert isinstance(resp, RAGResponse)
        assert resp.sources
        assert 0.0 <= resp.confidence <= 1.0
        assert "[1]" in resp.answer

    def test_invalid_citations_handled(self) -> None:

        retriever = StubRetriever([_generation_result(1, "only source")])
        llm = StubLLM("Grounded [1] but invalid [9].")
        orch = Orchestrator(retriever, llm)  # type: ignore[arg-type]

        resp = orch.query("q?")

        assert "[9]" not in resp.answer
        assert "[1]" in resp.answer
        assert any("out-of-range" in w for w in resp.warnings)

    def test_grounding_verifier_sets_faithfulness(self) -> None:
        retriever = StubRetriever([_generation_result(1, "source text")])
        llm = StubLLM("Answer [1].")
        grounding = StubGrounding(faithfulness=0.5, unsupported=["Answer [1]."])
        orch = Orchestrator(retriever, llm, grounding_verifier=grounding)  # type: ignore[arg-type]

        resp = orch.query("q?")

        assert resp.confidence == 0.5
        assert any("not grounded" in w for w in resp.warnings)

    @pytest.mark.parametrize("grounding_enabled", [False, True])
    def test_failed_generation_keeps_sources_without_scoring_error_text(
        self, grounding_enabled: bool
    ) -> None:
        retriever = StubRetriever([_generation_result(1, "Retrieved evidence", score=1.0)])
        llm = Mock(spec=StubLLM)
        llm.generate.side_effect = LLMError("model unavailable")
        grounding = Mock(spec=StubGrounding)
        grounding.verify.return_value = GroundingReport(faithfulness=0.8, sentences=[])
        orch = Orchestrator(  # type: ignore[arg-type]
            retriever, llm, grounding_verifier=grounding if grounding_enabled else None
        )

        response = orch.query("What did the study find?")

        assert response.answer == "Unable to generate response. Please try again."
        assert response.confidence == 0.0
        assert response.sources[0].content == "Retrieved evidence"
        assert response.warnings == ["Answer generation failed."]
        grounding.verify.assert_not_called()

    def test_unexpected_generation_errors_are_not_reported_as_backend_failures(self) -> None:
        retriever = StubRetriever([_generation_result(1, "Retrieved evidence", score=1.0)])
        llm = Mock(spec=StubLLM)
        llm.generate.side_effect = KeyError("prompt")

        with pytest.raises(KeyError):
            Orchestrator(retriever, llm).query("What did the study find?")  # type: ignore[arg-type]

    def test_empty_retrieval_returns_an_error_answer_without_answer_checks(self) -> None:
        llm = Mock(spec=LLMClient)
        grounding = Mock(spec=StubGrounding)
        orch = Orchestrator(StubRetriever([]), llm, grounding_verifier=grounding)  # type: ignore[arg-type]
        orch.verifier = Mock(spec=AnswerVerifier)
        error = "No relevant documents found. Please upload documents and try again."

        response = orch.query("What did the study find?")

        assert response.answer == error
        assert response.sources == []
        assert response.warnings == [error]
        llm.generate.assert_not_called()
        orch.verifier.verify.assert_not_called()
        grounding.verify.assert_not_called()

    def test_omitted_sources_cannot_be_cited_or_used_for_grounding(self) -> None:
        results = [_generation_result(i, f"Passage {i}: " + "evidence " * 145) for i in range(1, 6)]
        llm = Mock(spec=LLMClient)
        llm.generate.return_value = "Visible claim [1]. Omitted claim [5]."
        grounding = Mock(spec=StubGrounding)
        grounding.verify.return_value = GroundingReport(faithfulness=1.0, sentences=[])
        orch = Orchestrator(StubRetriever(results), llm, grounding_verifier=grounding)  # type: ignore[arg-type]

        response = orch.query("What did the study find?")

        assert [source.index for source in response.sources] == [1, 2, 3, 4]
        assert "[1]" in response.answer and "[5]" not in response.answer
        assert any("out-of-range" in warning for warning in response.warnings)
        assert grounding.verify.call_args.args == (response.answer, response.sources)
        prompt = llm.generate.call_args.args[0]
        assert all((source.full_content or source.content) in prompt for source in response.sources)
        assert results[4].chunk.content not in prompt

    def test_oversized_context_stops_before_generation_and_verification(self) -> None:
        first = _generation_result(1, "Short main passage.")
        first.chunk.context_before = "Nearby context. " * 500
        llm = Mock(spec=LLMClient)
        grounding = Mock(spec=StubGrounding)
        orch = Orchestrator(  # type: ignore[arg-type]
            StubRetriever([first, _generation_result(2, "A later passage that would fit.")]),
            llm,
            grounding_verifier=grounding,
        )
        error = "No source passages fit the context limit."

        response = orch.query("What did the study find?")

        assert response.answer == error
        assert response.sources == []
        assert response.confidence == 0.0
        assert response.warnings == [error]
        llm.generate.assert_not_called()
        grounding.verify.assert_not_called()


def _client_with(behaviors: list[httpx.Response | Exception]) -> tuple[LLMClient, Mock]:
    client = LLMClient(max_retries=2, backoff=0.0)
    for behavior in behaviors:
        if isinstance(behavior, httpx.Response):
            behavior.request = httpx.Request("POST", f"{client.base_url}/api/generate")
    fake = Mock(spec=httpx.Client)
    fake.post.side_effect = behaviors
    client._client = fake
    return client, fake


class TestLLMRetry:
    def test_retries_then_succeeds(self) -> None:
        client, fake = _client_with(
            [httpx.ConnectError("down"), httpx.Response(200, json={"response": "recovered"})]
        )
        assert client.generate("hi") == "recovered"
        assert fake.post.call_count == 2

    def test_raises_llmerror_after_exhaustion(self) -> None:
        client, fake = _client_with([httpx.ConnectError("down") for _ in range(3)])
        with pytest.raises(LLMError) as exc:
            client.generate("hi")

        assert "base_url" in str(exc.value)
        assert fake.post.call_count == 3

    def test_does_not_retry_on_4xx(self) -> None:
        client, fake = _client_with([httpx.Response(404)])
        with pytest.raises(LLMError, match="HTTP 404"):
            client.generate("hi")

        assert fake.post.call_count == 1

    def test_retries_on_5xx(self) -> None:
        client, fake = _client_with(
            [httpx.Response(503), httpx.Response(200, json={"response": "ok"})]
        )
        assert client.generate("hi") == "ok"
        assert fake.post.call_count == 2

    def test_missing_response_field_raises_llmerror(self) -> None:
        client, fake = _client_with([httpx.Response(200, json={"done": True})])
        with pytest.raises(LLMError, match="response"):
            client.generate("hi")
        assert fake.post.call_count == 1


@contextmanager
def _llm_response(response: httpx.Response, **options):
    with LLMClient(**options) as client:
        client._client = httpx.Client(
            base_url=client.base_url, transport=httpx.MockTransport(lambda request: response)
        )
        yield client


class TestLLMResponseValidation:
    @pytest.mark.parametrize("answer", ["", " \n\t"])
    def test_generate_rejects_empty_answers(self, answer: str) -> None:
        with _llm_response(httpx.Response(200, json={"response": answer})) as client:
            with pytest.raises(LLMError, match="empty response"):
                client.generate("question")

    @pytest.mark.parametrize("payload", [[], None, {"response": None}, {"response": 42}])
    def test_generate_rejects_invalid_response_fields(self, payload: Any) -> None:
        with _llm_response(httpx.Response(200, text=json.dumps(payload))) as client:
            with pytest.raises(LLMError, match="string 'response' field"):
                client.generate("question")

    @pytest.mark.parametrize("payload", [[], None])
    def test_readiness_rejects_non_object_json(self, payload: Any) -> None:
        with _llm_response(httpx.Response(200, text=json.dumps(payload))) as client:
            assert client.is_available() is False


@pytest.mark.parametrize(
    ("configured", "installed", "available"),
    [
        ("llama3.2:3b", "llama3.2:3b", True),
        ("llama3.2:3b", "llama3.2:3b-instruct", False),
        ("llama3.2", "llama3.2:3b", False),
        ("llama3.2", "llama3.2:latest", True),
        ("localhost:5000/model", "localhost:5000/model:latest", True),
    ],
)
def test_model_readiness_matches_exact_tag(configured, installed, available):
    with _llm_response(
        httpx.Response(200, json={"models": [{"name": installed}]}), model=configured
    ) as client:
        assert client.is_available() is available


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"response": "Partial answer", "done": False}, "before completion"),
        ({"response": "Partial answer", "done": "true"}, "must be a boolean"),
        ({"response": "Partial answer", "done": 1}, "must be a boolean"),
        (
            {"response": "Partial answer", "error": "model was unloaded"},
            "backend reported an error",
        ),
    ],
)
def test_generate_rejects_incomplete_or_failed_responses(payload, message):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    with LLMClient() as llm:
        llm._client = httpx.Client(base_url=llm.base_url, transport=httpx.MockTransport(respond))
        with pytest.raises(LLMError, match=message):
            llm.generate("Question")

    assert len(requests) == 1


@pytest.mark.parametrize("done", [True, None])
def test_generate_keeps_complete_response_and_legacy_metadata_contract(done):
    payload = {"response": "Answer.\n"}
    if done is not None:
        payload["done"] = done
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with LLMClient() as llm:
        llm._client = httpx.Client(base_url=llm.base_url, transport=transport)
        assert llm.generate("Question") == "Answer.\n"


def test_generate_wraps_invalid_unicode_body_as_backend_error():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"\xff"))
    with LLMClient() as llm:
        llm._client = httpx.Client(base_url=llm.base_url, transport=transport)
        with pytest.raises(LLMError, match="non-JSON body"):
            llm.generate("Question")


def test_model_readiness_rejects_invalid_unicode_body():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"\xff"))
    with LLMClient() as llm:
        llm._client = httpx.Client(base_url=llm.base_url, transport=transport)
        assert llm.is_available() is False


@pytest.mark.parametrize(("configured", "probe"), [(60.0, 3.0), (1.5, 1.5)])
def test_model_readiness_probe_does_not_wait_for_the_generation_timeout(configured, probe):
    timeouts = []

    def respond(request):
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"models": []})

    with LLMClient(timeout=configured) as llm:
        llm._client = httpx.Client(
            base_url=llm.base_url, transport=httpx.MockTransport(respond), timeout=configured
        )
        llm.is_available()

    assert timeouts == [dict.fromkeys(("connect", "read", "write", "pool"), probe)]


def test_generate_sends_generation_options():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "Answer.", "done": True})

    with LLMClient(model="custom-model") as llm:
        llm._client = httpx.Client(base_url=llm.base_url, transport=httpx.MockTransport(respond))
        assert (
            llm.generate("Question", system="Instructions", temperature=0.2, max_tokens=17)
            == "Answer."
        )

    assert requests == [
        {
            "model": "custom-model",
            "prompt": "Question",
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 17, "top_p": 0.9},
            "system": "Instructions",
        }
    ]


def test_generate_preserves_stop_sequences():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "Answer.", "done": True})

    with LLMClient() as llm:
        llm._client = httpx.Client(base_url=llm.base_url, transport=httpx.MockTransport(respond))
        assert llm.generate("Question", stop=["END", "STOP"]) == "Answer."

    assert requests[0]["options"]["stop"] == ["END", "STOP"]


@pytest.mark.parametrize(
    "options",
    [
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"max_retries": True},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": True},
        {"timeout": False},
        {"timeout": 10**1000},
        {"backoff": -0.1},
        {"backoff": float("nan")},
        {"backoff": float("inf")},
        {"backoff": True},
        {"backoff": False},
        {"backoff": 10**1000},
    ],
)
def test_invalid_client_configuration_fails_before_http_creation(monkeypatch, options):
    factory = Mock()
    monkeypatch.setattr("scinexusrag.generation.llm.httpx.Client", factory)

    with pytest.raises(ValueError):
        LLMClient(**options)

    factory.assert_not_called()


class _FailingCloseTransport(httpx.MockTransport):
    def close(self):
        raise OSError("Transport cleanup failed")


def test_close_failure_does_not_reuse_closed_http_client(monkeypatch):
    with LLMClient() as llm:
        original = httpx.Client(
            base_url=llm.base_url,
            transport=_FailingCloseTransport(lambda request: httpx.Response(200)),
        )
        llm._client = original
        with pytest.raises(OSError, match="Transport cleanup failed"):
            llm.close()

        assert original.is_closed
        assert llm._client is None
        replacement = httpx.Client(
            base_url=llm.base_url,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"response": "Recovered.", "done": True})
            ),
        )
        factory = Mock(return_value=replacement)
        monkeypatch.setattr("scinexusrag.generation.llm.httpx.Client", factory)

        assert llm.generate("Question") == "Recovered."
        factory.assert_called_once()

    assert replacement.is_closed


def test_finalizer_does_not_report_transport_cleanup_errors(monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "unraisablehook", errors.append)
    llm = LLMClient()
    llm._client = httpx.Client(
        base_url=llm.base_url,
        transport=_FailingCloseTransport(lambda request: httpx.Response(200)),
    )

    del llm
    gc.collect()

    assert errors == []


@pytest.fixture
def private_backend():
    secrets = (
        "fixture-password",
        "fixture-path-secret",
        "fixture-query-secret",
        "fixture-fragment",
    )
    password, path, query, fragment = secrets
    base_url = f"http://fixture-user:{password}@localhost:11434/{path}?token={query}#{fragment}"
    return base_url, (*secrets, "fixture-user")


def _failed_backend(base_url, failure):
    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout(base_url, request=request)
        if failure == "connect":
            raise httpx.ConnectError(base_url, request=request)
        if failure.startswith("http_"):
            return httpx.Response(int(failure[5:]), text=base_url)
        if failure == "unicode":
            return httpx.Response(200, content=base_url.encode() + b"\xff")
        if failure == "json":
            return httpx.Response(200, text=base_url)
        return httpx.Response(200, json={"error": base_url})

    client = LLMClient(base_url=base_url, max_retries=0)
    client._client = httpx.Client(base_url=base_url, transport=httpx.MockTransport(respond))
    return client


@pytest.mark.parametrize(
    "failure", ["timeout", "connect", "http_400", "http_503", "unicode", "json", "backend"]
)
def test_backend_errors_and_tracebacks_omit_private_url_components(private_backend, failure):
    base_url, secrets = private_backend
    with _failed_backend(base_url, failure) as client, pytest.raises(LLMError) as raised:
        client.generate("Question")
    error = raised.value
    diagnostic = str(error) + repr(error) + "".join(traceback.format_exception(error))
    assert "base_url=http://localhost:11434" in diagnostic
    assert all(secret not in diagnostic for secret in secrets)
    assert error.__cause__ is None
    if failure.startswith("http_"):
        assert f"HTTP {failure[5:]}" in str(error)


@pytest.mark.parametrize(
    "failure", ["timeout", "connect", "http_400", "http_503", "unicode", "json", "backend"]
)
def test_generation_logs_omit_private_backend_url(private_backend, failure, caplog):
    base_url, secrets = private_backend
    results = [RetrievalResult(Chunk(id="passage", document_id="paper", content="A finding."), 0.8)]
    with _failed_backend(base_url, failure) as client, caplog.at_level(logging.ERROR):
        result = Synthesizer(client).synthesize("Question", results)
        assert result.error == "Answer generation failed."
    assert caplog.records
    assert "base_url=http://localhost:11434" in caplog.text
    assert all(secret not in caplog.text for secret in secrets)


@pytest.fixture
def orchestrator():
    results = [
        RetrievalResult(
            Chunk(id="one", content="First evidence.", document_id="doc-one"), score=0.8
        ),
        RetrievalResult(
            Chunk(id="two", content="Second evidence.", document_id="doc-two"), score=0.8
        ),
    ]
    retriever = Mock(spec=CorrectiveRetriever)
    retriever.retrieve.return_value = results
    llm = Mock(spec=LLMClient)
    llm.generate.return_value = "A claim [1]."
    return Orchestrator(retriever, llm)


@pytest.mark.parametrize("stage", ["citations", "grounding"])
def test_verification_errors_propagate(orchestrator, stage):
    if stage == "grounding":
        orchestrator.grounding_verifier = Mock(spec=GroundingVerifier)
        verifier = orchestrator.grounding_verifier
    else:
        orchestrator.verifier = Mock(spec=AnswerVerifier)
        verifier = orchestrator.verifier
    verifier.verify.side_effect = RuntimeError("Verification unavailable")

    with pytest.raises(RuntimeError, match="Verification unavailable"):
        orchestrator.query("What was measured?")


@pytest.mark.parametrize(
    "scores,cited,expected",
    [
        ([-2.0, -1.0], False, 0.0),
        ([-2.0, -1.0], True, 0.25),
        ([0.4, 0.8], False, 0.3),
        ([0.4, 0.8], True, 0.55),
        ([2.0, 4.0], False, 0.5),
        ([2.0, 4.0], True, 0.75),
    ],
)
def test_answer_confidence_stays_bounded_for_finite_ranking_scores(
    orchestrator, scores, cited, expected
):
    results = orchestrator.retriever.retrieve.return_value
    for result, score in zip(results, scores, strict=True):
        result.score = score
    orchestrator.llm.generate.return_value = "A claim [1]." if cited else "A claim."

    response = orchestrator.query("What was measured?")

    assert response.confidence == expected
    assert 0 <= response.confidence <= 1


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "5", None])
def test_invalid_retrieval_limit_fails_before_models(orchestrator, top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        Orchestrator(orchestrator.retriever, orchestrator.llm, top_k=top_k)

    orchestrator.retriever.retrieve.assert_not_called()
    orchestrator.llm.generate.assert_not_called()


@pytest.mark.parametrize("question", ["", "  ", "\n\t", None, 1, b"Question"])
def test_invalid_question_fails_before_retrieval(orchestrator, question):
    error = ValueError if isinstance(question, str) else TypeError
    with pytest.raises(error, match="Question"):
        orchestrator.query(question)

    orchestrator.retriever.retrieve.assert_not_called()
    orchestrator.llm.generate.assert_not_called()


def test_question_whitespace_is_trimmed_before_retrieval(orchestrator):
    orchestrator.query("  What was measured? \n")

    orchestrator.retriever.retrieve.assert_called_once_with("What was measured?", top_k=5)


@pytest.mark.parametrize(
    "answer",
    [
        "    value =  42\n    print(value)\n\nClaim [1].\n",
        "\n\n```python\n    value =  42\n```\n\nClaim [1].\n",
        "  Claim [1].\n\n\n    Supporting detail.\n",
    ],
)
def test_complete_answers_preserve_their_layout(orchestrator, answer):
    orchestrator.llm.generate.return_value = answer

    assert orchestrator.query("What was measured?").answer == answer


def test_short_numeric_claims_are_each_checked_for_grounding(monkeypatch):
    verifier = GroundingVerifier()
    scores = Mock(side_effect=[[1.0], [0.0]])
    monkeypatch.setattr(verifier, "entailment_scores", scores)

    report = verifier.verify("42. 5.", ["The measured value is 42."])

    assert [item.sentence for item in report.sentences] == ["42.", "5."]
    assert report.faithfulness == 0.5
    assert report.unsupported == ["5."]
    assert scores.call_args_list == [
        call([("The measured value is 42.", "42.")]),
        call([("The measured value is 42.", "5.")]),
    ]
    assert verifier._model is None
