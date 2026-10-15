"Hermetic unit tests for the generation subsystem."

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from scinexusrag.generation.grounding import GroundingReport
from scinexusrag.generation.llm import LLMClient, LLMError
from scinexusrag.generation.orchestrator import Orchestrator, RAGResponse
from scinexusrag.generation.synthesizer import Source, Synthesizer
from scinexusrag.generation.verifier import AnswerVerifier
from scinexusrag.ingestion.chunker import Chunk
from scinexusrag.retrieval import RetrievalResult


def _result(idx: int, content: str, score: float = 0.9) -> RetrievalResult:
    chunk = Chunk(id=f"c{idx}", content=content, document_id=f"d{idx}")
    return RetrievalResult(chunk=chunk, score=score)


def _consume_stream(chunks: list[str]) -> tuple[str, list[dict[str, Any]]]:
    answer = ""
    events = []
    for chunk in chunks:
        if chunk.startswith("\ndata: "):
            event = json.loads(chunk.removeprefix("\ndata: "))
            events.append(event)
            if event["event"] == "verified_answer":
                answer = event["answer"]
        else:
            answer += chunk
    return answer, events


class StubLLM:
    """Stand-in for LLMClient that returns a canned answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        return self.answer


class StubRetriever:
    """Stand-in for CorrectiveRetriever."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.results

    def retrieve_traced(self, query: str, top_k: int = 5) -> tuple[list[RetrievalResult], bool]:
        return self.results, False


class StubGrounding:
    """Stand-in for GroundingVerifier with a fixed report."""

    def __init__(self, faithfulness: float, unsupported: list[str]) -> None:
        self.report = GroundingReport(
            faithfulness=faithfulness, sentences=[], unsupported=unsupported
        )

    def verify(self, answer: str, sources: list[Any]) -> GroundingReport:
        return self.report


class TestCitationValidation:
    def test_out_of_range_stripped_and_flagged(self) -> None:
        sources = [Source(index=1, chunk_id="c1", document_id="d1", content="a")]
        result = AnswerVerifier().verify("Claim one [1] and bogus [5].", sources)

        assert result.citations_valid == [1]
        assert result.citations_invalid == [5]
        assert "[5]" not in result.verified_answer
        assert "[1]" in result.verified_answer
        assert any("out-of-range" in w for w in result.warnings)

    def test_no_citations_warns(self) -> None:
        result = AnswerVerifier().verify("Plain answer.", [])
        assert result.citations_found == []
        assert any("no sources" in w for w in result.warnings)

    def test_citations_match_actual_source_indices(self) -> None:
        sources = [
            Source(index=2, chunk_id="c2", document_id="d2", content="alpha"),
            Source(index=4, chunk_id="c4", document_id="d4", content="beta"),
        ]

        result = AnswerVerifier().verify("Claim [1] [2] [4] [5].", sources)

        assert result.citations_valid == [2, 4]
        assert result.citations_invalid == [1, 5]
        assert "[2]" in result.verified_answer and "[4]" in result.verified_answer
        assert "[1]" not in result.verified_answer and "[5]" not in result.verified_answer

    def test_unparseable_citations_are_removed_without_inventing_indices(self) -> None:
        citation = "[" + "9" * 5000 + "]"
        sources = [Source(index=1, chunk_id="c1", document_id="d1", content="evidence")]

        result = AnswerVerifier().verify(
            f"Claim {citation} and {citation}. Known [1], invalid [9].", sources
        )

        assert result.verified_answer == "Claim and . Known [1], invalid ."
        assert result.citations_found == [1, 9]
        assert result.citations_valid == [1]
        assert result.citations_invalid == [9]
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
        out = synth.synthesize("q?", [_result(1, "evidence")])

        assert stub.calls == 1
        assert out.answer == "Answer with citation [1]."
        assert len(out.sources) == 1
        assert out.sources[0].index == 1

    def test_token_budget_capped_by_config(self) -> None:
        five = [_result(i, f"s{i}") for i in range(1, 6)]
        sources = Synthesizer(StubLLM("x"))._build_sources(five, {})  # type: ignore[arg-type]

        assert Synthesizer(StubLLM("x"), max_tokens=2048)._token_budget(sources) == 756  # type: ignore[arg-type]

        assert Synthesizer(StubLLM("x"), max_tokens=300)._token_budget(sources) == 300  # type: ignore[arg-type]

    @pytest.mark.parametrize("shortfall, expected_count", [(0, 2), (1, 1)])
    def test_context_budget_keeps_only_complete_blocks(
        self, shortfall: int, expected_count: int
    ) -> None:
        synth = Synthesizer(Mock(spec=LLMClient))
        results = [_result(i, f"Passage {i}.") for i in range(1, 4)]
        results[1].chunk.context_before = "Nearby evidence."
        sources = synth._build_sources(results, {})
        synth.max_context_length = len(synth._format_sources_for_llm(sources[:2])) - shortfall

        prepared = synth.prepare("What did the study find?", results)

        assert prepared.error is None
        assert prepared.sources == sources[:expected_count]
        for source in prepared.sources:
            assert synth._format_sources_for_llm([source]) in prepared.prompt
        assert results[expected_count].chunk.content not in prepared.prompt
        assert "[truncated]" not in prepared.prompt

    def test_source_limit_applies_after_deduplication(self) -> None:
        first = _result(1, "First passage.")
        synth = Synthesizer(Mock(spec=LLMClient), max_sources=2)

        prepared = synth.prepare(
            "What did the study find?",
            [first, first, _result(2, "Second passage."), _result(3, "Omitted passage.")],
        )

        assert [source.chunk_id for source in prepared.sources] == ["c1", "c2"]
        assert [source.index for source in prepared.sources] == [1, 2]
        assert prepared.prompt.count("First passage.") == 1
        assert "Omitted passage." not in prepared.prompt

    def test_regular_and_streaming_use_same_retained_sources_and_token_budget(self) -> None:
        results = [_result(i, f"Passage {i}: " + "evidence " * 145) for i in range(1, 6)]
        llm = Mock(spec=LLMClient)
        llm.generate.return_value = "Answer [2]."
        llm.stream.return_value = iter(["Answer ", "[2]."])
        synth = Synthesizer(llm)

        response = synth.synthesize("What did the study find?", results)
        streamed = "".join(synth.synthesize_streaming("What did the study find?", results))

        assert streamed == response.answer
        assert [source.chunk_id for source in response.sources] == ["c1", "c2", "c3", "c4"]
        assert llm.generate.call_args == llm.stream.call_args
        assert llm.generate.call_args.kwargs["max_tokens"] == 656
        prompt = llm.generate.call_args.args[0]
        assert results[4].chunk.content not in prompt
        assert all(source.content in prompt for source in response.sources)


class TestOrchestratorWiring:
    @pytest.mark.parametrize("streaming", [False, True])
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
        self, streaming: bool, digits: str, valid: bool, warning: str | None
    ) -> None:
        citation = f"[{digits}]"
        raw_answer = f"Claim {citation}. Known [1]."
        expected = raw_answer if valid else "Claim . Known [1]."

        def reply(request: httpx.Request) -> httpx.Response:
            if json.loads(request.content)["stream"]:
                records = [{"response": raw_answer}, {"done": True}]
                return httpx.Response(200, text="\n".join(json.dumps(row) for row in records))
            return httpx.Response(200, json={"response": raw_answer})

        with LLMClient() as llm:
            llm._client = httpx.Client(base_url=llm.base_url, transport=httpx.MockTransport(reply))
            orch = Orchestrator(StubRetriever([_result(1, "evidence")]), llm)  # type: ignore[arg-type]

            if streaming:
                chunks = list(orch.query_streaming("What was measured?"))
                answer, events = _consume_stream(chunks)
                assert chunks[:-2] == [raw_answer]
                assert events[-1] == {"event": "verified_answer", "answer": expected}
                assert events[0]["citations_valid"] == ([1, 1] if valid else [1])
                warnings = events[0]["warnings"]
            else:
                response = orch.query("What was measured?")
                answer, warnings = response.answer, response.warnings
                assert [source.index for source in response.sources] == [1]

        assert answer == expected
        assert warnings == ([] if warning is None else [warning])

    def test_returns_rag_response(self) -> None:
        retriever = StubRetriever([_result(1, "the sky is blue")])
        llm = StubLLM("The sky is blue [1].")
        orch = Orchestrator(retriever, llm)  # type: ignore[arg-type]

        resp = orch.query("why is the sky blue?")

        assert isinstance(resp, RAGResponse)
        assert resp.sources
        assert 0.0 <= resp.confidence <= 1.0
        assert "[1]" in resp.answer

    def test_invalid_citations_handled(self) -> None:

        retriever = StubRetriever([_result(1, "only source")])
        llm = StubLLM("Grounded [1] but invalid [9].")
        orch = Orchestrator(retriever, llm)  # type: ignore[arg-type]

        resp = orch.query("q?")

        assert "[9]" not in resp.answer
        assert "[1]" in resp.answer
        assert any("out-of-range" in w for w in resp.warnings)

    def test_grounding_verifier_sets_faithfulness(self) -> None:
        retriever = StubRetriever([_result(1, "source text")])
        llm = StubLLM("Answer [1].")
        grounding = StubGrounding(faithfulness=0.5, unsupported=["Answer [1]."])
        orch = Orchestrator(retriever, llm, grounding_verifier=grounding)  # type: ignore[arg-type]

        resp = orch.query("q?")

        assert resp.faithfulness == 0.5
        assert resp.confidence == 0.5
        assert any("not grounded" in w for w in resp.warnings)

    @pytest.mark.parametrize("grounding_enabled", [False, True])
    def test_failed_generation_keeps_sources_without_scoring_error_text(
        self, grounding_enabled: bool
    ) -> None:
        retriever = StubRetriever([_result(1, "Retrieved evidence", score=1.0)])
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
        assert response.faithfulness is None
        assert response.sources[0].content == "Retrieved evidence"
        assert response.warnings == ["Answer generation failed."]
        grounding.verify.assert_not_called()

    @pytest.mark.parametrize(
        ("tokens", "expected_answer", "expected_citations"),
        [
            (["Fact ", "[1]."], "Fact [1].", [1]),
            (["Claim ", "[99]."], "Claim .", []),
            (["Fact ", "[1] and [9]."], "Fact [1] and .", [1]),
        ],
    )
    def test_streaming_replaces_provisional_text_with_verified_answer(
        self, tokens: list[str], expected_answer: str, expected_citations: list[int]
    ) -> None:
        retriever = StubRetriever([_result(1, "fact")])
        llm = Mock(spec=LLMClient)
        llm.stream.return_value = iter(tokens)
        orch = Orchestrator(retriever, llm)  # type: ignore[arg-type]

        chunks = list(orch.query_streaming("q?"))
        answer, events = _consume_stream(chunks)

        assert chunks[:-2] == tokens
        assert [event["event"] for event in events] == ["verification", "verified_answer"]
        assert events[0]["citations_valid"] == expected_citations
        assert events[1]["answer"] == expected_answer
        assert answer == expected_answer
        assert "[99]" not in answer and "[9]" not in answer

    def test_empty_retrieval_stream_returns_final_error_without_answer_checks(self) -> None:
        llm = Mock(spec=LLMClient)
        grounding = Mock(spec=StubGrounding)
        orch = Orchestrator(StubRetriever([]), llm, grounding_verifier=grounding)  # type: ignore[arg-type]
        orch.verifier = Mock(spec=AnswerVerifier)
        error = "No relevant documents found. Please upload documents and try again."

        chunks = list(orch.query_streaming("What did the study find?"))
        answer, events = _consume_stream(chunks)

        assert chunks[:-2] == [error]
        assert answer == error
        assert events[-1] == {"event": "verified_answer", "answer": error}
        assert events[0]["citations_valid"] == []
        assert events[0]["faithfulness"] is None
        assert events[0]["warnings"] == [error]
        llm.stream.assert_not_called()
        orch.verifier.verify.assert_not_called()
        grounding.verify.assert_not_called()

    def test_omitted_sources_cannot_be_cited_or_used_for_grounding(self) -> None:
        results = [_result(i, f"Passage {i}: " + "evidence " * 145) for i in range(1, 6)]
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

    @pytest.mark.parametrize("streaming", [False, True])
    def test_oversized_context_stops_before_generation_and_verification(
        self, streaming: bool
    ) -> None:
        first = _result(1, "Short main passage.")
        first.chunk.context_before = "Nearby context. " * 500
        llm = Mock(spec=LLMClient)
        grounding = Mock(spec=StubGrounding)
        orch = Orchestrator(  # type: ignore[arg-type]
            StubRetriever([first, _result(2, "A later passage that would fit.")]),
            llm,
            grounding_verifier=grounding,
        )
        error = "No source passages fit the context limit."

        if streaming:
            chunks = list(orch.query_streaming("What did the study find?"))
            answer, events = _consume_stream(chunks)
            event = events[0]
            assert chunks[:-2] == [error]
            assert answer == error
            assert events[-1] == {"event": "verified_answer", "answer": error}
            assert event["citations_valid"] == []
            assert event["faithfulness"] is None
            assert event["warnings"] == [error]
        else:
            response = orch.query("What did the study find?")
            assert response.answer == error
            assert response.sources == []
            assert response.confidence == 0.0
            assert response.faithfulness is None
            assert response.warnings == [error]

        llm.generate.assert_not_called()
        llm.stream.assert_not_called()
        grounding.verify.assert_not_called()

    def test_stream_verifies_the_source_snapshot_sent_to_the_llm(self) -> None:
        results = [_result(i, f"Passage {i}: " + "evidence " * 145) for i in range(1, 6)]
        original_contents = [result.chunk.content for result in results[:4]]
        llm = Mock(spec=LLMClient)
        grounding = Mock(spec=StubGrounding)
        grounding.verify.return_value = GroundingReport(faithfulness=1.0, sentences=[])
        orch = Orchestrator(StubRetriever(results), llm, grounding_verifier=grounding)  # type: ignore[arg-type]

        def mutate_after_preparation(*args: Any, **kwargs: Any) -> Any:
            results[0].chunk.content = "Changed after prompt construction."
            results.clear()
            orch.synthesizer.max_sources = 0
            orch.synthesizer.max_context_length = 0
            yield "Visible claim [1]. Omitted claim [5]."

        llm.stream.side_effect = mutate_after_preparation

        chunks = list(orch.query_streaming("What did the study find?"))
        answer, events = _consume_stream(chunks)
        event = events[0]

        assert event["citations_valid"] == [1]
        assert any("out-of-range" in warning for warning in event["warnings"])
        verified_answer, sources = grounding.verify.call_args.args
        assert "[5]" not in verified_answer
        assert answer == verified_answer
        assert [source.content for source in sources] == original_contents
        assert [source.index for source in sources] == [1, 2, 3, 4]
        prompt = llm.stream.call_args.args[0]
        assert all(source.content in prompt for source in sources)

    @pytest.mark.parametrize("partial", [False, True])
    def test_failed_stream_reports_failure_without_grounding_partial_text(
        self, partial: bool
    ) -> None:
        def failed_stream(*args: Any, **kwargs: Any) -> Any:
            if partial:
                yield "Partial answer [1]."
            raise LLMError("model unavailable")

        llm = Mock(spec=LLMClient)
        llm.stream.side_effect = failed_stream
        grounding = Mock(spec=StubGrounding)
        grounding.verify.return_value = GroundingReport(faithfulness=0.8, sentences=[])
        orch = Orchestrator(  # type: ignore[arg-type]
            StubRetriever([_result(1, "evidence")]), llm, grounding_verifier=grounding
        )

        chunks = list(orch.query_streaming("What did the study find?"))
        answer, events = _consume_stream(chunks)
        event = events[0]

        assert chunks[-3] == "Unable to generate response."
        assert "Partial answer" not in answer
        assert answer == "Unable to generate response."
        assert events[-1] == {"event": "verified_answer", "answer": answer}
        assert event["warnings"] == ["Answer generation failed."]
        assert event["faithfulness"] is None
        assert event["citations_valid"] == []
        grounding.verify.assert_not_called()


class _Resp:
    """Minimal httpx.Response stand-in for retry tests."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.request = httpx.Request("POST", "http://x/api/generate")

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=self.request, response=self)  # type: ignore[arg-type]


class FakeHttpClient:
    """Fake httpx.Client whose post() follows a scripted sequence."""

    def __init__(self, behaviors: list[Any]) -> None:
        self.behaviors = behaviors
        self.posts = 0

    def post(self, path: str, json: dict[str, Any]) -> Any:
        item = self.behaviors[min(self.posts, len(self.behaviors) - 1)]
        self.posts += 1
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        pass


def _client_with(behaviors: list[Any], monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """LLMClient wired to a fake http client; no real network or sleeps."""
    monkeypatch.setattr("scinexusrag.generation.llm.time.sleep", lambda *_: None)
    fake = FakeHttpClient(behaviors)
    client = LLMClient(max_retries=2, backoff=0.0)
    client._client = fake  # type: ignore[assignment]
    return client, fake


class TestLLMRetry:
    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _client_with(
            [httpx.ConnectError("down"), _Resp(200, {"response": "recovered"})], monkeypatch
        )
        assert client.generate("hi") == "recovered"
        assert fake.posts == 2

    def test_raises_llmerror_after_exhaustion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _client_with([httpx.ConnectError("down")], monkeypatch)
        with pytest.raises(LLMError) as exc:
            client.generate("hi")

        assert "base_url" in str(exc.value)
        assert fake.posts == 3

    def test_does_not_retry_on_4xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _client_with([_Resp(404, {})], monkeypatch)
        with pytest.raises(httpx.HTTPStatusError):
            client.generate("hi")

        assert fake.posts == 1

    def test_retries_on_5xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _client_with([_Resp(503, {}), _Resp(200, {"response": "ok"})], monkeypatch)
        assert client.generate("hi") == "ok"
        assert fake.posts == 2

    def test_missing_response_field_raises_llmerror(self, monkeypatch: pytest.MonkeyPatch) -> None:

        client, fake = _client_with([_Resp(200, {"done": True})], monkeypatch)
        with pytest.raises(LLMError, match="response"):
            client.generate("hi")
        assert fake.posts == 1


class TestLLMResponseValidation:
    @pytest.mark.parametrize("answer", ["", " \n\t"])
    def test_generate_rejects_empty_answers(self, answer: str) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"response": answer})
        )
        with LLMClient() as client:
            client._client = httpx.Client(base_url=client.base_url, transport=transport)
            with pytest.raises(LLMError, match="empty response"):
                client.generate("question")

    @pytest.mark.parametrize("payload", [[], None, {"response": None}, {"response": 42}])
    def test_generate_rejects_invalid_response_fields(self, payload: Any) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=json.dumps(payload))
        )
        with LLMClient() as client:
            client._client = httpx.Client(base_url=client.base_url, transport=transport)
            with pytest.raises(LLMError, match="string 'response' field"):
                client.generate("question")

    @pytest.mark.parametrize(
        ("body", "error"),
        [
            ("[]", "JSON object"),
            ('{"response": null}', "must be a string"),
            ('{"response": 42}', "must be a string"),
            ('{"error": "model was unloaded"}', "model was unloaded"),
            ('{"done": "false"}', "must be a boolean"),
            ("invalid JSON", "stream failed"),
        ],
    )
    def test_stream_rejects_invalid_records(self, body: str, error: str) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        with LLMClient() as client:
            client._client = httpx.Client(base_url=client.base_url, transport=transport)
            with pytest.raises(LLMError, match=error):
                list(client.stream("question"))

    @pytest.mark.parametrize("body", ["", '{"response": "Partial answer"}\n'])
    def test_stream_requires_completion_record(self, body: str) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        with LLMClient() as client:
            client._client = httpx.Client(base_url=client.base_url, transport=transport)
            with pytest.raises(LLMError, match="before completion"):
                list(client.stream("question"))

    def test_stream_accepts_completed_answer(self) -> None:
        body = '{"response": "Complete answer"}\n{"done": true}\n'
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        with LLMClient() as client:
            client._client = httpx.Client(base_url=client.base_url, transport=transport)
            assert list(client.stream("question")) == ["Complete answer"]

    @pytest.mark.parametrize("answer", ["", " \n\t"])
    def test_stream_rejects_empty_completed_answers(self, answer: str) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"response": answer, "done": True})
        )
        with LLMClient() as client:
            client._client = httpx.Client(base_url=client.base_url, transport=transport)
            with pytest.raises(LLMError, match="empty response"):
                list(client.stream("question"))

    def test_stream_surfaces_an_error_after_partial_output(self) -> None:
        body = '{"response": "Partial answer"}\n{"error": "model was unloaded"}\n'
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        with LLMClient() as client:
            client._client = httpx.Client(base_url=client.base_url, transport=transport)
            tokens = client.stream("question")
            assert next(tokens) == "Partial answer"
            with pytest.raises(LLMError, match="model was unloaded"):
                next(tokens)

    @pytest.mark.parametrize("payload", [[], None])
    def test_readiness_rejects_non_object_json(self, payload: Any) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=json.dumps(payload))
        )
        with LLMClient() as client:
            client._client = httpx.Client(base_url=client.base_url, transport=transport)
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
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"models": [{"name": installed}]})
    )
    with LLMClient(model=configured) as client:
        client._client = httpx.Client(base_url=client.base_url, transport=transport)
        assert client.is_available() is available


class TestStripCitations:
    def test_preserves_line_structure(self) -> None:
        from scinexusrag.generation.citations import strip_citations

        text = "Main finding [1].\n\nDetails:\n1. First [2]\n2. Second [3]"
        out = strip_citations(text, {1, 2, 3})
        assert "\n" in out
        assert out.count("\n") >= 2

    def test_drops_out_of_range(self) -> None:
        from scinexusrag.generation.citations import strip_citations

        out = strip_citations("a [1] b [9].", {1})
        assert "[9]" not in out and "[1]" in out


class TestBuildSourcesIndex:
    def test_no_index_gaps_with_duplicates(self) -> None:
        syn = Synthesizer(StubLLM("x"))
        dup = _result(1, "alpha")
        results = [dup, dup, _result(2, "beta")]
        sources = syn._build_sources(results, {})
        assert [s.index for s in sources] == [1, 2]
