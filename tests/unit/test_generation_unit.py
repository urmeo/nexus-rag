"""Hermetic unit tests for the generation subsystem.

All heavy models are stubbed: no downloads, no network, no Ollama.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from nexusrag.generation.grounding import GroundingReport
from nexusrag.generation.llm import LLMClient, LLMError
from nexusrag.generation.orchestrator import Orchestrator, RAGResponse
from nexusrag.generation.synthesizer import Source, Synthesizer
from nexusrag.generation.verifier import AnswerVerifier
from nexusrag.ingestion.chunker import Chunk
from nexusrag.retrieval import RetrievalResult


def _result(idx: int, content: str, score: float = 0.9) -> RetrievalResult:
    chunk = Chunk(id=f"c{idx}", content=content, document_id=f"d{idx}")
    return RetrievalResult(chunk=chunk, score=score)


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


class TestOrchestratorWiring:
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
        # The synthesizer strips out-of-range markers before verification,
        # so [9] never reaches the answer while [1] survives.
        retriever = StubRetriever([_result(1, "only source")])
        llm = StubLLM("Grounded [1] but invalid [9].")
        orch = Orchestrator(retriever, llm)  # type: ignore[arg-type]

        resp = orch.query("q?")

        assert "[9]" not in resp.answer
        assert "[1]" in resp.answer

    def test_grounding_verifier_sets_faithfulness(self) -> None:
        retriever = StubRetriever([_result(1, "source text")])
        llm = StubLLM("Answer [1].")
        grounding = StubGrounding(faithfulness=0.5, unsupported=["Answer [1]."])
        orch = Orchestrator(retriever, llm, grounding_verifier=grounding)  # type: ignore[arg-type]

        resp = orch.query("q?")

        assert resp.faithfulness == 0.5
        assert resp.confidence == 0.5
        assert any("not grounded" in w for w in resp.warnings)

    def test_streaming_appends_verification_event(self) -> None:
        retriever = StubRetriever([_result(1, "fact")])
        llm = StubLLM("ignored")

        def fake_stream(query: str, results: Any, **kwargs: Any) -> Any:
            yield "Fact "
            yield "[1] and [9]."

        orch = Orchestrator(retriever, llm)  # type: ignore[arg-type]
        orch.synthesizer.synthesize_streaming = fake_stream  # type: ignore[assignment]

        chunks = list(orch.query_streaming("q?"))

        assert "Fact " in chunks
        final = chunks[-1]
        assert final.startswith("\ndata: ")
        assert "verification" in final
        assert "citations_valid" in final


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
    monkeypatch.setattr("nexusrag.generation.llm.time.sleep", lambda *_: None)
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
        assert fake.posts == 3  # initial + 2 retries

    def test_does_not_retry_on_4xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _client_with([_Resp(404, {})], monkeypatch)
        with pytest.raises(httpx.HTTPStatusError):
            client.generate("hi")

        assert fake.posts == 1

    def test_retries_on_5xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client, fake = _client_with(
            [_Resp(503, {}), _Resp(200, {"response": "ok"})], monkeypatch
        )
        assert client.generate("hi") == "ok"
        assert fake.posts == 2


class TestStripCitations:
    def test_preserves_line_structure(self) -> None:
        from nexusrag.generation.citations import strip_citations

        text = "Main finding [1].\n\nDetails:\n1. First [2]\n2. Second [3]"
        out = strip_citations(text, {1, 2, 3})
        assert "\n" in out  # newlines must survive, not collapse to one line
        assert out.count("\n") >= 2

    def test_drops_out_of_range(self) -> None:
        from nexusrag.generation.citations import strip_citations

        out = strip_citations("a [1] b [9].", {1})
        assert "[9]" not in out and "[1]" in out


class TestBuildSourcesIndex:
    def test_no_index_gaps_with_duplicates(self) -> None:
        syn = Synthesizer(StubLLM("x"))
        dup = _result(1, "alpha")
        results = [dup, dup, _result(2, "beta")]  # duplicate chunk id repeated
        sources = syn._build_sources(results, {})
        assert [s.index for s in sources] == [1, 2]  # contiguous, no gap
