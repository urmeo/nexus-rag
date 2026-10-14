"""RAG orchestrator: retrieve, synthesize, verify, ground."""

import json
import logging
import time
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from typing import Any

from scinexusrag.generation.grounding import GroundingVerifier
from scinexusrag.generation.llm import LLMClient
from scinexusrag.generation.query_analyzer import QueryAnalyzer
from scinexusrag.generation.synthesizer import Source, Synthesizer
from scinexusrag.generation.verifier import AnswerVerifier, VerificationResult
from scinexusrag.retrieval import CorrectiveRetriever, RetrievalResult
from scinexusrag.utils.filenames import resolve_display_name

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    stage: str
    detail: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source]
    confidence: float
    processing_time_ms: float = 0.0
    reasoning_trace: list[ReasoningStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    faithfulness: float | None = None
    corrected: bool = False


class Orchestrator:
    """Runs a question through retrieval, generation, and verification."""

    def __init__(
        self,
        retriever: CorrectiveRetriever,
        llm: LLMClient,
        top_k: int = 5,
        document_store: Any = None,
        grounding_verifier: GroundingVerifier | None = None,
        max_tokens: int = 768,
        temperature: float = 0.1,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.document_store = document_store
        self.grounding_verifier = grounding_verifier
        self.analyzer = QueryAnalyzer()
        self.synthesizer = Synthesizer(llm, max_tokens=max_tokens, temperature=temperature)
        self.verifier = AnswerVerifier()

    def query(self, question: str) -> RAGResponse:
        start = time.perf_counter()
        trace: list[ReasoningStep] = []

        analyzed = self.analyzer.analyze(question)
        question = self.analyzer.rewrite_vague_query(question)

        results, corrected = self.retriever.retrieve_traced(question, top_k=self.top_k)
        trace.append(ReasoningStep("retrieval", f"{len(results)} chunks, corrected={corrected}"))

        doc_names = self._document_names(results)
        synthesis = self.synthesizer.synthesize(
            question, results, doc_names=doc_names, query_type=analyzed.query_type
        )

        if synthesis.error:
            answer = synthesis.answer
            confidence = 0.0
            faithfulness = None
            warnings = [synthesis.error]
        else:
            verification, faithfulness, warnings = self._verify_answer(
                synthesis.answer, synthesis.sources
            )
            answer = verification.verified_answer
            confidence = self._confidence(
                synthesis.sources, verification.citations_valid, faithfulness
            )
        trace.append(ReasoningStep("synthesis", f"confidence={confidence:.2f}"))

        return RAGResponse(
            answer=answer,
            sources=synthesis.sources,
            confidence=confidence,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            reasoning_trace=trace,
            warnings=warnings,
            faithfulness=faithfulness,
            corrected=corrected,
        )

    def query_streaming(self, question: str) -> Generator[str, None, None]:
        analyzed = self.analyzer.analyze(question)
        question = self.analyzer.rewrite_vague_query(question)
        results = self.retriever.retrieve(question, top_k=self.top_k)
        doc_names = self._document_names(results)
        prepared = self.synthesizer.prepare(
            question, results, doc_names=doc_names, query_type=analyzed.query_type
        )

        chunks: list[str] = []
        citations_valid: list[int]
        try:
            for token in self.synthesizer.stream_prepared(prepared):
                chunks.append(token)
                yield token
        except Exception:
            logger.exception("Streaming generation failed")
            yield "Unable to generate response."
            citations_valid = []
            faithfulness = None
            warnings = ["Answer generation failed."]
        else:
            if prepared.error:
                citations_valid = []
                faithfulness = None
                warnings = [prepared.error]
            else:
                answer = "".join(chunks)
                verification, faithfulness, warnings = self._verify_answer(answer, prepared.sources)
                citations_valid = verification.citations_valid

        for warning in warnings:
            logger.warning("streaming verification: %s", warning)

        event = {
            "event": "verification",
            "citations_valid": citations_valid,
            "faithfulness": faithfulness,
            "warnings": warnings,
        }
        yield f"\ndata: {json.dumps(event)}\n\n"

    def _verify_answer(
        self, answer: str, sources: list[Source]
    ) -> tuple[VerificationResult, float | None, list[str]]:
        verification = self.verifier.verify(answer, sources)
        warnings = list(verification.warnings)
        faithfulness: float | None = None
        if self.grounding_verifier is not None and sources:
            report = self.grounding_verifier.verify(verification.verified_answer, sources)
            faithfulness = report.faithfulness
            if report.unsupported:
                warnings.append(f"{len(report.unsupported)} sentence(s) not grounded in sources")
        return verification, faithfulness, warnings

    @staticmethod
    def _confidence(sources: list[Source], cited: list[int], faithfulness: float | None) -> float:
        if faithfulness is not None:
            return round(faithfulness, 3)
        if not sources:
            return 0.0
        avg_score = sum(s.score for s in sources) / len(sources)
        coverage = len(set(cited)) / len(sources)
        return round(0.5 * min(1.0, avg_score) + 0.5 * coverage, 3)

    def _document_names(self, results: Sequence[RetrievalResult]) -> dict[str, str]:
        names: dict[str, str] = {}
        if self.document_store:
            try:
                for doc_id, meta in self.document_store.list_with_metadata().items():
                    names[doc_id] = resolve_display_name(meta, fallback=doc_id[:8])
            except Exception:
                logger.debug("could not load document names", exc_info=True)
        for r in results:
            doc_id = r.chunk.document_id
            if doc_id not in names:
                name = resolve_display_name(r.chunk.metadata, fallback="")
                if name:
                    names[doc_id] = name
        return names
