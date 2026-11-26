"""Answer orchestration."""

import logging
import time
from dataclasses import dataclass, field

from scinexusrag.generation.grounding import GroundingVerifier
from scinexusrag.generation.llm import LLMClient
from scinexusrag.generation.query_analyzer import QueryAnalyzer, QueryType
from scinexusrag.generation.synthesizer import Source, Synthesizer
from scinexusrag.generation.verifier import AnswerVerifier, VerificationResult
from scinexusrag.retrieval.corrective import CorrectiveRetriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source]
    confidence: float
    processing_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(
        self,
        retriever: CorrectiveRetriever,
        llm: LLMClient,
        top_k: int = 5,
        grounding_verifier: GroundingVerifier | None = None,
        max_tokens: int = 768,
        temperature: float = 0.1,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.grounding_verifier = grounding_verifier
        self.analyzer = QueryAnalyzer()
        self.synthesizer = Synthesizer(llm, max_tokens=max_tokens, temperature=temperature)
        self.verifier = AnswerVerifier()

    def query(self, question: str) -> RAGResponse:
        start = time.perf_counter()
        query_type, question = self._prepare_question(question)
        results = self.retriever.retrieve(question, top_k=self.top_k)
        synthesis = self.synthesizer.synthesize(question, results, query_type)

        if synthesis.error:
            answer = synthesis.answer
            confidence = 0.0
            warnings = [synthesis.error]
        else:
            verification, faithfulness, warnings = self._verify_answer(
                synthesis.answer, synthesis.sources
            )
            answer = verification.verified_answer
            confidence = self._confidence(
                synthesis.sources, verification.citations_valid, faithfulness
            )

        return RAGResponse(
            answer=answer,
            sources=synthesis.sources,
            confidence=confidence,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            warnings=warnings,
        )

    def _prepare_question(self, question: str) -> tuple[QueryType, str]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        return self.analyzer.classify(question), self.analyzer.rewrite_vague_query(question)

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
        return round(0.5 * max(0.0, min(1.0, avg_score)) + 0.5 * coverage, 3)
