"""Main orchestrator coordinating the RAG pipeline."""

import logging
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from nexusrag.agents.grounding import GroundingVerifier
from nexusrag.agents.llm import LLMClient
from nexusrag.agents.planner import QueryPlan, QueryPlanner
from nexusrag.agents.query_analyzer import QueryAnalyzer
from nexusrag.agents.retriever_agent import RetrievalQuality, RetrieverAgent, VerifiedResult
from nexusrag.agents.synthesizer import Source, Synthesizer
from nexusrag.agents.verifier import AnswerVerifier
from nexusrag.retrieval import HybridRetriever, Reranker, RetrievalResult
from nexusrag.utils.filenames import resolve_display_name

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""

    stage: str
    action: str
    result: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    answer: str
    sources: list[Source]
    confidence: float
    reasoning_trace: list[ReasoningStep]
    query_plan: QueryPlan | None = None
    retrieval_quality: RetrievalQuality = RetrievalQuality.CORRECT
    total_chunks_retrieved: int = 0
    processing_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    faithfulness: float | None = None


class Orchestrator:
    """
    Main coordinator for the self-correcting RAG pipeline.

    Flow:
    1. Query Planning: Analyze and decompose complex queries
    2. Retrieval: Execute CRAG-style retrieval with verification
    3. Synthesis: Generate answer with citations

    The orchestrator tracks reasoning at each step for transparency.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        reranker: Reranker | None = None,
        max_retrieval_attempts: int = 2,  # Reduced for speed
        top_k: int = 3,  # Reduced from 5 for speed
        document_store: Any = None,  # For getting document names
        relevance_threshold: float = 0.5,
        grounding_verifier: GroundingVerifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.top_k = top_k
        self.document_store = document_store
        self.grounding_verifier = grounding_verifier

        # Initialize components
        self.planner = QueryPlanner(llm)
        self.retriever_agent = RetrieverAgent(
            retriever=retriever,
            llm=llm,
            planner=self.planner,
            reranker=reranker,
            max_attempts=max_retrieval_attempts,
            relevance_threshold=relevance_threshold,
        )
        self.analyzer = QueryAnalyzer(llm)
        self.synthesizer = Synthesizer(llm)
        self.verifier = AnswerVerifier()

    def query(self, question: str) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Args:
            question: User's research question

        Returns:
            RAGResponse with answer, sources, and reasoning trace
        """
        import time

        start_time = time.perf_counter()

        trace: list[ReasoningStep] = []

        # Stage 0: Query Analysis
        analyzed = self.analyzer.analyze(question)
        rewritten = self.analyzer.rewrite_vague_query(question)
        if rewritten != question:
            question = rewritten

        trace.append(
            ReasoningStep(
                stage="analysis",
                action="Query analyzed",
                result=f"Type: {analyzed.query_type.value}, Keywords: {analyzed.keywords[:5]}",
            )
        )

        # Stage 1: Query Planning
        trace.append(
            ReasoningStep(
                stage="planning",
                action="Analyzing query complexity",
                result=f"Query: {question[:100]}...",
            )
        )

        plan = self.planner.plan(question)

        trace.append(
            ReasoningStep(
                stage="planning",
                action="Query plan created",
                result=f"Complexity: {plan.complexity.value}, Steps: {len(plan.steps)}",
            )
        )

        # Stage 2: Retrieval with Verification
        if plan.is_multi_step:
            verified_results = self._execute_multi_step(plan, trace)
        else:
            verified_results = [self._execute_single_step(question, trace)]

        # Merge results from all steps
        all_results = []
        overall_quality = RetrievalQuality.CORRECT
        total_attempts = 0

        for vr in verified_results:
            all_results.extend(vr.results)
            total_attempts += vr.attempts
            if vr.quality == RetrievalQuality.INCORRECT:
                overall_quality = RetrievalQuality.INCORRECT
            elif (
                vr.quality == RetrievalQuality.AMBIGUOUS
                and overall_quality != RetrievalQuality.INCORRECT
            ):
                overall_quality = RetrievalQuality.AMBIGUOUS

        # Deduplicate by chunk ID, keeping highest scores
        seen_ids: set[str] = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.chunk.id not in seen_ids:
                seen_ids.add(r.chunk.id)
                unique_results.append(r)

        unique_results = unique_results[: self.top_k]

        trace.append(
            ReasoningStep(
                stage="retrieval",
                action="Retrieval complete",
                result=f"Quality: {overall_quality.value}, Chunks: {len(unique_results)}, Attempts: {total_attempts}",
            )
        )

        # Stage 3: Synthesis
        trace.append(
            ReasoningStep(
                stage="synthesis",
                action="Generating answer with citations",
                result=f"Using {len(unique_results)} source chunks",
            )
        )

        # Get document names for better citations
        doc_names = self._get_document_names(unique_results)

        synthesis = self.synthesizer.synthesize(
            question, unique_results, doc_names=doc_names, query_type=analyzed.query_type
        )

        trace.append(
            ReasoningStep(
                stage="synthesis",
                action="Answer generated",
                result=f"Confidence: {synthesis.confidence:.2f}, Citations: {len(synthesis.sources)}",
            )
        )

        # Stage 4: Verification (heuristic, no LLM call)
        verification = self.verifier.verify(synthesis.answer, synthesis.sources, question)

        trace.append(
            ReasoningStep(
                stage="verification",
                action="Answer verified",
                result=(
                    f"Valid citations: {len(verification.citations_valid)}/{len(verification.citations_found)}, "
                    f"Confidence: {verification.confidence_score:.2f}"
                    + (f", Warnings: {verification.warnings}" if verification.warnings else "")
                ),
            )
        )

        # Stage 5: Grounding (optional NLI faithfulness check)
        faithfulness: float | None = None
        if self.grounding_verifier is not None:
            report = self.grounding_verifier.verify(verification.verified_answer, synthesis.sources)
            faithfulness = report.faithfulness
            if report.unsupported:
                verification.warnings.append(
                    f"{len(report.unsupported)} sentence(s) not grounded in sources"
                )
            trace.append(
                ReasoningStep(
                    stage="grounding",
                    action="Faithfulness checked",
                    result=f"Grounded: {faithfulness:.2f}, Unsupported: {len(report.unsupported)}",
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            answer=verification.verified_answer,
            sources=synthesis.sources,
            confidence=verification.confidence_score,
            reasoning_trace=trace,
            query_plan=plan,
            retrieval_quality=overall_quality,
            total_chunks_retrieved=len(unique_results),
            processing_time_ms=elapsed_ms,
            warnings=verification.warnings,
            faithfulness=faithfulness,
        )

    def _execute_single_step(self, query: str, trace: list[ReasoningStep]) -> VerifiedResult:
        """Execute retrieval for a single query."""
        trace.append(
            ReasoningStep(
                stage="retrieval",
                action="Starting retrieval with verification",
                result=f"Query: {query[:80]}...",
            )
        )

        result = self.retriever_agent.retrieve_with_verification(query, top_k=self.top_k)

        if result.reformulated_queries:
            trace.append(
                ReasoningStep(
                    stage="retrieval",
                    action="Query reformulation triggered",
                    result=f"Reformulations: {result.reformulated_queries}",
                )
            )

        return result

    def _get_document_names(self, results: Sequence[RetrievalResult]) -> dict[str, str]:
        """Get document ID to original filename mapping."""
        doc_names = {}

        # Primary: get from document store (has correct original filenames)
        if self.document_store:
            try:
                docs = self.document_store.list_with_metadata()
                for doc_id, meta in docs.items():
                    name = resolve_display_name(meta, fallback=doc_id[:8])
                    doc_names[doc_id] = name
            except Exception:
                logger.debug("Failed to load document names from store", exc_info=True)

        # Fallback: get from chunk metadata
        for r in results:
            doc_id = r.chunk.document_id
            if doc_id not in doc_names:
                name = resolve_display_name(r.chunk.metadata, fallback="")
                if name:
                    doc_names[doc_id] = name

        return doc_names

    def _execute_multi_step(
        self, plan: QueryPlan, trace: list[ReasoningStep]
    ) -> list[VerifiedResult]:
        """Execute retrieval for each step in the plan, respecting dependencies."""
        step_results: dict[int, VerifiedResult] = {}
        results: list[VerifiedResult] = []

        for i, step in enumerate(plan.steps, start=1):
            trace.append(
                ReasoningStep(
                    stage="retrieval",
                    action=f"Executing step {i}/{len(plan.steps)}",
                    result=f"Sub-query: {step.query[:60]}... | Purpose: {step.purpose}",
                )
            )

            # Build augmented query from dependency results
            query = step.query
            if step.depends_on:
                dep_context_parts: list[str] = []
                for dep_idx in step.depends_on:
                    if dep_idx not in step_results:
                        logger.warning(
                            f"Step {i} depends on step {dep_idx} which hasn't been executed yet "
                            f"(forward/invalid reference) — skipping"
                        )
                        continue
                    dep_result = step_results[dep_idx]
                    if dep_result.results:
                        # Take top-2 result summaries (200 chars each)
                        for r in dep_result.results[:2]:
                            dep_context_parts.append(r.chunk.content[:200])
                # Cap total dependency context at 500 chars
                dep_context = " ".join(dep_context_parts)[:500]
                if dep_context:
                    query = f"{step.query} [Context: {dep_context}]"

            step_result = self.retriever_agent.retrieve_with_verification(query, top_k=self.top_k)
            step_results[i] = step_result
            results.append(step_result)

            trace.append(
                ReasoningStep(
                    stage="retrieval",
                    action=f"Step {i} complete",
                    result=f"Found {len(step_result.results)} chunks, Quality: {step_result.quality.value}",
                )
            )

        return results

    def query_streaming(self, question: str) -> Generator[str, None, None]:
        """
        Execute pipeline with streaming synthesis.

        Yields tokens as they're generated, then yields final metadata.
        """
        # Query analysis
        analyzed = self.analyzer.analyze(question)
        rewritten = self.analyzer.rewrite_vague_query(question)
        if rewritten != question:
            question = rewritten

        # Non-streaming stages
        plan = self.planner.plan(question)

        if plan.is_multi_step:
            verified_results = self.retriever_agent.retrieve_multi_step(question, top_k=self.top_k)
        else:
            verified_results = [
                self.retriever_agent.retrieve_with_verification(question, top_k=self.top_k)
            ]

        # Collect results
        all_results = []
        for vr in verified_results:
            all_results.extend(vr.results)

        # Deduplicate
        seen_ids: set[str] = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.chunk.id not in seen_ids:
                seen_ids.add(r.chunk.id)
                unique_results.append(r)
        unique_results = unique_results[: self.top_k]

        # Streaming synthesis
        doc_names = self._get_document_names(unique_results)
        yield from self.synthesizer.synthesize_streaming(
            question, unique_results, doc_names=doc_names, query_type=analyzed.query_type
        )
