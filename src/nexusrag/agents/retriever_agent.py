"""CRAG-style retrieval agent with self-correction."""

import re
from dataclasses import dataclass, field
from enum import Enum

from nexusrag.agents.llm import LLMClient
from nexusrag.agents.planner import QueryPlanner
from nexusrag.retrieval import HybridRetriever, Reranker, RetrievalResult


class RetrievalQuality(Enum):
    """Assessment of retrieval quality."""

    CORRECT = "correct"  # Relevant documents found
    INCORRECT = "incorrect"  # Documents not relevant
    AMBIGUOUS = "ambiguous"  # Partially relevant


@dataclass
class VerifiedResult:
    """Retrieval result with quality verification."""

    results: list[RetrievalResult]
    quality: RetrievalQuality
    confidence: float
    attempts: int = 1
    reformulated_queries: list[str] = field(default_factory=list)


BATCH_GRADING_PROMPT = """Evaluate if each document is relevant to the query.

Query: {query}

{documents}

For each document, respond with its number followed by RELEVANT or IRRELEVANT.
Example format:
1: RELEVANT
2: IRRELEVANT
3: RELEVANT"""


class RetrieverAgent:
    """Self-correcting retrieval agent using CRAG methodology."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        planner: QueryPlanner | None = None,
        reranker: Reranker | None = None,
        max_attempts: int = 3,
        relevance_threshold: float = 0.5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.planner = planner or QueryPlanner(llm)
        self.reranker = reranker
        self.max_attempts = max(1, max_attempts)
        self.relevance_threshold = relevance_threshold

    def retrieve_with_verification(
        self,
        query: str,
        top_k: int = 5,
    ) -> VerifiedResult:
        """
        Retrieve documents with quality verification and self-correction.

        Implements CRAG (Corrective RAG) pattern:
        1. Retrieve initial results
        2. Grade each result for relevance
        3. If quality is low, reformulate query and retry
        4. Return verified results with confidence

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            VerifiedResult with quality assessment
        """
        reformulated_queries: list[str] = []
        current_query = query

        for attempt in range(1, self.max_attempts + 1):
            # Retrieve candidates
            results = self.retriever.retrieve(current_query, top_k=top_k * 2)

            # Optional reranking
            if self.reranker and results:
                results = self.reranker.rerank(current_query, results, top_k=top_k)
            else:
                results = results[:top_k]

            # Grade results
            quality, confidence, filtered = self._grade_results(query, results)

            if quality == RetrievalQuality.CORRECT:
                return VerifiedResult(
                    results=filtered,
                    quality=quality,
                    confidence=confidence,
                    attempts=attempt,
                    reformulated_queries=reformulated_queries,
                )

            if attempt < self.max_attempts:
                # Reformulate query for next attempt
                context = self._build_reformulation_context(results)
                current_query = self.planner.reformulate(query, context)
                reformulated_queries.append(current_query)

        # Return best effort after max attempts
        return VerifiedResult(
            results=filtered if filtered else results[:top_k],
            quality=quality,
            confidence=confidence,
            attempts=self.max_attempts,
            reformulated_queries=reformulated_queries,
        )

    def _grade_results(
        self, query: str, results: list[RetrievalResult]
    ) -> tuple[RetrievalQuality, float, list[RetrievalResult]]:
        """
        Grade all results for relevance in a single batched LLM call.

        Returns:
            Tuple of (overall quality, confidence, filtered results)
        """
        if not results:
            return RetrievalQuality.INCORRECT, 0.0, []

        # Identify results above threshold that need LLM grading
        above_threshold: list[tuple[int, RetrievalResult]] = []
        for i, result in enumerate(results):
            if result.score >= self.relevance_threshold:
                above_threshold.append((i, result))

        # Initialize per-result verdicts: False = irrelevant
        verdicts: dict[int, bool] = dict.fromkeys(range(len(results)), False)

        if above_threshold:
            # Build a single prompt with all above-threshold documents
            doc_sections = []
            for idx, (_, result) in enumerate(above_threshold, 1):
                # Use full_context (with surrounding text) when available
                chunk = result.chunk
                text = getattr(chunk, "full_context", None) or chunk.content
                doc_sections.append(f"Document {idx}:\n{text[:1500]}")
            documents_text = "\n\n".join(doc_sections)

            prompt = BATCH_GRADING_PROMPT.format(query=query, documents=documents_text)
            # Scale max_tokens with document count
            grading_tokens = max(100, 20 * len(above_threshold))
            response = self.llm.generate(prompt, temperature=0.0, max_tokens=grading_tokens)

            # Parse response — expect lines like "1: RELEVANT"
            for line in response.strip().split("\n"):
                match = re.match(r"(\d+)\s*:\s*(RELEVANT|IRRELEVANT)", line.upper())
                if match:
                    doc_num = int(match.group(1))
                    is_relevant = match.group(2) == "RELEVANT"
                    if 1 <= doc_num <= len(above_threshold):
                        original_idx = above_threshold[doc_num - 1][0]
                        verdicts[original_idx] = is_relevant

        # Build filtered results and scores
        relevant_results: list[RetrievalResult] = []
        relevance_scores: list[float] = []
        for i, result in enumerate(results):
            if verdicts[i]:
                relevant_results.append(result)
                relevance_scores.append(result.score)
            else:
                relevance_scores.append(result.score * 0.5)

        # Calculate quality
        relevance_ratio = len(relevant_results) / len(results)
        avg_score = sum(relevance_scores) / len(relevance_scores)

        if relevance_ratio >= 0.6:
            quality = RetrievalQuality.CORRECT
        elif relevance_ratio >= 0.3:
            quality = RetrievalQuality.AMBIGUOUS
        else:
            quality = RetrievalQuality.INCORRECT

        return quality, avg_score, relevant_results

    def _build_reformulation_context(self, results: list[RetrievalResult]) -> str:
        """Build context string for query reformulation."""
        if not results:
            return "No relevant documents found."

        topics = set()
        for r in results[:3]:
            words = r.chunk.content.split()[:20]
            topics.add(" ".join(words))

        return f"Retrieved documents discuss: {'; '.join(topics)}"

    def retrieve_multi_step(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VerifiedResult]:
        """
        Execute multi-step retrieval for complex queries.

        Uses QueryPlanner to decompose query, then retrieves for each step.
        """
        plan = self.planner.plan(query)

        if not plan.is_multi_step:
            return [self.retrieve_with_verification(query, top_k)]

        results: list[VerifiedResult] = []
        for step in plan.steps:
            step_result = self.retrieve_with_verification(step.query, top_k)
            results.append(step_result)

        return results
