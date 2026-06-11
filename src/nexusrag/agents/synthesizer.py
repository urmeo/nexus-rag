"""Production-grade answer synthesis with structured citations."""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from nexusrag.agents.citations import extract_citations, strip_citations
from nexusrag.agents.llm import LLMClient
from nexusrag.agents.query_analyzer import QueryType
from nexusrag.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A cited source with metadata."""

    index: int
    chunk_id: str
    document_id: str
    content: str
    score: float = 0.0
    page_number: int | None = None
    section_title: str | None = None
    document_name: str = ""
    full_content: str = ""

    @property
    def citation_key(self) -> str:
        """Format citation reference."""
        parts = [f"[{self.index}]"]
        if self.document_name:
            parts.append(self.document_name)
        if self.section_title:
            parts.append(self.section_title)
        return " - ".join(parts)


@dataclass
class SynthesisResult:
    """Result of answer synthesis."""

    answer: str
    sources: list[Source]
    confidence: float
    raw_response: str = ""
    tokens_used: int = 0


# Production system prompt - strict, concise, accurate
SYSTEM_PROMPT = """You are a precise research assistant. Answer questions using ONLY the provided sources.

RULES:
1. Use ONLY information from the sources. NEVER use outside knowledge.
2. If the answer is NOT in the sources, say: "This information is not found in the provided documents."
3. Cite every fact with [1], [2], etc. matching the source numbers.
4. First give a direct answer, then provide supporting details with citations.
5. Prefer sources with higher relevance scores."""


USER_PROMPT_TEMPLATE = """QUESTION: {question}

SOURCES:
{formatted_sources}

Answer using only the sources above. Cite with [1], [2], etc."""


class Synthesizer:
    """
    Production-grade answer synthesizer.

    Features:
    - Structured prompts for consistent output
    - Citation verification
    - Confidence scoring
    - Source deduplication
    """

    def __init__(
        self,
        llm: LLMClient,
        max_context_length: int = 6000,
        max_sources: int = 5,
    ):
        self.llm = llm
        self.max_context_length = max_context_length
        self.max_sources = max_sources
        logger.info(
            f"Initialized Synthesizer: max_context={max_context_length}, max_sources={max_sources}"
        )

    def _get_type_hint(self, query_type: QueryType) -> str | None:
        """Return type-specific synthesis instructions."""
        hints = {
            QueryType.COMPARISON: "Structure as a comparison, highlighting similarities and differences.",
            QueryType.METHODOLOGY: "Explain step by step.",
            QueryType.DEFINITION: "Begin with a clear definition, then elaborate.",
            QueryType.LIST: "Present as a numbered list.",
            QueryType.SUMMARY: "Provide a concise summary of key points.",
            QueryType.CAUSAL: "Explain the causal relationship clearly.",
            QueryType.TEMPORAL: "Present in chronological order.",
        }
        return hints.get(query_type)

    def synthesize(
        self,
        query: str,
        results: list[RetrievalResult],
        temperature: float = 0.1,
        doc_names: dict[str, str] | None = None,
        query_type: QueryType | None = None,
    ) -> SynthesisResult:
        """
        Generate answer with verified citations.

        Args:
            query: User's question
            results: Retrieved chunks with scores
            temperature: LLM temperature (lower = more focused, default 0.1)
            doc_names: Mapping of document_id to filename

        Returns:
            SynthesisResult with answer and sources
        """
        logger.info(f"Synthesizing answer for: {query[:50]}...")

        if not results:
            logger.warning("No results provided for synthesis")
            return SynthesisResult(
                answer="No relevant documents found. Please upload documents and try again.",
                sources=[],
                confidence=0.0,
            )

        # Build sources with document names
        doc_names = doc_names or {}
        sources = self._build_sources(results, doc_names)

        # Limit sources
        sources = sources[: self.max_sources]

        # Format sources for LLM
        formatted_sources = self._format_sources_for_llm(sources)

        # Truncate if needed
        if len(formatted_sources) > self.max_context_length:
            formatted_sources = formatted_sources[: self.max_context_length]
            formatted_sources += "\n\n[Additional sources truncated for length]"
            logger.warning("Sources truncated due to length")

        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=query, formatted_sources=formatted_sources
        )

        # Append query-type-specific hint
        if query_type is not None:
            type_hint = self._get_type_hint(query_type)
            if type_hint:
                user_prompt += f"\n\nInstruction: {type_hint}"

        # Generate answer
        logger.info(f"Calling LLM with {len(sources)} sources...")
        try:
            answer = self.llm.generate(
                user_prompt,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=min(768, 256 + 100 * len(sources)),
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return SynthesisResult(
                answer="Unable to generate response. Please try again.",
                sources=sources,
                confidence=0.0,
            )

        # Verify and clean citations
        answer = self._verify_citations(answer, len(sources))

        # Calculate confidence
        confidence = self._calculate_confidence(results, answer, sources)

        logger.info(f"Synthesis complete: confidence={confidence:.2f}")

        return SynthesisResult(
            answer=answer.strip(),
            sources=sources,
            confidence=confidence,
            raw_response=answer,
        )

    def synthesize_streaming(
        self,
        query: str,
        results: list[RetrievalResult],
        doc_names: dict[str, str] | None = None,
        query_type: QueryType | None = None,
    ) -> Generator[str, None, None]:
        """Stream the synthesis response token by token."""
        if not results:
            yield "No relevant documents found. Please upload documents and try again."
            return

        doc_names = doc_names or {}
        sources = self._build_sources(results, doc_names)[: self.max_sources]
        formatted_sources = self._format_sources_for_llm(sources)

        if len(formatted_sources) > self.max_context_length:
            formatted_sources = formatted_sources[: self.max_context_length]
            formatted_sources += "\n\n[truncated]"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=query, formatted_sources=formatted_sources
        )

        # Append query-type-specific hint
        if query_type is not None:
            type_hint = self._get_type_hint(query_type)
            if type_hint:
                user_prompt += f"\n\nInstruction: {type_hint}"

        try:
            yield from self.llm.stream(
                user_prompt,
                system=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=min(768, 256 + 100 * len(sources)),
            )
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "Unable to generate response."

    def _build_sources(
        self, results: list[RetrievalResult], doc_names: dict[str, str]
    ) -> list[Source]:
        """Convert retrieval results to sources with proper names."""
        sources: list[Source] = []
        seen_chunks = set()

        for i, result in enumerate(results, start=1):
            # Skip duplicates
            if result.chunk.id in seen_chunks:
                continue
            seen_chunks.add(result.chunk.id)

            chunk = result.chunk
            metadata = chunk.metadata
            doc_id = chunk.document_id

            # Get document name - multiple fallbacks
            doc_name = self._get_document_name(chunk, doc_id, doc_names, i)

            # Use full_context (with surrounding text) when available
            full_ctx = ""
            if hasattr(chunk, "full_context"):
                ctx = chunk.full_context
                if ctx and ctx != chunk.content:
                    full_ctx = ctx

            sources.append(
                Source(
                    index=i,
                    chunk_id=chunk.id,
                    document_id=doc_id,
                    content=chunk.content,
                    score=float(result.score),
                    page_number=getattr(chunk, "page_number", None) or metadata.get("page_number"),  # type: ignore[arg-type]
                    section_title=getattr(chunk, "section_title", "")
                    or metadata.get("section_title"),  # type: ignore[arg-type]
                    document_name=doc_name,
                    full_content=full_ctx,
                )
            )

        return sources

    def _get_document_name(
        self,
        chunk: Any,
        doc_id: str,
        doc_names: dict[str, str],
        fallback_index: int,
    ) -> str:
        """Get document name with multiple fallbacks."""
        # Try doc_names mapping
        if doc_id in doc_names:
            name = doc_names[doc_id]
            if not name.startswith("tmp"):
                return name

        # Try chunk attributes
        name = getattr(chunk, "document_name", "")
        if name and not name.startswith("tmp"):
            return name

        # Try metadata
        metadata = chunk.metadata
        for key in ["original_filename", "filename", "display_name", "document_name"]:
            name = str(metadata.get(key, ""))
            if name and not name.startswith("tmp"):
                return name

        return f"Document {fallback_index}"

    def _format_sources_for_llm(self, sources: list[Source]) -> str:
        """Format sources in a clear, numbered structure for the LLM."""
        formatted_parts = []

        for source in sources:
            # Build clear header with relevance score
            section_info = f", Section: {source.section_title}" if source.section_title else ""
            page_info = f", Page {source.page_number}" if source.page_number else ""
            relevance_pct = int(source.score * 100)

            # Use full_content (with surrounding context) when available
            text = source.full_content or source.content

            # Format source block with clear delineation
            source_block = f"""====================
[{source.index}] Source: {source.document_name}{section_info}{page_info} (relevance: {relevance_pct}%)
"{text}"
===================="""
            formatted_parts.append(source_block)

        return "\n".join(formatted_parts)

    def _verify_citations(self, answer: str, num_sources: int) -> str:
        """Drop citations that point past the available sources."""
        return strip_citations(answer, set(range(1, num_sources + 1)))

    def _calculate_confidence(
        self, results: list[RetrievalResult], answer: str, sources: list[Source]
    ) -> float:
        """Calculate confidence based on multiple factors."""
        if not results:
            return 0.0

        scores = []

        # 1. Source relevance (avg similarity score) - 30%
        avg_similarity = sum(r.score for r in results) / len(results)
        scores.append(min(1.0, avg_similarity) * 0.30)

        # 2. Citation coverage (does answer cite sources?) - 25%
        cited_sources = set(extract_citations(answer))
        citation_ratio = len(cited_sources) / len(sources) if sources else 0
        scores.append(min(1.0, citation_ratio) * 0.25)

        # 3. Answer completeness - 25%
        completeness = 0.5  # Base
        if "not found" in answer.lower() or "not in the" in answer.lower():
            completeness = 0.2
        elif len(answer) > 100:
            completeness = 0.9
        else:
            completeness = 0.6
        scores.append(completeness * 0.25)

        # 4. Query word coverage (does answer address the question?) - 20%
        # This is approximated by checking if sources are diverse
        unique_docs = {s.document_id for s in sources}
        doc_diversity = min(1.0, len(unique_docs) / 2)  # 2+ docs = full score

        # Multiple citations from same doc = lower diversity
        if len(cited_sources) >= 2:
            doc_diversity = min(1.0, doc_diversity + 0.3)

        scores.append(doc_diversity * 0.20)

        total = sum(scores)

        # Clamp to reasonable range
        return max(0.15, min(0.95, total))
