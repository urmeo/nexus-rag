"""Answer synthesis with inline citations."""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from nexusrag.generation.citations import strip_citations
from nexusrag.generation.llm import LLMClient
from nexusrag.generation.query_analyzer import QueryType
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
    confidence: float = 0.0
    raw_response: str = ""
    tokens_used: int = 0


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
    """Builds the prompt, calls the LLM, and keeps only valid citations."""

    def __init__(
        self,
        llm: LLMClient,
        max_context_length: int = 6000,
        max_sources: int = 5,
    ):
        self.llm = llm
        self.max_context_length = max_context_length
        self.max_sources = max_sources

    def _get_type_hint(self, query_type: QueryType) -> str | None:
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
        if not results:
            return SynthesisResult(
                answer="No relevant documents found. Please upload documents and try again.",
                sources=[],
                confidence=0.0,
            )

        doc_names = doc_names or {}
        sources = self._build_sources(results, doc_names)[: self.max_sources]
        formatted_sources = self._format_sources_for_llm(sources)

        if len(formatted_sources) > self.max_context_length:
            formatted_sources = formatted_sources[: self.max_context_length] + "\n\n[truncated]"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=query, formatted_sources=formatted_sources
        )
        if query_type is not None:
            hint = self._get_type_hint(query_type)
            if hint:
                user_prompt += f"\n\nInstruction: {hint}"

        try:
            answer = self.llm.generate(
                user_prompt,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=min(768, 256 + 100 * len(sources)),
            )
        except Exception:
            logger.error("LLM generation failed", exc_info=True)
            return SynthesisResult(
                answer="Unable to generate response. Please try again.",
                sources=sources,
                confidence=0.0,
            )

        answer = strip_citations(answer, set(range(1, len(sources) + 1)))
        return SynthesisResult(answer=answer.strip(), sources=sources, raw_response=answer)

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

        for result in results:
            # Skip duplicates
            if result.chunk.id in seen_chunks:
                continue
            seen_chunks.add(result.chunk.id)

            # number by final position so citation indices have no gaps
            index = len(sources) + 1
            chunk = result.chunk
            metadata = chunk.metadata
            doc_id = chunk.document_id

            # Get document name - multiple fallbacks
            doc_name = self._get_document_name(chunk, doc_id, doc_names, index)

            # Use full_context (with surrounding text) when available
            full_ctx = ""
            if hasattr(chunk, "full_context"):
                ctx = chunk.full_context
                if ctx and ctx != chunk.content:
                    full_ctx = ctx

            sources.append(
                Source(
                    index=index,
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
