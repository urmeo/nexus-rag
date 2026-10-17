"""Answer synthesis with inline citations."""

import logging
from collections.abc import Generator
from dataclasses import dataclass

from scinexusrag.generation.llm import LLMClient
from scinexusrag.generation.query_analyzer import QueryType
from scinexusrag.retrieval import RetrievalResult
from scinexusrag.utils.filenames import resolve_display_name

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


@dataclass
class SynthesisResult:
    answer: str
    sources: list[Source]
    confidence: float = 0.0
    raw_response: str = ""
    tokens_used: int = 0
    error: str | None = None


@dataclass
class SynthesisPrompt:
    prompt: str
    sources: list[Source]
    error: str | None = None


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

_QUERY_TYPE_HINTS = {
    QueryType.COMPARISON: "Structure as a comparison, highlighting similarities and differences.",
    QueryType.METHODOLOGY: "Explain step by step.",
    QueryType.DEFINITION: "Begin with a clear definition, then elaborate.",
    QueryType.LIST: "Present as a numbered list.",
    QueryType.SUMMARY: "Provide a concise summary of key points.",
    QueryType.CAUSAL: "Explain the causal relationship clearly.",
    QueryType.TEMPORAL: "Present in chronological order.",
}


class Synthesizer:
    """Build cited-source prompts and call the LLM."""

    def __init__(
        self,
        llm: LLMClient,
        max_context_length: int = 6000,
        max_sources: int = 5,
        max_tokens: int = 768,
        temperature: float = 0.1,
    ):
        self.llm = llm
        self.max_context_length = max_context_length
        self.max_sources = max_sources
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _token_budget(self, sources: list[Source]) -> int:
        """Scale the response budget with source count, capped by config."""
        return min(self.max_tokens, 256 + 100 * len(sources))

    def _build_prompt(self, query: str, sources: list[Source], query_type: QueryType | None) -> str:
        formatted_sources = self._format_sources_for_llm(sources)
        prompt = USER_PROMPT_TEMPLATE.format(question=query, formatted_sources=formatted_sources)
        if query_type is not None:
            hint = _QUERY_TYPE_HINTS.get(query_type)
            if hint:
                prompt += f"\n\nInstruction: {hint}"
        return prompt

    def prepare(
        self,
        query: str,
        results: list[RetrievalResult],
        doc_names: dict[str, str] | None = None,
        query_type: QueryType | None = None,
    ) -> SynthesisPrompt:
        if not results:
            return SynthesisPrompt(
                prompt="",
                sources=[],
                error="No relevant documents found. Please upload documents and try again.",
            )

        sources: list[Source] = []
        context_length = 0
        for source in self._build_sources(results, doc_names or {})[: self.max_sources]:
            block_length = len(self._format_sources_for_llm([source])) + (1 if sources else 0)
            if context_length + block_length > self.max_context_length:
                break
            sources.append(source)
            context_length += block_length

        if not sources:
            return SynthesisPrompt(
                prompt="", sources=[], error="No source passages fit the context limit."
            )
        return SynthesisPrompt(self._build_prompt(query, sources, query_type), sources)

    def synthesize(
        self,
        query: str,
        results: list[RetrievalResult],
        doc_names: dict[str, str] | None = None,
        query_type: QueryType | None = None,
    ) -> SynthesisResult:
        prepared = self.prepare(query, results, doc_names, query_type)
        if prepared.error:
            return SynthesisResult(answer=prepared.error, sources=[], error=prepared.error)

        try:
            answer = self.llm.generate(
                prepared.prompt,
                system=SYSTEM_PROMPT,
                temperature=self.temperature,
                max_tokens=self._token_budget(prepared.sources),
            )
        except Exception:
            logger.error("LLM generation failed", exc_info=True)
            return SynthesisResult(
                answer="Unable to generate response. Please try again.",
                sources=prepared.sources,
                error="Answer generation failed.",
            )

        return SynthesisResult(answer=answer.strip(), sources=prepared.sources, raw_response=answer)

    def synthesize_streaming(
        self,
        query: str,
        results: list[RetrievalResult],
        doc_names: dict[str, str] | None = None,
        query_type: QueryType | None = None,
    ) -> Generator[str, None, None]:
        """Stream the synthesis response token by token."""
        yield from self.stream_prepared(self.prepare(query, results, doc_names, query_type))

    def stream_prepared(self, prepared: SynthesisPrompt) -> Generator[str, None, None]:
        if prepared.error:
            yield prepared.error
            return
        yield from self.llm.stream(
            prepared.prompt,
            system=SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self._token_budget(prepared.sources),
        )

    def _build_sources(
        self, results: list[RetrievalResult], doc_names: dict[str, str]
    ) -> list[Source]:
        """Build numbered sources with document names."""
        sources: list[Source] = []
        seen_chunks = set()

        for result in results:
            if result.chunk.id in seen_chunks:
                continue
            seen_chunks.add(result.chunk.id)

            index = len(sources) + 1
            chunk = result.chunk
            metadata = chunk.metadata
            doc_id = chunk.document_id

            document_name = doc_names.get(
                doc_id,
                chunk.document_name or resolve_display_name(metadata, fallback=f"Document {index}"),
            )
            full_context = chunk.full_context

            sources.append(
                Source(
                    index=index,
                    chunk_id=chunk.id,
                    document_id=doc_id,
                    content=chunk.content,
                    score=float(result.score),
                    page_number=chunk.page_number or metadata.get("page_number"),  # type: ignore[arg-type]
                    section_title=chunk.section_title or metadata.get("section_title"),  # type: ignore[arg-type]
                    document_name=document_name,
                    full_content=full_context if full_context != chunk.content else "",
                )
            )

        return sources

    def _format_sources_for_llm(self, sources: list[Source]) -> str:
        """Number source passages for the prompt."""
        formatted_parts = []

        for source in sources:
            section_info = f", Section: {source.section_title}" if source.section_title else ""
            page_info = f", Page {source.page_number}" if source.page_number else ""
            relevance_pct = int(source.score * 100)

            text = source.full_content or source.content

            source_block = f"""====================
[{source.index}] Source: {source.document_name}{section_info}{page_info} (relevance: {relevance_pct}%)
"{text}"
===================="""
            formatted_parts.append(source_block)

        return "\n".join(formatted_parts)
