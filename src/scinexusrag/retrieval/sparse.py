"""Sparse retrieval using BM25."""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from scinexusrag.ingestion import Chunk
from scinexusrag.retrieval.dense import RetrievalResult
from scinexusrag.retrieval.stopwords import STOP_WORDS


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[Chunk]


class BM25Retriever:
    """Lexical retrieval with BM25."""

    def __init__(self, stopwords: set[str] | None = None):
        self._index: BM25Index | None = None
        self.stopwords = STOP_WORDS if stopwords is None else stopwords

    def tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        tokenized = [self.tokenize(chunk.content) for chunk in chunks]
        if not any(tokenized):
            tokenized = [["\x00"] for _ in chunks]

        self._index = BM25Index(bm25=BM25Okapi(tokenized), chunks=chunks)
        return len(chunks)

    def add_incremental(self, chunks: list[Chunk]) -> int:
        "Add chunks to the existing index (rebuilds internally)."
        existing_chunks = self._index.chunks if self._index is not None else []
        seen_ids = {chunk.id for chunk in existing_chunks}
        fresh: list[Chunk] = []
        for chunk in chunks:
            if chunk.id not in seen_ids:
                fresh.append(chunk)
                seen_ids.add(chunk.id)
        if not fresh:
            return len(existing_chunks)
        return self.add(existing_chunks + fresh)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        index = self._index
        if index is None or top_k <= 0:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        query_terms = set(tokenized_query)
        candidates = [
            i
            for i, frequencies in enumerate(index.bm25.doc_freqs)
            if any(term in frequencies for term in query_terms)
        ]
        if not candidates:
            return []

        scores = index.bm25.get_scores(tokenized_query)
        top_indices = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]
        peak_score = max(scores)
        max_score = peak_score if peak_score > 0 else 1.0

        return [
            RetrievalResult(
                chunk=index.chunks[i],
                score=max(0.0, float(scores[i] / max_score)),
                source="sparse",
            )
            for i in top_indices
        ]

    def remove(self, chunk_ids: set[str]) -> int:
        """Remove chunks and rebuild index."""
        if self._index is None:
            return 0

        remaining = [c for c in self._index.chunks if c.id not in chunk_ids]
        removed = len(self._index.chunks) - len(remaining)

        if remaining:
            self.add(remaining)
        else:
            self._index = None

        return removed

    def count(self) -> int:
        return len(self._index.chunks) if self._index else 0

    def clear(self) -> None:
        self._index = None
