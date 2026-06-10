"""Sparse retrieval using BM25."""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from nexusrag.ingestion import Chunk
from nexusrag.retrieval.dense import RetrievalResult
from nexusrag.retrieval.stopwords import STOP_WORDS


@dataclass
class BM25Index:
    """BM25 index with chunk mapping."""

    bm25: BM25Okapi
    chunks: list[Chunk]
    tokenized_corpus: list[list[str]] = field(default_factory=list)


class BM25Retriever:
    """Lexical retrieval using BM25 algorithm."""

    def __init__(self, stopwords: set[str] | None = None):
        self._index: BM25Index | None = None
        self._chunk_map: dict[str, Chunk] = {}
        self.stopwords = stopwords or STOP_WORDS

    def tokenize(self, text: str) -> list[str]:
        """Tokenize and normalize text."""
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def add(self, chunks: list[Chunk]) -> int:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunks to index

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        tokenized = [self.tokenize(chunk.content) for chunk in chunks]

        self._index = BM25Index(
            bm25=BM25Okapi(tokenized),
            chunks=chunks,
            tokenized_corpus=tokenized,
        )

        self._chunk_map = {chunk.id: chunk for chunk in chunks}
        return len(chunks)

    def add_incremental(self, chunks: list[Chunk]) -> int:
        """Add chunks to existing index (rebuilds internally)."""
        if self._index is None:
            return self.add(chunks)

        all_chunks = self._index.chunks + chunks
        return self.add(all_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant chunks using BM25.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of RetrievalResult sorted by BM25 score
        """
        if self._index is None:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self._index.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1.0

        return [
            RetrievalResult(
                chunk=self._index.chunks[i],
                score=scores[i] / max_score,
                source="sparse",
            )
            for i in top_indices
            if scores[i] > 0
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
            self._chunk_map.clear()

        return removed

    def count(self) -> int:
        """Number of indexed chunks."""
        return len(self._index.chunks) if self._index else 0

    def clear(self) -> None:
        """Clear the index."""
        self._index = None
        self._chunk_map.clear()
