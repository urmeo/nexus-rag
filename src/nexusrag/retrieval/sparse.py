"""Sparse retrieval using BM25."""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from nexusrag.ingestion import Chunk
from nexusrag.retrieval.dense import RetrievalResult
from nexusrag.retrieval.stopwords import STOP_WORDS


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[Chunk]
    tokenized_corpus: list[list[str]] = field(default_factory=list)


class BM25Retriever:
    """Lexical retrieval with BM25."""

    def __init__(self, stopwords: set[str] | None = None):
        self._index: BM25Index | None = None
        self.stopwords = stopwords or STOP_WORDS

    def tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        tokenized = [self.tokenize(chunk.content) for chunk in chunks]

        self._index = BM25Index(
            bm25=BM25Okapi(tokenized),
            chunks=chunks,
            tokenized_corpus=tokenized,
        )

        return len(chunks)

    def add_incremental(self, chunks: list[Chunk]) -> int:
        """Add chunks to the existing index (rebuilds internally).

        Chunks whose id is already indexed are skipped. Chunk ids are unique,
        so this makes the add idempotent: a lazy cold-start rebuild that
        happens to observe a document's just-written chunks, followed by this
        incremental add of the same chunks, cannot double-count them.
        """
        if self._index is None:
            return self.add(chunks)

        existing = {c.id for c in self._index.chunks}
        fresh = [c for c in chunks if c.id not in existing]
        if not fresh:
            return len(self._index.chunks)
        return self.add(self._index.chunks + fresh)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        # Pin one snapshot: add() rebinds self._index atomically, so a
        # concurrent ingest cannot desync scores from chunks mid-call.
        index = self._index
        if index is None:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = index.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        max_score = max(scores) if max(scores) > 0 else 1.0

        return [
            RetrievalResult(
                chunk=index.chunks[i],
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

        return removed

    def count(self) -> int:
        """Number of indexed chunks."""
        return len(self._index.chunks) if self._index else 0

    def clear(self) -> None:
        """Clear the index."""
        self._index = None
