"""Sparse retrieval."""

import re
from copy import deepcopy
from dataclasses import dataclass
from heapq import nlargest

from rank_bm25 import BM25Okapi

from scinexusrag.ingestion.chunker import Chunk
from scinexusrag.retrieval.dense import RetrievalResult
from scinexusrag.retrieval.stopwords import STOP_WORDS


def _validate_chunks(chunks: list[Chunk]) -> None:
    if not isinstance(chunks, list) or any(
        not isinstance(chunk, Chunk)
        or not isinstance(chunk.id, str)
        or not chunk.id.strip()
        or not isinstance(chunk.content, str)
        for chunk in chunks
    ):
        raise ValueError(
            "chunks must be a list of chunks with nonempty string IDs and text content"
        )


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[Chunk]


class BM25Retriever:
    """BM25 retrieval."""

    def __init__(self, stopwords: set[str] | frozenset[str] | None = None):
        self._index: BM25Index | None = None
        self._stopwords = STOP_WORDS if stopwords is None else frozenset(stopwords)

    @property
    def stopwords(self) -> frozenset[str]:
        return self._stopwords

    def tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return [token for token in tokens if token not in self.stopwords and len(token) > 1]

    def add(self, chunks: list[Chunk]) -> int:
        _validate_chunks(chunks)
        if not chunks:
            self.clear()
            return 0

        indexed_chunks = deepcopy(chunks)
        if len({chunk.id for chunk in indexed_chunks}) != len(indexed_chunks):
            raise ValueError("chunk IDs must be unique")
        tokenized = [self.tokenize(chunk.content) for chunk in indexed_chunks]
        if not any(tokenized):
            tokenized = [["\x00"] for _ in indexed_chunks]

        self._index = BM25Index(bm25=BM25Okapi(tokenized), chunks=indexed_chunks)
        return len(indexed_chunks)

    def add_incremental(self, chunks: list[Chunk]) -> int:
        _validate_chunks(chunks)
        index = self._index
        existing_chunks = index.chunks if index is not None else []
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
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        index = self._index
        if index is None or top_k <= 0 or not query.strip():
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
        top_indices = nlargest(top_k, candidates, key=scores.__getitem__)
        peak_score = max(scores)
        normalizer = peak_score if peak_score > 0 else 1.0

        return [
            RetrievalResult(
                chunk=deepcopy(index.chunks[i]),
                score=max(0.0, float(scores[i] / normalizer)),
                source="sparse",
            )
            for i in top_indices
        ]

    def remove(self, chunk_ids: set[str]) -> int:
        index = self._index
        if index is None or not chunk_ids:
            return 0

        remaining = [chunk for chunk in index.chunks if chunk.id not in chunk_ids]
        removed = len(index.chunks) - len(remaining)
        if not removed:
            return 0

        self.add(remaining)
        return removed

    def count(self) -> int:
        index = self._index
        return len(index.chunks) if index is not None else 0

    def clear(self) -> None:
        self._index = None
