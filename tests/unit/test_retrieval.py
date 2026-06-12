"""Tests for retrieval components."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from nexusrag.ingestion import Chunk
from nexusrag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalResult,
)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="chunk1",
            content="Machine learning is a subset of artificial intelligence.",
            document_id="doc1",
            metadata={"chunk_index": 0},
        ),
        Chunk(
            id="chunk2",
            content="Deep learning uses neural networks with many layers.",
            document_id="doc1",
            metadata={"chunk_index": 1},
        ),
        Chunk(
            id="chunk3",
            content="Natural language processing enables computers to understand text.",
            document_id="doc2",
            metadata={"chunk_index": 0},
        ),
        Chunk(
            id="chunk4",
            content="Computer vision allows machines to interpret images.",
            document_id="doc2",
            metadata={"chunk_index": 1},
        ),
        Chunk(
            id="chunk5",
            content="Reinforcement learning trains agents through rewards.",
            document_id="doc3",
            metadata={"chunk_index": 0},
        ),
    ]


class TestBM25Retriever:
    """Test suite for BM25Retriever."""

    def test_add_chunks(self, sample_chunks):
        """Adding chunks builds the index."""
        retriever = BM25Retriever()
        count = retriever.add(sample_chunks)

        assert count == 5
        assert retriever.count() == 5

    def test_retrieve_basic(self, sample_chunks):
        """Basic retrieval returns relevant results."""
        retriever = BM25Retriever()
        retriever.add(sample_chunks)

        results = retriever.retrieve("neural networks deep learning", top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.source == "sparse" for r in results)

        # Deep learning chunk should be highly ranked
        chunk_ids = [r.chunk.id for r in results]
        assert "chunk2" in chunk_ids

    def test_retrieve_scores_normalized(self, sample_chunks):
        """Retrieval scores are normalized to 0-1 range."""
        retriever = BM25Retriever()
        retriever.add(sample_chunks)

        results = retriever.retrieve("machine learning artificial intelligence", top_k=5)

        for result in results:
            assert 0 <= result.score <= 1

    def test_empty_index(self):
        """Retrieval from empty index returns empty list."""
        retriever = BM25Retriever()

        results = retriever.retrieve("test query", top_k=5)

        assert results == []

    def test_empty_query(self, sample_chunks):
        """Empty query returns empty results."""
        retriever = BM25Retriever()
        retriever.add(sample_chunks)

        results = retriever.retrieve("", top_k=5)

        assert results == []

    def test_stopwords_removed(self, sample_chunks):
        """Stopwords are removed during tokenization."""
        retriever = BM25Retriever()

        tokens = retriever.tokenize("the quick brown fox is jumping")

        assert "the" not in tokens
        assert "is" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_add_incremental(self, sample_chunks):
        """Incremental add extends the index."""
        retriever = BM25Retriever()
        retriever.add(sample_chunks[:3])
        assert retriever.count() == 3

        retriever.add_incremental(sample_chunks[3:])
        assert retriever.count() == 5

    def test_remove_chunks(self, sample_chunks):
        """Removing chunks updates the index."""
        retriever = BM25Retriever()
        retriever.add(sample_chunks)

        removed = retriever.remove({"chunk1", "chunk2"})

        assert removed == 2
        assert retriever.count() == 3

    def test_clear(self, sample_chunks):
        """Clear removes all chunks."""
        retriever = BM25Retriever()
        retriever.add(sample_chunks)
        retriever.clear()

        assert retriever.count() == 0


class TestDenseRetriever:
    """Test suite for DenseRetriever."""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        embedder.embed_query.return_value = np.random.rand(384).astype(np.float32)
        return embedder

    @pytest.fixture
    def mock_vector_store(self, sample_chunks):
        """Create mock vector store."""
        store = MagicMock()

        def mock_search(query_emb, top_k):
            from nexusrag.storage import SearchResult as StoreSearchResult

            return [
                StoreSearchResult(chunk=sample_chunks[1], score=0.9),
                StoreSearchResult(chunk=sample_chunks[0], score=0.7),
            ][:top_k]

        store.search.side_effect = mock_search
        return store

    def test_retrieve_basic(self, mock_embedder, mock_vector_store):
        """Basic dense retrieval works correctly."""
        retriever = DenseRetriever(mock_embedder, mock_vector_store)

        results = retriever.retrieve("deep learning query", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.source == "dense" for r in results)
        mock_embedder.embed_query.assert_called_once_with("deep learning query")

    def test_retrieve_with_threshold(self, mock_embedder, mock_vector_store):
        """Threshold filtering removes low-score results."""
        retriever = DenseRetriever(mock_embedder, mock_vector_store)

        results = retriever.retrieve_with_threshold("query", top_k=5, min_score=0.8)

        # Only chunk with score >= 0.8 should remain
        assert len(results) == 1
        assert results[0].score >= 0.8


class TestHybridRetriever:
    """Test suite for HybridRetriever."""

    @pytest.fixture
    def mock_dense_retriever(self, sample_chunks):
        """Create mock dense retriever."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            RetrievalResult(chunk=sample_chunks[0], score=0.9, source="dense"),
            RetrievalResult(chunk=sample_chunks[1], score=0.8, source="dense"),
            RetrievalResult(chunk=sample_chunks[2], score=0.6, source="dense"),
        ]
        return retriever

    @pytest.fixture
    def mock_sparse_retriever(self, sample_chunks):
        """Create mock sparse retriever."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            RetrievalResult(chunk=sample_chunks[1], score=0.85, source="sparse"),
            RetrievalResult(chunk=sample_chunks[3], score=0.7, source="sparse"),
            RetrievalResult(chunk=sample_chunks[0], score=0.5, source="sparse"),
        ]
        return retriever

    def test_hybrid_fusion(self, mock_dense_retriever, mock_sparse_retriever, sample_chunks):
        """Hybrid retrieval fuses results correctly."""
        hybrid = HybridRetriever(
            mock_dense_retriever,
            mock_sparse_retriever,
            dense_weight=0.7,
            sparse_weight=0.3,
        )

        results = hybrid.retrieve("test query", top_k=3)

        assert len(results) == 3
        assert all(r.source == "hybrid" for r in results)

        # chunk1 appears in both, should be ranked high
        top_ids = [r.chunk.id for r in results]
        assert "chunk2" in top_ids  # Appears in both retrievers

    def test_rrf_scoring(self, mock_dense_retriever, mock_sparse_retriever):
        """RRF scoring combines ranks correctly."""
        hybrid = HybridRetriever(
            mock_dense_retriever,
            mock_sparse_retriever,
            dense_weight=0.5,
            sparse_weight=0.5,
            rrf_k=60,
        )

        results = hybrid.retrieve("test", top_k=5)

        # Scores should be positive and ordered
        scores = [r.score for r in results]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_dense_only(self, mock_dense_retriever, mock_sparse_retriever):
        """Dense-only retrieval bypasses fusion."""
        hybrid = HybridRetriever(mock_dense_retriever, mock_sparse_retriever)

        results = hybrid.retrieve_dense_only("query", top_k=3)

        assert all(r.source == "dense" for r in results)
        mock_sparse_retriever.retrieve.assert_not_called()

    def test_sparse_only(self, mock_dense_retriever, mock_sparse_retriever):
        """Sparse-only retrieval bypasses fusion."""
        hybrid = HybridRetriever(mock_dense_retriever, mock_sparse_retriever)

        results = hybrid.retrieve_sparse_only("query", top_k=3)

        assert all(r.source == "sparse" for r in results)
        mock_dense_retriever.retrieve.assert_not_called()

    def test_weight_validation(self, mock_dense_retriever, mock_sparse_retriever):
        """Invalid weights raise ValueError."""
        with pytest.raises(ValueError):
            HybridRetriever(
                mock_dense_retriever,
                mock_sparse_retriever,
                dense_weight=1.5,
                sparse_weight=0.3,
            )

    def test_empty_results(self, sample_chunks):
        """Handles empty results from retrievers."""
        dense = MagicMock()
        dense.retrieve.return_value = []
        sparse = MagicMock()
        sparse.retrieve.return_value = []

        hybrid = HybridRetriever(dense, sparse)
        results = hybrid.retrieve("query", top_k=5)

        assert results == []
