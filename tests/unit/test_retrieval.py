from unittest.mock import MagicMock

import numpy as np
import pytest

from scinexusrag.ingestion import Chunk
from scinexusrag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Reranker,
    RetrievalResult,
)


@pytest.fixture
def sample_chunks():
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
    ]
    docs = ["doc1", "doc1", "doc2", "doc2", "doc3"]
    return [
        Chunk(id=f"chunk{i + 1}", content=t, document_id=d)
        for i, (t, d) in enumerate(zip(texts, docs, strict=True))
    ]


class TestBM25Retriever:
    def test_add_indexes_all_chunks(self, sample_chunks):
        r = BM25Retriever()
        assert r.add(sample_chunks) == 5
        assert r.count() == 5

    def test_retrieve_ranks_the_matching_chunk_first(self, sample_chunks):
        r = BM25Retriever()
        r.add(sample_chunks)
        results = r.retrieve("neural networks deep learning", top_k=3)
        assert results[0].chunk.id == "chunk2"
        assert all(res.source == "sparse" for res in results)
        assert all(0 <= res.score <= 1 for res in results)

    def test_empty_index_and_empty_query(self, sample_chunks):
        assert BM25Retriever().retrieve("anything", top_k=5) == []
        r = BM25Retriever()
        r.add(sample_chunks)
        assert r.retrieve("", top_k=5) == []

    def test_single_matching_passage_with_negative_idf_is_returned(self):
        retriever = BM25Retriever()
        retriever.add([Chunk(id="match", document_id="paper", content="kinase tumor inhibition")])

        results = retriever.retrieve("kinase")

        assert [result.chunk.id for result in results] == ["match"]
        assert results[0].score == 0.0

    @pytest.mark.parametrize(
        "texts",
        [
            ["cardiac disease", "kinase inhibition"],
            ["shared pathway", "shared pathway kinase", "shared pathway kinase kinase"],
        ],
    )
    def test_nonpositive_match_is_ranked_before_unmatched_passage(self, texts):
        retriever = BM25Retriever()
        retriever.add(
            [
                Chunk(id=f"passage{i}", document_id="paper", content=text)
                for i, text in enumerate(texts)
            ]
        )

        results = retriever.retrieve("kinase", top_k=1)

        assert [result.chunk.id for result in results] == ["passage1"]
        assert results[0].score == 0.0

    def test_query_without_token_overlap_returns_no_passages(self):
        retriever = BM25Retriever()
        retriever.add([Chunk(id="one", document_id="paper", content="kinase tumor inhibition")])

        assert retriever.retrieve("cardiac disease") == []

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_nonpositive_depth_returns_no_passages(self, sample_chunks, top_k):
        retriever = BM25Retriever()
        retriever.add(sample_chunks)

        assert retriever.retrieve("neural networks", top_k=top_k) == []

    def test_tokenize_drops_stopwords(self):
        tokens = BM25Retriever().tokenize("the quick brown fox is jumping")
        assert "the" not in tokens and "is" not in tokens
        assert "quick" in tokens and "brown" in tokens

    def test_explicit_empty_stopwords_keep_common_words(self):
        assert BM25Retriever(stopwords=set()).tokenize("the quick fox") == [
            "the",
            "quick",
            "fox",
        ]

    def test_incremental_add_and_remove(self, sample_chunks):
        r = BM25Retriever()
        r.add(sample_chunks[:3])
        r.add_incremental(sample_chunks[3:])
        assert r.count() == 5
        assert r.remove({"chunk1", "chunk2"}) == 2
        assert r.count() == 3

    def test_incremental_add_is_idempotent_by_id(self, sample_chunks):

        r = BM25Retriever()
        r.add(sample_chunks[:3])
        r.add_incremental(sample_chunks[:3])
        assert r.count() == 3
        r.add_incremental(sample_chunks[2:])
        assert r.count() == 5

    @pytest.mark.parametrize("preload", [False, True])
    def test_incremental_add_deduplicates_the_new_batch(self, sample_chunks, preload):
        retriever = BM25Retriever()
        if preload:
            retriever.add(sample_chunks[:1])

        assert retriever.add_incremental([*sample_chunks[:2], *sample_chunks[:2]]) == 2
        assert retriever.count() == 2
        assert retriever.remove({sample_chunks[0].id}) == 1
        assert retriever.count() == 1

    def test_add_survives_all_empty_tokenization(self):

        chunks = [
            Chunk(id=f"c{i}", content=c, document_id="d")
            for i, c in enumerate(["the a is", "of to in", "x y z"])
        ]
        r = BM25Retriever()
        assert r.add(chunks) == 3
        assert r.count() == 3
        assert r.retrieve("machine learning", top_k=5) == []


class TestReranker:
    def _results(self):
        return [
            RetrievalResult(
                chunk=Chunk(id=f"c{i}", content=t, document_id="d"), score=0.0, source="hybrid"
            )
            for i, t in enumerate(["alpha", "beta", "gamma"])
        ]

    def _reranker(self, scores):
        r = Reranker()
        model = MagicMock()
        model.predict.return_value = np.array(scores, dtype=np.float32)
        r._model = model
        return r

    def test_empty_results_returns_empty(self):
        assert self._reranker([]).rerank("q", []) == []

    def test_orders_by_cross_encoder_score_desc(self):
        out = self._reranker([0.1, 0.9, 0.5]).rerank("q", self._results())
        assert [x.chunk.id for x in out] == ["c1", "c2", "c0"]
        assert out[0].score == 1.0 and out[-1].score == 0.0
        assert all(x.source == "hybrid+rerank" for x in out)

    def test_all_equal_scores_map_to_neutral_one(self):
        out = self._reranker([0.4, 0.4, 0.4]).rerank("q", self._results())
        assert all(x.score == 1.0 for x in out)

    def test_top_k_zero_returns_empty_not_all(self):
        assert self._reranker([0.1, 0.9, 0.5]).rerank("q", self._results(), top_k=0) == []

    def test_top_k_truncates(self):
        out = self._reranker([0.1, 0.9, 0.5]).rerank("q", self._results(), top_k=2)
        assert [x.chunk.id for x in out] == ["c1", "c2"]


class TestDenseRetriever:
    @pytest.fixture
    def retriever(self, sample_chunks):
        embedder = MagicMock()
        embedder.embed_query.return_value = np.random.rand(384).astype(np.float32)
        store = MagicMock()
        from scinexusrag.storage import SearchResult

        store.search.return_value = [
            SearchResult(chunk=sample_chunks[1], score=0.9),
            SearchResult(chunk=sample_chunks[0], score=0.7),
        ]
        return DenseRetriever(embedder, store)

    def test_retrieve_passes_through_store(self, retriever):
        results = retriever.retrieve("deep learning", top_k=2)
        assert [r.chunk.id for r in results] == ["chunk2", "chunk1"]
        assert all(r.source == "dense" for r in results)

    def test_threshold_filters_low_scores(self, retriever):
        results = retriever.retrieve_with_threshold("q", top_k=5, min_score=0.8)
        assert len(results) == 1 and results[0].score >= 0.8


class TestHybridRetriever:
    @pytest.fixture
    def dense(self, sample_chunks):
        m = MagicMock()
        m.retrieve.return_value = [
            RetrievalResult(sample_chunks[0], 0.9, "dense"),
            RetrievalResult(sample_chunks[1], 0.8, "dense"),
            RetrievalResult(sample_chunks[2], 0.6, "dense"),
        ]
        return m

    @pytest.fixture
    def sparse(self, sample_chunks):
        m = MagicMock()
        m.retrieve.return_value = [
            RetrievalResult(sample_chunks[1], 0.85, "sparse"),
            RetrievalResult(sample_chunks[3], 0.7, "sparse"),
            RetrievalResult(sample_chunks[0], 0.5, "sparse"),
        ]
        return m

    def test_fusion_promotes_chunk_in_both_lists(self, dense, sparse):
        hybrid = HybridRetriever(dense, sparse, 0.5, 0.5)
        results = hybrid.retrieve("q", top_k=3)
        assert results[0].chunk.id == "chunk2"
        assert all(r.source == "hybrid" for r in results)

    def test_negative_weight_rejected(self, dense, sparse):
        with pytest.raises(ValueError):
            HybridRetriever(dense, sparse, dense_weight=-0.1, sparse_weight=0.3)

    def test_empty_results(self, dense, sparse):
        dense.retrieve.return_value = []
        sparse.retrieve.return_value = []
        assert HybridRetriever(dense, sparse).retrieve("q", top_k=5) == []


class TestSpladeEmptyCorpus:
    def test_empty_corpus_does_not_crash(self):
        from scinexusrag.retrieval.splade import SpladeRetriever

        r = SpladeRetriever([])
        assert r.retrieve("query") == []
