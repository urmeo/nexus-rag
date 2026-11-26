"""Retrieval checks."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from scipy import sparse
from support import PIN, make_config

from scinexusrag.eval.splade import DEFAULT_MODEL, SpladeRetriever
from scinexusrag.ingestion.embedder import EmbeddingConfigError
from scinexusrag.models import Chunk
from scinexusrag.retrieval.base import RetrievalResult
from scinexusrag.retrieval.corrective import CorrectiveRetriever
from scinexusrag.retrieval.dense import DenseRetriever
from scinexusrag.retrieval.hybrid import AdaptiveHybridRetriever, HybridRetriever, rrf_fuse
from scinexusrag.retrieval.reranker import Reranker
from scinexusrag.retrieval.sparse import BM25Retriever
from scinexusrag.storage.vector_store import SearchResult, VectorStoreError


@pytest.mark.parametrize(
    "options", [{"query": ""}, {"query": " \t\n"}, {"top_k": 0}, {"top_k": -1}]
)
def test_empty_corrective_requests_do_not_call_base(options):
    base = Mock()
    retriever = CorrectiveRetriever(base)

    assert retriever.retrieve_traced(**{"query": "query", **options}) == ([], False)
    base.retrieve_with_dense_top.assert_not_called()
    base.retrieve.assert_not_called()


def test_invalid_corrective_depth_does_not_call_base():
    base = Mock()
    retriever = CorrectiveRetriever(base)

    with pytest.raises(ValueError, match="depth"):
        retriever.retrieve_traced("query", depth=0)

    base.retrieve_with_dense_top.assert_not_called()
    base.retrieve.assert_not_called()


@pytest.mark.parametrize("tau", [-0.1, 1.1, float("nan"), float("inf"), 10**1000, -(10**1000)])
def test_invalid_corrective_thresholds_fail_during_construction(tau):
    with pytest.raises(ValueError, match="threshold"):
        CorrectiveRetriever(Mock(), tau=tau)


@pytest.mark.parametrize("option", ["feedback_docs", "feedback_terms"])
@pytest.mark.parametrize("value", [0, -1])
def test_feedback_sizes_require_positive_integers(option, value):
    with pytest.raises(ValueError, match=option):
        CorrectiveRetriever(Mock(), **{option: value})


@pytest.mark.parametrize(
    "query,content",
    [
        ("kinase inhibition", "kinase inhibition kinase"),
        (" kinase inhibition ", "kinase inhibition"),
        ("kinase inhibition", "the and with"),
    ],
)
def test_feedback_without_new_terms_does_not_repeat_the_query(query, content):
    base = Mock()
    first = [RetrievalResult(Chunk(id="first", content=content, document_id="paper"), 0.7)]
    base.retrieve_with_dense_top.return_value = first, 0.1
    base.sparse = BM25Retriever()

    results, triggered = CorrectiveRetriever(base).retrieve_traced(query)

    assert results == first
    assert results[0] is first[0]
    assert triggered is False
    base.retrieve.assert_not_called()


@pytest.mark.parametrize("query,top_k", [("evidence", 0), ("evidence", -1), ("", 5), (" \n", 5)])
def test_empty_requests_skip_model_and_storage_access(retriever, query, top_k):
    assert retriever.retrieve(query, top_k=top_k) == []

    retriever.embedder.embed_query.assert_not_called()
    retriever.vector_store.validate_embedding_config.assert_not_called()
    retriever.vector_store.search.assert_not_called()


def test_nonempty_query_and_result_order_are_preserved(retriever):
    chunks = [Chunk(str(i), f"Evidence {i}.", "paper") for i in range(2)]
    retriever.vector_store.search.return_value = [
        SearchResult(chunks[0], 0.8),
        SearchResult(chunks[1], 0.5),
    ]
    query = " evidence \n"

    results = retriever.retrieve(query, top_k=2)

    retriever.embedder.embed_query.assert_called_once_with(query)
    assert [(result.chunk, result.score, result.source) for result in results] == [
        (chunks[0], 0.8, "dense"),
        (chunks[1], 0.5, "dense"),
    ]


def test_search_failure_propagates(retriever):
    failure = VectorStoreError("storage unavailable")
    retriever.vector_store.search.side_effect = failure

    with pytest.raises(VectorStoreError) as caught:
        retriever.retrieve("evidence")

    assert caught.value is failure


def _fusion_result(chunk_id):
    return RetrievalResult(Chunk(id=chunk_id, content=chunk_id, document_id="paper"), 0.5)


@pytest.mark.parametrize("weight", [-1, -0.1, float("nan"), float("inf")])
def test_invalid_fusion_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="weights"):
        rrf_fuse([[_fusion_result("first")]], [weight])
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(Mock(), Mock(), dense_weight=weight)


@pytest.mark.parametrize("k", [-1])
def test_invalid_rrf_constants_are_rejected(k):
    with pytest.raises(ValueError, match="RRF k"):
        rrf_fuse([[_fusion_result("first")]], [1.0], k=k)
    with pytest.raises(ValueError, match="RRF k"):
        HybridRetriever(Mock(), Mock(), rrf_k=k)


def test_fusion_requires_one_weight_per_list():
    with pytest.raises(ValueError, match="one fusion weight"):
        rrf_fuse([[_fusion_result("first")]], [])


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_fusion_limits_return_no_results(top_k):
    assert rrf_fuse([[_fusion_result("first"), _fusion_result("second")]], [1.0], top_k=top_k) == []


def test_zero_rrf_constant_uses_reciprocal_rank():
    results = rrf_fuse([[_fusion_result("first"), _fusion_result("second")]], [1.0], k=0)

    assert [result.score for result in results] == [1.0, 0.5]


def test_duplicate_chunk_does_not_gain_extra_rank_votes():
    first = _fusion_result("first")
    second = _fusion_result("second")

    clean = rrf_fuse([[first, second], [second, first]], [0.5, 0.5])
    repeated = rrf_fuse([[first, first, second], [second, first]], [0.5, 0.5])

    assert repeated == clean


def test_first_occurrence_supplies_chunk_content_when_results_repeat():
    first = _fusion_result("first")
    conflicting = _fusion_result("first")
    conflicting.chunk.content = "Conflicting repeated content."

    result = rrf_fuse([[first, conflicting]], [1.0])[0]

    assert result.chunk is first.chunk


@pytest.mark.parametrize("query", ["", " \t\n"])
def test_empty_hybrid_queries_do_not_call_backends(query):
    dense, sparse = Mock(), Mock()
    retriever = HybridRetriever(dense, sparse)

    assert retriever.retrieve_with_dense_top(query) == ([], 0.0)
    assert retriever.retrieve(query) == []
    dense.retrieve.assert_not_called()
    sparse.retrieve.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_hybrid_limit_does_not_call_backends(top_k):
    dense, sparse = Mock(), Mock()

    assert HybridRetriever(dense, sparse).retrieve_with_dense_top("query", top_k=top_k) == (
        [],
        0.0,
    )
    dense.retrieve.assert_not_called()
    sparse.retrieve.assert_not_called()


def test_invalid_hybrid_depth_does_not_call_backends():
    dense, sparse = Mock(), Mock()
    retriever = HybridRetriever(dense, sparse)

    with pytest.raises(ValueError, match="depth"):
        retriever.retrieve("query", depth=0)

    dense.retrieve.assert_not_called()
    sparse.retrieve.assert_not_called()


def test_explicit_candidate_depth_and_query_text_are_preserved():
    dense, sparse = Mock(), Mock()
    dense.retrieve.return_value = [_fusion_result("dense")]
    sparse.retrieve.return_value = [_fusion_result("sparse")]

    results = HybridRetriever(dense, sparse).retrieve(" query ", top_k=10, depth=1)

    assert len(results) == 2
    dense.retrieve.assert_called_once_with(" query ", 1)
    sparse.retrieve.assert_called_once_with(" query ", 1)


@pytest.mark.parametrize("shift", [-0.1, float("nan"), float("inf")])
def test_invalid_adaptive_shifts_fail_during_construction(shift):
    with pytest.raises(ValueError, match="shift"):
        AdaptiveHybridRetriever(Mock(), Mock(), shift=shift)


@pytest.mark.parametrize(
    "query,weights",
    [
        ("kinase inhibition", (0.3, 0.7)),
        ("how does the treatment change the observed rate of recovery over time", (0.7, 0.3)),
        ("kinase inhibition in human cells", (0.5, 0.5)),
    ],
)
def test_default_adaptive_weight_rules_remain_unchanged(query, weights):
    retriever = AdaptiveHybridRetriever(Mock(), Mock())

    assert retriever._weights(query) == weights


@pytest.fixture
def reranker():
    instance = Reranker()
    instance._model = Mock()
    return instance


def test_concurrent_first_reranker_load_constructs_once(monkeypatch, observed_lock):
    reranker = Reranker()
    lock = observed_lock
    reranker._model_lock = lock
    start = Barrier(3)
    constructing, release = Event(), Event()
    model = object()

    def construct(*_args, **_kwargs):
        constructing.set()
        assert release.wait(timeout=5)
        return model

    factory = Mock(side_effect=construct)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=factory))

    def load():
        start.wait(timeout=5)
        return reranker.model

    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(load) for _ in range(3)]
        try:
            assert constructing.wait(timeout=5)
            assert len({lock.attempts.get(timeout=5) for _ in range(3)}) == 3
        finally:
            release.set()
        assert all(future.result(timeout=5) is model for future in futures)

    factory.assert_called_once_with(reranker.model_name, device=None, revision=reranker.revision)


def test_failed_reranker_load_can_retry(monkeypatch):
    model = object()
    factory = Mock(side_effect=[RuntimeError("load failed"), model])
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=factory))
    reranker = Reranker()

    with pytest.raises(RuntimeError, match="load failed"):
        _ = reranker.model

    assert reranker._model is None
    assert reranker.model is model
    assert reranker.model is model
    assert factory.call_count == 2


@pytest.mark.parametrize("stage", ["to", "eval"])
def test_failed_splade_initialization_does_not_cache_partial_model(monkeypatch, stage):
    tokenizer = Mock()
    model = Mock()
    model.to.return_value = model
    getattr(model, stage).side_effect = [RuntimeError("model failed"), model]
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForMaskedLM=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    monkeypatch.setattr(SpladeRetriever, "_encode", lambda *_: sparse.csr_matrix((0, 0)))
    retriever = SpladeRetriever([])

    with pytest.raises(RuntimeError, match="model failed"):
        retriever._load()

    assert retriever._model is None
    assert retriever._tok is None
    retriever._load()
    retriever._load()
    assert retriever._model is model
    assert retriever._tok is tokenizer
    assert model_loader.call_count == 2


def candidates(count=3):
    return [
        RetrievalResult(Chunk(str(index), f"passage {index}", "doc"), 0.5, "hybrid")
        for index in range(count)
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"model_name": ""},
        {"device": "automatic"},
        {"batch_size": 0},
        {"revision": ""},
    ],
)
def test_invalid_reranker_settings_fail_without_load(monkeypatch, options):
    factory = Mock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=factory))

    with pytest.raises(ValueError):
        Reranker(**options)

    factory.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, -99])
def test_nonpositive_reranking_depth_skips_model(reranker, top_k):
    assert reranker.rerank("query", candidates(), top_k=top_k) == []
    reranker._model.predict.assert_not_called()


@pytest.mark.parametrize("query", ["", " \n\t"])
def test_blank_reranking_query_skips_model(reranker, query):
    assert reranker.rerank(query, candidates()) == []
    reranker._model.predict.assert_not_called()


@pytest.mark.parametrize(
    "scores",
    [
        [0.1, 0.2],
        [0.1, np.nan, 0.3],
        [0.1, np.inf, 0.3],
        [[0.1, 0.2]] * 3,
        ["0.1", "0.2", "0.3"],
        [1j, 2j, 3j],
    ],
)
def test_malformed_reranker_scores_are_rejected(scores):
    reranker = Reranker()
    reranker._model = Mock(predict=Mock(return_value=scores))

    with pytest.raises(ValueError, match="reranker"):
        reranker.rerank("query", candidates())


def test_column_reranker_scores_have_scalar_outputs_and_stable_ties():
    reranker = Reranker(batch_size=2)
    reranker._model = Mock(predict=Mock(return_value=np.array([[0.2], [0.8], [0.8]])))
    results = candidates()

    reranked = reranker.rerank("query", results)

    assert [result.chunk.id for result in reranked] == ["1", "2", "0"]
    assert [result.score for result in reranked] == [1.0, 1.0, 0.0]
    assert all(type(result.score) is float for result in reranked)
    assert all(result.source == "hybrid+rerank" for result in reranked)
    assert [result.score for result in results] == [0.5] * 3
    reranker._model.predict.assert_called_once_with(
        [("query", f"passage {index}") for index in range(3)], batch_size=2
    )


def test_large_finite_reranker_scores_normalize_without_overflow():
    reranker = Reranker()
    reranker._model = Mock(predict=Mock(return_value=np.array([-1e308, 0, 1e308])))

    with np.errstate(all="raise"):
        reranked = reranker.rerank("query", candidates())

    assert [result.score for result in reranked] == [1.0, 0.5, 0.0]


def test_splade_index_keeps_a_snapshot_of_caller_chunks(monkeypatch):
    encode = Mock(
        side_effect=[
            sparse.csr_matrix([[1.0], [3.0]]),
            sparse.csr_matrix([[2.0]]),
            sparse.csr_matrix([[2.0]]),
        ]
    )
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    chunks = [
        Chunk("first", "first passage", "first-doc"),
        Chunk("second", "second passage", "second-doc", metadata={"name": "original"}),
    ]
    retriever = SpladeRetriever(chunks)
    chunks[1].content = "replacement passage"
    chunks[1].metadata["name"] = "replacement"
    chunks.append(Chunk("third", "third passage", "third-doc"))

    results = retriever.retrieve("question")

    assert [result.chunk.id for result in results] == ["second", "first"]
    assert results[0].chunk.content == "second passage"
    assert results[0].chunk.metadata["name"] == "original"
    assert results[0].score == 6.0
    assert encode.call_args_list[0].args == (["first passage", "second passage"],)
    results[0].chunk.content = "changed result"
    results[0].chunk.metadata["name"] = "changed result"

    repeated = retriever.retrieve("question")

    assert repeated[0].chunk.content == "second passage"
    assert repeated[0].chunk.metadata["name"] == "original"


@pytest.mark.parametrize(
    "options",
    [
        {"model_name": ""},
        {"model_name": None},
        {"device": "automatic"},
        {"batch_size": False},
        {"batch_size": 1.5},
        {"max_length": True},
        {"max_length": 1.5},
        {"revision": ""},
        {"revision": 1},
        {"chunks": None},
        {"chunks": ["passage"]},
        {"chunks": [Chunk("doc", None, "doc")]},
    ],
)
def test_invalid_splade_settings_fail_before_encoding(monkeypatch, options):
    encode = Mock()
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    settings = {"chunks": [], **options}

    with pytest.raises(ValueError):
        SpladeRetriever(**settings)

    encode.assert_not_called()


@pytest.mark.parametrize("top_k", [True, False, 1.5, "1", None])
def test_invalid_splade_depth_fails_without_encoding(monkeypatch, top_k):
    encode = Mock(return_value=sparse.csr_matrix([[1.0]]))
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    retriever = SpladeRetriever([Chunk("doc", "passage", "doc")])
    encode.reset_mock()

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("question", top_k=top_k)

    encode.assert_not_called()


@pytest.mark.parametrize("query", ["", " \n\t"])
def test_blank_splade_query_skips_encoding(monkeypatch, query):
    encode = Mock(return_value=sparse.csr_matrix([[1.0]]))
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    retriever = SpladeRetriever([Chunk("doc", "passage", "doc")])
    encode.reset_mock()

    assert retriever.retrieve(query) == []
    encode.assert_not_called()


@pytest.mark.parametrize("query", [None, 1, ["question"]])
def test_nontext_splade_query_fails_without_encoding(monkeypatch, query):
    encode = Mock(return_value=sparse.csr_matrix([[1.0]]))
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    retriever = SpladeRetriever([Chunk("doc", "passage", "doc")])
    encode.reset_mock()

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(query)

    encode.assert_not_called()


def splade_with_logits(logits, mask):
    import torch

    class TensorBatch(dict):
        def to(self, device):
            return self

    batch = TensorBatch(attention_mask=torch.tensor(mask))
    retriever = SpladeRetriever([])
    retriever._tok = Mock(return_value=batch)
    retriever._model = Mock(return_value=SimpleNamespace(logits=torch.tensor(logits)))
    return retriever


def test_splade_pooling_excludes_padding_and_preserves_encoding_defaults():
    retriever = splade_with_logits([[[1.0, -1.0, 3.0], [100.0, 100.0, 100.0]]], [[1, 0]])

    vectors = retriever._encode(["passage"])

    np.testing.assert_allclose(vectors.toarray(), [[np.log(2), 0, np.log(4)]])
    assert sparse.isspmatrix_csr(vectors)
    retriever._tok.assert_called_once_with(
        ["passage"], padding=True, truncation=True, max_length=256, return_tensors="pt"
    )


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_splade_weights_are_rejected(value):
    retriever = splade_with_logits([[[1.0, value, 3.0]]], [[1]])

    with pytest.raises(ValueError, match="finite"):
        retriever._encode(["passage"])


@pytest.mark.parametrize(
    "logits,mask",
    [
        ([[1.0, 2.0, 3.0]], [[1]]),
        ([[[1.0], [2.0]]], [[1]]),
        ([[[1.0]], [[2.0]]], [[1], [1]]),
    ],
)
def test_splade_logits_must_match_tokenized_batch(logits, mask):
    retriever = splade_with_logits(logits, mask)

    with pytest.raises(ValueError, match="align"):
        retriever._encode(["passage"])


@pytest.mark.parametrize(
    "query_matrix", [sparse.csr_matrix([[1.0, 2.0]]), sparse.csr_matrix((0, 1))]
)
def test_splade_query_vectors_must_align_with_index(monkeypatch, query_matrix):
    encode = Mock(side_effect=[sparse.csr_matrix([[1.0]]), query_matrix])
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    retriever = SpladeRetriever([Chunk("doc", "passage", "doc")])

    with pytest.raises(ValueError, match="align"):
        retriever.retrieve("question")


def test_nonfinite_splade_dot_products_are_rejected(monkeypatch):
    encode = Mock(side_effect=[sparse.csr_matrix([[1e30]], dtype=np.float32)] * 2)
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    retriever = SpladeRetriever([Chunk("doc", "passage", "doc")])

    with pytest.raises(ValueError, match="finite"):
        retriever.retrieve("question")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "other/model"),
        ("revision", "different-revision"),
        ("device", "cuda"),
        ("max_length", 512),
    ],
)
def test_splade_queries_cannot_use_changed_encoding_settings(monkeypatch, field, value):
    encode = Mock(return_value=sparse.csr_matrix([[1.0]]))
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    retriever = SpladeRetriever([Chunk("doc", "passage", "doc")])
    setattr(retriever, field, value)
    encode.reset_mock()

    with pytest.raises(ValueError, match="settings changed"):
        retriever.retrieve("question")

    encode.assert_not_called()


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
    def test_rebuild_indexes_all_chunks(self, sample_chunks):
        r = BM25Retriever()
        assert r.rebuild(sample_chunks) == 5
        assert r.count() == 5

    def test_retrieve_ranks_the_matching_chunk_first(self, sample_chunks):
        r = BM25Retriever()
        r.rebuild(sample_chunks)
        results = r.retrieve("neural networks deep learning", top_k=3)
        assert results[0].chunk.id == "chunk2"
        assert all(res.source == "sparse" for res in results)
        assert all(0 <= res.score <= 1 for res in results)

    def test_empty_index_and_empty_query(self, sample_chunks):
        assert BM25Retriever().retrieve("anything", top_k=5) == []
        r = BM25Retriever()
        r.rebuild(sample_chunks)
        assert r.retrieve("", top_k=5) == []

    def test_single_matching_passage_with_negative_idf_is_returned(self):
        retriever = BM25Retriever()
        retriever.rebuild(
            [Chunk(id="match", document_id="paper", content="kinase tumor inhibition")]
        )

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
        retriever.rebuild(
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
        retriever.rebuild([Chunk(id="one", document_id="paper", content="kinase tumor inhibition")])

        assert retriever.retrieve("cardiac disease") == []

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_nonpositive_depth_returns_no_passages(self, sample_chunks, top_k):
        retriever = BM25Retriever()
        retriever.rebuild(sample_chunks)

        assert retriever.retrieve("neural networks", top_k=top_k) == []

    def test_tokenize_drops_stopwords(self):
        tokens = BM25Retriever().tokenize("the quick brown fox is jumping")
        assert "the" not in tokens and "is" not in tokens
        assert "quick" in tokens and "brown" in tokens

    def test_incremental_add_and_remove(self, sample_chunks):
        r = BM25Retriever()
        r.rebuild(sample_chunks[:3])
        r.add(sample_chunks[3:])
        assert r.count() == 5
        assert r.remove({"chunk1", "chunk2"}) == 2
        assert r.count() == 3

    def test_incremental_add_is_idempotent_by_id(self, sample_chunks):

        r = BM25Retriever()
        r.rebuild(sample_chunks[:3])
        r.add(sample_chunks[:3])
        assert r.count() == 3
        r.add(sample_chunks[2:])
        assert r.count() == 5

    @pytest.mark.parametrize("preload", [False, True])
    def test_incremental_add_deduplicates_the_new_batch(self, sample_chunks, preload):
        retriever = BM25Retriever()
        if preload:
            retriever.rebuild(sample_chunks[:1])

        assert retriever.add([*sample_chunks[:2], *sample_chunks[:2]]) == 2
        assert retriever.count() == 2
        assert retriever.remove({sample_chunks[0].id}) == 1
        assert retriever.count() == 1

    def test_add_survives_all_empty_tokenization(self):

        chunks = [
            Chunk(id=f"c{i}", content=c, document_id="d")
            for i, c in enumerate(["the a is", "of to in", "x y z"])
        ]
        r = BM25Retriever()
        assert r.rebuild(chunks) == 3
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

    def test_top_k_truncates(self):
        out = self._reranker([0.1, 0.9, 0.5]).rerank("q", self._results(), top_k=2)
        assert [x.chunk.id for x in out] == ["c1", "c2"]


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

    def test_empty_results(self, dense, sparse):
        dense.retrieve.return_value = []
        sparse.retrieve.return_value = []
        assert HybridRetriever(dense, sparse).retrieve("q", top_k=5) == []

    def test_fusion_retains_lexical_match_with_zero_bm25_score(self):
        sparse = BM25Retriever()
        sparse.rebuild([Chunk(id="match", document_id="paper", content="kinase tumor inhibition")])
        assert sparse.retrieve("kinase")[0].score == 0.0
        dense = MagicMock()
        dense.retrieve.return_value = []

        results = HybridRetriever(dense, sparse).retrieve("kinase", top_k=1)

        assert [result.chunk.id for result in results] == ["match"]
        assert results[0].source == "hybrid"
        assert results[0].score == 1.0


class TestSpladeEmptyCorpus:
    def test_empty_corpus_does_not_crash(self):

        r = SpladeRetriever([])
        assert r.retrieve("query") == []


def _ranked_result(cid: str, score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(chunk=Chunk(id=cid, content=cid, document_id=cid), score=score)


def test_rrf_fuse_orders_by_summed_reciprocal_rank():
    a = [_ranked_result("c1"), _ranked_result("c2"), _ranked_result("c3")]
    b = [_ranked_result("c2"), _ranked_result("c3"), _ranked_result("c4")]
    fused = rrf_fuse([a, b], [1.0, 1.0], k=60)

    assert [r.chunk.id for r in fused] == ["c2", "c3", "c1", "c4"]
    assert fused[0].score == 1.0


def test_rrf_fuse_weights_shift_ranking():
    a = [_ranked_result("c1"), _ranked_result("c2")]
    b = [_ranked_result("c2"), _ranked_result("c1")]
    sparse_heavy = rrf_fuse([a, b], [0.1, 0.9], k=60)
    assert sparse_heavy[0].chunk.id == "c2"


def test_rrf_fuse_zero_weights_return_zero_scores_in_stable_order():
    first = [_ranked_result("c1"), _ranked_result("c2")]
    second = [_ranked_result("c2"), _ranked_result("c3")]

    fused = rrf_fuse([first, second], [0.0, 0.0], top_k=2)

    assert [result.chunk.id for result in fused] == ["c1", "c2"]
    assert all(result.score == 0.0 for result in fused)


class _FakeDense:
    def __init__(self, top_score: float):
        self.top_score = top_score
        self.calls = 0

    def retrieve(self, query, top_k=5):
        self.calls += 1
        return [_ranked_result("d1", self.top_score)]


class _FakeBase:
    def __init__(self, top_score: float):
        self.dense = _FakeDense(top_score)
        self.rrf_k = 60
        self.calls: list[str] = []

    def tokenize_sparse(self, text):
        return [t for t in text.lower().split() if len(t) > 1]

    @property
    def sparse(self):
        return self

    def tokenize(self, text):
        return self.tokenize_sparse(text)

    def retrieve(self, query, top_k=10, depth=50):
        self.calls.append(query)
        return [_ranked_result("d1", 0.9), _ranked_result("d2", 0.8)]

    def retrieve_with_dense_top(self, query, top_k=10, depth=50):

        self.calls.append(query)
        return [_ranked_result("d1", 0.9), _ranked_result("d2", 0.8)], self.dense.retrieve(
            query, 1
        )[0].score


def test_corrective_skips_when_confident():
    base = _FakeBase(top_score=0.9)
    cr = CorrectiveRetriever(base, tau=0.55)
    _, triggered = cr.retrieve_traced("kinase inhibits tumor growth", top_k=2)
    assert triggered is False
    assert len(base.calls) == 1
    assert base.dense.calls == 1


def test_corrective_expands_when_weak():
    base = _FakeBase(top_score=0.2)
    cr = CorrectiveRetriever(base, tau=0.55, feedback_terms=3)
    _, triggered = cr.retrieve_traced("kinase inhibits tumor growth", top_k=2)
    assert triggered is True
    assert len(base.calls) == 2
    assert base.calls[1] != base.calls[0]


def test_corrective_disabled_skips_second_pass():

    base = _FakeBase(top_score=0.2)
    cr = CorrectiveRetriever(base, tau=0.55, enabled=False)
    _, triggered = cr.retrieve_traced("kinase inhibits tumor growth", top_k=2)
    assert triggered is False
    assert len(base.calls) == 1


def _content_result(text: str) -> RetrievalResult:
    return RetrievalResult(chunk=Chunk(id="x", content=text, document_id="d"), score=0.5)


def test_expand_appends_frequent_non_query_terms():

    base = _FakeBase(top_score=0.2)
    cr = CorrectiveRetriever(base, feedback_terms=2)
    results = [
        _content_result("apoptosis apoptosis apoptosis signaling pathway"),
        _content_result("signaling pathway pathway"),
    ]
    added = cr.expand("kinase inhibits", results).split()[2:]

    assert added == ["apoptosis", "pathway"]
    assert "kinase" not in added and "inhibits" not in added
    assert "signaling" not in added


def test_expand_respects_feedback_terms_cap():
    base = _FakeBase(top_score=0.2)
    cr = CorrectiveRetriever(base, feedback_terms=1)
    added = cr.expand("kinase", [_content_result("apoptosis apoptosis signaling")]).split()[1:]
    assert added == ["apoptosis"]


def test_hybrid_retrieve_with_dense_top_single_pass():

    class _Dense:
        def __init__(self):
            self.calls = 0

        def retrieve(self, q, k):
            self.calls += 1
            return [_ranked_result("d1", 0.72)]

    class _Sparse:
        def retrieve(self, q, k):
            return [_ranked_result("d1", 0.4)]

    dense = _Dense()
    fused, top = HybridRetriever(dense, _Sparse()).retrieve_with_dense_top("q", top_k=5, depth=10)
    assert dense.calls == 1
    assert top == 0.72
    assert fused


def test_looks_technical_ignores_trailing_punctuation():
    from scinexusrag.retrieval.hybrid import _looks_technical

    assert _looks_technical("classification.") is False
    assert _looks_technical("immunodeficiency,") is True
    assert _looks_technical("TF-IDF") is True


def test_index_retains_its_own_corpus_when_the_input_list_changes():
    chunks = [
        Chunk(id="first", document_id="paper", content="kinase inhibition"),
        Chunk(id="second", document_id="paper", content="cardiac disease"),
    ]
    retriever = BM25Retriever()
    retriever.rebuild(chunks)
    chunks.clear()

    assert retriever.count() == 2
    assert [result.chunk.id for result in retriever.retrieve("cardiac")] == ["second"]


def test_index_copies_passage_text_identity_and_nested_metadata():
    chunk = Chunk(
        id="original",
        document_id="paper",
        content="kinase inhibition",
        metadata={"units": ["mA"]},
    )
    retriever = BM25Retriever()
    retriever.rebuild([chunk])
    chunk.id = "changed"
    chunk.content = "cardiac disease"
    chunk.metadata["units"].append("mV")

    result = retriever.retrieve("kinase")[0]

    assert result.chunk.id == "original"
    assert result.chunk.content == "kinase inhibition"
    assert result.chunk.metadata == {"units": ["mA"]}
    assert retriever.retrieve("cardiac") == []


def test_mutating_a_result_does_not_change_future_results_or_removal():
    retriever = BM25Retriever()
    retriever.rebuild(
        [
            Chunk(
                id="original",
                document_id="paper",
                content="kinase inhibition",
                metadata={"units": ["mA"]},
            )
        ]
    )
    result = retriever.retrieve("kinase")[0]
    result.chunk.id = "changed"
    result.chunk.content = "cardiac disease"
    result.chunk.metadata["units"].append("mV")

    next_result = retriever.retrieve("kinase")[0]

    assert next_result.chunk.id == "original"
    assert next_result.chunk.content == "kinase inhibition"
    assert next_result.chunk.metadata == {"units": ["mA"]}
    assert retriever.remove({"original"}) == 1
    assert retriever.count() == 0


def test_incremental_rebuild_copies_new_chunks():
    retriever = BM25Retriever()
    retriever.rebuild([Chunk(id="first", document_id="paper", content="kinase inhibition")])
    fresh = Chunk(id="second", document_id="paper", content="cardiac disease")
    retriever.add([fresh])
    fresh.id = "changed"
    fresh.content = "unrelated text"

    assert [result.chunk.id for result in retriever.retrieve("cardiac")] == ["second"]


def test_empty_corpus_replacement_clears_old_passages():
    retriever = BM25Retriever()
    retriever.rebuild([Chunk(id="old", document_id="paper", content="kinase inhibition")])

    assert retriever.rebuild([]) == 0
    assert retriever.count() == 0
    assert retriever.retrieve("kinase") == []


def test_empty_incremental_batch_keeps_existing_passages():
    retriever = BM25Retriever()
    retriever.rebuild([Chunk(id="old", document_id="paper", content="kinase inhibition")])

    assert retriever.add([]) == 1
    assert retriever.count() == 1
    assert [result.chunk.id for result in retriever.retrieve("kinase")] == ["old"]


def test_duplicate_passage_ids_are_rejected_without_changing_the_old_index():
    retriever = BM25Retriever()
    retriever.rebuild([Chunk(id="old", document_id="paper", content="kinase inhibition")])
    duplicate_ids = [
        Chunk(id="duplicate", document_id="paper", content="cardiac disease"),
        Chunk(id="duplicate", document_id="paper", content="neural response"),
    ]

    with pytest.raises(ValueError, match="IDs must be unique"):
        retriever.rebuild(duplicate_ids)

    assert retriever.count() == 1
    assert [result.chunk.id for result in retriever.retrieve("kinase")] == ["old"]
    assert retriever.retrieve("cardiac") == []


@pytest.mark.parametrize("operation", ["replace", "incremental", "remove"])
def test_failed_rebuild_keeps_the_previous_index(monkeypatch, operation):
    retriever = BM25Retriever()
    retriever.rebuild(
        [
            Chunk(id="first", document_id="paper", content="kinase inhibition"),
            Chunk(id="second", document_id="paper", content="cardiac disease"),
        ]
    )
    before = retriever.retrieve("kinase cardiac")
    new_chunks = [Chunk(id="new", document_id="paper", content="neural response")]

    def fail(_corpus):
        raise RuntimeError("index construction failed")

    monkeypatch.setattr("scinexusrag.retrieval.sparse.BM25Okapi", fail)

    with pytest.raises(RuntimeError, match="index construction failed"):
        if operation == "replace":
            retriever.rebuild(new_chunks)
        elif operation == "incremental":
            retriever.add(new_chunks)
        else:
            retriever.remove({"second"})

    assert retriever.count() == 2
    assert retriever.retrieve("kinase cardiac") == before


@pytest.mark.parametrize("chunk_ids", [set(), {"absent"}])
def test_removing_no_passages_does_not_rebuild(monkeypatch, chunk_ids):
    retriever = BM25Retriever()
    retriever.rebuild([Chunk(id="old", document_id="paper", content="kinase inhibition")])

    def fail(_corpus):
        raise AssertionError("unchanged index must not be rebuilt")

    monkeypatch.setattr("scinexusrag.retrieval.sparse.BM25Okapi", fail)

    assert retriever.remove(chunk_ids) == 0
    assert retriever.count() == 1
    assert [result.chunk.id for result in retriever.retrieve("kinase")] == ["old"]


@pytest.mark.parametrize("query,top_k", [("kinase", 0), ("kinase", -2), ("", 5), (" \n\t ", 5)])
def test_empty_requests_do_not_tokenize_or_score_the_index(monkeypatch, query, top_k):
    retriever = BM25Retriever()
    retriever.rebuild([Chunk(id="original", document_id="paper", content="kinase inhibition")])

    def fail(_query):
        raise AssertionError("empty request must not be tokenized")

    monkeypatch.setattr(retriever, "tokenize", fail)

    assert retriever.retrieve(query, top_k=top_k) == []


@pytest.mark.parametrize("method", ["rebuild", "add"])
@pytest.mark.parametrize(
    "chunks",
    [[Chunk("", "passage", "paper")], [Chunk(" \t ", "passage", "paper")]],
)
def test_blank_passage_ids_never_change_the_existing_index(method, chunks):
    retriever = BM25Retriever()
    retriever.rebuild(
        [
            Chunk("first", "kinase inhibition", "paper"),
            Chunk("second", "cardiac disease", "paper"),
        ]
    )
    before = retriever.retrieve("kinase cardiac")

    with pytest.raises(ValueError, match="nonempty IDs"):
        getattr(retriever, method)(chunks)

    assert retriever.count() == 2
    assert retriever.retrieve("kinase cardiac") == before


@pytest.mark.parametrize("field", ["batch_size", "max_length"])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_encoding_parameters_fail_before_loading(monkeypatch, field, value):
    encode = Mock()
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)

    with pytest.raises(ValueError, match="positive"):
        SpladeRetriever([], **{field: value})
    encode.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, -99])
def test_nonpositive_depth_skips_query_encoding(monkeypatch, top_k):
    encode = Mock(return_value=sparse.csr_matrix([[1.0]]))
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    retriever = SpladeRetriever([Chunk(id="doc", content="Evidence", document_id="doc")])

    assert retriever.retrieve("query", top_k=top_k) == []
    assert encode.call_count == 1


@pytest.mark.parametrize("top_k,expected", [(1, [1]), (2, [1, 2]), (99, [1, 2, 3, 0, 4])])
def test_tied_scores_keep_corpus_order_at_depth_boundary(monkeypatch, top_k, expected):
    encode = Mock(
        side_effect=[
            sparse.csr_matrix([[0.3], [0.7], [0.7], [0.7], [0.1]]),
            sparse.csr_matrix([[2.0]]),
        ]
    )
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    chunks = [Chunk(id=str(i), content=str(i), document_id=str(i)) for i in range(5)]
    retriever = SpladeRetriever(chunks)
    results = retriever.retrieve("query", top_k=top_k)

    assert [result.chunk.id for result in results] == [str(i) for i in expected]
    assert results[0].score == pytest.approx(1.4)


@pytest.mark.parametrize(
    "model_name,revision,expected",
    [
        (DEFAULT_MODEL, None, PIN),
        (DEFAULT_MODEL, "explicit-pin", "explicit-pin"),
        ("custom/sparse-model", "custom-pin", "custom-pin"),
        ("custom/sparse-model", None, None),
    ],
)
def test_tokenizer_and_model_share_selected_revision(monkeypatch, model_name, revision, expected):
    tokenizer = Mock()
    model = Mock()
    model.to.return_value = model
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForMaskedLM=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    monkeypatch.setattr(SpladeRetriever, "_encode", lambda *_: sparse.csr_matrix((0, 0)))
    retriever = SpladeRetriever([], model_name=model_name, revision=revision)
    retriever._load()
    retriever._load()

    assert retriever.revision == expected
    tokenizer_loader.assert_called_once_with(model_name, revision=expected)
    model_loader.assert_called_once_with(model_name, revision=expected)
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once()


def test_dense_retrieval_checks_active_configuration_before_embedding(vector_store) -> None:
    embedder = MagicMock()
    embedder.embedding_config = make_config(query_prefix="another query prefix: ")
    retriever = DenseRetriever(embedder, vector_store)

    with pytest.raises(EmbeddingConfigError):
        retriever.retrieve("Which measurement?")

    embedder.embed_query.assert_not_called()
    assert vector_store._db is None
