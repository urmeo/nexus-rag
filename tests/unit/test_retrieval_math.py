import math

from nexusrag.eval.metrics import ece, holm_correction, pr_auc, risk_coverage_auc, roc_auc
from nexusrag.ingestion import Chunk
from nexusrag.retrieval import CorrectiveRetriever, RetrievalResult, rrf_fuse


def _result(cid: str, score: float = 0.0) -> RetrievalResult:
    return RetrievalResult(chunk=Chunk(id=cid, content=cid, document_id=cid), score=score)


def test_rrf_fuse_orders_by_summed_reciprocal_rank():
    a = [_result("c1"), _result("c2"), _result("c3")]
    b = [_result("c2"), _result("c3"), _result("c4")]
    fused = rrf_fuse([a, b], [1.0, 1.0], k=60)

    assert [r.chunk.id for r in fused] == ["c2", "c3", "c1", "c4"]
    assert fused[0].score == 1.0  # top normalized to 1


def test_rrf_fuse_weights_shift_ranking():
    a = [_result("c1"), _result("c2")]
    b = [_result("c2"), _result("c1")]
    sparse_heavy = rrf_fuse([a, b], [0.1, 0.9], k=60)
    assert sparse_heavy[0].chunk.id == "c2"


def test_roc_auc_known_value():
    assert math.isclose(roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)
    assert roc_auc([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
    assert roc_auc([0.4, 0.3, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_roc_auc_handles_ties():
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5


def test_pr_auc_known_value():
    assert math.isclose(pr_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.8333333, rel_tol=1e-5)


def test_holm_is_monotone_and_scales_smallest():
    adj = holm_correction({"a": 0.01, "b": 0.02, "c": 0.5})
    assert math.isclose(adj["a"], 0.03)  # smallest * m
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_risk_coverage_auc_rewards_confident_correct():
    assert risk_coverage_auc([0.9, 0.8, 0.7], [1, 1, 1]) == 0.0
    assert risk_coverage_auc([0.9, 0.8, 0.7], [0, 0, 0]) == 1.0
    assert math.isclose(risk_coverage_auc([0.9, 0.8], [1, 0]), 0.25)


def test_ece_perfect_and_overconfident():
    assert ece([0.0, 1.0], [0, 1]) == 0.0
    assert math.isclose(ece([1.0, 1.0], [0, 1]), 0.5)


class _FakeDense:
    def __init__(self, top_score: float):
        self.top_score = top_score

    def retrieve(self, query, top_k=5):
        return [_result("d1", self.top_score)]


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
        return [_result("d1", 0.9), _result("d2", 0.8)]


def test_corrective_skips_when_confident():
    base = _FakeBase(top_score=0.9)
    cr = CorrectiveRetriever(base, tau=0.55)
    _, triggered = cr.retrieve_traced("kinase inhibits tumor growth", top_k=2)
    assert triggered is False
    assert len(base.calls) == 1


def test_corrective_expands_when_weak():
    base = _FakeBase(top_score=0.2)
    cr = CorrectiveRetriever(base, tau=0.55, feedback_terms=3)
    _, triggered = cr.retrieve_traced("kinase inhibits tumor growth", top_k=2)
    assert triggered is True
    assert len(base.calls) == 2
    assert base.calls[1] != base.calls[0]


def _content_result(text: str) -> RetrievalResult:
    return RetrievalResult(chunk=Chunk(id="x", content=text, document_id="d"), score=0.5)


def test_expand_appends_frequent_non_query_terms():
    # PRF term selection is the headline of the corrective loop; test it directly.
    base = _FakeBase(top_score=0.2)
    cr = CorrectiveRetriever(base, feedback_terms=2)
    results = [
        _content_result("apoptosis apoptosis apoptosis signaling pathway"),
        _content_result("signaling pathway pathway"),
    ]
    added = cr.expand("kinase inhibits", results).split()[2:]  # after the query terms

    assert added == ["apoptosis", "pathway"]  # top-2 by frequency, in order
    assert "kinase" not in added and "inhibits" not in added  # query terms excluded
    assert "signaling" not in added  # capped out by feedback_terms=2


def test_expand_respects_feedback_terms_cap():
    base = _FakeBase(top_score=0.2)
    cr = CorrectiveRetriever(base, feedback_terms=1)
    added = cr.expand("kinase", [_content_result("apoptosis apoptosis signaling")]).split()[1:]
    assert added == ["apoptosis"]
