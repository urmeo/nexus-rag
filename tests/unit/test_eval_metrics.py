"""Tests for retrieval metrics."""

import math

import pytest

from scinexusrag.eval import metrics as M


class TestRankingMetrics:
    def test_precision_recall(self) -> None:
        ranked = ["a", "b", "c", "d"]
        rel = {"a", "c"}
        assert M.precision_at_k(ranked, rel, 2) == 0.5
        assert M.recall_at_k(ranked, rel, 4) == 1.0
        assert M.recall_at_k(ranked, rel, 1) == 0.5

    def test_hit_and_mrr(self) -> None:
        assert M.hit_at_k(["x", "a"], {"a"}, 3) == 1.0
        assert M.hit_at_k(["x", "y"], {"a"}, 3) == 0.0
        assert M.reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
        assert M.reciprocal_rank(["x"], {"a"}) == 0.0

    def test_ndcg_perfect_and_empty(self) -> None:
        ranked = ["a", "b", "c"]
        assert M.ndcg_at_k(ranked, {"a", "b", "c"}, 3) == 1.0
        assert M.ndcg_at_k(ranked, set(), 3) == 0.0

    def test_ndcg_order_matters(self) -> None:
        good = M.ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        worse = M.ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert good > worse

    def test_ndcg_matches_trec_linear_graded_gain(self) -> None:
        discount = 1 / math.log2(3)
        expected = (1 + 2 * discount) / (2 + discount)
        assert M.ndcg_at_k(["low", "high"], {"high": 2, "low": 1}, 2) == pytest.approx(expected)

    @pytest.mark.parametrize("k", [0, -1])
    def test_nonpositive_cutoffs_are_empty(self, k: int) -> None:
        assert M.precision_at_k(["a", "b"], {"a"}, k) == 0
        assert M.recall_at_k(["a", "b"], {"a"}, k) == 0
        assert M.hit_at_k(["a", "b"], {"a"}, k) == 0
        assert M.ndcg_at_k(["a", "b"], {"a"}, k) == 0

    def test_average_precision(self) -> None:
        ap = M.average_precision(["a", "x", "b"], {"a", "b"})
        assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-9


class TestAggregateAndSignificance:
    def test_duplicate_document_ids_cannot_inflate_scores(self) -> None:
        with pytest.raises(ValueError, match="duplicate ranked"):
            M.per_query({"q": ["a", "a"]}, {"q": {"a"}})

    def test_per_query_skips_empty_qrels(self) -> None:
        run = {"q1": ["a", "b"], "q2": ["c"]}
        qrels = {"q1": {"a"}, "q2": set()}
        scores = M.per_query(run, qrels)
        assert set(scores) == {"q1"}

    def test_aggregate_means(self) -> None:
        run = {"q1": ["a"], "q2": ["b"]}
        qrels = {"q1": {"a"}, "q2": {"b"}}
        means = M.aggregate(M.per_query(run, qrels))
        assert means["P@1"] == 1.0

    def test_bootstrap_ci_bounds(self) -> None:
        mean, lo, hi = M.bootstrap_ci([0.5, 0.5, 0.5, 0.5], seed=1)
        assert mean == 0.5 and lo == 0.5 and hi == 0.5

    def test_randomization_identical_is_insignificant(self) -> None:
        a = [0.4, 0.6, 0.5]
        assert M.paired_randomization_test(a, a, seed=1) == 1.0

    def test_randomization_clear_difference(self) -> None:
        a = [1.0] * 10
        b = [0.0] * 10
        assert M.paired_randomization_test(a, b, seed=1) < 0.05

    @pytest.mark.parametrize("fn", [M.paired_delta_ci, M.paired_randomization_test])
    def test_incomplete_pairs_are_rejected(self, fn) -> None:
        with pytest.raises(ValueError, match="equal length"):
            fn([1.0, 0.5], [1.0])


class TestBinaryMetrics:
    def test_pr_auc_ties_share_a_threshold(self) -> None:
        assert M.pr_auc([0.5, 0.5], [1, 0]) == 0.5
        assert M.pr_auc([0.5, 0.5], [0, 1]) == 0.5
        assert M.pr_auc([0.9, 0.5, 0.5], [1, 0, 1]) == pytest.approx(5 / 6)

    def test_risk_coverage_ties_do_not_depend_on_row_order(self) -> None:
        assert M.risk_coverage_auc([0.5, 0.5], [1, 0]) == 0.5
        assert M.risk_coverage_auc([0.5, 0.5], [0, 1]) == 0.5
        assert M.risk_coverage_auc([0.9, 0.5, 0.5], [1, 0, 1]) == pytest.approx(7 / 36)

    @pytest.mark.parametrize("fn", [M.pr_auc, M.roc_auc, M.risk_coverage_auc])
    def test_invalid_binary_scores_are_rejected(self, fn) -> None:
        with pytest.raises(ValueError, match="equal length"):
            fn([0.5], [0, 1])
        with pytest.raises(ValueError, match="finite"):
            fn([float("nan"), 0.5], [0, 1])
        with pytest.raises(ValueError, match="binary"):
            fn([0.5, 0.6], [0, 2])
