"""Check that plotted curves preserve tied thresholds and recorded metrics."""

import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from scinexusrag.eval import report
from scinexusrag.eval.metrics import pr_auc, roc_auc
from scinexusrag.eval.report import binary_curve_points, plot_evidence_curves, write_figures


@pytest.mark.parametrize(
    "scores,labels",
    [
        ([0.9, 0.5, 0.5, 0.1], [1, 0, 1, 0]),
        ([1.0, 1.0, 1.0, 1.0], [1, 0, 1, 0]),
        ([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]),
        ([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]),
    ],
)
def test_curve_areas_match_metrics_with_tied_scores(scores, labels):
    fpr, tpr, recall, precision = binary_curve_points(scores, labels)
    # The ROC trapezoid and PR steps must represent the metrics used in the report.
    roc_area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    average_precision = np.sum(np.diff(recall) * precision[1:])
    assert roc_area == pytest.approx(roc_auc(scores, labels))
    assert average_precision == pytest.approx(pr_auc(scores, labels))
    assert len(fpr) == len(set(scores)) + 1
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)


@pytest.mark.parametrize(
    "scores,labels",
    [
        ([], []),
        ([0.1], [1]),
        ([0.1, 0.2], [0, 0]),
        ([0.1], [0, 1]),
        ([0.1, np.nan], [0, 1]),
        ([0.1, 0.2], [0, 2]),
        ([[0.1, 0.2]], [[0, 1]]),
    ],
)
def test_curves_reject_invalid_or_single_class_inputs(scores, labels):
    with pytest.raises(ValueError):
        binary_curve_points(scores, labels)


def test_committed_evidence_curves_match_recorded_areas():
    results = Path(__file__).resolve().parents[2] / "outputs/results/faithfulness_dev.json"
    faith = json.loads(results.read_text())
    candidates = faith["per_candidate"]["dev"]
    labels = candidates["labels"]
    assert len(labels) == faith["num_candidates"]
    for name, scores in candidates["scores"].items():
        fpr, tpr, recall, precision = binary_curve_points(scores, labels)
        assert np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2) == pytest.approx(
            faith["methods"][name]["roc_auc"], abs=1e-12
        )
        assert np.sum(np.diff(recall) * precision[1:]) == pytest.approx(
            faith["methods"][name]["pr_auc"], abs=1e-12
        )


def test_evidence_chart_rejects_stale_recorded_metrics(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    open_figures = set(plt.get_fignums())
    scores, labels = [0.9, 0.5, 0.5, 0.1], [1, 0, 1, 0]
    faith = {
        "num_candidates": 4,
        "gold_base_rate": 0.5,
        "per_candidate": {"dev": {"labels": labels, "scores": {"nli": scores}}},
        "methods": {"nli": {"roc_auc": 0.1, "pr_auc": pr_auc(scores, labels)}},
    }
    with pytest.raises(ValueError, match="do not match recorded AUC/AP"):
        plot_evidence_curves(faith, tmp_path / "evidence.svg")
    assert set(plt.get_fignums()) == open_figures


def test_embedding_macro_uses_saved_randomization_seed(monkeypatch):
    scores = {"BM25": [0.4, 0.6], "Dense": [0.5, 0.7], "Hybrid (RRF)": [0.6, 0.8]}
    retrieval = {
        "seed": 7,
        "num_queries": 2,
        "query_ids": ["q1", "q2"],
        "corpus_size": 10,
        "per_query_ndcg": scores,
        "systems": {
            name: {"means": {"nDCG@10": float(np.mean(values))}} for name, values in scores.items()
        },
    }
    randomization = Mock(return_value=0.25)
    monkeypatch.setattr(report, "paired_randomization_test", randomization)
    report.build_macros(retrieval, retrieval, retrieval, None, None)
    randomization.assert_called_with(scores["Dense"], scores["Dense"], seed=7)


def test_mismatched_embedding_chart_does_not_leave_a_figure_open(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    open_figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match="matching datasets and query_ids"):
        report.plot_retrieval_comparison(
            {"dataset": "scifact"}, {}, {"dataset": "nfcorpus"}, None, tmp_path / "retrieval.svg"
        )
    assert set(plt.get_fignums()) == open_figures
    assert not (tmp_path / "retrieval.svg").exists()


def test_export_uses_custom_result_provenance_and_only_plotted_files(tmp_path):
    pytest.importorskip("matplotlib")
    committed = Path(__file__).resolve().parents[2] / "outputs/results"
    custom_results = tmp_path / "fresh-results"
    custom_results.mkdir()
    for filename in (
        "scifact_test.json",
        "nfcorpus_test.json",
        "faithfulness_dev.json",
        "corrective_scifact.json",
    ):
        shutil.copyfile(committed / filename, custom_results / filename)
    figures = tmp_path / "charts"
    write_figures(figures, custom_results)
    for chart in figures.glob("*.svg"):
        svg = chart.read_text()
        assert f"Source: {custom_results}/" in svg
        assert "outputs/results" not in svg
        assert all(line == line.rstrip() for line in svg.splitlines())
    retrieval = (figures / "retrieval-comparison.svg").read_text()
    assert "{scifact,nfcorpus}_test.json" in retrieval
    assert "minilm" not in retrieval.lower()
