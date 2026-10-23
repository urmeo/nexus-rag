"""Evaluation checks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scinexusrag.eval import (
    corrective,
    datasets,
    faithfulness,
    gate,
    generation,
    metrics,
    provenance,
    ragtruth,
    report,
    reproduce,
    run,
)
from scinexusrag.eval.datasets import IRDataset
from scinexusrag.eval.indexes import ExactDenseRetriever, corpus_to_chunks
from scinexusrag.eval.metrics import pr_auc, roc_auc
from scinexusrag.eval.provenance import evaluation_provenance
from scinexusrag.eval.report import (
    binary_curve_points,
    plot_evidence_curves,
    write_figures,
)
from scinexusrag.eval.result_files import load_result, write_csv_results
from scinexusrag.ingestion.embedder import Embedder
from scinexusrag.retrieval.dense import RetrievalResult
from scinexusrag.retrieval.splade import DEFAULT_MODEL
from scinexusrag.settings import HF_REVISIONS


class TestVendoredIR:
    def test_load_scifact_sample(self) -> None:
        ds = datasets.load("scifact", prefer_vendored=True)
        assert ds.corpus and ds.queries and ds.qrels
        qid = next(iter(ds.qrels))
        assert ds.qrels[qid]
        assert all(d in ds.corpus for rel in ds.qrels.values() for d in rel)

    @pytest.mark.parametrize(
        "title,body,expected",
        [
            (" Study ", " Evidence ", "Study. Evidence"),
            ("", " Evidence ", "Evidence"),
            ("Study", "", "Study"),
            (" \t", " Evidence ", "Evidence"),
        ],
    )
    def test_doc_text_joins_available_title_and_body(self, title, body, expected) -> None:
        dataset = datasets.IRDataset(
            "fixture", {"document": {"title": title, "text": body}}, {}, {}
        )

        assert dataset.doc_text("document") == expected


class TestClaims:
    @pytest.mark.parametrize("split", ["train", "dev"])
    def test_load_claims_have_gold_candidates(self, split) -> None:
        claims = faithfulness.load_claims(split, prefer_vendored=True)
        assert claims
        for claim in claims:
            candidates = {(doc_id, index) for doc_id, index, _ in claim.candidates}
            assert claim.gold
            assert claim.gold <= candidates
            assert claim.label in {"SUPPORT", "CONTRADICT"}

    def test_lexical_overlap_is_jaccard(self) -> None:
        assert faithfulness.lexical_overlap("kinase inhibits tumor", "tumor kinase growth") == 0.5
        assert faithfulness.lexical_overlap("abc def", "xyz wuv") == 0.0

    def test_f1_at_perfect_separation(self) -> None:
        assert faithfulness._f1_at([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 0.5) == 1.0

    def test_bootstrap_auroc_brackets_perfect(self) -> None:
        lo, hi = faithfulness._bootstrap_auroc(
            [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], [0, 0, 1, 1], n_boot=200
        )
        assert 0.5 <= lo <= 1.0 and hi == 1.0


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"hallucination_labels": "[]"}, 0),
        ({"hallucination_labels": '[{"start": 0, "end": 3}]'}, 1),
        ({"hallucination_labels": []}, 0),
        ({"hallucination_labels": [{"start": 0, "end": 3}]}, 1),
        ({"hallucination_labels_processed": {"evident_conflict": 0, "baseless_info": 0}}, 0),
        ({"hallucination_labels_processed": {"evident_conflict": 0, "baseless_info": 1}}, 1),
    ],
)
def test_ragtruth_parses_annotations_without_string_truthiness(row, expected):
    assert ragtruth._is_hallucinated(row) == expected


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"hallucination_labels": "not json"},
        {"hallucination_labels": "{}"},
        {"hallucination_labels": 0},
        {"hallucination_labels_processed": {"baseless_info": -1}},
        {"hallucination_labels_processed": {"baseless_info": float("nan")}},
        {"hallucination_labels_processed": "0"},
    ],
)
def test_ragtruth_rejects_missing_or_invalid_annotations(row):
    with pytest.raises(ValueError, match="RAGTruth"):
        ragtruth._is_hallucinated(row)


@pytest.mark.parametrize("output,context", [("", "Evidence."), ("Answer.", " ")])
def test_ragtruth_rejects_empty_text_before_loading_models(monkeypatch, output, context):
    row = {"output": output, "context": context, "hallucination_labels": "[]"}
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: [row])
    )
    verifier = Mock()
    monkeypatch.setattr(ragtruth, "GroundingVerifier", verifier)

    with pytest.raises(ValueError, match="nonempty output and context"):
        ragtruth.evaluate()

    verifier.assert_not_called()


@pytest.mark.parametrize("labels,expected_auc", [(["[]", '[{"start": 0}]'], 1.0), (["[]"], None)])
def test_ragtruth_reports_observed_labels_and_undefined_auc(monkeypatch, labels, expected_auc):
    rows = [
        {"output": str(index), "context": "Evidence.", "hallucination_labels": label}
        for index, label in enumerate(labels)
    ]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: rows))
    monkeypatch.setattr(
        ragtruth, "GroundingVerifier", lambda **kwargs: SimpleNamespace(model_name="fixture")
    )
    monkeypatch.setattr(
        ragtruth, "_groundedness", lambda verifier, output, context: 0.9 - int(output) * 0.8
    )

    result = ragtruth.evaluate()

    assert result["hallucination_rate"] == (0.5 if len(labels) == 2 else 0)
    assert result["roc_auc"] == expected_auc
    json.dumps(result, allow_nan=False)


def test_ragtruth_records_the_dataset_revision_passed_to_the_loader(monkeypatch):
    row = {"output": "Answer.", "context": "Evidence.", "hallucination_labels": "[]"}
    load_dataset = Mock(return_value=[row])
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(
        ragtruth, "GroundingVerifier", lambda **kwargs: SimpleNamespace(model_name="fixture")
    )
    monkeypatch.setattr(ragtruth, "_groundedness", lambda *args: 0.9)

    result = ragtruth.evaluate(n=1)

    revision = datasets.DATASET_REVISIONS["ragtruth"]
    load_dataset.assert_called_once_with(
        "wandb/RAGTruth-processed", split="train", revision=revision
    )
    assert result["dataset_revision"] == revision


@pytest.mark.parametrize("scores", [[], [0.5], [0.5, np.nan], [0.5, 1.1], [0.5, -0.1]])
def test_ragtruth_rejects_invalid_nli_probabilities(scores):
    verifier = SimpleNamespace(entailment_scores=lambda pairs: scores)
    with pytest.raises(ValueError, match="finite probabilities"):
        ragtruth._groundedness(verifier, "Answer.", "First evidence. Second evidence.")


def test_ragtruth_averages_best_context_probabilities():
    verifier = SimpleNamespace(entailment_scores=Mock(side_effect=[[0.2, 0.8], [0.6, 0.4]]))
    score = ragtruth._groundedness(
        verifier, "First answer. Second answer.", "First evidence. Second evidence."
    )
    assert score == pytest.approx(0.7)


def test_ragtruth_cli_preserves_undefined_auc_as_null(monkeypatch, tmp_path, capsys):
    result = {"num_responses": 1, "hallucination_rate": 0, "roc_auc": None, "risk_coverage_auc": 0}
    destination = tmp_path / "ragtruth.json"
    monkeypatch.setattr(ragtruth, "evaluate", lambda **kwargs: result)
    monkeypatch.setattr(sys, "argv", ["ragtruth", "--out", str(destination)])

    ragtruth.main()

    assert json.loads(destination.read_text(encoding="utf-8")) == result
    assert "AUROC=undefined" in capsys.readouterr().out


def _load_dataset_rows(monkeypatch, tmp_path, loader, rows):
    if loader == "vendored":
        base = tmp_path / "scifact_sample"
        base.mkdir()
        for name, records in rows.items():
            (base / f"{name}.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
        monkeypatch.setattr(datasets, "VENDORED_DIR", tmp_path)
        return datasets.load_vendored("scifact")

    def load_dataset(repo, config=None, **kwargs):
        return {config or "test": rows[config or "qrels"]}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    return datasets.load_beir("scifact")


@pytest.fixture
def dataset_rows():
    return {
        "corpus": [{"_id": "doc", "title": "Study", "text": "Evidence."}],
        "queries": [{"_id": "query", "text": "Which evidence?"}],
        "qrels": [{"query-id": "query", "corpus-id": "doc", "score": 1}],
    }


@pytest.mark.parametrize("loader", ["vendored", "beir"])
@pytest.mark.parametrize(
    "part,records,message",
    [
        ("corpus", [{"_id": "doc"}, {"_id": "doc"}], "duplicate corpus"),
        (
            "queries",
            [{"_id": "query", "text": "One"}, {"_id": "query", "text": "Two"}],
            "duplicate query",
        ),
        ("qrels", [{"query-id": "query", "corpus-id": "missing", "score": 1}], "missing document"),
        ("qrels", [{"query-id": "missing", "corpus-id": "doc", "score": 1}], "missing query"),
        ("qrels", [{"query-id": "query", "corpus-id": "doc", "score": 0.5}], "finite integers"),
        ("qrels", [{"query-id": "query", "corpus-id": "doc", "score": True}], "finite integers"),
        (
            "qrels",
            [{"query-id": "query", "corpus-id": "doc", "score": 10**1000}],
            "finite integers",
        ),
        (
            "qrels",
            [{"query-id": "query", "corpus-id": "doc", "score": -(10**1000)}],
            "finite integers",
        ),
        ("qrels", [{"query-id": "query", "corpus-id": "doc"}], "finite integers"),
        (
            "qrels",
            [
                {"query-id": "query", "corpus-id": "doc", "score": 1},
                {"query-id": "query", "corpus-id": "doc", "score": 2},
            ],
            "conflicting qrels",
        ),
    ],
)
def test_dataset_loaders_reject_ambiguous_or_broken_judgments(
    monkeypatch, tmp_path, dataset_rows, loader, part, records, message
):
    dataset_rows[part] = records
    with pytest.raises(ValueError, match=message):
        _load_dataset_rows(monkeypatch, tmp_path, loader, dataset_rows)


@pytest.mark.parametrize("loader", ["vendored", "beir"])
def test_dataset_loaders_preserve_grade_and_query_selection(
    monkeypatch, tmp_path, dataset_rows, loader
):
    dataset_rows["queries"].append({"_id": "unjudged", "text": "No judgments."})
    dataset_rows["qrels"][0]["score"] = 2
    dataset_rows["qrels"] *= 2
    dataset_rows["qrels"].append({"query-id": "unjudged", "corpus-id": "doc", "score": 0})

    dataset = _load_dataset_rows(monkeypatch, tmp_path, loader, dataset_rows)

    assert dataset.corpus == {"doc": {"title": "Study", "text": "Evidence."}}
    assert dataset.qrels == {"query": {"doc": 2}}
    assert list(dataset.queries) == (["query", "unjudged"] if loader == "vendored" else ["query"])
    assert dataset.source == ("vendored" if loader == "vendored" else "huggingface")


@pytest.mark.parametrize(
    "evidence,message",
    [
        ({"missing": [{"label": "SUPPORT", "sentences": [0]}]}, "missing document"),
        ({"doc": [{"label": "UNKNOWN", "sentences": [0]}]}, "invalid SciFact evidence label"),
        ({"doc": [{"label": "SUPPORT", "sentences": [-1]}]}, "out of range"),
        ({"doc": [{"label": "SUPPORT", "sentences": [2]}]}, "out of range"),
        ({"doc": [{"label": "SUPPORT", "sentences": [0.5]}]}, "out of range"),
        ({"doc": []}, "requires annotated sentences"),
        ({"doc": [{"label": "SUPPORT", "sentences": []}]}, "requires annotated sentences"),
    ],
)
def test_scifact_rejects_unusable_gold_evidence(tmp_path, evidence, message):
    path = tmp_path / "claims.jsonl"
    row = {"id": 1, "claim": "Claim.", "evidence": evidence}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        faithfulness._read_claims(path, {"doc": ["First sentence.", "Second sentence."]})


def test_scifact_rejects_duplicate_claim_ids(tmp_path):
    path = tmp_path / "claims.jsonl"
    row = {
        "id": 1,
        "claim": "Claim.",
        "evidence": {"doc": [{"label": "SUPPORT", "sentences": [0]}]},
    }
    path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate SciFact claim"):
        faithfulness._read_claims(path, {"doc": ["Evidence."]})


@pytest.mark.parametrize(
    "records,message",
    [
        (
            [{"doc_id": 1, "abstract": ["One."]}, {"doc_id": 1, "abstract": ["Two."]}],
            "duplicate SciFact document",
        ),
        ([{"doc_id": 1, "abstract": "One sentence."}], "sentence strings"),
    ],
)
def test_scifact_rejects_ambiguous_corpus_rows(tmp_path, records, message):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        faithfulness._read_corpus(path)


@pytest.fixture
def evidence_claim():
    return faithfulness.Claim(
        1, "Claim.", "SUPPORT", {("doc", 0)}, [("doc", 0, "Evidence."), ("doc", 1, "Other text.")]
    )


@pytest.mark.parametrize(
    "probabilities,entailment,contradiction",
    [
        ([[0.2, 0.8], [0.3, 0.7]], 1, 0),
        ([[0.2, 0.5, 0.3]], 1, 0),
        ([[0.2, 0.5, 0.3], [0.1, 0.5, 0.4]], 3, 0),
        ([[0.2, 0.5, 0.3], [0.1, 0.5, 0.4]], 1, 1),
        ([[0.2, float("nan"), 0.3], [0.1, 0.5, 0.4]], 1, 0),
        ([[0.2, 0.7, 0.3], [0.1, 0.5, 0.4]], 1, 0),
    ],
)
def test_evidence_nli_requires_valid_class_probabilities(
    evidence_claim, probabilities, entailment, contradiction
):
    verifier = SimpleNamespace(
        class_probs=lambda pairs: probabilities, entail_idx=entailment, contra_idx=contradiction
    )
    with pytest.raises(ValueError, match="NLI probabilities"):
        faithfulness._nli_relatedness(verifier, [evidence_claim])


def test_evidence_relatedness_adds_entailment_and_contradiction(evidence_claim):
    probabilities = [[0.2, 0.5, 0.3], [0.1, 0.5, 0.4]]
    verifier = SimpleNamespace(
        class_probs=Mock(return_value=probabilities), entail_idx=1, contra_idx=0
    )

    scores = faithfulness._nli_relatedness(verifier, [evidence_claim])

    assert scores[0] == pytest.approx([0.7, 0.6])
    verifier.class_probs.assert_called_once_with(
        [("Evidence.", "Claim."), ("Other text.", "Claim.")]
    )


@pytest.mark.parametrize("scores", [[0.5], [[0.5, 0.4], [0.3, 0.2]], [np.nan, 0.5]])
def test_evidence_reranker_scores_align_with_candidates(monkeypatch, evidence_claim, scores):
    monkeypatch.setattr(faithfulness, "_nli_relatedness", lambda *args: [[0.7, 0.6]])
    monkeypatch.setattr(
        "scinexusrag.retrieval.reranker.Reranker",
        lambda **kwargs: SimpleNamespace(model=SimpleNamespace(predict=lambda pairs: scores)),
    )

    with pytest.raises(ValueError, match="finite and align"):
        faithfulness._score_methods([evidence_claim], None, with_reranker=True)


@pytest.mark.parametrize(
    "scores,labels,claims",
    [
        ([[0.5, 0.6]], [0], [0]),
        ([0.5, 0.6], [0, 1], [[0], [1]]),
        ([0.5, np.nan], [0, 1], [0, 1]),
        ([0.5, 0.6], [0, 2], [0, 1]),
        ([0.5, 0.6], [0, 1], [0.5, 1.5]),
    ],
)
def test_claim_bootstrap_rejects_invalid_arrays(scores, labels, claims):
    with pytest.raises(ValueError, match="bootstrap"):
        faithfulness._bootstrap_auroc(scores, labels, claims, n_boot=2)


def test_generation_sources_are_numbered_and_truncated():
    long_passage = "x" * 600
    formatted_sources = generation._format_sources(["first passage", long_passage])

    assert formatted_sources.splitlines() == ["[1] first passage", "[2] " + "x" * 400]


@pytest.mark.parametrize("empty", ["corpus", "queries"])
def test_empty_generation_dataset_fails_before_models(monkeypatch, empty):
    dataset = IRDataset(
        name="scifact",
        corpus={"doc": {"text": "Evidence."}},
        queries={"query": "What evidence?"},
        qrels={},
    )
    setattr(dataset, empty, {})
    monkeypatch.setattr(generation.datasets, "load", lambda *args, **kwargs: dataset)
    embedder = Mock()
    generator = Mock()
    verifier = Mock()
    monkeypatch.setattr(generation, "Embedder", embedder)
    monkeypatch.setattr(generation, "LocalGenerator", generator)
    monkeypatch.setattr(generation, "GroundingVerifier", verifier)

    with pytest.raises(ValueError, match="nonempty corpus and queries"):
        generation.evaluate()

    embedder.assert_not_called()
    generator.assert_not_called()
    verifier.assert_not_called()


def test_generation_averages_baseline_and_gated_answer_scores(monkeypatch):
    dataset = IRDataset(
        name="scifact",
        corpus={"doc": {"text": "Evidence."}},
        queries={"first": "First question?", "second": "Second question?"},
        qrels={},
    )
    passage = corpus_to_chunks({"doc": "Evidence."})[0]
    results = [RetrievalResult(chunk=passage, score=1)]
    hybrid = SimpleNamespace(retrieve=lambda *args, **kwargs: results, rrf_k=60)
    generator = SimpleNamespace(
        model_name="fixture", generate=lambda prompt: "Answer.", close=Mock()
    )
    verifier = SimpleNamespace(
        model_name="fixture",
        verify=Mock(side_effect=[SimpleNamespace(faithfulness=value) for value in (0.6, 0.4, 0.9)]),
    )
    monkeypatch.setattr(generation.datasets, "load", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(generation, "Embedder", Mock())
    monkeypatch.setattr(generation, "ExactDenseRetriever", Mock())
    monkeypatch.setattr(generation, "AdaptiveHybridRetriever", lambda *args: hybrid)
    monkeypatch.setattr(
        generation,
        "CorrectiveRetriever",
        lambda *args, **kwargs: SimpleNamespace(expand=lambda query, sources: query),
    )
    monkeypatch.setattr(generation, "LocalGenerator", lambda *args: generator)
    monkeypatch.setattr(generation, "GroundingVerifier", lambda **kwargs: verifier)

    result = generation.evaluate(n=2, tau=0.5)

    assert result["num_queries"] == 2
    assert result["num_corrected"] == 1
    assert result["baseline_faithfulness"] == pytest.approx(0.5)
    assert result["corrective_faithfulness"] == pytest.approx(0.75)
    assert result["improvement"] == pytest.approx(0.25)
    assert verifier.verify.call_count == 3
    generator.close.assert_called_once_with()


class TestRankingMetrics:
    def test_precision_recall(self) -> None:
        ranked = ["a", "b", "c", "d"]
        rel = {"a", "c"}
        assert metrics.precision_at_k(ranked, rel, 2) == 0.5
        assert metrics.recall_at_k(ranked, rel, 4) == 1.0
        assert metrics.recall_at_k(ranked, rel, 1) == 0.5

    def test_hit_and_mrr(self) -> None:
        assert metrics.hit_at_k(["x", "a"], {"a"}, 3) == 1.0
        assert metrics.hit_at_k(["x", "y"], {"a"}, 3) == 0.0
        assert metrics.reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
        assert metrics.reciprocal_rank(["x"], {"a"}) == 0.0

    def test_ndcg_perfect_and_empty(self) -> None:
        ranked = ["a", "b", "c"]
        assert metrics.ndcg_at_k(ranked, {"a", "b", "c"}, 3) == 1.0
        assert metrics.ndcg_at_k(ranked, set(), 3) == 0.0

    def test_ndcg_order_matters(self) -> None:
        good = metrics.ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        worse = metrics.ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert good > worse

    def test_ndcg_matches_trec_linear_graded_gain(self) -> None:
        discount = 1 / math.log2(3)
        expected = (1 + 2 * discount) / (2 + discount)
        assert metrics.ndcg_at_k(["low", "high"], {"high": 2, "low": 1}, 2) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize("k", [0, -1])
    def test_nonpositive_cutoffs_are_empty(self, k: int) -> None:
        assert metrics.precision_at_k(["a", "b"], {"a"}, k) == 0
        assert metrics.recall_at_k(["a", "b"], {"a"}, k) == 0
        assert metrics.hit_at_k(["a", "b"], {"a"}, k) == 0
        assert metrics.ndcg_at_k(["a", "b"], {"a"}, k) == 0

    def test_average_precision(self) -> None:
        ap = metrics.average_precision(["a", "x", "b"], {"a", "b"})
        assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-9


class TestAggregateAndSignificance:
    def test_duplicate_document_ids_cannot_inflate_scores(self) -> None:
        with pytest.raises(ValueError, match="duplicate ranked"):
            metrics.per_query({"q": ["a", "a"]}, {"q": {"a"}})

    def test_per_query_skips_empty_qrels(self) -> None:
        run = {"q1": ["a", "b"], "q2": ["c"]}
        qrels = {"q1": {"a"}, "q2": set()}
        scores = metrics.per_query(run, qrels)
        assert set(scores) == {"q1"}

    def test_aggregate_means(self) -> None:
        run = {"q1": ["a"], "q2": ["irrelevant"]}
        qrels = {"q1": {"a"}, "q2": {"b"}}
        means = metrics.aggregate(metrics.per_query(run, qrels))
        assert means["P@1"] == 0.5

    def test_bootstrap_ci_bounds(self) -> None:
        mean, lo, hi = metrics.bootstrap_ci([0.5, 0.5, 0.5, 0.5], seed=1)
        assert mean == 0.5 and lo == 0.5 and hi == 0.5

    def test_randomization_identical_is_insignificant(self) -> None:
        a = [0.4, 0.6, 0.5]
        assert metrics.paired_randomization_test(a, a, seed=1) == 1.0

    def test_randomization_clear_difference(self) -> None:
        a = [1.0] * 10
        b = [0.0] * 10
        assert metrics.paired_randomization_test(a, b, seed=1) < 0.05

    @pytest.mark.parametrize("fn", [metrics.paired_delta_ci, metrics.paired_randomization_test])
    def test_incomplete_pairs_are_rejected(self, fn) -> None:
        with pytest.raises(ValueError, match="equal length"):
            fn([1.0, 0.5], [1.0])


class TestBinaryMetrics:
    def test_pr_auc_ties_share_a_threshold(self) -> None:
        assert metrics.pr_auc([0.5, 0.5], [1, 0]) == 0.5
        assert metrics.pr_auc([0.5, 0.5], [0, 1]) == 0.5
        assert metrics.pr_auc([0.9, 0.5, 0.5], [1, 0, 1]) == pytest.approx(5 / 6)

    def test_risk_coverage_ties_do_not_depend_on_row_order(self) -> None:
        assert metrics.risk_coverage_auc([0.5, 0.5], [1, 0]) == 0.5
        assert metrics.risk_coverage_auc([0.5, 0.5], [0, 1]) == 0.5
        assert metrics.risk_coverage_auc([0.9, 0.5, 0.5], [1, 0, 1]) == pytest.approx(7 / 36)

    @pytest.mark.parametrize("fn", [metrics.pr_auc, metrics.roc_auc, metrics.risk_coverage_auc])
    def test_invalid_binary_scores_are_rejected(self, fn) -> None:
        with pytest.raises(ValueError, match="equal length"):
            fn([0.5], [0, 1])
        with pytest.raises(ValueError, match="finite"):
            fn([float("nan"), 0.5], [0, 1])
        with pytest.raises(ValueError, match="binary"):
            fn([0.5, 0.6], [0, 2])


@pytest.mark.parametrize("state", ["missing", "file", "empty"])
def test_provenance_rejects_missing_python_source(tmp_path, state):
    directory = tmp_path / "source"
    if state == "file":
        directory.write_text("not a source directory", encoding="utf-8")
    elif state == "empty":
        directory.mkdir()

    with pytest.raises(ValueError, match="Source directory"):
        provenance.evaluation_provenance(directory)


def test_provenance_records_bm25_package_version(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "version", lambda name: f"{name}-version")

    result = provenance.evaluation_provenance(tmp_path)

    assert result["packages"]["rank-bm25"] == "rank-bm25-version"


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
    results = Path(__file__).resolve().parents[1] / "outputs/results"
    faith = load_result("faithfulness_dev", results)
    assert faith is not None
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
    from matplotlib import pyplot as plt

    open_figures = set(plt.get_fignums())
    with pytest.raises(ValueError, match="matching datasets and query_ids"):
        report.plot_retrieval_comparison(
            {"dataset": "scifact"}, {}, {"dataset": "nfcorpus"}, None, tmp_path / "retrieval.svg"
        )
    assert set(plt.get_fignums()) == open_figures
    assert not (tmp_path / "retrieval.svg").exists()


@pytest.mark.parametrize("result_format", ["json", "csv"])
def test_export_uses_custom_result_provenance_and_only_plotted_files(tmp_path, result_format):
    committed = Path(__file__).resolve().parents[1] / "outputs/results"
    custom_results = tmp_path / "fresh-results"
    custom_results.mkdir()
    results = {}
    for name in (
        "scifact_test",
        "nfcorpus_test",
        "faithfulness_dev",
        "corrective_scifact",
    ):
        result = load_result(name, committed)
        assert result is not None
        results[name] = result
    if result_format == "csv":
        write_csv_results(results, custom_results)
    else:
        for name, result in results.items():
            (custom_results / f"{name}.json").write_text(json.dumps(result), encoding="utf-8")
    figures = tmp_path / "charts"
    write_figures(figures, custom_results)
    assert {chart.name for chart in figures.glob("*.svg")} == {
        "retrieval-comparison.svg",
        "retrieval-query-changes.svg",
        "retrieval-cost.svg",
        "evidence-curves.svg",
        "corrective-sweep.svg",
    }
    for chart in figures.glob("*.svg"):
        svg = chart.read_text()
        assert f"Source: {custom_results}/" in svg
        assert "outputs/results" not in svg
        assert all(line == line.rstrip() for line in svg.splitlines())
    retrieval = (figures / "retrieval-comparison.svg").read_text()
    expected_source = (
        "retrieval_metrics.csv" if result_format == "csv" else "{scifact,nfcorpus}_test.json"
    )
    assert expected_source in retrieval
    assert "minilm" not in retrieval.lower()
    for chart, name in (
        ("evidence-curves.svg", "faithfulness_candidates"),
        ("corrective-sweep.svg", "corrective_sweeps"),
        ("retrieval-query-changes.svg", "retrieval_queries"),
        ("retrieval-cost.svg", "corrective_cost_quality"),
    ):
        svg = (figures / chart).read_text()
        if result_format == "csv":
            assert f"{name}.csv" in svg
            assert ".json" not in svg
        else:
            assert ".csv" not in svg


def test_cli_renders_charts_from_exported_csv(monkeypatch, tmp_path):
    original = load_result("scifact_test", Path(__file__).resolve().parents[1] / "outputs/results")
    assert original is not None
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "scifact_test.json").write_text(json.dumps(original), encoding="utf-8")
    exported = tmp_path / "csv"
    charts = tmp_path / "charts"
    render = Mock()
    monkeypatch.setattr(report, "write_figures", render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--results",
            str(generated),
            "--csv",
            str(exported),
            "--figures",
            str(charts),
        ],
    )

    report.main()

    assert load_result("scifact_test", exported) == original
    assert not list(exported.glob("*.json"))
    render.assert_called_once_with(charts, exported)


def test_cli_rejects_csv_export_without_saved_results(monkeypatch, tmp_path):
    exported = tmp_path / "csv"
    monkeypatch.setattr(sys, "argv", ["report", "--results", str(tmp_path), "--csv", str(exported)])
    with pytest.raises(SystemExit, match="No evaluation results found"):
        report.main()
    assert not exported.exists()


@pytest.fixture
def sample_gate(monkeypatch):
    spec = json.loads(gate.THRESHOLDS.read_text())
    retrieval = {
        key: spec["retrieval"][key] for key in ("dataset", "split", "num_queries", "corpus_size")
    }
    retrieval["systems"] = {
        name: {"means": deepcopy(values)} for name, values in spec["retrieval"]["floors"].items()
    }
    evidence = {key: spec["faithfulness"][key] for key in ("dataset", "split", "num_claims")}
    evidence["methods"] = deepcopy(spec["faithfulness"]["floors"])
    monkeypatch.setattr(gate, "run_retrieval", lambda **kwargs: retrieval)
    monkeypatch.setattr(gate.faithfulness, "evaluate", lambda **kwargs: evidence)
    return retrieval, evidence


def test_gate_accepts_matching_sample_protocol_at_floors(sample_gate):
    assert gate.main() == 0


@pytest.mark.parametrize("score", [float("inf"), float("nan"), -0.1, 1.1, True])
def test_gate_rejects_invalid_observed_scores(sample_gate, score):
    retrieval, _ = sample_gate
    retrieval["systems"]["Hybrid (RRF)"]["means"]["nDCG@10"] = score
    assert gate.main() == 1


@pytest.mark.parametrize(
    "section,field,value",
    [
        (0, "dataset", "nfcorpus"),
        (0, "split", "test"),
        (0, "num_queries", 1),
        (0, "corpus_size", 1),
        (1, "dataset", "another-dataset"),
        (1, "split", "dev"),
        (1, "num_claims", 1),
    ],
)
def test_gate_rejects_changed_sample_protocol(sample_gate, section, field, value):
    sample_gate[section][field] = value
    assert gate.main() == 1


def test_gate_rejects_invalid_floor_before_running_models(monkeypatch, tmp_path):
    spec = json.loads(gate.THRESHOLDS.read_text())
    spec["retrieval"]["floors"]["Hybrid (RRF)"]["nDCG@10"] = float("inf")
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(gate, "THRESHOLDS", path)

    def unexpected(**kwargs):
        pytest.fail("Invalid thresholds must fail before model evaluation.")

    monkeypatch.setattr(gate, "run_retrieval", unexpected)
    with pytest.raises(ValueError, match="regression floor"):
        gate.main()


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
def test_json_results_reject_nonfinite_numbers(tmp_path, number):
    (tmp_path / "fresh.json").write_text(f'{{"score": {number}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_result("fresh", tmp_path)


@pytest.fixture
def retrieval_result():
    result = load_result("scifact_test", Path(__file__).resolve().parents[1] / "outputs/results")
    assert result is not None
    return result


@pytest.mark.parametrize("existing", [False, True])
def test_csv_publish_failure_restores_previous_directory(
    monkeypatch, tmp_path, retrieval_result, existing
):
    if existing:
        write_csv_results({"retrieval": retrieval_result}, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    updated = deepcopy(retrieval_result)
    updated["seed"] += 1
    replace = Path.replace
    failed = False

    def fail_once(path, target):
        nonlocal failed
        if path.name == "retrieval_queries.csv" and path.parent != tmp_path and not failed:
            failed = True
            raise OSError("Publication failed")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="Publication failed"):
        write_csv_results({"retrieval": updated}, tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_failed_csv_recovery_retains_backups(monkeypatch, tmp_path, retrieval_result):
    write_csv_results({"retrieval": retrieval_result}, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    updated = deepcopy(retrieval_result)
    updated["seed"] += 1
    replace = Path.replace

    def fail_publication_and_recovery(path, target):
        if path.name == "retrieval_systems.csv" or (
            path.name == "metadata.csv" and path.parent.name.startswith(".results-backup-")
        ):
            raise OSError("Storage unavailable")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication_and_recovery)
    with pytest.raises(OSError, match="backups retained"):
        write_csv_results({"retrieval": updated}, tmp_path)
    backups = list(tmp_path.glob(".results-backup-*"))
    assert len(backups) == 1
    assert {path.name: path.read_bytes() for path in backups[0].iterdir()} == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "another-dataset"),
        ("split", "train"),
        ("corpus_size", 1),
        ("dataset_revision", "another-revision"),
        ("qrels_revision", "another-revision"),
    ],
)
def test_embedding_macros_reject_incompatible_results(retrieval_result, field, value):
    comparison = deepcopy(retrieval_result)
    comparison[field] = value
    with pytest.raises(ValueError, match="matching datasets and query_ids"):
        report.build_macros(retrieval_result, retrieval_result, comparison, None, None)


def test_embedding_chart_rejects_different_dataset_revision(tmp_path, retrieval_result):
    comparison = deepcopy(retrieval_result)
    comparison["dataset_revision"] = "another-revision"
    with pytest.raises(ValueError, match="matching datasets and query_ids"):
        report.plot_retrieval_comparison(
            retrieval_result, retrieval_result, comparison, None, tmp_path / "retrieval.svg"
        )
    assert not (tmp_path / "retrieval.svg").exists()


def _draw_chart(kind, output):
    results = Path(__file__).resolve().parents[1] / "outputs/results"
    if kind == "retrieval":
        report.plot_retrieval_comparison(
            load_result("scifact_test", results),
            load_result("nfcorpus_test", results),
            None,
            None,
            output,
        )
    elif kind == "evidence":
        report.plot_evidence_curves(load_result("faithfulness_dev", results), output)
    elif kind == "corrective":
        report.plot_corrective_sweep(load_result("corrective_scifact", results), output)
    elif kind == "query_changes":
        report.plot_retrieval_query_changes(
            load_result("scifact_test", results), load_result("nfcorpus_test", results), output
        )
    else:
        report.plot_retrieval_cost(load_result("corrective_scifact", results), output)


@pytest.mark.parametrize("kind", ["retrieval", "evidence", "corrective", "query_changes", "cost"])
def test_chart_save_failure_closes_figure(monkeypatch, tmp_path, kind):
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure

    open_figures = set(plt.get_fignums())

    def fail(*args, **kwargs):
        raise OSError("Chart destination unavailable")

    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="destination unavailable"):
        _draw_chart(kind, tmp_path / "chart.svg")
    assert set(plt.get_fignums()) == open_figures


@pytest.mark.parametrize(
    "kind,method",
    [
        ("retrieval", "errorbar"),
        ("evidence", "plot"),
        ("corrective", "plot"),
        ("query_changes", "scatter"),
        ("cost", "barh"),
    ],
)
def test_chart_drawing_failure_closes_figure(monkeypatch, tmp_path, kind, method):
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

    open_figures = set(plt.get_fignums())

    def fail(*args, **kwargs):
        raise ValueError("Chart drawing failed")

    monkeypatch.setattr(Axes, method, fail)
    with pytest.raises(ValueError, match="Chart drawing failed"):
        _draw_chart(kind, tmp_path / "chart.svg")
    assert set(plt.get_fignums()) == open_figures


def test_query_change_chart_uses_recorded_query_pairs(monkeypatch, tmp_path, published_results):
    results_dir = Path(__file__).resolve().parents[1] / "outputs/results"
    saved = Mock()
    monkeypatch.setattr(report, "_save_figure", saved)

    report.plot_retrieval_query_changes(
        published_results["scifact_test"],
        published_results["nfcorpus_test"],
        tmp_path / "query-changes.svg",
        results_dir=results_dir,
    )

    figure = saved.call_args.args[0]
    with (results_dir / "retrieval_queries.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for axis, run_name in zip(figure.axes, ("scifact_test", "nfcorpus_test"), strict=True):
        scores = {
            system: {
                row["query_id"]: float(row["ndcg_at_10"])
                for row in rows
                if row["run"] == run_name and row["system"] == system
            }
            for system in ("Dense", "+ Adaptive weights")
        }
        query_ids = published_results[run_name]["query_ids"]
        expected = [
            (scores["Dense"][query_id], scores["+ Adaptive weights"][query_id])
            for query_id in query_ids
        ]
        observed = [
            tuple(point)
            for collection in axis.collections
            for point in np.asarray(collection.get_offsets())
        ]
        assert sorted(observed) == sorted(expected)
        assert len(observed) == published_results[run_name]["num_queries"]
        differences = np.asarray([adaptive - dense for dense, adaptive in expected])
        expected_counts = {
            "Higher": np.count_nonzero(differences > 1e-12),
            "Equal": np.count_nonzero(np.abs(differences) <= 1e-12),
            "Lower": np.count_nonzero(differences < -1e-12),
        }
        for collection in axis.collections:
            group = collection.get_label().split()[0]
            offsets = np.asarray(collection.get_offsets())
            assert len(offsets) == expected_counts[group]
            delta = offsets[:, 1] - offsets[:, 0]
            assert np.all(
                {
                    "Higher": delta > 1e-12,
                    "Equal": np.abs(delta) <= 1e-12,
                    "Lower": delta < -1e-12,
                }[group]
            )
        annotations = " ".join(text.get_text() for text in axis.texts)
        assert f"Mean paired Δ = {differences.mean():+.4f}" in annotations
        for group, count in expected_counts.items():
            assert f"{group} {count}" in annotations
        assert axis.get_xlim() == axis.get_ylim() == (0, 1)
        assert "dense" in axis.get_xlabel().lower()
        assert "adaptive" in axis.get_ylabel().lower()
    assert any(
        str(results_dir / "retrieval_queries.csv") in text.get_text() for text in figure.texts
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "count",
        "boolean_count",
        "duplicate_ids",
        "length",
        "nonfinite",
        "range",
        "overflow",
    ],
)
def test_query_change_chart_rejects_unaligned_records(tmp_path, published_results, invalid):
    from matplotlib import pyplot as plt

    result = deepcopy(published_results["scifact_test"])
    if invalid == "count":
        result["num_queries"] -= 1
    elif invalid == "boolean_count":
        result["num_queries"] = True
    elif invalid == "duplicate_ids":
        result["query_ids"][0] = result["query_ids"][1]
    elif invalid == "length":
        result["per_query_ndcg"]["Dense"].pop()
    else:
        result["per_query_ndcg"]["+ Adaptive weights"][0] = {
            "nonfinite": np.nan,
            "range": 1.1,
            "overflow": 10**1000,
        }[invalid]
    output = tmp_path / "invalid.svg"
    open_figures = set(plt.get_fignums())

    with pytest.raises(ValueError):
        report.plot_retrieval_query_changes(result, published_results["nfcorpus_test"], output)

    assert set(plt.get_fignums()) == open_figures
    assert not output.exists()


def test_retrieval_cost_chart_uses_recorded_measurements(monkeypatch, tmp_path, published_results):
    results_dir = Path(__file__).resolve().parents[1] / "outputs/results"
    saved = Mock()
    monkeypatch.setattr(report, "_save_figure", saved)
    result = published_results["corrective_scifact"]

    report.plot_retrieval_cost(result, tmp_path / "retrieval-cost.svg", results_dir=results_dir)

    figure = saved.call_args.args[0]
    with (results_dir / "corrective_cost_quality.csv").open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["run"] == "corrective_scifact"]
    quality, latency = figure.axes
    assert [bar.get_width() for bar in quality.patches] == [float(row["ndcg"]) for row in rows]
    assert [bar.get_width() for bar in latency.patches] == [
        float(row["latency_ms"]) for row in rows
    ]
    assert quality.get_xlim() == (0, 1)
    assert latency.get_xlim()[0] == 0
    assert [bar.get_y() for bar in quality.patches] == [bar.get_y() for bar in latency.patches]
    assert latency.get_xscale() == "linear"
    assert [tick.get_text() for tick in quality.get_yticklabels()] == [
        "Adaptive hybrid",
        "Corrective PRF",
        "Cross-encoder",
    ]
    caption = " ".join(text.get_text() for text in figure.texts)
    assert f"first {result['cost_quality']['timed_queries']} test queries" in caption
    assert str(results_dir / "corrective_cost_quality.csv") in caption
    for term in (
        "CPU",
        "One timed pass; no dedicated warm-up",
        "Excludes indexing, embedding-model loading and answer generation",
        "Cross-encoder time includes its initial model load",
    ):
        assert term in caption


@pytest.mark.parametrize(
    "invalid",
    [
        "zero_count",
        "boolean_count",
        "count",
        "prefix",
        "duplicate_system",
        "empty_systems",
        "zero_latency",
        "negative_latency",
        "nonfinite_latency",
        "overflow_latency",
        "quality",
    ],
)
def test_retrieval_cost_chart_rejects_invalid_measurement_scope(
    tmp_path, published_results, invalid
):
    from matplotlib import pyplot as plt

    result = deepcopy(published_results["corrective_scifact"])
    cost = result["cost_quality"]
    if invalid == "zero_count":
        cost["timed_queries"] = 0
    elif invalid == "boolean_count":
        cost["timed_queries"] = True
    elif invalid == "count":
        cost["timed_queries"] -= 1
    elif invalid == "prefix":
        cost["query_ids"] = list(reversed(cost["query_ids"]))
    elif invalid == "duplicate_system":
        cost["systems"][1]["system"] = cost["systems"][0]["system"]
    elif invalid == "empty_systems":
        cost["systems"] = []
    elif invalid == "quality":
        cost["systems"][0]["ndcg"] = 1.1
    else:
        cost["systems"][0]["latency_ms"] = {
            "zero_latency": 0,
            "negative_latency": -1,
            "nonfinite_latency": np.nan,
            "overflow_latency": 10**1000,
        }[invalid]
    output = tmp_path / "invalid.svg"
    open_figures = set(plt.get_fignums())

    with pytest.raises(ValueError):
        report.plot_retrieval_cost(result, output)

    assert set(plt.get_fignums()) == open_figures
    assert not output.exists()


@pytest.mark.parametrize("output", ["table", "chart"])
@pytest.mark.parametrize("unsupported", [False, True])
def test_evidence_reports_reject_missing_or_unsupported_methods(tmp_path, output, unsupported):
    faith = {"methods": {"unrecorded_scorer": {}} if unsupported else {}}
    message = "Unsupported evidence methods: unrecorded_scorer" if unsupported else "no methods"
    if output == "table":
        with pytest.raises(ValueError, match=message):
            report.faithfulness_table(faith)
    else:
        with pytest.raises(ValueError, match=message):
            report.plot_evidence_curves(faith, tmp_path / "evidence.svg")
        assert not (tmp_path / "evidence.svg").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_csv_interruption_after_rename_restores_previous_directory(
    monkeypatch, tmp_path, retrieval_result, existing
):
    if existing:
        write_csv_results({"retrieval": retrieval_result}, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    updated = deepcopy(retrieval_result)
    updated["seed"] += 1
    replace = Path.replace
    interrupted = False

    def interrupt_after_rename(path, target):
        nonlocal interrupted
        result = replace(path, target)
        if path.name == "retrieval_queries.csv" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(Path, "replace", interrupt_after_rename)
    with pytest.raises(KeyboardInterrupt):
        write_csv_results({"retrieval": updated}, tmp_path)
    assert interrupted
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), 1j])
def test_invalid_measurement_preserves_previous_result(monkeypatch, tmp_path, score):
    monkeypatch.setattr(reproduce, "RESULTS_DIR", tmp_path)
    result = tmp_path / "retrieval.json"
    original = '{"score": 0.5}\n'
    result.write_text(original, encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        reproduce._write_json({"score": score}, result.name)

    assert result.read_text(encoding="utf-8") == original


def test_rank_metrics_count_only_positive_judgments() -> None:
    ranked = ["zero", "negative", "evidence"]
    relevant = {"zero": 0, "negative": -1, "evidence": 2}

    assert metrics.precision_at_k(ranked, relevant, 3) == pytest.approx(1 / 3)
    assert metrics.recall_at_k(ranked, relevant, 3) == 1.0
    assert metrics.hit_at_k(ranked, relevant, 2) == 0.0
    assert metrics.reciprocal_rank(ranked, relevant) == pytest.approx(1 / 3)
    assert metrics.average_precision(ranked, relevant) == pytest.approx(1 / 3)
    assert metrics.ndcg_at_k(ranked, relevant, 3) == 0.5


def test_unjudged_queries_do_not_enter_metric_averages() -> None:
    scores = metrics.per_query(
        {"unjudged": ["zero"], "judged": ["evidence"]},
        {"unjudged": {"zero": 0, "negative": -1}, "judged": {"evidence": 2}},
    )

    assert list(scores) == ["judged"]
    assert metrics.aggregate(scores)["R@10"] == 1.0


@pytest.mark.parametrize("grade", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_relevance_cannot_produce_reported_scores(grade: float) -> None:
    with pytest.raises(ValueError, match="relevance grades must be finite"):
        metrics.per_query({"query": ["evidence"]}, {"query": {"evidence": grade}})


@pytest.mark.parametrize("pvalue", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_holm_rejects_invalid_pvalues(pvalue: float) -> None:
    with pytest.raises(ValueError, match="p-values must be finite and between 0 and 1"):
        metrics.holm_correction({"invalid": pvalue, "valid": 0.05})


def test_holm_preserves_monotonic_adjustments() -> None:
    assert metrics.holm_correction({"first": 0.01, "last": 0.04, "middle": 0.03}) == {
        "first": 0.03,
        "middle": 0.06,
        "last": 0.06,
    }
    assert metrics.holm_correction({}) == {}


def test_dense_index_keeps_its_own_chunk_snapshot() -> None:
    chunks = corpus_to_chunks({"first": "Original evidence.", "second": "Other evidence."})
    labels = ["original"]
    chunks[0].metadata["labels"] = labels
    embedder = SimpleNamespace(
        normalize=True,
        embed=Mock(return_value=np.eye(2, dtype=np.float32)),
        embed_query=lambda query: np.array([1.0, 0.0], dtype=np.float32),
    )
    retriever = ExactDenseRetriever(embedder, chunks)
    chunks[0].content = "Unindexed replacement."
    chunks[0].document_id = "wrong-document"
    labels.append("changed")
    chunks.reverse()
    chunks.clear()

    results = retriever.retrieve("evidence", top_k=2)

    assert [result.chunk.document_id for result in results] == ["first", "second"]
    assert results[0].chunk.content == "Original evidence."
    assert results[0].chunk.metadata["labels"] == ["original"]
    embedder.embed.assert_called_once_with(
        ["Original evidence.", "Other evidence."], batch_size=64, show_progress=False
    )


@pytest.mark.parametrize("depth,expected", [(7, 50), (50, 50), (60, 60)])
def test_reported_candidate_depth_matches_hybrid_searches(monkeypatch, depth, expected) -> None:
    dataset = datasets.IRDataset(
        name="fixture",
        corpus={
            "first": {"text": "Original evidence."},
            "second": {"text": "Other evidence."},
        },
        queries={"query": "Original evidence"},
        qrels={"query": {"first": 1}},
    )
    embedder = SimpleNamespace(
        normalize=True,
        revision="fixture",
        query_prefix="",
        doc_prefix="",
        embed=lambda *args, **kwargs: np.eye(2, dtype=np.float32),
        embed_query=lambda query: np.array([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(datasets, "load", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(run, "Embedder", lambda **kwargs: embedder)
    retrieve = ExactDenseRetriever.retrieve
    observed_depths = []

    def record_depth(self, query, top_k=5):
        observed_depths.append(top_k)
        return retrieve(self, query, top_k)

    monkeypatch.setattr(ExactDenseRetriever, "retrieve", record_depth)

    result = run.evaluate(depth=depth)

    assert result["depth"] == depth
    assert result["hybrid_candidate_depth"] == expected
    assert observed_depths == [depth, expected, expected, expected]


@pytest.mark.parametrize(
    "model,revision",
    [(None, None), ("custom/splade", "a" * 40)],
)
def test_splade_provenance_survives_csv_export(monkeypatch, tmp_path, model, revision):
    dataset = datasets.IRDataset(
        name="fixture",
        corpus={"document": {"text": "Evidence."}},
        queries={"query": "Which evidence?"},
        qrels={"query": {"document": 1}},
    )
    monkeypatch.setattr(datasets, "load", lambda *args, **kwargs: dataset)
    build_systems = Mock(return_value={"BM25": lambda *args: ["document"]})
    monkeypatch.setattr(run, "build_systems", build_systems)

    result = run.evaluate(include_splade=True, splade_model=model, splade_revision=revision)
    write_csv_results({"splade": result}, tmp_path)
    restored = load_result("splade", tmp_path)

    assert restored == result
    expected_model = model or DEFAULT_MODEL
    expected_revision = revision or HF_REVISIONS[expected_model]
    assert restored["provenance"]["splade_model"] == expected_model
    assert restored["provenance"]["splade_revision"] == expected_revision
    assert build_systems.call_args.kwargs["splade_model"] == expected_model
    assert build_systems.call_args.kwargs["splade_revision"] == expected_revision


@pytest.mark.parametrize("module", [run, corrective], ids=["retrieval", "corrective"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_invalid_result_cannot_overwrite_previous_output(monkeypatch, tmp_path, module, nonfinite):
    output = tmp_path / "results.json"
    previous = json.dumps({"previous": "valid measurement"})
    output.write_text(previous, encoding="utf-8")
    result = {"tau_sweep": [], "best_tau": 0.5, "base_ndcg": 0.5, "nonfinite": nonfinite}
    monkeypatch.setattr(module, "evaluate", lambda **kwargs: result)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--out", str(output)])

    with pytest.raises(ValueError, match="Out of range float values"):
        module.main()

    assert output.read_text(encoding="utf-8") == previous


def _dataset(qid: str = "judged") -> datasets.IRDataset:
    return datasets.IRDataset(
        name="scifact",
        corpus={"doc": {"title": "", "text": "Scientific evidence."}},
        queries={qid: "Which evidence?", "unjudged": "No judgement."},
        qrels={qid: {"doc": 1}},
        revision="corpus-revision",
        qrels_revision="qrels-revision",
        source="test-fixture",
    )


def test_download_failure_never_becomes_full_benchmark(monkeypatch) -> None:
    failure = RuntimeError("download unavailable")
    monkeypatch.setattr(datasets, "load_beir", Mock(side_effect=failure))
    vendored = Mock()
    monkeypatch.setattr(datasets, "load_vendored", vendored)
    with pytest.raises(RuntimeError, match="download unavailable"):
        datasets.load("scifact")
    vendored.assert_not_called()


def test_sample_request_never_attempts_download(monkeypatch) -> None:
    downloaded = Mock()
    sample = _dataset()
    monkeypatch.setattr(datasets, "load_beir", downloaded)
    monkeypatch.setattr(datasets, "load_vendored", Mock(return_value=sample))
    assert datasets.load("scifact", prefer_vendored=True) is sample
    downloaded.assert_not_called()


def test_claims_download_failure_never_becomes_dev_benchmark(monkeypatch) -> None:
    monkeypatch.setattr(faithfulness, "ensure_raw", lambda: False)
    with pytest.raises(RuntimeError, match="select --sample"):
        faithfulness.load_claims("dev")


def test_exact_dense_ties_and_nonpositive_depths(monkeypatch) -> None:
    embedder = Embedder()
    monkeypatch.setattr(embedder, "embed", lambda *a, **k: np.ones((8, 2)))
    monkeypatch.setattr(embedder, "embed_query", lambda q: np.ones(2))
    chunks = corpus_to_chunks({str(i): str(i) for i in range(8)})
    dense = ExactDenseRetriever(embedder, chunks)
    assert [r.chunk.id for r in dense.retrieve("query", top_k=3)] == ["0", "1", "2"]
    assert dense.retrieve("query", top_k=0) == []
    assert dense.retrieve("query", top_k=-1) == []


def test_provenance_identifies_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    first = evaluation_provenance(tmp_path)
    assert evaluation_provenance(tmp_path)["source_sha256"] == first["source_sha256"]
    source.write_text("value = 2\n")
    second = evaluation_provenance(tmp_path)
    assert first["source_sha256"] != second["source_sha256"]
    assert len(first["source_sha256"]) == 64
    assert first["python"] and first["platform"]
    assert {"torch", "sentence-transformers", "transformers", "numpy", "datasets"} <= first[
        "packages"
    ].keys()


def test_retrieval_scores_and_reports_only_judged_queries(monkeypatch) -> None:
    ds = _dataset()
    monkeypatch.setattr(datasets, "load", lambda *a, **k: ds)
    retrieve = Mock(return_value=["doc"])
    monkeypatch.setattr(run, "build_systems", lambda *a, **k: {"BM25": retrieve, "Dense": retrieve})
    result = run.evaluate(use_sample=True, depth=20)
    assert result["num_queries"] == 1
    assert result["query_ids"] == ["judged"]
    assert result["dataset_revision"] == "corpus-revision"
    assert result["qrels_revision"] == "qrels-revision"
    assert result["ndcg_gain"] == "linear"
    assert result["per_query_ndcg"] == {"BM25": [1.0], "Dense": [1.0]}
    assert retrieve.call_count == 2


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": -1}, {"depth": 0}])
def test_invalid_run_settings_rejected_before_models(kwargs) -> None:
    with pytest.raises(ValueError, match="positive"):
        run.evaluate(**kwargs)


def test_corrective_tuning_cannot_use_test_split() -> None:
    with pytest.raises(ValueError, match="splits must be different"):
        corrective.evaluate(split="test", tune_split="test")


def test_corrective_tuning_rejects_overlapping_queries_before_models(monkeypatch) -> None:
    monkeypatch.setattr(datasets, "load", lambda *a, **k: _dataset())
    with pytest.raises(ValueError, match="queries overlap"):
        corrective.evaluate()


def test_corrective_tuning_rejects_a_different_corpus(monkeypatch) -> None:
    test = _dataset("test")
    train = _dataset("train")
    train.corpus["doc"]["text"] = "A different corpus revision."
    monkeypatch.setattr(datasets, "load", Mock(side_effect=[test, train]))
    with pytest.raises(ValueError, match="same corpus"):
        corrective.evaluate()


def test_degenerate_claim_bootstrap_has_clear_error() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        faithfulness._bootstrap_auroc([], [], [])
    with pytest.raises(ValueError, match="both positive and negative"):
        faithfulness._bootstrap_auroc([0.8, 0.7], [1, 1], [0, 1], n_boot=5)


def test_faithfulness_preserves_dev_and_threshold_training_scores(
    monkeypatch, tmp_path: Path
) -> None:
    names = ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl")
    for name in names:
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(faithfulness, "VENDORED", tmp_path)
    dev = [
        faithfulness.Claim(
            2, "claim", "SUPPORT", {("doc", 1)}, [("doc", 0, "no"), ("doc", 1, "yes")]
        )
    ]
    train = [
        faithfulness.Claim(
            1, "claim", "SUPPORT", {("doc", 0)}, [("doc", 0, "yes"), ("doc", 1, "no")]
        )
    ]
    monkeypatch.setattr(
        faithfulness, "load_claims", lambda split, **k: dev if split == "dev" else train
    )
    monkeypatch.setattr(
        faithfulness,
        "GroundingVerifier",
        lambda **k: SimpleNamespace(model_name="fake-nli", revision="model-pin"),
    )
    monkeypatch.setattr(
        faithfulness,
        "_score_methods",
        Mock(
            side_effect=[
                ({"nli": [0.1, 0.9]}, [0, 1], [0, 0]),
                ({"nli": [0.9, 0.1]}, [1, 0], [0, 0]),
            ]
        ),
    )
    monkeypatch.setattr(faithfulness, "_bootstrap_auroc", lambda *a, **k: (1.0, 1.0))
    output = faithfulness.evaluate(prefer_vendored=True, with_reranker=False)
    assert output["per_candidate"]["dev"] == {
        "claim_ids": [2, 2],
        "document_ids": ["doc", "doc"],
        "sentence_indices": [0, 1],
        "labels": [0, 1],
        "scores": {"nli": [0.1, 0.9]},
    }
    assert output["per_candidate"]["train"]["labels"] == [1, 0]
    assert output["per_candidate"]["train"]["scores"] == {"nli": [0.9, 0.1]}
    assert output["threshold_train_claims"] == 1
    assert output["data_sha256"] == {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest() for name in names
    }


def test_sample_reproduction_preserves_full_result_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reproduce, "RESULTS_DIR", tmp_path)
    names = [
        "scifact_test.json",
        "nfcorpus_test.json",
        "scifact_minilm.json",
        "nfcorpus_minilm.json",
        "faithfulness_dev.json",
        "metadata.csv",
        "retrieval_metrics.csv",
    ]
    for name in names:
        (tmp_path / name).write_text("published measurement")
    result = {"query_ids": ["claim-42"], "per_query_ndcg": {"BM25": [0.123456789]}}
    monkeypatch.setattr(reproduce, "run_retrieval", lambda **kwargs: result)
    monkeypatch.setattr(faithfulness, "evaluate", lambda **kwargs: {"split": "sample"})
    tune = Mock()
    monkeypatch.setattr(corrective, "evaluate", tune)
    reproduce.run_all(sample=True)
    assert all((tmp_path / name).read_text() == "published measurement" for name in names)
    assert (tmp_path / "scifact_sample.json").exists()
    assert (tmp_path / "faithfulness_sample.json").exists()
    with (tmp_path / "scifact_sample_per_query.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"query_id": "claim-42", "system": "BM25", "ndcg_at_10": "0.123456789"}]
    tune.assert_not_called()


def test_csv_rejects_scores_without_matching_query_ids(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reproduce, "RESULTS_DIR", tmp_path)
    with pytest.raises(ValueError, match="align with query_ids"):
        reproduce.export_retrieval_csv(
            {"query_ids": ["q"], "per_query_ndcg": {"BM25": [0.1, 0.2]}}, "scores.csv"
        )
    assert not (tmp_path / "scores.csv").exists()


@pytest.mark.parametrize(
    "module,result",
    [
        (run, {}),
        (corrective, {"tau_sweep": [], "best_tau": 0.5, "base_ndcg": 0.5}),
        (
            faithfulness,
            {"num_claims": 1, "num_candidates": 2, "gold_base_rate": 0.5, "methods": {}},
        ),
        (
            generation,
            {
                "generator": "fixture",
                "num_queries": 1,
                "num_corrected": 0,
                "baseline_faithfulness": 0.5,
                "corrective_faithfulness": 0.5,
                "improvement": 0.0,
            },
        ),
        (
            ragtruth,
            {
                "num_responses": 2,
                "hallucination_rate": 0.5,
                "roc_auc": 0.5,
                "risk_coverage_auc": 0.5,
            },
        ),
    ],
    ids=["retrieval", "corrective", "evidence", "generation", "ragtruth"],
)
def test_cli_creates_only_the_requested_output_directory(monkeypatch, tmp_path, module, result):
    default_dir = tmp_path / "generated"
    output = tmp_path / "exports" / "results.json"
    monkeypatch.setattr(module, "RESULTS_DIR", default_dir)
    monkeypatch.setattr(module, "evaluate", lambda **kwargs: result)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--out", str(output)])

    module.main()

    assert json.loads(output.read_text()) == result
    assert not default_dir.exists()


def test_report_accepts_corrective_results_without_reranker() -> None:
    scores = {"BM25": [0.4, 0.6], "Dense": [0.5, 0.7], "Hybrid (RRF)": [0.6, 0.8]}
    retrieval = {
        "num_queries": 2,
        "corpus_size": 10,
        "per_query_ndcg": scores,
        "systems": {
            name: {"means": {"nDCG@10": float(np.mean(values))}} for name, values in scores.items()
        },
    }
    correction = {
        "tau_sweep": [{"trigger_rate": 0.5}],
        "cost_quality": {
            "systems": [
                {"system": "Adaptive", "ndcg": 0.7, "latency_ms": 10},
                {"system": "Corrective PRF", "ndcg": 0.8, "latency_ms": 12},
            ]
        },
    }
    macros = "\n".join(report.build_macros(retrieval, retrieval, None, correction, None))
    assert "\\BaseMs}{10}" in macros
    assert "\\CorrMs}{12}" in macros
    assert "\\BaseND}{0.700}" in macros
    assert "Rerank" not in macros


def test_generation_verifies_only_sources_shown_to_generator(monkeypatch) -> None:
    ds = _dataset()
    ds.queries = {"judged": ds.queries["judged"]}
    text = "a" * 400 + " HIDDEN EVIDENCE"
    chunk = corpus_to_chunks({"doc": text})[0]
    result = [RetrievalResult(chunk=chunk, score=1.0)]
    corrected_chunk = corpus_to_chunks({"corrected": "b" * 400 + " OTHER HIDDEN EVIDENCE"})[0]
    corrected_result = [RetrievalResult(chunk=corrected_chunk, score=1.0)]
    retrieved = Mock(side_effect=[result, corrected_result])
    retriever = SimpleNamespace(retrieve=retrieved, rrf_k=60)
    monkeypatch.setattr(datasets, "load", lambda *a, **k: ds)
    monkeypatch.setattr(generation, "ExactDenseRetriever", lambda *a, **k: object())
    monkeypatch.setattr(generation, "BM25Retriever", lambda: SimpleNamespace(add=lambda c: None))
    monkeypatch.setattr(generation, "AdaptiveHybridRetriever", lambda *a, **k: retriever)
    expansion = Mock(return_value="Expanded query")
    monkeypatch.setattr(
        generation, "CorrectiveRetriever", lambda *a, **k: SimpleNamespace(expand=expansion)
    )
    monkeypatch.setattr(generation, "rrf_fuse", lambda *a: corrected_result)
    prompts = []
    fake_gen = SimpleNamespace(model_name="fake", close=lambda: None)
    fake_gen.generate = lambda prompt: prompts.append(prompt) or "Answer."
    monkeypatch.setattr(generation, "LocalGenerator", lambda *a: fake_gen)
    verified = []
    verifier = SimpleNamespace(model_name="fake-nli")
    verifier.verify = lambda answer, sources: (
        verified.append(sources) or SimpleNamespace(faithfulness=0.2)
    )
    monkeypatch.setattr(generation, "GroundingVerifier", lambda **k: verifier)
    output = generation.evaluate(n=1)
    assert all("HIDDEN EVIDENCE" not in prompt for prompt in prompts)
    assert verified == [["a" * 400], ["b" * 400]]
    expansion.assert_called_once_with(ds.queries["judged"], result)
    assert retrieved.call_args_list[1].args == ("Expanded query",)
    assert output["dataset"] == "scifact"
    assert output["num_corrected"] == 1
    assert output["correction_strategy"] == "forced_prf_on_answer_gate"


@pytest.mark.parametrize("answers_before_failure", [0, 1])
def test_generation_closes_model_when_an_answer_fails(monkeypatch, answers_before_failure) -> None:
    ds = _dataset()
    results = [RetrievalResult(chunk=corpus_to_chunks({"doc": "Scientific evidence."})[0], score=1)]
    retriever = SimpleNamespace(retrieve=lambda *a, **k: results, rrf_k=60)
    monkeypatch.setattr(datasets, "load", lambda *a, **k: ds)
    monkeypatch.setattr(generation, "ExactDenseRetriever", lambda *a, **k: object())
    monkeypatch.setattr(generation, "AdaptiveHybridRetriever", lambda *a, **k: retriever)
    monkeypatch.setattr(
        generation,
        "CorrectiveRetriever",
        lambda *a, **k: SimpleNamespace(expand=lambda *a: "query"),
    )
    generator = SimpleNamespace(
        model_name="fake-generator",
        generate=Mock(
            side_effect=["Answer."] * answers_before_failure + [RuntimeError("failed answer")]
        ),
        close=Mock(),
    )
    monkeypatch.setattr(generation, "LocalGenerator", lambda *a: generator)
    verifier = Mock()
    monkeypatch.setattr(generation, "GroundingVerifier", verifier)
    with pytest.raises(RuntimeError, match="failed answer"):
        generation.evaluate(n=1)
    generator.close.assert_called_once_with()
    verifier.assert_not_called()


@pytest.mark.parametrize("rows,task_type", [([], None), ([{"task_type": "Summary"}], "QA")])
def test_ragtruth_rejects_empty_selection_before_models(monkeypatch, rows, task_type) -> None:
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: rows))
    verifier = Mock()
    monkeypatch.setattr(ragtruth, "GroundingVerifier", verifier)
    with pytest.raises(ValueError, match="no RAGTruth responses"):
        ragtruth.evaluate(task_type=task_type)
    verifier.assert_not_called()


def test_ragtruth_stops_reading_after_requested_response_count(monkeypatch) -> None:
    def responses():
        yield {
            "task_type": "QA",
            "output": "Answer.",
            "context": "Evidence.",
            "hallucination_labels": "[]",
        }
        raise AssertionError("read beyond the requested response count")

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: responses())
    )
    monkeypatch.setattr(
        ragtruth, "GroundingVerifier", lambda **k: SimpleNamespace(model_name="fake-nli")
    )
    monkeypatch.setattr(ragtruth, "_groundedness", lambda *a: 0.9)
    result = ragtruth.evaluate(n=1, task_type="QA")
    assert result["num_responses"] == 1
    assert result["hallucination_rate"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        OSError("cache unavailable"),
        urllib.error.URLError("offline"),
        urllib.error.HTTPError("https://example.test/data.tar.gz", 503, "unavailable", None, None),
        TimeoutError("download timed out"),
    ],
)
def test_claims_expected_download_errors_return_false(monkeypatch, tmp_path, failure) -> None:
    monkeypatch.setattr(faithfulness, "RAW_DIR", tmp_path / "data")
    monkeypatch.setattr(urllib.request, "urlopen", Mock(side_effect=failure))
    assert faithfulness.ensure_raw() is False


def test_claims_unreadable_archive_returns_false(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(faithfulness, "RAW_DIR", tmp_path / "data")
    monkeypatch.setattr(urllib.request, "urlopen", Mock(return_value=io.BytesIO()))
    monkeypatch.setattr(tarfile, "open", Mock(side_effect=tarfile.ReadError("bad archive")))
    assert faithfulness.ensure_raw() is False


def test_claims_cache_directory_error_returns_false(monkeypatch, tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.write_text("a file blocks the required cache directory")
    monkeypatch.setattr(faithfulness, "RAW_DIR", cache / "data")
    download = Mock()
    monkeypatch.setattr(urllib.request, "urlopen", download)
    assert faithfulness.ensure_raw() is False
    download.assert_not_called()


def test_claims_unexpected_download_errors_are_not_hidden(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(faithfulness, "RAW_DIR", tmp_path / "data")
    monkeypatch.setattr(
        urllib.request, "urlopen", Mock(side_effect=RuntimeError("programming error"))
    )
    with pytest.raises(RuntimeError, match="programming error"):
        faithfulness.ensure_raw()


def _claims_archive(*, missing=None, nonregular=None):
    contents = io.BytesIO()
    with tarfile.open(fileobj=contents, mode="w:gz") as archive:
        for name in ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "../outside.txt"):
            if name == missing:
                continue
            member = tarfile.TarInfo(f"data/{name}")
            if name == nonregular:
                member.type = tarfile.SYMTYPE
                member.linkname = "corpus.jsonl"
                archive.addfile(member)
            else:
                payload = (name + "\n").encode()
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    return contents.getvalue()


def test_claims_complete_cache_does_not_download(monkeypatch, tmp_path):
    raw = tmp_path / "data"
    raw.mkdir()
    for name in ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"):
        (raw / name).write_text("existing data", encoding="utf-8")
    download = Mock()
    monkeypatch.setattr(faithfulness, "RAW_DIR", raw)
    monkeypatch.setattr(urllib.request, "urlopen", download)

    assert faithfulness.ensure_raw() is True

    download.assert_not_called()
    assert all(path.read_text() == "existing data" for path in raw.iterdir())


def test_claims_download_publishes_only_required_regular_files(monkeypatch, tmp_path):
    monkeypatch.setattr(faithfulness, "RAW_DIR", tmp_path / "data")
    download = Mock(return_value=io.BytesIO(_claims_archive()))
    monkeypatch.setattr(urllib.request, "urlopen", download)

    assert faithfulness.ensure_raw() is True

    download.assert_called_once_with(faithfulness.SCIFACT_URL, timeout=30)
    assert {path.name for path in tmp_path.iterdir()} == {"data"}
    assert {path.name for path in (tmp_path / "data").iterdir()} == {
        "corpus.jsonl",
        "claims_train.jsonl",
        "claims_dev.jsonl",
    }


@pytest.mark.parametrize("invalid", ["missing", "nonregular"])
def test_claims_invalid_archive_preserves_existing_cache(monkeypatch, tmp_path, invalid):
    raw = tmp_path / "data"
    raw.mkdir()
    previous = raw / "corpus.jsonl"
    previous.write_text("existing corpus", encoding="utf-8")
    monkeypatch.setattr(faithfulness, "RAW_DIR", raw)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        Mock(return_value=io.BytesIO(_claims_archive(**{invalid: "claims_dev.jsonl"}))),
    )

    assert faithfulness.ensure_raw() is False

    assert list(tmp_path.iterdir()) == [raw]
    assert list(raw.iterdir()) == [previous]
    assert previous.read_text() == "existing corpus"


def test_claims_failed_extraction_cannot_become_a_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setattr(faithfulness, "RAW_DIR", tmp_path / "data")
    download = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(_claims_archive()))
    monkeypatch.setattr(urllib.request, "urlopen", download)
    copy = shutil.copyfileobj
    copied = 0

    def fail_after_first_file(source, destination):
        nonlocal copied
        copied += 1
        if copied == 3:
            destination.write(b"partial")
            raise tarfile.ReadError("truncated member")
        return copy(source, destination)

    monkeypatch.setattr(shutil, "copyfileobj", fail_after_first_file)
    assert faithfulness.ensure_raw() is False
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(shutil, "copyfileobj", copy)
    assert faithfulness.ensure_raw() is True
    assert download.call_count == 2


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_claims_failed_publication_preserves_existing_cache(monkeypatch, tmp_path, recovery_fails):
    raw = tmp_path / "data"
    raw.mkdir()
    (raw / "corpus.jsonl").write_text("existing corpus", encoding="utf-8")
    monkeypatch.setattr(faithfulness, "RAW_DIR", raw)
    monkeypatch.setattr(urllib.request, "urlopen", Mock(return_value=io.BytesIO(_claims_archive())))
    replace = Path.replace
    failed = False

    def fail_publication(path, target):
        nonlocal failed
        if target == raw and (recovery_fails or not failed):
            failed = True
            raise OSError("cache unavailable")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    if recovery_fails:
        with pytest.raises(RuntimeError, match="backups retained"):
            faithfulness.ensure_raw()
        backups = list(tmp_path.glob(".scifact-backup-*"))
        assert len(backups) == 1
        assert (backups[0] / "data/corpus.jsonl").read_text() == "existing corpus"
    else:
        assert faithfulness.ensure_raw() is False
        assert list(tmp_path.iterdir()) == [raw]
        assert (raw / "corpus.jsonl").read_text() == "existing corpus"


RESULTS = Path(__file__).resolve().parents[1] / "outputs/results"


PUBLISHED_HASHES = {
    "corrective_scifact": "e676ab547efaee01d044d7861e52866d621ecae7f8c0d1e2f5999f94e7b45d81",
    "faithfulness_dev": "bc2aa20673e6460d501a84f2a254dd301ee13d740fed4e45c3c73443b742638b",
    "nfcorpus_minilm": "5086f461d83c83d30ed30060c593b2cd0a41fe0f67a68e727d35f5cfb2ab41a4",
    "nfcorpus_test": "ebd18ce98465b46551441c9df486a18188f6c6f4ee3e467f8ad09f6d847f4eea",
    "scifact_minilm": "3b0ca296f92b648f9bfa75246f7e8a10910ee632c14d3b5953bb053488dfa52f",
    "scifact_test": "7d6b86fc0544e0dd6fc89d5017bbdf71bb59bae34a26915feb8e09f5232f44fc",
}


@pytest.fixture(scope="module")
def published_results():
    results = {}
    for name in PUBLISHED_HASHES:
        result = load_result(name, RESULTS)
        assert result is not None, f"Missing published result: {name}"
        results[name] = result
    return results


@pytest.mark.parametrize("name", PUBLISHED_HASHES)
def test_published_csv_preserves_original_measurements_and_provenance(published_results, name):
    encoded = json.dumps(
        published_results[name], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == PUBLISHED_HASHES[name]


def test_all_published_results_round_trip_through_csv(tmp_path, published_results):
    write_csv_results(published_results, tmp_path)
    assert all(path.suffix == ".csv" for path in tmp_path.iterdir())
    for name, original in published_results.items():
        restored = load_result(name, tmp_path)
        assert restored == original
        if "systems" in original:
            assert restored is not None
            assert list(restored["systems"]) == list(original["systems"])


def test_csv_preserves_scalar_types_and_quoted_text(tmp_path, published_results):
    original = deepcopy(published_results["scifact_test"])
    scalars = {
        "empty_string": "",
        "numeric_string": "0007",
        "optional_value": None,
        "enabled": False,
        "count": 7,
        "precise_value": 0.12345678901234568,
        "description": 'Comma, quote " and\nnew line; café',
    }
    original.update(scalars)
    original["query_ids"][0] = '001,\n"quoted"'
    write_csv_results({"fixture": original}, tmp_path)
    restored = load_result("fixture", tmp_path)

    assert restored == original
    for key, value in scalars.items():
        assert type(restored[key]) is type(value)


@pytest.mark.parametrize("name", ["fresh", "fresh.json", "fresh.csv"])
def test_json_results_take_precedence_over_csv(tmp_path, name):
    result = {"dataset": "fresh", "score": 0.12345678901234568}
    (tmp_path / "fresh.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "metadata.csv").write_text("malformed CSV", encoding="utf-8")
    assert load_result(name, tmp_path) == result


def test_invalid_json_is_not_hidden_by_csv_fallback(tmp_path, published_results):
    write_csv_results({"scifact_test": published_results["scifact_test"]}, tmp_path)
    (tmp_path / "scifact_test.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        load_result("scifact_test", tmp_path)


def test_missing_results_are_distinct_from_damaged_csv(tmp_path, published_results):
    assert load_result("missing", tmp_path) is None
    write_csv_results({"scifact_test": published_results["scifact_test"]}, tmp_path)
    assert load_result("missing", tmp_path) is None
    (tmp_path / "retrieval_metrics.csv").unlink()
    with pytest.raises(ValueError, match="retrieval_metrics"):
        load_result("scifact_test", tmp_path)


@pytest.mark.parametrize(
    "run,filename",
    [
        ("scifact_test", "retrieval_queries.csv"),
        ("faithfulness_dev", "faithfulness_candidates.csv"),
        ("corrective_scifact", "corrective_sweeps.csv"),
    ],
)
def test_duplicate_measurement_rows_are_rejected(tmp_path, published_results, run, filename):
    write_csv_results({run: published_results[run]}, tmp_path)
    path = tmp_path / filename
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert len(rows) > 1
    with path.open("a", newline="", encoding="utf-8") as output:
        csv.writer(output).writerow(rows[1])
    with pytest.raises(ValueError):
        load_result(run, tmp_path)


def test_unknown_metadata_types_are_rejected(tmp_path, published_results):
    write_csv_results({"scifact_test": published_results["scifact_test"]}, tmp_path)
    path = tmp_path / "metadata.csv"
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[0]["type"] = "unknown"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_result("scifact_test", tmp_path)


@pytest.mark.parametrize("field,value", [("label", "2"), ("score_nli", "nan")])
def test_invalid_candidate_values_are_rejected(tmp_path, published_results, field, value):
    write_csv_results({"faithfulness_dev": published_results["faithfulness_dev"]}, tmp_path)
    path = tmp_path / "faithfulness_candidates.csv"
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_result("faithfulness_dev", tmp_path)


@pytest.mark.parametrize(
    "empty,value",
    [(False, float("nan")), (True, float("nan")), (False, 10**1000), (False, -(10**1000))],
)
def test_invalid_export_preserves_existing_tables(tmp_path, published_results, empty, value):
    write_csv_results(published_results, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    invalid = deepcopy(published_results["scifact_test"])
    invalid["per_query_ndcg"]["Dense"][0] = value

    with pytest.raises(ValueError):
        write_csv_results({} if empty else {"scifact_test": invalid}, tmp_path)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_published_directory_contains_only_csv_and_images():
    paths = list(RESULTS.iterdir())
    assert paths
    extensions = {".csv", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"}
    assert all(path.is_file() and path.suffix.lower() in extensions for path in paths)
