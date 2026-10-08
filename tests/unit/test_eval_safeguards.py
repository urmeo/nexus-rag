"""Offline regressions for benchmark provenance and held-out evaluation."""

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scinexusrag.eval import corrective, datasets, faithfulness, generation, report, reproduce, run
from scinexusrag.eval.indexes import ExactDenseRetriever, corpus_to_chunks
from scinexusrag.eval.provenance import evaluation_provenance
from scinexusrag.ingestion import Embedder
from scinexusrag.retrieval.dense import RetrievalResult


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
    assert set(output["data_sha256"]) == set(names)
    assert all(len(value) == 64 for value in output["data_sha256"].values())


def test_sample_reproduction_preserves_full_result_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reproduce, "RESULTS_DIR", tmp_path)
    names = [
        "scifact_test.json",
        "nfcorpus_test.json",
        "scifact_minilm.json",
        "nfcorpus_minilm.json",
        "faithfulness_dev.json",
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
