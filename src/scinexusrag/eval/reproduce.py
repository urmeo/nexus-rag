"""Benchmark reproduction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from scinexusrag.eval import corrective, faithfulness
from scinexusrag.eval.run import evaluate as run_retrieval
from scinexusrag.eval.result_files import write_result

RESULTS_DIR = Path("outputs/generated")
SEED = 0
BGE = "BAAI/bge-small-en-v1.5"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _write_json(obj: dict[str, Any], name: str) -> Path:
    path = write_result(obj, RESULTS_DIR / name, trailing_newline=True)
    print(f"  wrote {path}")
    return path


def export_retrieval_csv(result: dict[str, Any], name: str) -> Path:
    per_query_scores = result["per_query_ndcg"]
    query_ids = result["query_ids"]
    num_queries = len(query_ids)
    if any(len(scores) != num_queries for scores in per_query_scores.values()):
        raise ValueError("per-query scores must align with query_ids")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "system", "ndcg_at_10"])
        for system, scores in per_query_scores.items():
            for query_id, score in zip(query_ids, scores, strict=True):
                writer.writerow([query_id, system, repr(score)])
    print(f"  wrote {path} ({num_queries} queries x {len(per_query_scores)} systems)")
    return path


def run_all(sample: bool = False) -> None:
    tag = "sample" if sample else "test"
    for step, (dataset, label, model, model_label) in enumerate(
        (
            ("scifact", "SciFact", BGE, "BGE-small"),
            ("nfcorpus", "NFCorpus", BGE, "BGE-small"),
            ("scifact", "SciFact", MINILM, "MiniLM"),
            ("nfcorpus", "NFCorpus", MINILM, "MiniLM"),
        ),
        start=1,
    ):
        analysis = "ablation" if model == BGE else "baseline"
        print(f"[{step}/6] {label} retrieval {analysis} ({model_label})")
        result = run_retrieval(
            dataset=dataset, split="test", use_sample=sample, embedding_model=model, seed=SEED
        )
        suffix = tag if model == BGE else "minilm_sample" if sample else "minilm"
        _write_json(result, f"{dataset}_{suffix}.json")
        if model == BGE:
            export_retrieval_csv(result, f"{dataset}_{tag}_per_query.csv")

    print("[5/6] Corrective-loop tau selection (SciFact)")
    if sample:
        print("  skipped in --sample (no disjoint tune split in the offline subset)")
    else:
        correction = corrective.evaluate(dataset="scifact", split="test", embedding_model=BGE)
        _write_json(correction, "corrective_scifact.json")

    print("[6/6] Evidence-sentence detection (SciFact claims)")
    evidence = faithfulness.evaluate(prefer_vendored=sample, with_reranker=True, seed=SEED)
    _write_json(evidence, "faithfulness_sample.json" if sample else "faithfulness_dev.json")

    if sample:
        print("\nDone. Sample outputs are smoke checks, not the published benchmark.")
    else:
        print(
            "\nDone. Render these results with: python -m scinexusrag.eval.report --results outputs/generated"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval and evidence benchmarks")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="use the committed offline subsets for a smoke check",
    )
    args = parser.parse_args()
    run_all(sample=args.sample)


if __name__ == "__main__":
    main()
