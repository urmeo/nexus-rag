"""RAGTruth evaluation."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from scinexusrag.eval.datasets import DATASET_REVISIONS
from scinexusrag.eval.metrics import risk_coverage_auc, roc_auc
from scinexusrag.eval.provenance import evaluation_provenance
from scinexusrag.eval.result_files import write_result
from scinexusrag.generation.grounding import GroundingVerifier, split_sentences

RESULTS_DIR = Path("outputs/generated")
MAX_CTX_SENTS = 50
MAX_OUT_SENTS = 12


def _groundedness(verifier: GroundingVerifier, output: str, context: str) -> float:
    out_sents = split_sentences(output)[:MAX_OUT_SENTS]
    if not out_sents or not context.strip():
        raise ValueError("groundedness requires nonempty output and context")
    ctx_sents = split_sentences(context)[:MAX_CTX_SENTS] or [context[:2000]]
    best = []
    for sent in out_sents:
        scores = np.asarray(verifier.entailment_scores([(c, sent) for c in ctx_sents]), dtype=float)
        if (
            scores.shape != (len(ctx_sents),)
            or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
        ):
            raise ValueError("NLI scores must be finite probabilities for every context sentence")
        best.append(float(scores.max()))
    return float(np.mean(best))


def _is_hallucinated(row: dict[str, Any]) -> int:
    labels = row.get("hallucination_labels")
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except json.JSONDecodeError as exc:
            raise ValueError("RAGTruth hallucination_labels must contain a JSON list") from exc
    if labels is not None and not isinstance(labels, list):
        raise ValueError("RAGTruth hallucination_labels must be a list")
    if labels:
        return 1
    processed = row.get("hallucination_labels_processed")
    if processed is None:
        if labels is None:
            raise ValueError("RAGTruth response is missing hallucination annotations")
        return 0
    if not isinstance(processed, dict) or any(
        not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
        for value in processed.values()
    ):
        raise ValueError("RAGTruth processed annotations must contain nonnegative finite counts")
    return int(any(processed.values()))


def evaluate(
    n: int = 400, task_type: str | None = None, model_name: str | None = None
) -> dict[str, Any]:
    from datasets import load_dataset

    if n <= 0:
        raise ValueError("n must be positive")
    provenance = evaluation_provenance()
    revision = DATASET_REVISIONS["ragtruth"]
    data = load_dataset("wandb/RAGTruth-processed", split="train", revision=revision)
    selected_rows = (row for row in data if task_type is None or row["task_type"] == task_type)
    rows = list(islice(selected_rows, n))
    if not rows:
        raise ValueError(f"no RAGTruth responses match task_type={task_type!r}")
    for row in rows:
        if any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("output", "context")
        ):
            raise ValueError("RAGTruth responses require nonempty output and context")
    labels = [_is_hallucinated(row) for row in rows]

    verifier = (
        GroundingVerifier(model_name=model_name, device="cpu")
        if model_name
        else GroundingVerifier(device="cpu")
    )
    grounded = [_groundedness(verifier, row["output"], row["context"]) for row in rows]

    halluc_score = [1.0 - g for g in grounded]
    faithful = [1 - y for y in labels]
    auroc = roc_auc(halluc_score, labels)
    return {
        "provenance": provenance,
        "dataset": "ragtruth",
        "dataset_revision": revision,
        "task_type": task_type or "all",
        "num_responses": len(rows),
        "hallucination_rate": sum(labels) / len(labels),
        "nli_model": verifier.model_name,
        "roc_auc": None if np.isnan(auroc) else auroc,
        "risk_coverage_auc": risk_coverage_auc(grounded, faithful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGTruth hallucination-detection evaluation")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--task-type", default=None, help="QA / Summary / Data2txt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate(n=args.n, task_type=args.task_type, model_name=args.model)
    auroc = "undefined" if result["roc_auc"] is None else f"{result['roc_auc']:.3f}"
    print(
        f"n={result['num_responses']}  halluc_rate={result['hallucination_rate']:.3f}  "
        f"AUROC={auroc}  AURC={result['risk_coverage_auc']:.3f}"
    )

    out = Path(args.out) if args.out else RESULTS_DIR / "ragtruth.json"
    write_result(result, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
