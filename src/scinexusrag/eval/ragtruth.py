"""Hallucination detection on RAGTruth with the NLI grounding verifier.

For each generated response we score how well each response sentence is
entailed by the source context, then test whether that groundedness signal
separates faithful from hallucinated responses (ROC-AUC, risk-coverage).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scinexusrag.eval.metrics import risk_coverage_auc, roc_auc
from scinexusrag.generation.grounding import GroundingVerifier, split_sentences

RESULTS_DIR = Path("benchmarks/results")
MAX_CTX_SENTS = 50
MAX_OUT_SENTS = 12


def _groundedness(verifier: GroundingVerifier, output: str, context: str) -> float:
    out_sents = split_sentences(output)[:MAX_OUT_SENTS]
    ctx_sents = split_sentences(context)[:MAX_CTX_SENTS] or [context[:2000]]
    if not out_sents:
        return 1.0
    best = []
    for sent in out_sents:
        scores = verifier.entailment_scores([(c, sent) for c in ctx_sents])
        best.append(max(scores) if scores else 0.0)
    return float(np.mean(best))


def _is_hallucinated(row: dict[str, Any]) -> int:
    if row.get("hallucination_labels"):
        return 1
    processed = row.get("hallucination_labels_processed") or {}
    return 1 if sum(processed.values()) > 0 else 0


def evaluate(
    n: int = 400, task_type: str | None = None, model_name: str | None = None
) -> dict[str, Any]:
    from datasets import load_dataset

    data = load_dataset("wandb/RAGTruth-processed", split="train")
    rows = [r for r in data if task_type is None or r["task_type"] == task_type][:n]

    verifier = (
        GroundingVerifier(model_name=model_name, device="cpu")
        if model_name
        else GroundingVerifier(device="cpu")
    )
    grounded: list[float] = []
    labels: list[int] = []
    for r in rows:
        grounded.append(_groundedness(verifier, r["output"], r["context"]))
        labels.append(_is_hallucinated(r))

    halluc_score = [1.0 - g for g in grounded]
    faithful = [1 - y for y in labels]
    return {
        "dataset": "ragtruth",
        "task_type": task_type or "all",
        "num_responses": len(rows),
        "hallucination_rate": sum(labels) / len(labels) if labels else 0.0,
        "nli_model": verifier.model_name,
        "roc_auc": roc_auc(halluc_score, labels),
        "risk_coverage_auc": risk_coverage_auc(grounded, faithful),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="RAGTruth hallucination-detection evaluation")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--task-type", default=None, help="QA / Summary / Data2txt")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = evaluate(n=args.n, task_type=args.task_type, model_name=args.model)
    print(
        f"n={res['num_responses']}  halluc_rate={res['hallucination_rate']:.3f}  "
        f"AUROC={res['roc_auc']:.3f}  AURC={res['risk_coverage_auc']:.3f}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / "ragtruth.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
