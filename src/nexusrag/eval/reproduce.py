"""Single entrypoint that regenerates every committed README benchmark number.

Runs the full retrieval ablation on both datasets (BGE-small and the MiniLM
baseline), the corrective-loop tau selection, and the evidence-detection eval,
all with fixed seeds, writing to ``benchmarks/results/``. Then exports the
per-example (per-query / per-method) raw scores as CSV alongside the JSON so the
headline table can be audited row by row.

    python -m nexusrag.eval        # or: make reproduce

The full run downloads the BEIR corpora and small models on first use and takes
~15-25 min per dataset on CPU; seeds are 0 throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from nexusrag.eval import corrective as C
from nexusrag.eval import faithfulness as F
from nexusrag.eval.run import evaluate as run_retrieval

RESULTS_DIR = Path("benchmarks/results")
SEED = 0
BGE = "BAAI/bge-small-en-v1.5"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _write_json(obj: dict[str, Any], name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2))
    print(f"  wrote {path}")
    return path


def export_retrieval_csv(result: dict[str, Any], name: str) -> Path:
    """Flatten per-query nDCG@10 for every system into one tidy CSV."""
    pq = result["per_query_ndcg"]
    n = len(next(iter(pq.values())))
    path = RESULTS_DIR / name
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_index", "system", "ndcg_at_10"])
        for system, scores in pq.items():
            for i, s in enumerate(scores):
                w.writerow([i, system, f"{s:.6f}"])
    print(f"  wrote {path} ({n} queries x {len(pq)} systems)")
    return path


def run_all(sample: bool = False) -> None:
    print("[1/6] SciFact retrieval ablation (BGE-small)")
    sci = run_retrieval(
        dataset="scifact", split="test", use_sample=sample, embedding_model=BGE, seed=SEED
    )
    _write_json(sci, "scifact_test.json")
    export_retrieval_csv(sci, "scifact_test_per_query.csv")

    print("[2/6] NFCorpus retrieval ablation (BGE-small)")
    nf = run_retrieval(
        dataset="nfcorpus", split="test", use_sample=sample, embedding_model=BGE, seed=SEED
    )
    _write_json(nf, "nfcorpus_test.json")
    export_retrieval_csv(nf, "nfcorpus_test_per_query.csv")

    print("[3/6] SciFact retrieval baseline (MiniLM)")
    sci_m = run_retrieval(
        dataset="scifact", split="test", use_sample=sample, embedding_model=MINILM, seed=SEED
    )
    _write_json(sci_m, "scifact_minilm.json")

    print("[4/6] NFCorpus retrieval baseline (MiniLM)")
    nf_m = run_retrieval(
        dataset="nfcorpus", split="test", use_sample=sample, embedding_model=MINILM, seed=SEED
    )
    _write_json(nf_m, "nfcorpus_minilm.json")

    print("[5/6] Corrective-loop tau selection (SciFact)")
    if sample:
        # The vendored sample has a single 50-query split, so tune and test would
        # collapse to the same queries; skip rather than tune-and-test on a leak.
        print("  skipped in --sample (no disjoint tune split in the offline subset)")
    else:
        corr = C.evaluate(dataset="scifact", split="test", embedding_model=BGE)
        _write_json(corr, "corrective_scifact.json")

    print("[6/6] Evidence-detection eval (SciFact claims)")
    faith = F.evaluate(prefer_vendored=sample, with_reranker=True, seed=SEED)
    _write_json(faith, "faithfulness_dev.json")

    print("\nDone. Regenerate the paper tables with: python -m nexusrag.eval.report")


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate all committed README benchmark numbers")
    p.add_argument(
        "--sample",
        action="store_true",
        help="use the vendored offline subset (fast smoke, not the headline numbers)",
    )
    args = p.parse_args()
    run_all(sample=args.sample)


if __name__ == "__main__":
    main()
