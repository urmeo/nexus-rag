"Single entrypoint that regenerates every committed README benchmark number."

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scinexusrag.eval import corrective as C
from scinexusrag.eval import faithfulness as F
from scinexusrag.eval.run import evaluate as run_retrieval

RESULTS_DIR = Path("outputs/generated")
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
    qids = result["query_ids"]
    n = len(qids)
    if any(len(scores) != n for scores in pq.values()):
        raise ValueError("per-query scores must align with query_ids")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "system", "ndcg_at_10"])
        for system, scores in pq.items():
            for qid, s in zip(qids, scores, strict=True):
                w.writerow([qid, system, repr(s)])
    print(f"  wrote {path} ({n} queries x {len(pq)} systems)")
    return path


def run_all(sample: bool = False) -> None:
    tag = "sample" if sample else "test"
    print("[1/6] SciFact retrieval ablation (BGE-small)")
    sci = run_retrieval(
        dataset="scifact", split="test", use_sample=sample, embedding_model=BGE, seed=SEED
    )
    _write_json(sci, f"scifact_{tag}.json")
    export_retrieval_csv(sci, f"scifact_{tag}_per_query.csv")

    print("[2/6] NFCorpus retrieval ablation (BGE-small)")
    nf = run_retrieval(
        dataset="nfcorpus", split="test", use_sample=sample, embedding_model=BGE, seed=SEED
    )
    _write_json(nf, f"nfcorpus_{tag}.json")
    export_retrieval_csv(nf, f"nfcorpus_{tag}_per_query.csv")

    print("[3/6] SciFact retrieval baseline (MiniLM)")
    sci_m = run_retrieval(
        dataset="scifact", split="test", use_sample=sample, embedding_model=MINILM, seed=SEED
    )
    _write_json(sci_m, "scifact_minilm_sample.json" if sample else "scifact_minilm.json")

    print("[4/6] NFCorpus retrieval baseline (MiniLM)")
    nf_m = run_retrieval(
        dataset="nfcorpus", split="test", use_sample=sample, embedding_model=MINILM, seed=SEED
    )
    _write_json(nf_m, "nfcorpus_minilm_sample.json" if sample else "nfcorpus_minilm.json")

    print("[5/6] Corrective-loop tau selection (SciFact)")
    if sample:
        print("  skipped in --sample (no disjoint tune split in the offline subset)")
    else:
        corr = C.evaluate(dataset="scifact", split="test", embedding_model=BGE)
        _write_json(corr, "corrective_scifact.json")

    print("[6/6] Evidence-detection eval (SciFact claims)")
    faith = F.evaluate(prefer_vendored=sample, with_reranker=True, seed=SEED)
    _write_json(faith, "faithfulness_sample.json" if sample else "faithfulness_dev.json")

    if sample:
        print("\nDone. Sample outputs are smoke checks, not the published benchmark.")
    else:
        print(
            "\nDone. Render these results with: python -m scinexusrag.eval.report --results outputs/generated"
        )


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
