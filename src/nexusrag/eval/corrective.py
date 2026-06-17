"""Trigger analysis and cost/quality for the corrective retriever."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from nexusrag.eval import datasets as D
from nexusrag.eval import metrics as M
from nexusrag.eval.indexes import ExactDenseRetriever, corpus_to_chunks
from nexusrag.ingestion import Embedder
from nexusrag.retrieval import (
    AdaptiveHybridRetriever,
    BM25Retriever,
    CorrectiveRetriever,
    Reranker,
)
from nexusrag.retrieval.dense import RetrievalResult

RESULTS_DIR = Path("benchmarks/results")
NDCG = M.METRIC_FNS["nDCG@10"]
R20 = M.METRIC_FNS["R@20"]

# Held-out split used to pick tau, so the reported test numbers never tune on
# themselves. NFCorpus ships a validation split; SciFact only has train/test.
TUNE_SPLITS = {"nfcorpus": "validation", "scifact": "train"}
TUNE_LIMIT = 250  # cap tuning queries for a tractable, deterministic sweep


def _ids(results: list[RetrievalResult]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        d = r.chunk.document_id
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _sweep_tau(
    adaptive: AdaptiveHybridRetriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    qids: list[str],
    base_ndcg: dict[str, float],
    taus: tuple[float, ...],
    depth: int,
) -> list[dict[str, Any]]:
    """nDCG, trigger rate and triggered-only delta for each tau over ``qids``."""
    sweep: list[dict[str, Any]] = []
    for tau in taus:
        cr = CorrectiveRetriever(adaptive, tau=tau)
        corr_ndcg: dict[str, float] = {}
        triggered: list[str] = []
        for q in qids:
            res, trig = cr.retrieve_traced(queries[q], top_k=depth, depth=depth)
            corr_ndcg[q] = NDCG(_ids(res), qrels[q])
            if trig:
                triggered.append(q)
        a = [base_ndcg[q] for q in triggered]
        b = [corr_ndcg[q] for q in triggered]
        sweep.append(
            {
                "tau": tau,
                "ndcg": float(np.mean([corr_ndcg[q] for q in qids])),
                "trigger_rate": len(triggered) / len(qids),
                "triggered_n": len(triggered),
                "triggered_base_ndcg": float(np.mean(a)) if a else None,
                "triggered_corr_ndcg": float(np.mean(b)) if b else None,
                "triggered_delta": float(np.mean(b) - np.mean(a)) if a else None,
                "triggered_p": M.paired_randomization_test(b, a) if a else None,
            }
        )
    return sweep


def evaluate(
    dataset: str = "scifact",
    split: str = "test",
    tune_split: str | None = None,
    taus: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    depth: int = 50,
    with_reranker: bool = True,
) -> dict[str, Any]:
    ds = D.load(dataset, split=split)
    qids = [q for q in ds.queries if ds.qrels.get(q)]
    if not qids:
        raise ValueError(f"no judged queries for {dataset}/{split}")
    chunks = corpus_to_chunks({d: ds.doc_text(d) for d in ds.corpus})

    embedder = Embedder(model_name=embedding_model, device="cpu")
    dense = ExactDenseRetriever(embedder, chunks)
    bm25 = BM25Retriever()
    bm25.add(chunks)
    adaptive = AdaptiveHybridRetriever(dense, bm25, 0.5, 0.5)

    def base_ndcg_for(qs: dict[str, str], rels: dict[str, dict[str, int]], ids: list[str]) -> dict[str, float]:
        base = {q: _ids(adaptive.retrieve(qs[q], top_k=depth, depth=depth)) for q in ids}
        return {q: NDCG(base[q], rels[q]) for q in ids}

    # Pick tau on a held-out split (same shared corpus, different queries), so the
    # reported test sweep is never used to select its own hyperparameter.
    tune_split = tune_split or TUNE_SPLITS.get(dataset, "train")
    ds_tune = D.load(dataset, split=tune_split)
    tune_qids = [q for q in ds_tune.queries if ds_tune.qrels.get(q)][:TUNE_LIMIT]
    if not tune_qids:
        raise ValueError(f"no judged queries in tune split {dataset}/{tune_split}")
    tune_base = base_ndcg_for(ds_tune.queries, ds_tune.qrels, tune_qids)
    tune_sweep = _sweep_tau(
        adaptive, ds_tune.queries, ds_tune.qrels, tune_qids, tune_base, taus, depth
    )
    best_tau = float(max(tune_sweep, key=lambda s: float(s["ndcg"]))["tau"])

    # Report the full sweep on the test split for transparency.
    base_ndcg = base_ndcg_for(ds.queries, ds.qrels, qids)
    sweep = _sweep_tau(adaptive, ds.queries, ds.qrels, qids, base_ndcg, taus, depth)
    selected = next(s for s in sweep if abs(float(s["tau"]) - best_tau) < 1e-9)
    cost = _cost_quality(adaptive, ds, qids, best_tau, depth, with_reranker)

    return {
        "dataset": dataset,
        "split": split,
        "tune_split": tune_split,
        "tune_queries": len(tune_qids),
        "embedding_model": embedding_model,
        "num_queries": len(qids),
        "base_system": "+ Adaptive weights",
        "base_ndcg": float(np.mean(list(base_ndcg.values()))),
        "tau_sweep": sweep,
        "tune_sweep": tune_sweep,
        "best_tau": best_tau,
        "best_tau_selected_on": tune_split,
        "test_ndcg_at_best_tau": float(selected["ndcg"]),
        "cost_quality": cost,
    }


def _cost_quality(
    adaptive: AdaptiveHybridRetriever,
    ds: D.IRDataset,
    qids: list[str],
    tau: float,
    depth: int,
    with_reranker: bool = True,
) -> dict[str, Any]:
    rows = []
    sample = qids[:120]
    sub_qrels = {q: ds.qrels[q] for q in sample}

    def timed(name: str, fn: Any) -> None:
        t0 = time.perf_counter()
        run = {q: fn(ds.queries[q]) for q in sample}
        ms = (time.perf_counter() - t0) / len(sample) * 1000
        scores = M.per_query(run, sub_qrels)
        ndcg = float(np.mean([s["nDCG@10"] for s in scores.values()]))
        r20 = float(np.mean([s["R@20"] for s in scores.values()]))
        rows.append({"system": name, "ndcg": ndcg, "r20": r20, "latency_ms": ms})

    corrective = CorrectiveRetriever(adaptive, tau=tau)

    timed("Adaptive", lambda q: _ids(adaptive.retrieve(q, top_k=depth, depth=depth)))
    timed("Corrective PRF", lambda q: _ids(corrective.retrieve(q, top_k=depth, depth=depth)))
    if with_reranker:
        reranker = Reranker(device="cpu")
        timed(
            "Rerank (cross-enc)",
            lambda q: _ids(
                reranker.rerank(q, adaptive.retrieve(q, top_k=depth, depth=depth), top_k=depth)
            ),
        )
    return {"systems": rows, "timed_queries": len(sample)}


def main() -> None:
    p = argparse.ArgumentParser(description="Corrective retrieval analysis")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--tune-split", default=None, help="held-out split for tau selection")
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--no-reranker", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = evaluate(
        dataset=args.dataset,
        split=args.split,
        tune_split=args.tune_split,
        embedding_model=args.embedding_model,
        with_reranker=not args.no_reranker,
    )
    for s in res["tau_sweep"]:
        print(
            f"tau={s['tau']:.2f}  nDCG={s['ndcg']:.3f}  fire={s['trigger_rate']:.2f}  "
            f"triggered Δ={s['triggered_delta']}"
        )
    print(f"best tau={res['best_tau']}  base nDCG={res['base_ndcg']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"corrective_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
