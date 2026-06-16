"""Run the retrieval ablation and save results."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from nexusrag.eval import datasets as D
from nexusrag.eval import metrics as M
from nexusrag.eval.indexes import corpus_to_chunks
from nexusrag.eval.systems import build_systems
from nexusrag.ingestion import Embedder

RESULTS_DIR = Path("benchmarks/results")


def evaluate(
    dataset: str = "scifact",
    split: str = "test",
    use_sample: bool = False,
    depth: int = 50,
    include_rerank: bool = False,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    limit: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    ds = D.load(dataset, split=split, prefer_vendored=use_sample)
    qids = list(ds.queries)
    if limit:
        qids = qids[:limit]
        ds.qrels = {q: ds.qrels[q] for q in qids if q in ds.qrels}

    corpus_text = {doc_id: ds.doc_text(doc_id) for doc_id in ds.corpus}
    chunks = corpus_to_chunks(corpus_text)

    embedder = Embedder(model_name=embedding_model, device="cpu")
    systems = build_systems(chunks, embedder, include_rerank=include_rerank)

    results: dict[str, dict[str, Any]] = {}
    per_query_ndcg: dict[str, list[float]] = {}

    for name, fn in systems.items():
        t0 = time.perf_counter()
        run = {qid: fn(ds.queries[qid], depth) for qid in qids}
        elapsed = time.perf_counter() - t0

        scores = M.per_query(run, ds.qrels)
        means = M.aggregate(scores)
        cis = {
            metric: M.bootstrap_ci([s[metric] for s in scores.values()], seed=seed)
            for metric in M.METRIC_FNS
        }
        per_query_ndcg[name] = [scores[q]["nDCG@10"] for q in sorted(scores)]
        results[name] = {
            "means": means,
            "ci": {m: {"mean": c[0], "lo": c[1], "hi": c[2]} for m, c in cis.items()},
            "latency_ms_per_query": elapsed / max(1, len(qids)) * 1000,
        }
        print(f"{name:24s} nDCG@10={means['nDCG@10']:.3f}  R@10={means['R@10']:.3f}")

    # significance vs full
    ref = list(systems)[-1]
    for name in systems:
        if name == ref:
            results[name]["p_vs_final"] = None
            continue
        p = M.paired_randomization_test(per_query_ndcg[ref], per_query_ndcg[name], seed=seed)
        results[name]["p_vs_final"] = p

    return {
        "dataset": dataset,
        "split": "sample" if use_sample else split,
        "num_queries": len(qids),
        "corpus_size": len(ds.corpus),
        "depth": depth,
        "embedding_model": embedding_model,
        "reranker": include_rerank,
        "reference_system": ref,
        "seed": seed,
        "systems": results,
        "per_query_ndcg": per_query_ndcg,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="NexusRAG retrieval evaluation")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--sample", action="store_true", help="use vendored offline subset")
    p.add_argument("--depth", type=int, default=50)
    p.add_argument("--rerank", action="store_true", help="add the cross-encoder rerank rung")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = evaluate(
        dataset=args.dataset,
        split=args.split,
        use_sample=args.sample,
        depth=args.depth,
        include_rerank=args.rerank,
        limit=args.limit,
        seed=args.seed,
        embedding_model=args.embedding_model,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "sample" if args.sample else args.split
    path = Path(args.out) if args.out else RESULTS_DIR / f"{args.dataset}_{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
