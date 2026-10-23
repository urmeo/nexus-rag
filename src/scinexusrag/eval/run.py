"""Retrieval evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from scinexusrag.eval import datasets, metrics
from scinexusrag.eval.indexes import corpus_to_chunks
from scinexusrag.eval.provenance import evaluation_provenance
from scinexusrag.eval.result_files import write_result
from scinexusrag.eval.systems import MIN_CANDIDATE_DEPTH, build_systems
from scinexusrag.ingestion.embedder import Embedder
from scinexusrag.settings import HF_REVISIONS

RESULTS_DIR = Path("outputs/generated")


def evaluate(
    dataset: str = "scifact",
    split: str = "test",
    use_sample: bool = False,
    depth: int = 50,
    include_rerank: bool = False,
    include_splade: bool = False,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    limit: int | None = None,
    seed: int = 0,
    splade_model: str | None = None,
    splade_revision: str | None = None,
) -> dict[str, Any]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    provenance = evaluation_provenance()
    if include_splade:
        from scinexusrag.retrieval.splade import DEFAULT_MODEL

        splade_model = splade_model or DEFAULT_MODEL
        if splade_revision is None:
            splade_revision = HF_REVISIONS.get(splade_model)
        provenance.update(splade_model=splade_model, splade_revision=splade_revision)
    ds = datasets.load(dataset, split=split, prefer_vendored=use_sample)
    qids = [q for q in ds.queries if ds.qrels.get(q)]
    if limit is not None:
        qids = qids[:limit]
    if not qids:
        raise ValueError(f"no judged queries for {dataset}/{split}")
    qrels = {q: ds.qrels[q] for q in qids}

    corpus_text = {doc_id: ds.doc_text(doc_id) for doc_id in ds.corpus}
    chunks = corpus_to_chunks(corpus_text)

    embedder = Embedder(model_name=embedding_model, device="cpu")
    systems = build_systems(
        chunks,
        embedder,
        include_rerank=include_rerank,
        include_splade=include_splade,
        splade_model=splade_model,
        splade_revision=splade_revision,
    )

    results: dict[str, dict[str, Any]] = {}
    per_query_ndcg: dict[str, list[float]] = {}

    for name, fn in systems.items():
        t0 = time.perf_counter()
        run = {qid: fn(ds.queries[qid], depth) for qid in qids}
        elapsed = time.perf_counter() - t0

        scores = metrics.per_query(run, qrels)
        means = metrics.aggregate(scores)
        cis = {
            metric: metrics.bootstrap_ci([s[metric] for s in scores.values()], seed=seed)
            for metric in metrics.METRIC_FNS
        }
        per_query_ndcg[name] = [scores[q]["nDCG@10"] for q in sorted(scores)]
        results[name] = {
            "means": means,
            "ci": {m: {"mean": c[0], "lo": c[1], "hi": c[2]} for m, c in cis.items()},
            "latency_ms_per_query": elapsed / len(qids) * 1000,
        }
        print(f"{name:24s} nDCG@10={means['nDCG@10']:.3f}  R@10={means['R@10']:.3f}")

    ref = "+ Corrective PRF" if "+ Corrective PRF" in systems else list(systems)[-1]
    for name in systems:
        if name == ref:
            results[name]["p_vs_final"] = None
            continue
        p = metrics.paired_randomization_test(per_query_ndcg[ref], per_query_ndcg[name], seed=seed)
        results[name]["p_vs_final"] = p

    if "BM25" in per_query_ndcg:
        base = per_query_ndcg["BM25"]
        for name in systems:
            if name == "BM25":
                results[name]["delta_vs_bm25"] = None
                results[name]["p_vs_bm25"] = None
                continue
            d, lo, hi = metrics.paired_delta_ci(per_query_ndcg[name], base, seed=seed)
            results[name]["delta_vs_bm25"] = {"mean": d, "lo": lo, "hi": hi}
            results[name]["p_vs_bm25"] = metrics.paired_randomization_test(
                per_query_ndcg[name], base, seed=seed
            )

    return {
        "provenance": provenance,
        "dataset": dataset,
        "split": "sample" if use_sample else split,
        "dataset_revision": ds.revision,
        "qrels_revision": ds.qrels_revision,
        "dataset_source": ds.source,
        "num_queries": len(qids),
        "query_ids": sorted(qids),
        "corpus_size": len(ds.corpus),
        "depth": depth,
        "hybrid_candidate_depth": max(depth, MIN_CANDIDATE_DEPTH),
        "rrf_k": 60,
        "ndcg_gain": "linear",
        "embedding_model": embedding_model,
        "embedding_revision": embedder.revision,
        "query_prefix": embedder.query_prefix,
        "doc_prefix": embedder.doc_prefix,
        "reranker": include_rerank,
        "splade": include_splade,
        "reference_system": ref,
        "seed": seed,
        "systems": results,
        "per_query_ndcg": per_query_ndcg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NexusRAG retrieval evaluation")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", action="store_true", help="use vendored offline subset")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--rerank", action="store_true", help="include cross-encoder reranking")
    parser.add_argument("--splade", action="store_true", help="add the SPLADE baseline")
    parser.add_argument("--splade-model", default=None, help="override the SPLADE model repository")
    parser.add_argument(
        "--splade-revision", default=None, help="override the SPLADE model revision"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate(
        dataset=args.dataset,
        split=args.split,
        use_sample=args.sample,
        depth=args.depth,
        include_rerank=args.rerank,
        include_splade=args.splade,
        splade_model=args.splade_model,
        splade_revision=args.splade_revision,
        limit=args.limit,
        seed=args.seed,
        embedding_model=args.embedding_model,
    )

    tag = "sample" if args.sample else args.split
    path = Path(args.out) if args.out else RESULTS_DIR / f"{args.dataset}_{tag}.json"
    write_result(result, path)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
