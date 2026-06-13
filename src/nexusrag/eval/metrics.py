"""Ranking metrics with bootstrap significance testing."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

Qrels = Mapping[str, set[str]]
Run = Mapping[str, Sequence[str]]
MetricFn = Callable[[Sequence[str], set[str]], float]


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for d in top if d in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for d in top if d in relevant)
    return hits / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    # binary gains
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


METRIC_FNS: dict[str, MetricFn] = {
    "P@1": lambda r, rel: precision_at_k(r, rel, 1),
    "R@5": lambda r, rel: recall_at_k(r, rel, 5),
    "R@10": lambda r, rel: recall_at_k(r, rel, 10),
    "R@20": lambda r, rel: recall_at_k(r, rel, 20),
    "nDCG@10": lambda r, rel: ndcg_at_k(r, rel, 10),
    "MRR": reciprocal_rank,
    "MAP": average_precision,
    "Hit@3": lambda r, rel: hit_at_k(r, rel, 3),
}


def per_query(run: Run, qrels: Qrels) -> dict[str, dict[str, float]]:
    """Score every query, every metric."""
    out: dict[str, dict[str, float]] = {}
    for qid, relevant in qrels.items():
        if not relevant:
            continue
        ranked = run.get(qid, [])
        out[qid] = {name: fn(ranked, relevant) for name, fn in METRIC_FNS.items()}
    return out


def aggregate(scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Mean over queries."""
    if not scores:
        return dict.fromkeys(METRIC_FNS, 0.0)
    means: dict[str, float] = {}
    for name in METRIC_FNS:
        vals = [s[name] for s in scores.values()]
        means[name] = sum(vals) / len(vals)
    return means


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Mean with percentile bootstrap interval."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return float(arr.mean()), lo, hi


def paired_randomization_test(
    a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0
) -> float:
    """Two-sided p-value for mean(a) - mean(b)."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size != y.size or x.size == 0:
        return 1.0
    diff = x - y
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    perm_means = np.abs((signs * diff).mean(axis=1))
    return float((perm_means >= observed - 1e-12).mean())
