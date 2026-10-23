"""Ranking metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Relevance = Collection[str]
Qrels = Mapping[str, Relevance]
Run = Mapping[str, Sequence[str]]
MetricFn = Callable[[Sequence[str], Relevance], float]


def _positive_relevance(relevant: Relevance) -> Relevance:
    if not isinstance(relevant, Mapping):
        return relevant
    grades = {str(document_id): float(grade) for document_id, grade in relevant.items()}
    if not all(math.isfinite(grade) for grade in grades.values()):
        raise ValueError("relevance grades must be finite")
    return {document_id: grade for document_id, grade in grades.items() if grade > 0}


def precision_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    relevant = _positive_relevance(relevant)
    if k <= 0:
        return 0.0
    return sum(document_id in relevant for document_id in ranked[:k]) / k


def recall_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    relevant = _positive_relevance(relevant)
    if not relevant or k <= 0:
        return 0.0
    return sum(document_id in relevant for document_id in ranked[:k]) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    relevant = _positive_relevance(relevant)
    return 1.0 if k > 0 and any(d in relevant for d in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Relevance) -> float:
    relevant = _positive_relevance(relevant)
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], relevant: Relevance) -> float:
    relevant = _positive_relevance(relevant)
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Linear gains."""
    relevant = _positive_relevance(relevant)
    if k <= 0:
        return 0.0
    grades = dict(relevant) if isinstance(relevant, Mapping) else dict.fromkeys(relevant, 1.0)
    dcg = sum(
        grades.get(document_id, 0.0) / math.log2(i + 1)
        for i, document_id in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1) if g > 0)
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
    out: dict[str, dict[str, float]] = {}
    for qid, relevant in qrels.items():
        relevant = _positive_relevance(relevant)
        if not relevant:
            continue
        ranked = run.get(qid, [])
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"duplicate ranked document ids for query {qid}")
        out[qid] = {name: fn(ranked, relevant) for name, fn in METRIC_FNS.items()}
    return out


def aggregate(scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not scores:
        return dict.fromkeys(METRIC_FNS, 0.0)
    return {name: sum(row[name] for row in scores.values()) / len(scores) for name in METRIC_FNS}


def _avg_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    return np.asarray(ranks[inverse], dtype=np.float64)


def _binary_inputs(
    scores: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    s = np.asarray(scores, dtype=np.float64)
    raw_y = np.asarray(labels)
    if s.ndim != 1 or raw_y.ndim != 1 or s.size != raw_y.size:
        raise ValueError("scores and labels must be one-dimensional with equal length")
    if not np.all(np.isfinite(s)) or not np.all(np.isin(raw_y, [0, 1])):
        raise ValueError("scores must be finite and labels must be binary")
    return s, raw_y.astype(np.int64)


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-sum AUROC."""
    s, y = _binary_inputs(scores, labels)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision."""
    s, y = _binary_inputs(scores, labels)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]

    ends = np.r_[np.flatnonzero(np.diff(s[order])), s.size - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    recall = tp / n_pos
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def risk_coverage_auc(confidence: ArrayLike, correct: ArrayLike) -> float:
    """Tie-averaged risk."""
    conf, labels = _binary_inputs(confidence, correct)
    err = 1 - labels
    if conf.size == 0:
        return float("nan")
    order = np.argsort(-conf, kind="mergesort")
    sorted_err = err[order]
    ends = np.r_[np.flatnonzero(np.diff(conf[order])) + 1, conf.size]
    expected_err = np.empty(conf.size, dtype=np.float64)
    start = 0
    previous_errors = 0.0
    for end in ends:
        group_errors = float(sorted_err[start:end].sum())
        expected_err[start:end] = previous_errors + np.arange(1, end - start + 1) * (
            group_errors / (end - start)
        )
        previous_errors += group_errors
        start = end
    risk = expected_err / np.arange(1, conf.size + 1)
    return float(risk.mean())


def bootstrap_ci(
    values: ArrayLike, n_boot: int = 10000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap interval."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError("values must be a finite one-dimensional sequence")
    if n_boot <= 0 or not 0 < ci < 1:
        raise ValueError("n_boot must be positive and ci must be between 0 and 1")
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return float(arr.mean()), lo, hi


def paired_delta_ci(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Paired interval."""
    return bootstrap_ci(_paired_difference(a, b), n_boot=n_boot, ci=ci, seed=seed)


def _paired_difference(a: Sequence[float], b: Sequence[float]) -> NDArray[np.float64]:
    first, second = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1 or first.size != second.size or not first.size:
        raise ValueError("paired scores must be nonempty one-dimensional sequences of equal length")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired scores must be finite")
    return first - second


def paired_randomization_test(
    a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0
) -> float:
    """Paired randomization."""
    diff = _paired_difference(a, b)
    if n_perm <= 0:
        raise ValueError("paired scores must be finite and n_perm must be positive")
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    perm_means = np.abs((signs * diff).mean(axis=1))

    exceed = int((perm_means >= observed - 1e-12).sum())
    return (exceed + 1) / (n_perm + 1)


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm correction."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError("p-values must be finite and between 0 and 1")
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
