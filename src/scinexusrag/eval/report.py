"""Evaluation reports."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scinexusrag.eval.metrics import (
    _binary_inputs,
    holm_correction,
    paired_delta_ci,
    paired_randomization_test,
    pr_auc,
    roc_auc,
)
from scinexusrag.eval.result_files import load_result, write_csv_results

RESULTS = Path("outputs/results")
COLS = ["nDCG@10", "R@5", "R@10", "R@20", "MRR", "MAP"]
PLOT_STYLE: dict[Any, Any] = {
    "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": "#334155",
    "text.color": "#172b45",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "axes.edgecolor": "#cbd5e1",
    "axes.axisbelow": True,
    "svg.fonttype": "none",
    "svg.hashsalt": "nexus-rag-evaluation",
}


def _save_figure(figure: Any, path: Path) -> None:
    figure.savefig(path, metadata={"Date": None, "Title": figure.get_label()})
    if path.suffix.lower() == ".svg":
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _source_path(json_name: str, csv_name: str, results_dir: Path) -> Path:
    path = results_dir / json_name
    return path if path.is_file() else results_dir / csv_name


def _p_vs_baseline(res: dict[str, Any], baseline: str = "BM25") -> dict[str, float]:
    per_query = res["per_query_ndcg"]
    baseline_scores = per_query[baseline]
    return {
        name: paired_randomization_test(scores, baseline_scores, seed=res.get("seed", 0))
        for name, scores in per_query.items()
        if name != baseline
    }


def _fmt_p(p: float, significant: bool = False) -> str:
    star = "*" if significant else ""
    return ("$<$0.001" if p < 1e-3 else f"{p:.3f}") + star


def _check_embedding_pair(reference: dict[str, Any], comparison: dict[str, Any]) -> None:
    if any(
        reference.get(key) != comparison.get(key)
        for key in (
            "dataset",
            "split",
            "query_ids",
            "corpus_size",
            "dataset_revision",
            "qrels_revision",
        )
    ):
        raise ValueError(
            "embedding comparisons require matching datasets and query_ids, "
            "including split, corpus size and revisions"
        )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, divide: bool = False) -> str:
    lines = [
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(rows):
        lines.append(" & ".join(row) + " \\\\")
        if divide and index == 1:
            lines.append("\\midrule")
    return "\n".join([*lines, "\\bottomrule", "\\end{tabular}"])


def _best_score(value: float, best: float) -> str:
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if abs(value - best) < 1e-9 else text


def _figure_title(figure: Any, title: str, y: float = 0.99) -> None:
    figure.set_label(title)
    figure.suptitle(title, x=0.03, y=y, ha="left", fontsize=17, weight="bold")


@contextmanager
def _figure(size: tuple[float, float], **options: Any) -> Iterator[tuple[Any, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=size, **options)
        try:
            yield figure, axes
        finally:
            plt.close(figure)


def ablation_table(res: dict[str, Any]) -> str:
    pvalues = _p_vs_baseline(res)
    adjusted = holm_correction(pvalues)
    best = {
        column: max(system["means"][column] for system in res["systems"].values())
        for column in COLS
    }
    rows = []
    for name, system in res["systems"].items():
        cells = [_best_score(system["means"][column], best[column]) for column in COLS]
        pvalue = "--" if name == "BM25" else _fmt_p(pvalues[name], adjusted[name] < 0.05)
        rows.append([name.replace("&", "\\&"), *cells, pvalue])
    return _table(["System", *COLS, "$p$ vs BM25"], rows, divide=True)


def cost_quality_table(corr: dict[str, Any]) -> str:
    rows = [
        [row["system"], f"{row['ndcg']:.3f}", f"{row['r20']:.3f}", f"{row['latency_ms']:.0f}"]
        for row in corr["cost_quality"]["systems"]
    ]
    return _table(["System", "nDCG@10", "R@20", "ms/query"], rows)


def faithfulness_table(faith: dict[str, Any]) -> str:
    labels = {
        "nli": "NLI (DeBERTa)",
        "lexical_overlap": "Lexical overlap",
        "cross_encoder": "Cross-encoder",
    }
    methods = faith["methods"]
    _check_evidence_methods(methods)
    columns = ["roc_auc", "pr_auc", "f1"]
    best = {column: max(method[column] for method in methods.values()) for column in columns}
    rows = []
    for name in ("lexical_overlap", "cross_encoder", "nli"):
        if name not in methods:
            continue
        method = methods[name]
        cells = [_best_score(method[column], best[column]) for column in columns]
        low, high = method["roc_auc_ci"]
        cells[0] += f" [{low:.2f}, {high:.2f}]"
        rows.append([labels[name], *cells])
    return _table(["Scorer", "ROC-AUC [95\\% CI]", "Average precision", "F1"], rows)


def _check_evidence_methods(methods: dict[str, Any]) -> None:
    if not methods:
        raise ValueError("Evidence results contain no methods")
    unknown = set(methods) - {"nli", "lexical_overlap", "cross_encoder"}
    if unknown:
        raise ValueError(f"Unsupported evidence methods: {', '.join(sorted(unknown))}")


def binary_curve_points(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Threshold curves."""
    values, truth = _binary_inputs(scores, labels)
    positives = int(truth.sum())
    negatives = truth.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("curves require both positive and negative examples")
    order = np.argsort(-values, kind="mergesort")
    ends = np.r_[np.flatnonzero(np.diff(values[order])), values.size - 1]
    true_positives = np.cumsum(truth[order], dtype=np.float64)[ends]
    false_positives = ends + 1 - true_positives
    recall = np.r_[0.0, true_positives / positives]
    precision = np.r_[1.0, true_positives / (ends + 1)]
    return np.r_[0.0, false_positives / negatives], recall, recall, precision


def plot_retrieval_comparison(
    sci: dict[str, Any],
    nf: dict[str, Any],
    sci_mini: dict[str, Any] | None,
    nf_mini: dict[str, Any] | None,
    out_path: Path,
    *,
    results_dir: Path = RESULTS,
) -> None:
    from matplotlib.lines import Line2D

    labels = {
        "BM25": "BM25",
        "Dense": "Dense",
        "Hybrid (RRF)": "Hybrid (RRF)",
        "+ Adaptive weights": "Adaptive weights",
        "+ Corrective PRF": "Corrective PRF",
    }
    source_files = "{scifact,nfcorpus}_test.json"
    if sci_mini and nf_mini:
        source_files = "{scifact,nfcorpus}_{test,minilm}.json"
    elif sci_mini:
        source_files += ", scifact_minilm.json"
    elif nf_mini:
        source_files += ", nfcorpus_minilm.json"
    plotted_files = ["scifact_test.json", "nfcorpus_test.json"]
    if sci_mini:
        plotted_files.append("scifact_minilm.json")
    if nf_mini:
        plotted_files.append("nfcorpus_minilm.json")
    if any(not (results_dir / name).is_file() for name in plotted_files):
        source_files = ", ".join(
            dict.fromkeys(
                _source_path(name, "retrieval_metrics.csv", results_dir).name
                for name in plotted_files
            )
        )
    for reference, mini in ((sci, sci_mini), (nf, nf_mini)):
        if mini:
            _check_embedding_pair(reference, mini)
    with _figure((12, 5.9), sharey=True) as (fig, axes):
        for ax, res, mini, title in (
            (axes[0], sci, sci_mini, "SciFact"),
            (axes[1], nf, nf_mini, "NFCorpus"),
        ):
            names = list(res["systems"])
            for row, name in enumerate(names):
                if row % 2 == 0:
                    ax.axhspan(row - 0.45, row + 0.45, color="#f1f5f9", zorder=0)
                series = [(res, "#176b91", -0.16 if mini else 0)]
                if mini:
                    series.append((mini, "#c66d19", 0.16))
                if name == "BM25":
                    series = [(res, "#64748b", 0)]
                for data, color, offset in series:
                    interval = data["systems"][name]["ci"]["nDCG@10"]
                    mean, low, high = (interval[key] for key in ("mean", "lo", "hi"))
                    ax.errorbar(
                        mean,
                        row + offset,
                        xerr=[[max(0, mean - low)], [max(0, high - mean)]],
                        fmt="o",
                        color=color,
                        markersize=6.5,
                        capsize=3,
                        elinewidth=1.5,
                    )
                    ax.text(
                        1.025,
                        row + offset,
                        f"{mean:.3f}",
                        transform=ax.get_yaxis_transform(),
                        va="center",
                        color=color,
                        fontsize=10,
                    )
            ax.set_yticks(range(len(names)), [labels.get(name, name) for name in names])
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_ylim(len(names) - 0.5, -0.7)
            ax.set_xlabel("nDCG@10 · higher is better")
            ax.set_title(f"{title} · {res['num_queries']} test queries", loc="left", pad=15)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=8)
            ax.grid(axis="x", color="#e2e8f0")
        legend_items = [("BM25", "#64748b"), (sci["embedding_model"].rsplit("/", 1)[-1], "#176b91")]
        mini_result = sci_mini or nf_mini
        if mini_result:
            legend_items.append((mini_result["embedding_model"].rsplit("/", 1)[-1], "#c66d19"))
        legend = [
            Line2D([], [], color=color, marker="o", linestyle="none", label=label)
            for label, color in legend_items
        ]
        _figure_title(fig, "Retrieval quality", y=0.98)
        fig.legend(
            handles=legend, loc="upper left", bbox_to_anchor=(0.025, 0.91), ncol=3, frameon=False
        )
        fig.text(
            0.03,
            0.05,
            "Dots: recorded means. Whiskers: 95% bootstrap intervals over queries.",
            fontsize=10,
        )
        fig.text(
            0.03,
            0.015,
            f"Source: {results_dir}/{source_files}",
            fontsize=9,
            color="#64748b",
        )
        fig.tight_layout(rect=(0.02, 0.1, 0.955, 0.8), w_pad=4)
        _save_figure(fig, out_path)


def plot_evidence_curves(
    faith: dict[str, Any], out_path: Path, *, results_dir: Path = RESULTS
) -> None:
    _check_evidence_methods(faith["methods"])
    candidates = faith["per_candidate"]["dev"]
    labels = candidates["labels"]
    if len(labels) != faith["num_candidates"]:
        raise ValueError("candidate labels do not match num_candidates")
    base_rate = float(np.mean(labels))
    if not np.isclose(base_rate, faith["gold_base_rate"], rtol=0, atol=1e-12):
        raise ValueError("candidate labels do not match gold_base_rate")
    methods = [
        ("cross_encoder", "Cross-encoder", "#176b91"),
        ("nli", "NLI (DeBERTa)", "#c66d19"),
        ("lexical_overlap", "Lexical overlap", "#6c6b9e"),
    ]
    curves = []
    for key, name, color in methods:
        if key not in faith["methods"]:
            continue
        scores = candidates["scores"][key]
        auc = roc_auc(scores, labels)
        ap = pr_auc(scores, labels)
        recorded = faith["methods"][key]
        if not np.allclose(
            [auc, ap], [recorded["roc_auc"], recorded["pr_auc"]], rtol=0, atol=1e-12
        ):
            raise ValueError(f"{key} candidate scores do not match recorded AUC/AP")
        curves.append((name, color, auc, ap, *binary_curve_points(scores, labels)))
    with _figure((12, 7.1)) as (fig, (roc_ax, pr_ax)):
        for name, color, auc, ap, fpr, tpr, recall, precision in curves:
            roc_ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} · AUC {auc:.3f}")
            pr_ax.step(
                recall,
                precision,
                where="pre",
                color=color,
                linewidth=1.8,
                label=f"{name} · AP {ap:.3f}",
            )
        roc_ax.plot(
            [0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="Random ranking · 0.500"
        )
        pr_ax.axhline(
            base_rate,
            linestyle="--",
            color="#94a3b8",
            linewidth=1,
            label=f"Positive rate · {base_rate:.3f}",
        )
        for ax, title, xlabel, ylabel in (
            (roc_ax, "ROC curve", "False positive rate", "True positive rate"),
            (pr_ax, "Precision–recall curve", "Recall", "Precision"),
        ):
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.grid(color="#e2e8f0")
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(0, -0.2),
                frameon=False,
                borderaxespad=0,
                fontsize=10,
            )
        _figure_title(fig, "Evidence detection on SciFact dev")
        fig.text(
            0.03,
            0.91,
            f"{faith['num_claims']} annotated claims · {faith['num_candidates']:,} candidate sentences · positive = SUPPORT or CONTRADICT",
            fontsize=10,
        )
        fig.text(
            0.03,
            0.055,
            "AUC: ROC area. AP: average precision. Evidence detection does not measure generated-answer accuracy.",
            fontsize=9,
        )
        fig.text(
            0.03,
            0.02,
            f"Source: {_source_path('faithfulness_dev.json', 'faithfulness_candidates.csv', results_dir)}",
            fontsize=9,
            color="#64748b",
        )
        fig.tight_layout(rect=(0.015, 0.13, 0.99, 0.87), w_pad=3)
        _save_figure(fig, out_path)


def plot_corrective_sweep(
    corr: dict[str, Any], out_path: Path, *, results_dir: Path = RESULTS
) -> None:
    rows = sorted(corr["tau_sweep"], key=lambda row: row["tau"])
    thresholds = [row["tau"] for row in rows]
    selected = corr["best_tau"]
    selected_row = next(row for row in rows if np.isclose(row["tau"], selected))
    quality_values = [corr["base_ndcg"], *(row["ndcg"] for row in rows)]
    with _figure((12, 5.8)) as (fig, axes):
        quality, retries = axes
        quality.plot(thresholds, [row["ndcg"] for row in rows], "o-", color="#176b91", linewidth=2)
        quality.axhline(
            corr["base_ndcg"],
            linestyle="--",
            color="#94a3b8",
            label=f"Adaptive baseline · {corr['base_ndcg']:.3f}",
        )
        quality.set(
            ylabel="nDCG@10",
            title="Retrieval quality · zoomed axis",
            ylim=(min(quality_values) - 0.004, max(quality_values) + 0.006),
        )
        for row in rows:
            quality.annotate(
                f"{row['ndcg']:.3f}",
                (row["tau"], row["ndcg"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        retries.plot(
            thresholds,
            [row["trigger_rate"] * 100 for row in rows],
            "o-",
            color="#c66d19",
            linewidth=2,
            clip_on=False,
        )
        retry_ceiling = min(100, max(8, max(row["trigger_rate"] for row in rows) * 120))
        retries.set(
            ylabel="Queries retried (%)",
            title="Retry frequency · zoomed axis" if retry_ceiling < 100 else "Retry frequency",
            ylim=(0, retry_ceiling),
        )
        retries.set_yticks(np.linspace(0, retry_ceiling, 5))
        for row in rows:
            retries.annotate(
                f"{row['triggered_n']}/{corr['num_queries']}",
                (row["tau"], row["trigger_rate"] * 100),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        for ax in axes:
            ax.axvline(
                selected,
                color="#64748b",
                linestyle=":",
                linewidth=1,
                label=f"Recorded threshold · τ = {selected:g}",
            )
            ax.set_xlabel("Retry threshold τ")
            ax.set_xticks(thresholds)
            ax.grid(color="#e2e8f0")
        _figure_title(fig, "Corrective retrieval threshold sweep")
        fig.text(
            0.03,
            0.895,
            f"SciFact · {corr['num_queries']} test queries · threshold selected on {corr['best_tau_selected_on']}",
            fontsize=10,
        )
        fig.text(
            0.03,
            0.045,
            f"Measured thresholds; connecting lines are guides. Recorded τ: {selected_row['triggered_n']}/{corr['num_queries']} queries retried.",
            fontsize=9,
        )
        fig.text(
            0.03,
            0.01,
            f"Source: {_source_path('corrective_scifact.json', 'corrective_sweeps.csv', results_dir)} "
            "· No threshold tuning on test data.",
            fontsize=9,
            color="#64748b",
        )
        fig.legend(
            *quality.get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.025, 0.86),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        fig.tight_layout(rect=(0.01, 0.1, 0.99, 0.77), w_pad=3)
        _save_figure(fig, out_path)


def _plot_numbers(values: Sequence[Any], label: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or raw.dtype.kind not in "iuf"
        or any(isinstance(value, (bool, np.bool_)) for value in values)
    ):
        raise ValueError(f"{label} must contain numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        numbers = raw.astype(np.float64)
    if not np.isfinite(numbers).all():
        raise ValueError(f"{label} must contain finite values")
    return numbers


def plot_retrieval_query_changes(
    sci: dict[str, Any],
    nf: dict[str, Any],
    out_path: Path,
    *,
    results_dir: Path = RESULTS,
) -> None:
    pairs = []
    for result in (sci, nf):
        ids = result["query_ids"]
        if (
            not ids
            or type(result["num_queries"]) is not int
            or len(ids) != result["num_queries"]
            or any(not isinstance(query_id, str) or not query_id for query_id in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Paired query IDs must be unique and match num_queries")
        dense, adaptive = (
            _plot_numbers(result["per_query_ndcg"][name], "Paired nDCG scores")
            for name in ("Dense", "+ Adaptive weights")
        )
        if any(scores.shape != (len(ids),) for scores in (dense, adaptive)):
            raise ValueError("Paired nDCG scores must align with query IDs")
        if any(np.any((scores < 0) | (scores > 1)) for scores in (dense, adaptive)):
            raise ValueError("Paired nDCG scores must be between 0 and 1")
        pairs.append((dense, adaptive))

    with _figure((12, 6.7)) as (fig, axes):
        for ax, result, (dense, adaptive) in zip(axes, (sci, nf), pairs, strict=True):
            title = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}.get(
                result["dataset"], result["dataset"]
            )
            difference = adaptive - dense
            groups = (
                (difference > 1e-12, "Higher", "#16836b"),
                (np.abs(difference) <= 1e-12, "Equal", "#64748b"),
                (difference < -1e-12, "Lower", "#bd573b"),
            )
            ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
            for mask, label, color in groups:
                ax.scatter(
                    dense[mask],
                    adaptive[mask],
                    label=label,
                    color=color,
                    s=25,
                    alpha=0.65,
                    edgecolors="white",
                    linewidths=0.4,
                    clip_on=False,
                )
            ax.set(
                title=f"{title} · {result['num_queries']} {result['split']} queries",
                xlabel="Dense nDCG@10",
                ylabel="Adaptive hybrid nDCG@10",
                xlim=(0, 1),
                ylim=(0, 1),
            )
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="#e2e8f0")
            counts = " · ".join(f"{label} {int(mask.sum())}" for mask, label, _ in groups)
            ax.text(
                0,
                -0.25,
                f"{counts}\nMean paired Δ = {float(difference.mean()):+.4f}",
                transform=ax.transAxes,
                fontsize=10,
                linespacing=1.6,
            )
        _figure_title(fig, "Query-level changes")
        models = " / ".join(
            dict.fromkeys(result["embedding_model"].rsplit("/", 1)[-1] for result in (sci, nf))
        )
        fig.text(
            0.03,
            0.91,
            f"{models} · adaptive weights versus dense · one marker per query; markers may overlap.",
            fontsize=10,
        )
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.025, 0.88),
            ncol=3,
            frameon=False,
            fontsize=10,
        )
        fig.text(
            0.03,
            0.055,
            "Above the diagonal: adaptive improves. Below: it declines. Equal means |Δ| ≤ 10⁻¹².",
            fontsize=9,
        )
        files = ", ".join(
            dict.fromkeys(
                _source_path(name, "retrieval_queries.csv", results_dir).name
                for name in ("scifact_test.json", "nfcorpus_test.json")
            )
        )
        fig.text(0.03, 0.02, f"Source: {results_dir}/{files}", fontsize=9, color="#64748b")
        fig.tight_layout(rect=(0.01, 0.12, 0.99, 0.81), w_pad=3)
        _save_figure(fig, out_path)


def plot_retrieval_cost(
    corr: dict[str, Any], out_path: Path, *, results_dir: Path = RESULTS
) -> None:
    cost = corr["cost_quality"]
    timed = cost["timed_queries"]
    ids = cost["query_ids"]
    if (
        type(timed) is not int
        or timed <= 0
        or len(ids) != timed
        or any(not isinstance(query_id, str) or not query_id for query_id in ids)
        or len(set(ids)) != timed
        or len(corr["query_ids"]) != corr["num_queries"]
        or ids != corr["query_ids"][:timed]
    ):
        raise ValueError("Timed query IDs must match the first timed_queries test queries")
    rows = cost["systems"]
    if not rows:
        raise ValueError("Retrieval cost results contain no systems")
    names = [row["system"] for row in rows]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("Retrieval cost systems must have unique nonempty names")
    quality = _plot_numbers([row["ndcg"] for row in rows], "Cost nDCG scores")
    latency = _plot_numbers([row["latency_ms"] for row in rows], "Query latency")
    if np.any((quality < 0) | (quality > 1)) or np.any(latency <= 0):
        raise ValueError("Cost nDCG must be between 0 and 1; latency must be positive")
    labels = {
        "Adaptive": "Adaptive hybrid",
        "Corrective PRF": "Corrective PRF",
        "Rerank (cross-enc)": "Cross-encoder",
    }
    colors = ["#176b91", "#7865a3", "#c66d19"]
    colors = [colors[index % len(colors)] for index in range(len(rows))]
    positions = np.arange(len(rows))
    with _figure((12, 5.6), sharey=True) as (fig, (quality_ax, latency_ax)):
        for ax, values, title, xlabel in (
            (quality_ax, quality, "Retrieval quality", "nDCG@10 · higher is better"),
            (latency_ax, latency, "Mean time per query", "Milliseconds · lower is better"),
        ):
            bars = ax.barh(positions, values, color=colors, height=0.48)
            ax.bar_label(
                bars,
                labels=[
                    f"{value:.3f}" if ax is quality_ax else f"{value:,.1f}" for value in values
                ],
                padding=6,
                fontsize=10,
            )
            ax.set(title=title, xlabel=xlabel)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color="#e2e8f0")
        quality_ax.set_yticks(positions, [labels.get(name, name) for name in names])
        quality_ax.set_ylim(len(rows) - 0.6, -0.6)
        quality_ax.set_xlim(0, 1)
        quality_ax.set_xticks(np.linspace(0, 1, 5))
        latency_ax.set_xlim(0, float(latency.max()) * 1.22)
        _figure_title(fig, "Retrieval cost and quality")
        dataset = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}.get(
            corr["dataset"], corr["dataset"]
        )
        fig.text(
            0.03, 0.89, f"{dataset} · first {timed} {corr['split']} queries · CPU", fontsize=10
        )
        fig.text(
            0.03,
            0.095,
            "One timed pass; no dedicated warm-up. Excludes indexing, embedding-model loading and answer generation.",
            fontsize=9,
        )
        if "Rerank (cross-enc)" in names:
            fig.text(0.03, 0.06, "Cross-encoder time includes its initial model load.", fontsize=9)
        fig.text(
            0.03,
            0.02,
            f"Source: {_source_path('corrective_scifact.json', 'corrective_cost_quality.csv', results_dir)}",
            fontsize=9,
            color="#64748b",
        )
        fig.tight_layout(rect=(0.01, 0.15, 0.985, 0.83), w_pad=3)
        _save_figure(fig, out_path)


def write_figures(figures_dir: Path, results_dir: Path = RESULTS) -> None:
    sci = load_result("scifact_test.json", results_dir)
    nf = load_result("nfcorpus_test.json", results_dir)
    faith = load_result("faithfulness_dev.json", results_dir)
    corr = load_result("corrective_scifact.json", results_dir)
    if not sci or not nf or not faith or not corr:
        raise SystemExit("Charts require saved retrieval, evidence, and corrective results.")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_retrieval_comparison(
        sci,
        nf,
        load_result("scifact_minilm.json", results_dir),
        load_result("nfcorpus_minilm.json", results_dir),
        figures_dir / "retrieval-comparison.svg",
        results_dir=results_dir,
    )
    plot_evidence_curves(faith, figures_dir / "evidence-curves.svg", results_dir=results_dir)
    plot_corrective_sweep(corr, figures_dir / "corrective-sweep.svg", results_dir=results_dir)
    plot_retrieval_query_changes(
        sci, nf, figures_dir / "retrieval-query-changes.svg", results_dir=results_dir
    )
    plot_retrieval_cost(corr, figures_dir / "retrieval-cost.svg", results_dir=results_dir)
    print(f"wrote evaluation charts to {figures_dir}")


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def build_macros(
    sci: dict[str, Any],
    nf: dict[str, Any],
    mini: dict[str, Any] | None,
    corr_sci: dict[str, Any] | None,
    faith: dict[str, Any] | None,
) -> list[str]:
    if mini:
        _check_embedding_pair(sci, mini)

    def nd(res: dict[str, Any], sysname: str) -> str:
        return f"{res['systems'][sysname]['means']['nDCG@10']:.3f}"

    def delta(res: dict[str, Any], sysname: str) -> tuple[float, float, float]:
        pq = res["per_query_ndcg"]
        return paired_delta_ci(pq[sysname], pq["BM25"], seed=res.get("seed", 0))

    sci_p = _p_vs_baseline(sci)
    nf_p = _p_vs_baseline(nf)
    sci_hd = delta(sci, "Hybrid (RRF)")
    nf_hd = delta(nf, "Hybrid (RRF)")
    macros = [
        _macro("NumQueries", str(sci["num_queries"])),
        _macro("CorpusSize", str(sci["corpus_size"])),
        _macro("NFqueries", str(nf["num_queries"])),
        _macro("NFcorpus", str(nf["corpus_size"])),
        _macro("RRFk", str(sci.get("rrf_k", 60))),
        _macro("EvalSeed", str(sci.get("seed", 0))),
        _macro("SciBM", nd(sci, "BM25")),
        _macro("SciDense", nd(sci, "Dense")),
        _macro("SciHybridP", _fmt_p(sci_p["Hybrid (RRF)"])),
        _macro("SciHybridDelta", f"{sci_hd[0]:+.3f}"),
        _macro("SciHybridDeltaCI", f"[{sci_hd[1]:+.3f}, {sci_hd[2]:+.3f}]"),
        _macro("NFhybridP", _fmt_p(nf_p["Hybrid (RRF)"])),
        _macro("NFhybridDelta", f"{nf_hd[0]:+.3f}"),
        _macro("NFhybridDeltaCI", f"[{nf_hd[1]:+.3f}, {nf_hd[2]:+.3f}]"),
    ]
    if mini:
        bge = sci["per_query_ndcg"]["Dense"]
        ml = mini["per_query_ndcg"]["Dense"]
        gain = np.mean(bge) - np.mean(ml)
        macros += [
            _macro("SciDenseMini", nd(mini, "Dense")),
            _macro("SciEmbGain", f"{gain:+.3f}"),
            _macro(
                "SciEmbP",
                _fmt_p(paired_randomization_test(bge, ml, seed=sci.get("seed", 0))),
            ),
        ]
    if corr_sci:
        cq = {r["system"]: r for r in corr_sci["cost_quality"]["systems"]}

        max_fire = max(s["trigger_rate"] for s in corr_sci["tau_sweep"])
        macros += [
            _macro("SciCorrMaxFire", f"{max_fire * 100:.0f}\\%"),
            _macro("CorrMs", f"{cq['Corrective PRF']['latency_ms']:.0f}"),
            _macro("BaseMs", f"{cq['Adaptive']['latency_ms']:.0f}"),
            _macro("BaseND", f"{cq['Adaptive']['ndcg']:.3f}"),
        ]
        if "Rerank (cross-enc)" in cq:
            rerank = cq["Rerank (cross-enc)"]
            speedup = rerank["latency_ms"] / max(cq["Adaptive"]["latency_ms"], 1e-9)
            macros += [
                _macro("RerankMs", f"{rerank['latency_ms']:.0f}"),
                _macro("RerankSlowdown", f"{speedup:.0f}"),
                _macro("RerankND", f"{rerank['ndcg']:.3f}"),
            ]
    if faith:
        m = faith["methods"]
        lo, hi = m["nli"]["roc_auc_ci"]
        macros += [
            _macro("FaithClaims", str(faith["num_claims"])),
            _macro("FaithBaseRate", f"{faith['gold_base_rate']:.2f}"),
            _macro("FaithNliAuroc", f"{m['nli']['roc_auc']:.3f}"),
            _macro("FaithNliAurocLo", f"{lo:.3f}"),
            _macro("FaithNliAurocHi", f"{hi:.3f}"),
            _macro("FaithLexAuroc", f"{m['lexical_overlap']['roc_auc']:.3f}"),
        ]
        if "cross_encoder" in m:
            macros.append(_macro("FaithCeAuroc", f"{m['cross_encoder']['roc_auc']:.3f}"))
        if faith.get("ce_minus_nli_auroc") is not None:
            macros.append(_macro("FaithCeMinusNli", f"{faith['ce_minus_nli_auroc']:+.3f}"))
    return macros


def write_paper_bundle(paper_dir: Path, results_dir: Path = RESULTS) -> None:
    sci = load_result("scifact_test.json", results_dir)
    nf = load_result("nfcorpus_test.json", results_dir)
    if not sci or not nf:
        raise SystemExit(
            "Missing SciFact or NFCorpus retrieval results (JSON or CSV). "
            "Run `python -m scinexusrag.eval`, "
            "then `python -m scinexusrag.eval.report --results outputs/generated`."
        )
    mini = load_result("scifact_minilm.json", results_dir)
    corr_sci = load_result("corrective_scifact.json", results_dir)
    faith = load_result("faithfulness_dev.json", results_dir)

    (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_dir / "figures").mkdir(parents=True, exist_ok=True)
    (paper_dir / "tables" / "scifact.tex").write_text(ablation_table(sci) + "\n")
    (paper_dir / "tables" / "nfcorpus.tex").write_text(ablation_table(nf) + "\n")
    if corr_sci:
        (paper_dir / "tables" / "cost_quality.tex").write_text(cost_quality_table(corr_sci) + "\n")
    if faith:
        (paper_dir / "tables" / "faithfulness.tex").write_text(faithfulness_table(faith) + "\n")
    plot_retrieval_comparison(
        sci, nf, None, None, paper_dir / "figures" / "ablation.png", results_dir=results_dir
    )
    (paper_dir / "generated.tex").write_text(
        "\n".join(build_macros(sci, nf, mini, corr_sci, faith)) + "\n"
    )
    print(f"wrote paper bundle to {paper_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Render evaluation charts, tables, and macros")
    p.add_argument("--paper", default="outputs/generated/paper", help="paper directory to populate")
    p.add_argument(
        "--results", default=str(RESULTS), help="directory containing evaluation results"
    )
    p.add_argument("--figures", help="write standalone SVG charts here instead of a paper bundle")
    p.add_argument("--csv", help="export saved results as CSV tables to this directory")
    args = p.parse_args()
    results_dir = Path(args.results)
    if args.csv:
        runs = {}
        for name in (
            "scifact_test",
            "nfcorpus_test",
            "scifact_minilm",
            "nfcorpus_minilm",
            "faithfulness_dev",
            "corrective_scifact",
        ):
            result = load_result(name, results_dir)
            if result is not None:
                runs[name] = result
        if not runs:
            raise SystemExit(f"No evaluation results found in {results_dir}.")
        results_dir = Path(args.csv)
        write_csv_results(runs, results_dir)
        print(f"wrote CSV results to {results_dir}")
    if args.figures:
        write_figures(Path(args.figures), results_dir)
    elif not args.csv:
        write_paper_bundle(Path(args.paper), results_dir)


if __name__ == "__main__":
    main()
