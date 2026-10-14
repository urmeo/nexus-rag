"""Render evaluation charts, tables, and LaTeX macros from saved results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scinexusrag.eval.metrics import (
    holm_correction,
    paired_delta_ci,
    paired_randomization_test,
    pr_auc,
    roc_auc,
)

RESULTS = Path("outputs/results")
COLS = ["nDCG@10", "R@5", "R@10", "R@20", "MRR", "MAP"]
PLOT_STYLE: dict[Any, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
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
    figure.savefig(path, metadata={"Date": None})
    if path.suffix.lower() == ".svg":
        # Matplotlib adds trailing spaces to SVG path attributes.
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _load(name: str, results_dir: Path = RESULTS) -> dict[str, Any] | None:
    path = results_dir / name
    return json.loads(path.read_text()) if path.exists() else None


def _p_vs_baseline(res: dict[str, Any], baseline: str = "BM25") -> dict[str, float]:
    per_query = res["per_query_ndcg"]
    baseline_scores = per_query[baseline]
    return {
        name: paired_randomization_test(scores, baseline_scores, seed=res.get("seed", 0))
        for name, scores in per_query.items()
        if name != baseline
    }


def _fmt_p(p: float, significant: bool) -> str:
    star = "*" if significant else ""
    return ("$<$0.001" if p < 1e-3 else f"{p:.3f}") + star


def ablation_table(res: dict[str, Any]) -> str:
    pvs = _p_vs_baseline(res)
    holm = holm_correction(pvs)
    names = list(res["systems"])
    colmax = {c: max(res["systems"][n]["means"][c] for n in names) for c in COLS}

    lines = [
        "\\begin{tabular}{l" + "r" * (len(COLS) + 1) + "}",
        "\\toprule",
        "System & " + " & ".join(COLS) + " & $p$ vs BM25 \\\\",
        "\\midrule",
    ]
    for i, (name, s) in enumerate(res["systems"].items()):
        cells = []
        for c in COLS:
            v = s["means"][c]
            txt = f"{v:.3f}"
            if abs(v - colmax[c]) < 1e-9:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        pcell = "--" if name == "BM25" else _fmt_p(pvs[name], holm[name] < 0.05)
        lines.append(name.replace("&", "\\&") + " & " + " & ".join(cells) + f" & {pcell} \\\\")
        if i == 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def cost_quality_table(corr: dict[str, Any]) -> str:
    rows = corr["cost_quality"]["systems"]
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "System & nDCG@10 & R@20 & ms/query \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['system']} & {r['ndcg']:.3f} & {r['r20']:.3f} & {r['latency_ms']:.0f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def faithfulness_table(faith: dict[str, Any]) -> str:
    label = {
        "nli": "NLI (DeBERTa)",
        "lexical_overlap": "Lexical overlap",
        "cross_encoder": "Cross-encoder",
    }
    m = faith["methods"]
    cols = ["roc_auc", "pr_auc", "f1"]
    colmax = {c: max(m[k][c] for k in m) for c in cols}
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Scorer & ROC-AUC [95\\% CI] & Average precision & F1 \\\\",
        "\\midrule",
    ]
    for key in ("lexical_overlap", "cross_encoder", "nli"):
        if key not in m:
            continue
        cells = []
        for c in cols:
            v = m[key][c]
            txt = f"{v:.3f}"
            if abs(v - colmax[c]) < 1e-9:
                txt = f"\\textbf{{{txt}}}"
            if c == "roc_auc":
                clo, chi = m[key]["roc_auc_ci"]
                txt += f" [{clo:.2f}, {chi:.2f}]"
            cells.append(txt)
        lines.append(f"{label.get(key, key)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def binary_curve_points(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ROC and precision-recall points at distinct score thresholds.

    Equal scores enter together. Plot PR with ``where="pre"`` so its step area
    matches average precision, rather than a trapezoidal PR integral.
    """
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels)
    if values.ndim != 1 or truth.ndim != 1 or values.size != truth.size:
        raise ValueError("scores and labels must be one-dimensional with equal length")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(truth, [0, 1])):
        raise ValueError("scores must be finite and labels must be binary")
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
    """Compare recorded nDCG means and bootstrap intervals on both test sets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    for reference, mini in ((sci, sci_mini), (nf, nf_mini)):
        if mini and any(
            reference.get(key) != mini.get(key)
            for key in ("dataset", "split", "query_ids", "corpus_size")
        ):
            raise ValueError("embedding comparisons require matching datasets and query_ids")
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True)
        for ax, res, mini, title in (
            (axes[0], sci, sci_mini, "SciFact"),
            (axes[1], nf, nf_mini, "NFCorpus"),
        ):
            names = list(res["systems"])
            for row, name in enumerate(names):
                series = [(res, "#176b91", -0.12 if mini else 0)]
                if mini:
                    series.append((mini, "#c66d19", 0.12))
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
                        markersize=6,
                        capsize=3,
                        elinewidth=1.5,
                    )
                    ax.text(high + 0.015, row + offset, f"{mean:.3f}", va="center", fontsize=9)
            ax.set_yticks(range(len(names)), [labels.get(name, name) for name in names])
            ax.set_xlim(0, 1)
            ax.set_ylim(len(names) - 0.5, -0.7)
            ax.set_xlabel("nDCG@10 · higher is better")
            ax.set_title(f"{title} · {res['num_queries']} test queries", loc="left", pad=15)
            ax.grid(axis="x", color="#e2e8f0")
        legend = [
            Line2D([], [], color="#64748b", marker="o", linestyle="none", label="BM25"),
            Line2D(
                [],
                [],
                color="#176b91",
                marker="o",
                linestyle="none",
                label=sci["embedding_model"].rsplit("/", 1)[-1],
            ),
        ]
        mini_result = sci_mini or nf_mini
        if mini_result:
            legend.append(
                Line2D(
                    [],
                    [],
                    color="#c66d19",
                    marker="o",
                    linestyle="none",
                    label=mini_result["embedding_model"].rsplit("/", 1)[-1],
                )
            )
        fig.suptitle("Retrieval results", x=0.03, y=0.98, ha="left", fontsize=17, weight="bold")
        fig.legend(
            handles=legend, loc="upper center", bbox_to_anchor=(0.62, 1.0), ncol=3, frameon=False
        )
        fig.text(
            0.03,
            0.05,
            "Dots: recorded means. Whiskers: 95% bootstrap intervals over queries.",
            fontsize=9,
        )
        fig.text(
            0.03,
            0.015,
            f"Source: {results_dir}/{source_files}",
            fontsize=8,
            color="#64748b",
        )
        fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.9))
        _save_figure(fig, out_path)
        plt.close(fig)


def plot_evidence_curves(
    faith: dict[str, Any], out_path: Path, *, results_dir: Path = RESULTS
) -> None:
    """Plot candidate evidence detection, with areas checked against saved metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    with plt.rc_context(PLOT_STYLE):
        fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(12, 5.8))
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
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
            ax.grid(color="#e2e8f0")
            ax.legend(
                loc="lower right" if ax is roc_ax else "upper right",
                frameon=True,
                edgecolor="#e2e8f0",
                fontsize=9,
            )
        fig.suptitle(
            "Evidence detection on SciFact dev",
            x=0.03,
            y=0.99,
            ha="left",
            fontsize=17,
            weight="bold",
        )
        fig.text(
            0.03,
            0.91,
            f"{faith['num_candidates']:,} candidate sentences · {faith['num_claims']} claims · positive = SUPPORT or CONTRADICT",
            fontsize=10,
        )
        fig.text(
            0.03,
            0.055,
            "AUC: ROC area. AP: average precision, using steps at distinct score thresholds.",
            fontsize=9,
        )
        fig.text(
            0.03,
            0.02,
            f"This task detects evidence; it does not measure generated-answer accuracy. Source: {results_dir / 'faithfulness_dev.json'}",
            fontsize=8,
            color="#64748b",
        )
        fig.tight_layout(rect=(0.015, 0.09, 0.99, 0.87))
        _save_figure(fig, out_path)
        plt.close(fig)


def plot_corrective_sweep(
    corr: dict[str, Any], out_path: Path, *, results_dir: Path = RESULTS
) -> None:
    """Show measured test results across retry thresholds without implying tuning on test."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(corr["tau_sweep"], key=lambda row: row["tau"])
    thresholds = [row["tau"] for row in rows]
    selected = corr["best_tau"]
    selected_row = next(row for row in rows if np.isclose(row["tau"], selected))
    quality_values = [corr["base_ndcg"], *(row["ndcg"] for row in rows)]
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
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
        quality.legend(frameon=False, loc="lower left", fontsize=9)
        for row in rows:
            quality.annotate(
                f"{row['ndcg']:.3f}",
                (row["tau"], row["ndcg"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        retries.plot(
            thresholds,
            [row["trigger_rate"] * 100 for row in rows],
            "o-",
            color="#c66d19",
            linewidth=2,
        )
        retries.set(
            ylabel="Queries retried (%)",
            title="Retry frequency",
            ylim=(-0.5, max(8, max(row["trigger_rate"] for row in rows) * 115)),
        )
        for row in rows:
            retries.annotate(
                f"{row['triggered_n']}/{corr['num_queries']}",
                (row["tau"], row["trigger_rate"] * 100),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        for ax in axes:
            ax.axvline(selected, color="#64748b", linestyle=":", linewidth=1)
            ax.set_xlabel("Retry threshold τ")
            ax.set_xticks(thresholds)
            ax.grid(color="#e2e8f0")
        fig.suptitle(
            "Corrective retrieval threshold sweep",
            x=0.03,
            y=0.99,
            ha="left",
            fontsize=17,
            weight="bold",
        )
        fig.text(
            0.03,
            0.895,
            f"SciFact · {corr['num_queries']} test queries · dotted line: τ = {selected:g}, selected on {corr['best_tau_selected_on']}",
            fontsize=10,
        )
        fig.text(
            0.03,
            0.045,
            f"Markers are measured settings; lines are guides. Selected threshold: {selected_row['triggered_n']}/{corr['num_queries']} queries retried.",
            fontsize=9,
        )
        fig.text(
            0.03,
            0.01,
            f"Source: {results_dir / 'corrective_scifact.json'} · No automatic threshold tuning on test data.",
            fontsize=8,
            color="#64748b",
        )
        fig.tight_layout(rect=(0.01, 0.08, 0.99, 0.85))
        _save_figure(fig, out_path)
        plt.close(fig)


def write_figures(figures_dir: Path, results_dir: Path = RESULTS) -> None:
    sci = _load("scifact_test.json", results_dir)
    nf = _load("nfcorpus_test.json", results_dir)
    faith = _load("faithfulness_dev.json", results_dir)
    corr = _load("corrective_scifact.json", results_dir)
    if not sci or not nf or not faith or not corr:
        raise SystemExit("Charts require saved retrieval, evidence, and corrective results.")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_retrieval_comparison(
        sci,
        nf,
        _load("scifact_minilm.json", results_dir),
        _load("nfcorpus_minilm.json", results_dir),
        figures_dir / "retrieval-comparison.svg",
        results_dir=results_dir,
    )
    plot_evidence_curves(faith, figures_dir / "evidence-curves.svg", results_dir=results_dir)
    plot_corrective_sweep(corr, figures_dir / "corrective-sweep.svg", results_dir=results_dir)
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
        _macro("SciHybridP", _fmt_p(sci_p["Hybrid (RRF)"], True).replace("*", "")),
        _macro("SciHybridDelta", f"{sci_hd[0]:+.3f}"),
        _macro("SciHybridDeltaCI", f"[{sci_hd[1]:+.3f}, {sci_hd[2]:+.3f}]"),
        _macro("NFhybridP", _fmt_p(nf_p["Hybrid (RRF)"], True).replace("*", "")),
        _macro("NFhybridDelta", f"{nf_hd[0]:+.3f}"),
        _macro("NFhybridDeltaCI", f"[{nf_hd[1]:+.3f}, {nf_hd[2]:+.3f}]"),
    ]
    if mini:
        if sci.get("query_ids") != mini.get("query_ids"):
            raise ValueError("embedding comparisons require matching query_ids in the same order")
        bge = sci["per_query_ndcg"]["Dense"]
        ml = mini["per_query_ndcg"]["Dense"]
        gain = np.mean(bge) - np.mean(ml)
        macros += [
            _macro("SciDenseMini", nd(mini, "Dense")),
            _macro("SciEmbGain", f"{gain:+.3f}"),
            _macro(
                "SciEmbP",
                _fmt_p(paired_randomization_test(bge, ml, seed=sci.get("seed", 0)), True).replace(
                    "*", ""
                ),
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
    sci = _load("scifact_test.json", results_dir)
    nf = _load("nfcorpus_test.json", results_dir)
    if not sci or not nf:
        raise SystemExit(
            "Missing scifact_test.json / nfcorpus_test.json. Run `python -m scinexusrag.eval`, "
            "then `python -m scinexusrag.eval.report --results outputs/generated`."
        )
    mini = _load("scifact_minilm.json", results_dir)
    corr_sci = _load("corrective_scifact.json", results_dir)
    faith = _load("faithfulness_dev.json", results_dir)

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
    args = p.parse_args()
    if args.figures:
        write_figures(Path(args.figures), Path(args.results))
    else:
        write_paper_bundle(Path(args.paper), Path(args.results))


if __name__ == "__main__":
    main()
