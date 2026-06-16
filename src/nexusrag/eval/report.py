"""Render tables, a figure, and LaTeX macros for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nexusrag.eval.metrics import holm_correction
from nexusrag.eval.metrics import paired_randomization_test as prt

RESULTS = Path("benchmarks/results")
COLS = ["nDCG@10", "R@5", "R@10", "R@20", "MRR", "MAP"]


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _p_vs_baseline(res: dict[str, Any], baseline: str = "BM25") -> dict[str, float]:
    pq = res["per_query_ndcg"]
    base = pq[baseline]
    return {n: prt(pq[n], base) for n in pq if n != baseline}


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
        "Scorer & ROC-AUC & PR-AUC & F1 \\\\",
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
            cells.append(txt)
        lines.append(f"{label.get(key, key)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def plot_ablation(sci: dict[str, Any], nf: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, title in ((axes[0], sci, "SciFact"), (axes[1], nf, "NFCorpus")):
        names = list(res["systems"])
        means = [res["systems"][n]["ci"]["nDCG@10"]["mean"] for n in names]
        lo = [res["systems"][n]["ci"]["nDCG@10"]["lo"] for n in names]
        hi = [res["systems"][n]["ci"]["nDCG@10"]["hi"] for n in names]
        err = [
            [max(0.0, m - x) for m, x in zip(means, lo, strict=True)],
            [max(0.0, x - m) for m, x in zip(means, hi, strict=True)],
        ]
        ax.bar(range(len(names)), means, yerr=err, capsize=4, color="#3b6ea5")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("nDCG@10")
        ax.set_title(f"{title} ({res['num_queries']} queries, 95% CI)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    sci_p = _p_vs_baseline(sci)
    nf_p = _p_vs_baseline(nf)
    macros = [
        _macro("NumQueries", str(sci["num_queries"])),
        _macro("CorpusSize", str(sci["corpus_size"])),
        _macro("NFqueries", str(nf["num_queries"])),
        _macro("NFcorpus", str(nf["corpus_size"])),
        _macro("SciBM", nd(sci, "BM25")),
        _macro("SciDense", nd(sci, "Dense")),
        _macro("SciHybrid", nd(sci, "Hybrid (RRF)")),
        _macro("SciHybridP", _fmt_p(sci_p["Hybrid (RRF)"], True).replace("*", "")),
        _macro("NFbm", nd(nf, "BM25")),
        _macro("NFhybrid", nd(nf, "Hybrid (RRF)")),
        _macro("NFhybridP", _fmt_p(nf_p["Hybrid (RRF)"], True).replace("*", "")),
    ]
    if mini:
        import numpy as np

        bge = sci["per_query_ndcg"]["Dense"]
        ml = mini["per_query_ndcg"]["Dense"]
        gain = np.mean(bge) - np.mean(ml)
        macros += [
            _macro("SciDenseMini", nd(mini, "Dense")),
            _macro("SciEmbGain", f"{gain:+.3f}"),
            _macro("SciEmbP", _fmt_p(prt(bge, ml), True).replace("*", "")),
        ]
    if corr_sci:
        cq = {r["system"]: r for r in corr_sci["cost_quality"]["systems"]}
        from nexusrag.config import SelfCorrectionSettings

        max_fire = max(s["trigger_rate"] for s in corr_sci["tau_sweep"])
        speedup = cq["Rerank (cross-enc)"]["latency_ms"] / max(cq["Adaptive"]["latency_ms"], 1e-9)
        macros += [
            _macro("SciCorrTau", f"{SelfCorrectionSettings().confidence_tau:.2f}"),
            _macro("SciCorrMaxFire", f"{max_fire * 100:.0f}\\%"),
            _macro("RerankMs", f"{cq['Rerank (cross-enc)']['latency_ms']:.0f}"),
            _macro("CorrMs", f"{cq['Corrective PRF']['latency_ms']:.0f}"),
            _macro("BaseMs", f"{cq['Adaptive']['latency_ms']:.0f}"),
            _macro("RerankSlowdown", f"{speedup:.0f}"),
            _macro("BaseND", f"{cq['Adaptive']['ndcg']:.3f}"),
            _macro("RerankND", f"{cq['Rerank (cross-enc)']['ndcg']:.3f}"),
        ]
    if faith:
        m = faith["methods"]
        macros += [
            _macro("FaithClaims", str(faith["num_claims"])),
            _macro("FaithBaseRate", f"{faith['gold_base_rate']:.2f}"),
            _macro("FaithNliAuroc", f"{m['nli']['roc_auc']:.3f}"),
            _macro("FaithLexAuroc", f"{m['lexical_overlap']['roc_auc']:.3f}"),
        ]
        if "cross_encoder" in m:
            macros.append(_macro("FaithCeAuroc", f"{m['cross_encoder']['roc_auc']:.3f}"))
    return macros


def write_paper_bundle(paper_dir: Path) -> None:
    sci = _load("scifact_test.json")
    nf = _load("nfcorpus_test.json")
    if not sci or not nf:
        raise SystemExit("missing scifact_test.json / nfcorpus_test.json; run `make eval` first")
    mini = _load("scifact_minilm.json")
    corr_sci = _load("corrective_scifact.json")
    faith = _load("faithfulness_dev.json")

    (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_dir / "figures").mkdir(parents=True, exist_ok=True)
    (paper_dir / "tables" / "scifact.tex").write_text(ablation_table(sci) + "\n")
    (paper_dir / "tables" / "nfcorpus.tex").write_text(ablation_table(nf) + "\n")
    if corr_sci:
        (paper_dir / "tables" / "cost_quality.tex").write_text(cost_quality_table(corr_sci) + "\n")
    if faith:
        (paper_dir / "tables" / "faithfulness.tex").write_text(faithfulness_table(faith) + "\n")
    plot_ablation(sci, nf, paper_dir / "figures" / "ablation.png")
    (paper_dir / "generated.tex").write_text(
        "\n".join(build_macros(sci, nf, mini, corr_sci, faith)) + "\n"
    )
    print(f"wrote paper bundle to {paper_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Render eval tables, figure, and macros")
    p.add_argument("--paper", default="paper", help="paper directory to populate")
    args = p.parse_args()
    write_paper_bundle(Path(args.paper))


if __name__ == "__main__":
    main()
