"""Render result tables and figures for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COLUMNS = ["nDCG@10", "R@5", "R@10", "R@20", "MRR", "MAP"]


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
        return data


def markdown_table(res: dict[str, Any]) -> str:
    head = "| System | " + " | ".join(COLUMNS) + " | p vs final |"
    sep = "|" + "---|" * (len(COLUMNS) + 2)
    lines = [head, sep]
    for name, s in res["systems"].items():
        cells = [f"{s['means'][c]:.3f}" for c in COLUMNS]
        p = s.get("p_vs_final")
        pcell = "—" if p is None else (f"{p:.3f}" + ("*" if p < 0.05 else ""))
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {pcell} |")
    return "\n".join(lines)


def latex_table(res: dict[str, Any]) -> str:
    cols = "l" + "r" * len(COLUMNS)
    names = list(res["systems"])
    column_max = {c: max(res["systems"][n]["means"][c] for n in names) for c in COLUMNS}
    lines = [
        "\\begin{tabular}{" + cols + "}",
        "\\toprule",
        "System & " + " & ".join(COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for i, (name, s) in enumerate(res["systems"].items()):
        cells = []
        for c in COLUMNS:
            v = s["means"][c]
            txt = f"{v:.3f}"
            if abs(v - column_max[c]) < 1e-9:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        safe = name.replace("&", "\\&")
        lines.append(safe + " & " + " & ".join(cells) + " \\\\")
        if i == 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def plot_ablation(res: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res["systems"])
    means = [res["systems"][n]["ci"]["nDCG@10"]["mean"] for n in names]
    los = [res["systems"][n]["ci"]["nDCG@10"]["lo"] for n in names]
    his = [res["systems"][n]["ci"]["nDCG@10"]["hi"] for n in names]
    err = [
        [max(0.0, m - lo) for m, lo in zip(means, los, strict=True)],
        [max(0.0, hi - m) for m, hi in zip(means, his, strict=True)],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = range(len(names))
    ax.bar(x, means, yerr=err, capsize=4, color="#3b6ea5")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("nDCG@10")
    ax.set_title(f"Retrieval ablation on {res['dataset']} ({res['num_queries']} queries, 95% CI)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def write_paper_bundle(
    res: dict[str, Any], paper_dir: Path, faith: dict[str, Any] | None = None
) -> None:
    """Emit LaTeX macros, table and figure into the paper tree."""
    (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_dir / "figures").mkdir(parents=True, exist_ok=True)
    (paper_dir / "tables" / "ablation.tex").write_text(latex_table(res) + "\n")
    plot_ablation(res, paper_dir / "figures" / "ablation.png")

    sysd = res["systems"]
    names = list(sysd)

    def mean(name: str, metric: str) -> str:
        return f"{sysd[name]['means'][metric]:.3f}"

    best = max(sysd, key=lambda n: sysd[n]["means"]["nDCG@10"])
    dense_p = sysd["Dense (MiniLM)"].get("p_vs_final")
    macros = [
        _macro("DatasetName", res["dataset"]),
        _macro("NumQueries", str(res["num_queries"])),
        _macro("CorpusSize", str(res["corpus_size"])),
        _macro("BMNDCG", mean("BM25", "nDCG@10")),
        _macro("DenseNDCG", mean("Dense (MiniLM)", "nDCG@10")),
        _macro("HybridNDCG", mean("Hybrid-RRF", "nDCG@10")),
        _macro("FullNDCG", mean(names[-1], "nDCG@10")),
        _macro("FullRten", mean(names[-1], "R@10")),
        _macro("FullName", names[-1].replace("&", "\\&")),
        _macro("BestName", best.replace("&", "\\&")),
        _macro("BestNDCG", mean(best, "nDCG@10")),
        _macro("BMRtwenty", mean("BM25", "R@20")),
        _macro("DenseRtwenty", mean("Dense (MiniLM)", "R@20")),
        _macro("HybridRtwenty", mean("Hybrid-RRF", "R@20")),
        _macro("DenseP", f"{dense_p:.3f}" if dense_p is not None else "n/a"),
    ]
    if faith:
        v = faith["nli_verifier"]
        macros += [
            _macro("FaithFone", f"{v['rationale_f1']:.3f}"),
            _macro("FaithPrec", f"{v['rationale_precision']:.3f}"),
            _macro("FaithRecall", f"{v['rationale_recall']:.3f}"),
            _macro("FaithAcc", f"{v['label_accuracy']:.3f}"),
            _macro("FaithTau", str(faith["tuned_threshold"])),
            _macro("NumClaims", str(faith["num_claims"])),
        ]
    (paper_dir / "generated.tex").write_text("\n".join(macros) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Render eval tables and figures")
    p.add_argument("results")
    p.add_argument("--outdir", default="benchmarks/results")
    p.add_argument("--paper", default=None, help="paper dir to populate")
    p.add_argument("--faith", default=None, help="faithfulness results json")
    args = p.parse_args()

    res = _load(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.results).stem

    (outdir / f"{stem}_table.md").write_text(markdown_table(res) + "\n")
    (outdir / f"{stem}_table.tex").write_text(latex_table(res) + "\n")
    plot_ablation(res, outdir / f"{stem}_ablation.png")

    if args.paper:
        faith = _load(args.faith) if args.faith else None
        write_paper_bundle(res, Path(args.paper), faith=faith)

    print(markdown_table(res))
    print(f"\nWrote tables + figure to {outdir}")


if __name__ == "__main__":
    main()
