"""Evidence-sentence detection: NLI vs lexical and cross-encoder baselines.

For each SciFact claim and its cited abstract, every abstract sentence is a
candidate; gold rationale sentences are positives. We score candidates with a
zero-shot NLI cross-encoder (relatedness = 1 - P(neutral)) and compare against
lexical overlap and a relevance cross-encoder, as threshold-free detection
(ROC-AUC, PR-AUC) plus F1 at a tuned threshold.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nexusrag.agents.grounding import GroundingVerifier
from nexusrag.eval.metrics import pr_auc, roc_auc
from nexusrag.retrieval.stopwords import STOP_WORDS

RAW_DIR = Path("data/scifact_raw/data")
VENDORED = Path(__file__).resolve().parents[3] / "benchmarks" / "datasets" / "scifact_claims_sample"
RESULTS_DIR = Path("benchmarks/results")
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]


@dataclass
class Claim:
    id: int
    text: str
    label: str
    gold: set[tuple[str, int]]
    candidates: list[tuple[str, int, str]]


def ensure_raw() -> bool:
    if (RAW_DIR / "corpus.jsonl").exists():
        return True
    import tarfile
    import urllib.request

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR.parent / "data.tar.gz"
    try:
        urllib.request.urlretrieve(SCIFACT_URL, archive)  # fixed https URL
        with tarfile.open(archive) as tar:
            tar.extractall(RAW_DIR.parent, filter="data")  # block path traversal
    except Exception:
        return False
    return (RAW_DIR / "corpus.jsonl").exists()


def _read_corpus(path: Path) -> dict[str, list[str]]:
    corpus: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["doc_id"])] = row["abstract"]
    return corpus


def _read_claims(path: Path, corpus: dict[str, list[str]]) -> list[Claim]:
    claims: list[Claim] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            evidence = row.get("evidence", {})
            if not evidence:
                continue
            gold: set[tuple[str, int]] = set()
            labels: set[str] = set()
            for doc_id, groups in evidence.items():
                for g in groups:
                    labels.add(g["label"])
                    for s in g["sentences"]:
                        gold.add((str(doc_id), int(s)))
            label = "CONTRADICT" if "CONTRADICT" in labels else "SUPPORT"
            candidates = [
                (str(doc_id), i, sent)
                for doc_id in evidence
                for i, sent in enumerate(corpus.get(str(doc_id), []))
            ]
            if candidates:
                claims.append(Claim(row["id"], row["claim"], label, gold, candidates))
    return claims


def load_claims(split: str, prefer_vendored: bool = False) -> list[Claim]:
    base = VENDORED if (prefer_vendored or not ensure_raw()) else RAW_DIR
    corpus = _read_corpus(base / "corpus.jsonl")
    return _read_claims(base / f"claims_{split}.jsonl", corpus)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP_WORDS and len(t) > 1}


def lexical_overlap(claim: str, sentence: str) -> float:
    c, s = _tokens(claim), _tokens(sentence)
    return len(c & s) / len(c | s) if c and s else 0.0


def _nli_relatedness(verifier: GroundingVerifier, claims: list[Claim]) -> list[list[float]]:
    out = []
    for c in claims:
        probs = verifier.class_probs([(sent, c.text) for (_, _, sent) in c.candidates])
        entail = probs[:, min(verifier.entail_idx, probs.shape[1] - 1)]
        contra = probs[:, min(verifier.contra_idx, probs.shape[1] - 1)]
        out.append([float(e + co) for e, co in zip(entail, contra, strict=True)])
    return out


def _f1_at(scores: list[float], labels: list[int], tau: float) -> float:
    tp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= tau and y)
    fp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= tau and not y)
    fn = sum(1 for s, y in zip(scores, labels, strict=True) if s < tau and y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _detection(scores: list[float], labels: list[int]) -> dict[str, float]:
    best_tau = max(THRESHOLD_GRID, key=lambda t: _f1_at(scores, labels, t))
    return {
        "roc_auc": roc_auc(scores, labels),
        "pr_auc": pr_auc(scores, labels),
        "f1": _f1_at(scores, labels, best_tau),
        "f1_tau": best_tau,
    }


def evaluate(
    prefer_vendored: bool = False,
    model_name: str | None = None,
    with_reranker: bool = True,
) -> dict[str, Any]:
    dev = load_claims("dev", prefer_vendored=prefer_vendored)
    verifier = (
        GroundingVerifier(model_name=model_name, device="cpu")
        if model_name
        else GroundingVerifier(device="cpu")
    )

    nli = _nli_relatedness(verifier, dev)
    labels: list[int] = []
    nli_flat: list[float] = []
    lex_flat: list[float] = []
    for ci, c in enumerate(dev):
        for j, (doc, idx, sent) in enumerate(c.candidates):
            labels.append(1 if (doc, idx) in c.gold else 0)
            nli_flat.append(nli[ci][j])
            lex_flat.append(lexical_overlap(c.text, sent))

    methods = {
        "nli": _detection(nli_flat, labels),
        "lexical_overlap": _detection(lex_flat, labels),
    }

    if with_reranker:
        from nexusrag.retrieval import Reranker

        rr = Reranker(device="cpu")
        ce_flat: list[float] = []
        for c in dev:
            scores = rr.model.predict([(c.text, sent) for (_, _, sent) in c.candidates])
            ce_flat.extend(float(s) for s in np.asarray(scores).reshape(-1))
        methods["cross_encoder"] = _detection(ce_flat, labels)

    base_rate = sum(labels) / len(labels) if labels else 0.0
    return {
        "dataset": "scifact-claims",
        "split": "sample" if prefer_vendored else "dev",
        "num_claims": len(dev),
        "num_candidates": len(labels),
        "gold_base_rate": base_rate,
        "nli_model": verifier.model_name,
        "methods": methods,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="SciFact evidence-detection evaluation")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--no-reranker", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = evaluate(
        prefer_vendored=args.sample, model_name=args.model, with_reranker=not args.no_reranker
    )
    print(
        f"claims={res['num_claims']}  candidates={res['num_candidates']}  base_rate={res['gold_base_rate']:.3f}"
    )
    for name, m in res["methods"].items():
        print(f"{name:16s} AUROC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  F1={m['f1']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "sample" if args.sample else "dev"
    out = Path(args.out) if args.out else RESULTS_DIR / f"faithfulness_{tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
