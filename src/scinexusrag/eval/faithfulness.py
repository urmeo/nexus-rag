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

from scinexusrag.eval.metrics import pr_auc, roc_auc
from scinexusrag.generation.grounding import GroundingVerifier
from scinexusrag.retrieval.stopwords import STOP_WORDS

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


def _score_methods(
    claims: list[Claim], verifier: GroundingVerifier, with_reranker: bool
) -> tuple[dict[str, list[float]], list[int], list[int]]:
    nli = _nli_relatedness(verifier, claims)
    labels: list[int] = []
    out: dict[str, list[float]] = {"nli": [], "lexical_overlap": []}
    claim_idx: list[int] = []
    for ci, c in enumerate(claims):
        for j, (doc, idx, sent) in enumerate(c.candidates):
            labels.append(1 if (doc, idx) in c.gold else 0)
            out["nli"].append(nli[ci][j])
            out["lexical_overlap"].append(lexical_overlap(c.text, sent))
            claim_idx.append(ci)
    if with_reranker:
        from scinexusrag.retrieval import Reranker

        rr = Reranker(device="cpu")
        ce: list[float] = []
        for c in claims:
            scores = rr.model.predict([(c.text, sent) for (_, _, sent) in c.candidates])
            ce.extend(float(s) for s in np.asarray(scores).reshape(-1))
        out["cross_encoder"] = ce
    return out, labels, claim_idx


def _bootstrap_auroc(
    scores: list[float], labels: list[int], claim_idx: list[int], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    s = np.asarray(scores)
    y = np.asarray(labels)
    idx = np.asarray(claim_idx)
    members = {c: np.where(idx == c)[0] for c in np.unique(idx)}
    rng = np.random.default_rng(seed)
    keys = list(members)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([members[c] for c in rng.choice(keys, len(keys), replace=True)])
        a = roc_auc(s[pick], y[pick])
        if a == a:  # skip NaN resamples
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(
    prefer_vendored: bool = False,
    model_name: str | None = None,
    with_reranker: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    dev = load_claims("dev", prefer_vendored=prefer_vendored)
    train = load_claims("train", prefer_vendored=prefer_vendored)[:120]
    verifier = (
        GroundingVerifier(model_name=model_name, device="cpu")
        if model_name
        else GroundingVerifier(device="cpu")
    )

    dev_s, labels, claim_idx = _score_methods(dev, verifier, with_reranker)
    train_s, train_y, _ = _score_methods(train, verifier, with_reranker)

    methods: dict[str, Any] = {}
    for name, scores in dev_s.items():
        tau = max(THRESHOLD_GRID, key=lambda t: _f1_at(train_s[name], train_y, t))
        lo, hi = _bootstrap_auroc(scores, labels, claim_idx, seed=seed)
        methods[name] = {
            "roc_auc": roc_auc(scores, labels),
            "roc_auc_ci": [lo, hi],
            "pr_auc": pr_auc(scores, labels),
            "f1": _f1_at(scores, labels, tau),
            "f1_tau": tau,  # tuned on train split
        }

    gap = None
    if "cross_encoder" in dev_s:
        d = roc_auc(dev_s["cross_encoder"], labels) - roc_auc(dev_s["nli"], labels)
        gap = round(d, 3)

    base_rate = sum(labels) / len(labels) if labels else 0.0
    return {
        "dataset": "scifact-claims",
        "split": "sample" if prefer_vendored else "dev",
        "num_claims": len(dev),
        "num_candidates": len(labels),
        "gold_base_rate": base_rate,
        "threshold_tuned_on": "train",
        "ce_minus_nli_auroc": gap,
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
