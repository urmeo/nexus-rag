"""Faithfulness: NLI rationale selection on SciFact gold evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nexusrag.agents.grounding import GroundingVerifier

RAW_DIR = Path("data/scifact_raw/data")
VENDORED = Path(__file__).resolve().parents[3] / "benchmarks" / "datasets" / "scifact_claims_sample"
RESULTS_DIR = Path("benchmarks/results")
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DEFAULT_TAU = 0.5


def ensure_raw() -> bool:
    """Download the SciFact release if absent."""
    if (RAW_DIR / "corpus.jsonl").exists():
        return True
    import tarfile
    import urllib.request

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR.parent / "data.tar.gz"
    try:
        print("Downloading SciFact release...")
        urllib.request.urlretrieve(SCIFACT_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(RAW_DIR.parent)
    except Exception:
        return False
    return (RAW_DIR / "corpus.jsonl").exists()


@dataclass
class Claim:
    id: int
    text: str
    label: str  # SUPPORT / CONTRADICT
    gold: set[tuple[str, int]]  # (doc_id, sentence_idx)
    candidates: list[tuple[str, int, str]]  # (doc_id, idx, sentence)


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
            candidates: list[tuple[str, int, str]] = []
            for doc_id in evidence:
                for i, sent in enumerate(corpus.get(str(doc_id), [])):
                    candidates.append((str(doc_id), i, sent))
            if candidates:
                claims.append(
                    Claim(
                        id=row["id"],
                        text=row["claim"],
                        label=label,
                        gold=gold,
                        candidates=candidates,
                    )
                )
    return claims


def load_claims(split: str, prefer_vendored: bool = False) -> list[Claim]:
    use_vendored = prefer_vendored or not ensure_raw()
    base = VENDORED if use_vendored else RAW_DIR
    corpus = _read_corpus(base / "corpus.jsonl")
    return _read_claims(base / f"claims_{split}.jsonl", corpus)


def _score_claims(verifier: GroundingVerifier, claims: list[Claim]) -> list[dict[str, Any]]:
    """Run NLI once per candidate sentence."""
    rows: list[dict[str, Any]] = []
    for c in claims:
        pairs = [(sent, c.text) for (_, _, sent) in c.candidates]
        probs = verifier.class_probs(pairs)
        entail = probs[:, min(verifier.entail_idx, probs.shape[1] - 1)]
        contra = probs[:, min(verifier.contra_idx, probs.shape[1] - 1)]
        for j, (doc_id, idx, _) in enumerate(c.candidates):
            rows.append(
                {
                    "claim_id": c.id,
                    "evidence_strength": float(max(entail[j], contra[j])),
                    "entail": float(entail[j]),
                    "contra": float(contra[j]),
                    "is_gold": (doc_id, idx) in c.gold,
                }
            )
    return rows


def _prf(rows: list[dict[str, Any]], tau: float) -> tuple[float, float, float]:
    tp = sum(1 for r in rows if r["evidence_strength"] >= tau and r["is_gold"])
    fp = sum(1 for r in rows if r["evidence_strength"] >= tau and not r["is_gold"])
    fn = sum(1 for r in rows if r["evidence_strength"] < tau and r["is_gold"])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _label_accuracy(claims: list[Claim], rows: list[dict[str, Any]], tau: float) -> float:
    by_claim: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_claim.setdefault(r["claim_id"], []).append(r)
    correct = 0
    for c in claims:
        picked = [r for r in by_claim.get(c.id, []) if r["evidence_strength"] >= tau]
        if not picked:
            pred = "NOINFO"
        else:
            pred = (
                "SUPPORT"
                if sum(r["entail"] for r in picked) >= sum(r["contra"] for r in picked)
                else "CONTRADICT"
            )
        if pred == c.label:
            correct += 1
    return correct / len(claims) if claims else 0.0


def evaluate(prefer_vendored: bool = False, model_name: str | None = None) -> dict[str, Any]:
    verifier = GroundingVerifier(model_name=model_name) if model_name else GroundingVerifier()

    dev = load_claims("dev", prefer_vendored=prefer_vendored)
    try:
        train = load_claims("train", prefer_vendored=prefer_vendored)[:120]
        train_rows = _score_claims(verifier, train)
        tau = max(THRESHOLD_GRID, key=lambda t: _prf(train_rows, t)[2])
    except FileNotFoundError:
        print(f"No held-out train split; using default tau={DEFAULT_TAU}")
        tau = DEFAULT_TAU

    dev_rows = _score_claims(verifier, dev)
    p, r, f1 = _prf(dev_rows, tau)
    acc = _label_accuracy(dev, dev_rows, tau)

    return {
        "dataset": "scifact-claims",
        "split": "sample" if prefer_vendored else "dev",
        "num_claims": len(dev),
        "num_candidate_sentences": len(dev_rows),
        "tuned_threshold": tau,
        "model": verifier.model_name,
        "nli_verifier": {
            "rationale_precision": p,
            "rationale_recall": r,
            "rationale_f1": f1,
            "label_accuracy": acc,
        },
        "citation_index_baseline": {
            "rationale_precision": 0.0,
            "rationale_recall": 0.0,
            "rationale_f1": 0.0,
            "note": "checks citation numbers only; cannot localize evidence sentences",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="SciFact faithfulness evaluation")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = evaluate(prefer_vendored=args.sample, model_name=args.model)
    v = res["nli_verifier"]
    print(f"claims={res['num_claims']}  tau={res['tuned_threshold']}")
    print(
        f"NLI rationale  P={v['rationale_precision']:.3f} R={v['rationale_recall']:.3f} F1={v['rationale_f1']:.3f}"
    )
    print(f"NLI label accuracy={v['label_accuracy']:.3f}")
    print("citation-index baseline F1=0.000 (no grounding signal)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "sample" if args.sample else "dev"
    out = Path(args.out) if args.out else RESULTS_DIR / f"faithfulness_{tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
