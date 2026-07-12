"""Regression gate for the offline sample eval.

Reruns the vendored SciFact retrieval ablation and the SciFact-claims
evidence-detection eval (both seed 0, CPU, deterministic) and fails if any
tracked metric drops below the committed floor in ``benchmarks/thresholds.json``.
Run in CI so a change that quietly degrades retrieval or faithfulness cannot
merge. Update the floors deliberately when a real improvement lands.
"""

from __future__ import annotations

import json
from pathlib import Path

from scinexusrag.eval import faithfulness as F
from scinexusrag.eval.run import evaluate as run_retrieval

THRESHOLDS = Path("benchmarks/thresholds.json")


def _check(observed: float, floor: float, label: str) -> str | None:
    status = "ok" if observed >= floor else "FAIL"
    print(f"  {label:38s} {observed:.4f}  (floor {floor:.4f})  {status}")
    return None if observed >= floor else f"{label}: {observed:.4f} < floor {floor:.4f}"


def main() -> int:
    spec = json.loads(THRESHOLDS.read_text())
    failures: list[str] = []

    print("Retrieval (sample):")
    ret = run_retrieval(dataset="scifact", split="test", use_sample=True, seed=0)
    systems = ret["systems"]
    for name, metrics in spec["retrieval"]["floors"].items():
        got = systems[name]["means"]
        for metric, floor in metrics.items():
            fail = _check(got[metric], floor, f"{name} {metric}")
            if fail:
                failures.append(fail)

    print("Faithfulness (sample):")
    faith = F.evaluate(prefer_vendored=True, with_reranker=True, seed=0)
    for name, metrics in spec["faithfulness"]["floors"].items():
        got = faith["methods"][name]
        for metric, floor in metrics.items():
            fail = _check(got[metric], floor, f"{name} {metric}")
            if fail:
                failures.append(fail)

    if failures:
        print("\nEVAL REGRESSION GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nEval regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
