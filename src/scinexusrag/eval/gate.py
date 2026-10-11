"""Check offline sample results against the committed regression floors."""

from __future__ import annotations

import json
from pathlib import Path

from scinexusrag.eval import faithfulness
from scinexusrag.eval.run import evaluate as run_retrieval

THRESHOLDS = Path(__file__).resolve().parent / "data" / "thresholds.json"


def _check(observed: float, floor: float, label: str) -> str | None:
    passed = observed >= floor
    status = "ok" if passed else "FAIL"
    print(f"  {label:38s} {observed:.4f}  (floor {floor:.4f})  {status}")
    return None if passed else f"{label}: {observed:.4f} < floor {floor:.4f}"


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

    print("Evidence-sentence detection (sample):")
    faith = faithfulness.evaluate(prefer_vendored=True, with_reranker=True, seed=0)
    for name, metrics in spec["faithfulness"]["floors"].items():
        got = faith["methods"][name]
        for metric, floor in metrics.items():
            fail = _check(got[metric], floor, f"{name} {metric}")
            if fail:
                failures.append(fail)

    if failures:
        print("\nEVAL REGRESSION GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nEval regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
