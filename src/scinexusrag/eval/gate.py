"""Regression checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scinexusrag.eval import faithfulness
from scinexusrag.eval.run import evaluate as run_retrieval

THRESHOLDS = Path(__file__).resolve().parent / "data" / "thresholds.json"


def _check(observed: float, floor: float, label: str) -> str | None:
    if not _valid_score(observed):
        failure = f"{label}: invalid score {observed!r}"
        print(f"  {failure}  FAIL")
        return failure
    passed = observed >= floor
    status = "ok" if passed else "FAIL"
    print(f"  {label:38s} {observed:.4f}  (floor {floor:.4f})  {status}")
    return None if passed else f"{label}: {observed:.4f} < floor {floor:.4f}"


def _valid_score(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _protocol_failures(
    result: dict[str, Any], expected: dict[str, Any], fields: tuple[str, ...], label: str
) -> list[str]:
    return [
        f"{label} {field}: expected {expected[field]!r}, got {result.get(field)!r}"
        for field in fields
        if result.get(field) != expected[field]
    ]


def main() -> int:
    spec = json.loads(THRESHOLDS.read_text())
    failures: list[str] = []
    for section in ("retrieval", "faithfulness"):
        for name, floors in spec[section]["floors"].items():
            for metric, floor in floors.items():
                if not _valid_score(floor):
                    raise ValueError(f"Invalid regression floor for {name} {metric}: {floor!r}")

    print("Retrieval (sample):")
    ret = run_retrieval(dataset="scifact", split="test", use_sample=True, seed=0)
    failures.extend(
        _protocol_failures(
            ret, spec["retrieval"], ("dataset", "split", "num_queries", "corpus_size"), "Retrieval"
        )
    )
    systems = ret["systems"]
    for name, metrics in spec["retrieval"]["floors"].items():
        got = systems[name]["means"]
        for metric, floor in metrics.items():
            fail = _check(got[metric], floor, f"{name} {metric}")
            if fail:
                failures.append(fail)

    print("Evidence-sentence detection (sample):")
    faith = faithfulness.evaluate(prefer_vendored=True, with_reranker=True, seed=0)
    failures.extend(
        _protocol_failures(
            faith, spec["faithfulness"], ("dataset", "split", "num_claims"), "Evidence"
        )
    )
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
