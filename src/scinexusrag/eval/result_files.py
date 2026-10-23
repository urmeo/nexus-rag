"""Result tables."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any, NoReturn

SCHEMAS = {
    "metadata": ("run", "section", "key", "type", "value"),
    "retrieval_systems": (
        "run",
        "system",
        "latency_ms_per_query",
        "p_vs_final",
        "delta_vs_bm25_mean",
        "delta_vs_bm25_lo",
        "delta_vs_bm25_hi",
        "p_vs_bm25",
    ),
    "retrieval_metrics": ("run", "system", "metric", "mean", "ci_mean", "ci_lo", "ci_hi"),
    "retrieval_queries": ("run", "system", "query_order", "query_id", "ndcg_at_10"),
    "corrective_sweeps": (
        "run",
        "split",
        "tau",
        "ndcg",
        "trigger_rate",
        "triggered_n",
        "triggered_base_ndcg",
        "triggered_corr_ndcg",
        "triggered_delta",
        "triggered_p",
    ),
    "corrective_queries": ("run", "role", "query_order", "query_id"),
    "corrective_cost_quality": ("run", "system", "ndcg", "r20", "latency_ms"),
    "faithfulness_methods": (
        "run",
        "method",
        "roc_auc",
        "roc_auc_ci_lo",
        "roc_auc_ci_hi",
        "pr_auc",
        "f1",
        "f1_tau",
    ),
    "faithfulness_candidates": (
        "run",
        "split",
        "candidate_order",
        "claim_id",
        "document_id",
        "sentence_index",
        "label",
    ),
}
_STRUCTURED = {
    "retrieval": {"systems", "per_query_ndcg", "query_ids"},
    "corrective": {"tau_sweep", "tune_sweep", "query_ids", "tune_query_ids", "cost_quality"},
    "faithfulness": {"methods", "per_candidate", "data_sha256"},
}


def write_result(result: dict[str, Any], path: Path, *, trailing_newline: bool = False) -> Path:
    content = json.dumps(result, indent=2, allow_nan=False) + ("\n" if trailing_newline else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _stem(name: str) -> str:
    path = Path(name)
    if path.name != name or not path.stem or path.stem in {".", ".."}:
        raise ValueError(f"Invalid result name: {name!r}")
    return path.stem if path.suffix in {".json", ".csv"} else name


def _fields(value: Mapping[str, Any], expected: Sequence[str] | set[str]) -> None:
    if set(value) != set(expected):
        raise ValueError(f"Expected fields {sorted(expected)}, got {sorted(value)}")


def _scalar(value: Any) -> tuple[str, str]:
    if value is None:
        return "null", ""
    if isinstance(value, bool):
        return "bool", str(value).lower()
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float) and math.isfinite(value):
        return "float", repr(value)
    if isinstance(value, str):
        return "str", value
    raise ValueError(f"Expected a finite scalar, got {type(value).__name__}")


def _decode_scalar(kind: str, value: str) -> Any:
    if kind == "str":
        return value
    if kind == "null" and value == "":
        return None
    if kind == "bool" and value in {"true", "false"}:
        return value == "true"
    if kind == "int":
        return int(value)
    if kind == "float":
        return _number(value)
    raise ValueError(f"Invalid metadata type or value: {kind!r}, {value!r}")


def _number(value: Any, *, optional: bool = False) -> float | None:
    if optional and (value is None or value == ""):
        return None
    if isinstance(value, bool):
        raise ValueError("Expected a number, got a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid number: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Expected a finite number: {value!r}")
    return number


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Invalid integer: {value!r}")
    return int(value)


def _unique(values: Sequence[Any], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label}")


def _ordered(rows: list[dict[str, str]], column: str) -> list[dict[str, str]]:
    if [_integer(row[column]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{column} must be consecutive and start at zero")
    return rows


def _rows(directory: Path, table: str, run: str) -> list[dict[str, str]]:
    path = directory / f"{table}.csv"
    if not path.is_file():
        raise ValueError(f"Missing result table: {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        expected = list(SCHEMAS[table])
        extra = headers[len(expected) :]
        if (
            headers[: len(expected)] != expected
            or (
                extra
                and (
                    table != "faithfulness_candidates"
                    or any(not key.startswith("score_") or key == "score_" for key in extra)
                )
            )
            or len(set(headers)) != len(headers)
        ):
            raise ValueError(f"Invalid columns in {path.name}")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed row in {path.name}")
            if row["run"] == run:
                rows.append(row)
    return rows


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"Nonfinite JSON number: {value}")


def load_result(name: str, results_dir: Path) -> dict[str, Any] | None:
    """Load measurements."""
    run = _stem(name)
    json_path = results_dir / f"{run}.json"
    if json_path.is_file():
        value = json.loads(
            json_path.read_text(encoding="utf-8"),
            parse_float=_number,
            parse_constant=_reject_constant,
        )
        if not isinstance(value, dict):
            raise ValueError(f"Expected a result object in {json_path.name}")
        return value
    if not (results_dir / "metadata.csv").is_file():
        return None
    metadata = _rows(results_dir, "metadata", run)
    if not metadata:
        return None
    result: dict[str, Any] = {}
    sections: dict[str, dict[str, Any]] = {}
    for row in metadata:
        section = row["section"]
        if section not in {
            "format",
            "config",
            "provenance",
            "packages",
            "data_sha256",
            "cost_quality",
        }:
            raise ValueError(f"Unknown metadata section: {section}")
        values = sections.setdefault(section, {})
        if not row["key"] or row["key"] in values:
            raise ValueError(f"Duplicate or empty metadata key in {run}: {section}/{row['key']}")
        values[row["key"]] = _decode_scalar(row["type"], row["value"])
    _fields(sections.get("format", {}), {"kind"})
    kind = sections["format"]["kind"]
    if kind not in _STRUCTURED:
        raise ValueError(f"Unknown result kind: {kind!r}")
    result.update(sections.get("config", {}))
    if set(result) & (_STRUCTURED[kind] | {"provenance"}):
        raise ValueError("Structured result fields cannot be scalar metadata")
    if "provenance" in sections or "packages" in sections:
        result["provenance"] = sections.get("provenance", {})
        if "packages" in sections:
            if "packages" in result["provenance"]:
                raise ValueError("Duplicate package provenance")
            result["provenance"]["packages"] = sections["packages"]
    if kind == "retrieval":
        _load_retrieval(result, results_dir, run)
    elif kind == "corrective":
        result["cost_quality"] = sections.get("cost_quality", {})
        _load_corrective(result, results_dir, run)
    else:
        if "data_sha256" in sections:
            result["data_sha256"] = sections["data_sha256"]
        _load_faithfulness(result, results_dir, run)
    allowed = {"corrective": {"cost_quality"}, "faithfulness": {"data_sha256"}}.get(kind, set())
    unexpected = set(sections) & ({"cost_quality", "data_sha256"} - allowed)
    if unexpected:
        raise ValueError(f"Unexpected metadata sections for {kind}: {sorted(unexpected)}")
    return result


def _load_retrieval(result: dict[str, Any], directory: Path, run: str) -> None:
    systems: dict[str, Any] = {}
    for row in _rows(directory, "retrieval_systems", run):
        name = row["system"]
        if not name or name in systems:
            raise ValueError(f"Duplicate or empty retrieval system: {name!r}")
        delta = [
            _number(row[f"delta_vs_bm25_{key}"], optional=True) for key in ("mean", "lo", "hi")
        ]
        if any(value is None for value in delta) and any(value is not None for value in delta):
            raise ValueError("Incomplete BM25 difference interval")
        systems[name] = {
            "means": {},
            "ci": {},
            "latency_ms_per_query": _number(row["latency_ms_per_query"]),
            "p_vs_final": _number(row["p_vs_final"], optional=True),
            "delta_vs_bm25": None
            if delta[0] is None
            else dict(zip(("mean", "lo", "hi"), delta, strict=True)),
            "p_vs_bm25": _number(row["p_vs_bm25"], optional=True),
        }
    if not systems:
        raise ValueError(f"Missing retrieval systems for {run}")
    for row in _rows(directory, "retrieval_metrics", run):
        if row["system"] not in systems:
            raise ValueError(f"Unknown retrieval system: {row['system']}")
        system = systems[row["system"]]
        metric = row["metric"]
        if not metric or metric in system["means"]:
            raise ValueError(f"Duplicate or empty retrieval metric: {metric!r}")
        system["means"][metric] = _number(row["mean"])
        system["ci"][metric] = {key: _number(row[f"ci_{key}"]) for key in ("mean", "lo", "hi")}
    metrics = set(next(iter(systems.values()))["means"])
    if not metrics or any(set(system["means"]) != metrics for system in systems.values()):
        raise ValueError("Retrieval systems must have the same nonempty metric set")
    queries: dict[str, list[dict[str, str]]] = {name: [] for name in systems}
    for row in _rows(directory, "retrieval_queries", run):
        if row["system"] not in queries:
            raise ValueError(f"Unknown retrieval system: {row['system']}")
        queries[row["system"]].append(row)
    ids: list[str] | None = None
    scores = {}
    for name, rows in queries.items():
        _ordered(rows, "query_order")
        query_ids = [row["query_id"] for row in rows]
        _unique(query_ids, "query IDs")
        if not query_ids or (ids is not None and query_ids != ids):
            raise ValueError("Retrieval query IDs must align across systems")
        ids = query_ids
        scores[name] = [_number(row["ndcg_at_10"]) for row in rows]
    if "num_queries" in result and len(ids or []) != result["num_queries"]:
        raise ValueError("Retrieval query count does not match metadata")
    result.update(systems=systems, query_ids=ids, per_query_ndcg=scores)


def _load_corrective(result: dict[str, Any], directory: Path, run: str) -> None:
    sweeps: dict[str, list[dict[str, Any]]] = {"test": [], "tune": []}
    for row in _rows(directory, "corrective_sweeps", run):
        if row["split"] not in sweeps:
            raise ValueError(f"Unknown corrective sweep split: {row['split']}")
        sweep = {
            key: _integer(row[key])
            if key == "triggered_n"
            else _number(row[key], optional=key.startswith("triggered_"))
            for key in SCHEMAS["corrective_sweeps"][2:]
        }
        sweeps[row["split"]].append(sweep)
    for rows in sweeps.values():
        if not rows:
            raise ValueError(f"Missing corrective sweep for {run}")
        _unique([row["tau"] for row in rows], "corrective thresholds")
    queries: dict[str, list[dict[str, str]]] = {"test": [], "tune": [], "timed": []}
    for row in _rows(directory, "corrective_queries", run):
        if row["role"] not in queries:
            raise ValueError(f"Unknown corrective query role: {row['role']}")
        queries[row["role"]].append(row)
    ids = {}
    for role, rows in queries.items():
        _ordered(rows, "query_order")
        ids[role] = [row["query_id"] for row in rows]
        _unique(ids[role], "corrective query IDs")
        if not ids[role]:
            raise ValueError(f"Missing {role} corrective queries for {run}")
    cost = result["cost_quality"]
    if set(cost) & {"systems", "query_ids"}:
        raise ValueError("Structured cost fields cannot be scalar metadata")
    for role, count in (
        ("test", result.get("num_queries")),
        ("tune", result.get("tune_queries")),
        ("timed", cost.get("timed_queries")),
    ):
        if count is not None and count != len(ids[role]):
            raise ValueError(f"Corrective {role} query count does not match metadata")
    if not set(ids["timed"]) <= set(ids["test"]):
        raise ValueError("Timed corrective queries must belong to the test split")
    systems = []
    for row in _rows(directory, "corrective_cost_quality", run):
        systems.append(
            {
                key: row[key] if key == "system" else _number(row[key])
                for key in SCHEMAS["corrective_cost_quality"][1:]
            }
        )
    if not systems:
        raise ValueError(f"Missing corrective cost measurements for {run}")
    _unique([row["system"] for row in systems], "corrective cost systems")
    cost.update(systems=systems, query_ids=ids["timed"])
    result.update(
        tau_sweep=sweeps["test"],
        tune_sweep=sweeps["tune"],
        query_ids=ids["test"],
        tune_query_ids=ids["tune"],
    )


def _load_faithfulness(result: dict[str, Any], directory: Path, run: str) -> None:
    methods: dict[str, Any] = {}
    for row in _rows(directory, "faithfulness_methods", run):
        name = row["method"]
        if not name or name in methods:
            raise ValueError(f"Duplicate or empty evidence method: {name!r}")
        methods[name] = {key: _number(row[key]) for key in ("roc_auc", "pr_auc", "f1", "f1_tau")}
        methods[name]["roc_auc_ci"] = [
            _number(row[key]) for key in ("roc_auc_ci_lo", "roc_auc_ci_hi")
        ]
    if not methods:
        raise ValueError(f"Missing evidence methods for {run}")
    splits: dict[str, list[dict[str, str]]] = {}
    for row in _rows(directory, "faithfulness_candidates", run):
        if not row["split"]:
            raise ValueError("Evidence candidate split cannot be empty")
        score_names = {key.removeprefix("score_") for key in row if key.startswith("score_")}
        if not set(methods) <= score_names or any(
            row[f"score_{name}"] != "" for name in score_names - set(methods)
        ):
            raise ValueError("Evidence candidate scores do not match the recorded methods")
        splits.setdefault(row["split"], []).append(row)
    if not splits:
        raise ValueError(f"Missing evidence candidates for {run}")
    candidates: dict[str, Any] = {}
    for split, rows in splits.items():
        _ordered(rows, "candidate_order")
        labels = [_integer(row["label"]) for row in rows]
        if any(label not in {0, 1} for label in labels):
            raise ValueError("Evidence labels must be zero or one")
        candidates[split] = {
            "claim_ids": [_integer(row["claim_id"]) for row in rows],
            "document_ids": [row["document_id"] for row in rows],
            "sentence_indices": [_integer(row["sentence_index"]) for row in rows],
            "labels": labels,
            "scores": {name: [_number(row[f"score_{name}"]) for row in rows] for name in methods},
        }
    if "num_candidates" in result and (
        "dev" not in candidates or len(candidates["dev"]["labels"]) != result["num_candidates"]
    ):
        raise ValueError("Evidence candidate count does not match metadata")
    result.update(methods=methods, per_candidate=candidates)


def write_csv_results(results: Mapping[str, dict[str, Any]], output_dir: Path) -> None:
    """Export measurements."""
    if not results:
        raise ValueError("No result runs to export")
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in SCHEMAS}
    score_columns: list[str] = []

    def metadata(run: str, section: str, key: str, value: Any) -> None:
        kind, text = _scalar(value)
        if not key:
            raise ValueError("Metadata keys cannot be empty")
        tables["metadata"].append(
            {"run": run, "section": section, "key": key, "type": kind, "value": text}
        )

    runs = set()
    for name, result in results.items():
        run = _stem(name)
        if run in runs:
            raise ValueError(f"Duplicate result run: {run}")
        runs.add(run)
        kinds = [kind for kind, keys in _STRUCTURED.items() if keys <= set(result)]
        if len(kinds) != 1:
            raise ValueError(f"Unrecognized or incomplete result shape: {run}")
        kind = kinds[0]
        metadata(run, "format", "kind", kind)
        for key, value in result.items():
            if key not in _STRUCTURED[kind] | {"provenance"}:
                metadata(run, "config", key, value)
        provenance = result.get("provenance")
        if "provenance" in result:
            if not isinstance(provenance, dict) or not provenance:
                raise ValueError("Provenance must be a nonempty object")
            for key, value in provenance.items():
                if key == "packages":
                    if not isinstance(value, dict) or not value:
                        raise ValueError("Package versions must be a nonempty object")
                    for package, version in value.items():
                        metadata(run, "packages", package, version)
                else:
                    metadata(run, "provenance", key, value)
        if kind == "retrieval":
            _write_retrieval(tables, run, result)
        elif kind == "corrective":
            _write_corrective(tables, run, result, metadata)
        else:
            _write_faithfulness(tables, run, result, metadata, score_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".results-", dir=output_dir) as temporary:
        directory = Path(temporary)
        for table, rows in tables.items():
            fields = list(SCHEMAS[table])
            if table == "faithfulness_candidates":
                fields.extend(score_columns)
            for row in rows:
                for value in row.values():
                    _scalar(value)
            with (directory / f"{table}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                writer.writerows(rows)
        for name, expected in results.items():
            if load_result(name, directory) != expected:
                raise ValueError(f"CSV export did not preserve result data: {name}")
        _publish_tables(directory, output_dir, list(tables))


def _publish_tables(directory: Path, output_dir: Path, tables: Sequence[str]) -> None:
    backup = Path(mkdtemp(prefix=".results-backup-", dir=output_dir))
    attempted = []
    try:
        for table in tables:
            target = output_dir / f"{table}.csv"
            if target.exists() or target.is_symlink():
                shutil.copy2(target, backup / target.name, follow_symlinks=False)
        for table in tables:
            target = output_dir / f"{table}.csv"
            attempted.append(target)
            (directory / target.name).replace(target)
    except BaseException:
        try:
            for target in reversed(attempted):
                previous = backup / target.name
                if previous.exists() or previous.is_symlink():
                    previous.replace(target)
                else:
                    target.unlink(missing_ok=True)
        except BaseException as recovery_error:
            raise OSError(
                f"CSV export recovery failed; backups retained at {backup}"
            ) from recovery_error
        shutil.rmtree(backup)
        raise
    shutil.rmtree(backup)


def _write_retrieval(
    tables: dict[str, list[dict[str, Any]]], run: str, result: dict[str, Any]
) -> None:
    systems = result["systems"]
    scores = result["per_query_ndcg"]
    _fields(scores, systems)
    ids = result["query_ids"]
    _unique(ids, "query IDs")
    for name, system in systems.items():
        _fields(
            system,
            {"means", "ci", "latency_ms_per_query", "p_vs_final", "delta_vs_bm25", "p_vs_bm25"},
        )
        row = {
            key: value
            for key, value in system.items()
            if key not in {"means", "ci", "delta_vs_bm25"}
        }
        delta = system["delta_vs_bm25"]
        if delta is not None:
            _fields(delta, {"mean", "lo", "hi"})
        row.update(
            {
                f"delta_vs_bm25_{key}": None if delta is None else delta[key]
                for key in ("mean", "lo", "hi")
            }
        )
        tables["retrieval_systems"].append(dict(run=run, system=name, **row))
        _fields(system["ci"], system["means"])
        for metric, mean in system["means"].items():
            interval = system["ci"][metric]
            _fields(interval, {"mean", "lo", "hi"})
            tables["retrieval_metrics"].append(
                dict(
                    run=run,
                    system=name,
                    metric=metric,
                    mean=mean,
                    **{f"ci_{key}": value for key, value in interval.items()},
                )
            )
        if len(scores[name]) != len(ids):
            raise ValueError("Retrieval scores must align with query IDs")
        for index, (query_id, score) in enumerate(zip(ids, scores[name], strict=True)):
            tables["retrieval_queries"].append(
                {
                    "run": run,
                    "system": name,
                    "query_order": index,
                    "query_id": query_id,
                    "ndcg_at_10": score,
                }
            )


def _write_corrective(
    tables: dict[str, list[dict[str, Any]]], run: str, result: dict[str, Any], metadata: Any
) -> None:
    cost = result["cost_quality"]
    for key, value in cost.items():
        if key not in {"systems", "query_ids"}:
            metadata(run, "cost_quality", key, value)
    for role, ids in (
        ("test", result["query_ids"]),
        ("tune", result["tune_query_ids"]),
        ("timed", cost["query_ids"]),
    ):
        _unique(ids, "corrective query IDs")
        for index, query_id in enumerate(ids):
            tables["corrective_queries"].append(
                {"run": run, "role": role, "query_order": index, "query_id": query_id}
            )
    for split, key in (("test", "tau_sweep"), ("tune", "tune_sweep")):
        for sweep in result[key]:
            _fields(sweep, SCHEMAS["corrective_sweeps"][2:])
            tables["corrective_sweeps"].append(dict(run=run, split=split, **sweep))
    for system in cost["systems"]:
        _fields(system, SCHEMAS["corrective_cost_quality"][1:])
        tables["corrective_cost_quality"].append(dict(run=run, **system))


def _write_faithfulness(
    tables: dict[str, list[dict[str, Any]]],
    run: str,
    result: dict[str, Any],
    metadata: Any,
    score_columns: list[str],
) -> None:
    if not result["data_sha256"]:
        raise ValueError("Evidence source hashes cannot be empty")
    for filename, digest in result["data_sha256"].items():
        metadata(run, "data_sha256", filename, digest)
    for name, method in result["methods"].items():
        _fields(method, {"roc_auc", "roc_auc_ci", "pr_auc", "f1", "f1_tau"})
        if len(method["roc_auc_ci"]) != 2:
            raise ValueError("Evidence ROC-AUC intervals need two limits")
        row = {key: value for key, value in method.items() if key != "roc_auc_ci"}
        tables["faithfulness_methods"].append(
            dict(
                run=run,
                method=name,
                roc_auc_ci_lo=method["roc_auc_ci"][0],
                roc_auc_ci_hi=method["roc_auc_ci"][1],
                **row,
            )
        )
        column = f"score_{name}"
        if column not in score_columns:
            score_columns.append(column)
    for split, candidates in result["per_candidate"].items():
        _fields(candidates, {"claim_ids", "document_ids", "sentence_indices", "labels", "scores"})
        _fields(candidates["scores"], result["methods"])
        length = len(candidates["labels"])
        columns = [
            candidates[key] for key in ("claim_ids", "document_ids", "sentence_indices", "labels")
        ]
        columns.extend(candidates["scores"].values())
        if any(len(values) != length for values in columns):
            raise ValueError("Evidence candidate columns must have equal lengths")
        for index in range(length):
            row = {
                key: candidates[field][index]
                for key, field in (
                    ("claim_id", "claim_ids"),
                    ("document_id", "document_ids"),
                    ("sentence_index", "sentence_indices"),
                    ("label", "labels"),
                )
            }
            row.update(
                {f"score_{name}": values[index] for name, values in candidates["scores"].items()}
            )
            tables["faithfulness_candidates"].append(
                dict(run=run, split=split, candidate_order=index, **row)
            )
