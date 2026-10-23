"""Dataset loaders."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from sys import float_info
from typing import Any

VENDORED_DIR = Path(__file__).resolve().parent / "data"


BEIR_REPOS = {
    "scifact": "BeIR/scifact",
    "nfcorpus": "BeIR/nfcorpus",
    "arguana": "BeIR/arguana",
    "scidocs": "BeIR/scidocs",
    "fiqa": "BeIR/fiqa",
    "trec-covid": "BeIR/trec-covid",
    "touche2020": "BeIR/webis-touche2020",
}


DATASET_REVISIONS = {
    "scifact": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
    "scifact-qrels": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
    "nfcorpus": "b5026a0e96e8a7ac4f95f482a596389289d46269",
    "nfcorpus-qrels": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
    "ragtruth": "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b",
}


@dataclass
class IRDataset:
    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    revision: str | None = None
    qrels_revision: str | None = None
    source: str = "unknown"

    def doc_text(self, doc_id: str) -> str:
        document = self.corpus[doc_id]
        title = document.get("title", "").strip()
        body = document.get("text", "").strip()
        return f"{title}. {body}".strip(". ").strip() if title else body


def _identifier(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise ValueError("dataset identifiers must be nonempty strings or integers")
    return str(value)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"dataset rows must be JSON objects: {path}")
            yield row


def _from_rows(
    name: str,
    corpus_rows: Iterable[dict[str, Any]],
    query_rows: Iterable[dict[str, Any]],
    qrel_rows: Iterable[dict[str, Any]],
) -> IRDataset:
    corpus: dict[str, dict[str, str]] = {}
    for row in corpus_rows:
        doc_id = _identifier(row["_id"])
        if doc_id in corpus:
            raise ValueError(f"duplicate corpus document: {doc_id}")
        document = {"title": row.get("title", ""), "text": row.get("text", "")}
        if not all(isinstance(value, str) for value in document.values()):
            raise ValueError(f"document title and text must be strings: {doc_id}")
        corpus[doc_id] = document

    queries: dict[str, str] = {}
    for row in query_rows:
        query_id = _identifier(row["_id"])
        if query_id in queries:
            raise ValueError(f"duplicate query: {query_id}")
        if not isinstance(row["text"], str):
            raise ValueError(f"query text must be a string: {query_id}")
        queries[query_id] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    judgments: dict[tuple[str, str], int] = {}
    for row in qrel_rows:
        query_id = _identifier(row["query-id"])
        doc_id = _identifier(row["corpus-id"])
        score = row.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not -float_info.max <= score <= float_info.max
            or score != int(score)
        ):
            raise ValueError("qrels scores must be finite integers")
        if query_id not in queries:
            raise ValueError(f"qrels reference missing query: {query_id}")
        if doc_id not in corpus:
            raise ValueError(f"qrels reference missing document: {doc_id}")
        pair = (query_id, doc_id)
        if pair in judgments and judgments[pair] != score:
            raise ValueError(f"conflicting qrels judgments: {query_id}, {doc_id}")
        judgments[pair] = int(score)
        if score > 0:
            qrels.setdefault(query_id, {})[doc_id] = int(score)
    return IRDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def load_vendored(name: str) -> IRDataset:
    if name not in BEIR_REPOS:
        raise ValueError(f"unsupported dataset: {name}")
    base = VENDORED_DIR / f"{name}_sample"
    dataset = _from_rows(
        name,
        _read_jsonl(base / "corpus.jsonl"),
        _read_jsonl(base / "queries.jsonl"),
        _read_jsonl(base / "qrels.jsonl"),
    )
    dataset.source = "vendored"
    return dataset


def load_beir(name: str, split: str = "test", cache_dir: str | None = None) -> IRDataset:
    if name not in BEIR_REPOS:
        raise ValueError(f"unsupported dataset: {name}")
    from datasets import load_dataset

    repo = BEIR_REPOS[name]
    rev = DATASET_REVISIONS.get(name)
    qrev = DATASET_REVISIONS.get(f"{name}-qrels")
    corpus_ds = load_dataset(repo, "corpus", cache_dir=cache_dir, revision=rev)["corpus"]
    queries_ds = load_dataset(repo, "queries", cache_dir=cache_dir, revision=rev)["queries"]
    qrels_ds = load_dataset(f"{repo}-qrels", cache_dir=cache_dir, revision=qrev)[split]

    dataset = _from_rows(name, corpus_ds, queries_ds, qrels_ds)
    dataset.queries = {query_id: dataset.queries[query_id] for query_id in dataset.qrels}
    dataset.revision = rev
    dataset.qrels_revision = qrev
    dataset.source = "huggingface"
    return dataset


def load(
    name: str = "scifact",
    split: str = "test",
    prefer_vendored: bool = False,
    cache_dir: str | None = None,
) -> IRDataset:
    if prefer_vendored:
        return load_vendored(name)
    return load_beir(name, split=split, cache_dir=cache_dir)
