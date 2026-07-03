"""BEIR-format scientific IR datasets (SciFact, NFCorpus)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VENDORED_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "datasets"

# BEIR HuggingFace repos
BEIR_REPOS = {
    "scifact": "BeIR/scifact",
    "nfcorpus": "BeIR/nfcorpus",
    "arguana": "BeIR/arguana",
    "scidocs": "BeIR/scidocs",
    "fiqa": "BeIR/fiqa",
    "trec-covid": "BeIR/trec-covid",
    "touche2020": "BeIR/webis-touche2020",
}

# Pinned dataset git revisions, so reported numbers reference a fixed snapshot
# (BEIR mirrors can change). Keys are "<name>" (corpus+queries) and "<name>-qrels".
DATASET_REVISIONS = {
    "scifact": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
    "scifact-qrels": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
    "nfcorpus": "b5026a0e96e8a7ac4f95f482a596389289d46269",
    "nfcorpus-qrels": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
}


@dataclass
class IRDataset:
    """Corpus, queries and relevance judgements."""

    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance grade}
    revision: str | None = None

    def doc_text(self, doc_id: str) -> str:
        d = self.corpus[doc_id]
        title = d.get("title", "").strip()
        body = d.get("text", "").strip()
        return f"{title}. {body}".strip(". ").strip() if title else body


def _vendored_path(name: str) -> Path:
    return VENDORED_DIR / f"{name}_sample"


def load_vendored(name: str) -> IRDataset:
    """Load the committed offline sample."""
    base = _vendored_path(name)
    corpus: dict[str, dict[str, str]] = {}
    with open(base / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            corpus[str(row["_id"])] = {"title": row.get("title", ""), "text": row.get("text", "")}
    queries: dict[str, str] = {}
    with open(base / "queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(base / "qrels.jsonl") as f:
        for line in f:
            row = json.loads(line)
            score = int(row.get("score", 1))
            if score > 0:  # match load_beir; non-relevant judgements are not relevant
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score
    return IRDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def load_beir(name: str, split: str = "test", cache_dir: str | None = None) -> IRDataset:
    """Download a BEIR dataset via HuggingFace datasets."""
    from datasets import load_dataset

    repo = BEIR_REPOS[name]
    rev = DATASET_REVISIONS.get(name)
    qrev = DATASET_REVISIONS.get(f"{name}-qrels")
    corpus_ds = load_dataset(repo, "corpus", cache_dir=cache_dir, revision=rev)["corpus"]
    queries_ds = load_dataset(repo, "queries", cache_dir=cache_dir, revision=rev)["queries"]
    qrels_ds = load_dataset(f"{repo}-qrels", cache_dir=cache_dir, revision=qrev)[split]

    corpus = {
        str(r["_id"]): {"title": r.get("title", ""), "text": r.get("text", "")} for r in corpus_ds
    }
    all_queries = {str(r["_id"]): r["text"] for r in queries_ds}

    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        score = int(r["score"])
        if score > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = score

    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    return IRDataset(name=name, corpus=corpus, queries=queries, qrels=qrels, revision=rev)


def load(
    name: str = "scifact",
    split: str = "test",
    prefer_vendored: bool = False,
    cache_dir: str | None = None,
) -> IRDataset:
    """Load a dataset, falling back to the offline sample."""
    if prefer_vendored and _vendored_path(name).exists():
        return load_vendored(name)
    try:
        return load_beir(name, split=split, cache_dir=cache_dir)
    except Exception:
        if _vendored_path(name).exists():
            return load_vendored(name)
        raise
