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


@dataclass
class IRDataset:
    """Corpus, queries and relevance judgements."""

    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance grade}

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
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(
                row.get("score", 1)
            )
    return IRDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def load_beir(name: str, split: str = "test", cache_dir: str | None = None) -> IRDataset:
    """Download a BEIR dataset via HuggingFace datasets."""
    from datasets import load_dataset

    repo = BEIR_REPOS[name]
    corpus_ds = load_dataset(repo, "corpus", cache_dir=cache_dir)["corpus"]
    queries_ds = load_dataset(repo, "queries", cache_dir=cache_dir)["queries"]
    qrels_ds = load_dataset(f"{repo}-qrels", cache_dir=cache_dir)[split]

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
    return IRDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


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


def write_sample(ds: IRDataset, max_queries: int = 50, distractors: int = 600) -> Path:
    """Persist a small real subset for offline/CI use."""
    base = _vendored_path(ds.name)
    base.mkdir(parents=True, exist_ok=True)

    qids = list(ds.queries)[:max_queries]
    keep_docs: set[str] = set()
    for qid in qids:
        keep_docs |= set(ds.qrels.get(qid, {}))

    extra = [d for d in ds.corpus if d not in keep_docs][:distractors]
    keep_docs |= set(extra)

    with open(base / "corpus.jsonl", "w") as f:
        for doc_id in keep_docs:
            d = ds.corpus[doc_id]
            f.write(json.dumps({"_id": doc_id, "title": d["title"], "text": d["text"]}) + "\n")
    with open(base / "queries.jsonl", "w") as f:
        for qid in qids:
            f.write(json.dumps({"_id": qid, "text": ds.queries[qid]}) + "\n")
    with open(base / "qrels.jsonl", "w") as f:
        for qid in qids:
            for doc_id, score in ds.qrels.get(qid, {}).items():
                f.write(
                    json.dumps({"query-id": qid, "corpus-id": doc_id, "score": score}) + "\n"
                )
    return base
