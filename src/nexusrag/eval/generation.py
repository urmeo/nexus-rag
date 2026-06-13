"""Does the faithfulness-gated corrective loop reduce ungrounded answers?

Generates answers over retrieved scientific passages with a local instruct
model, scores each answer's grounding with the NLI verifier, and re-retrieves +
regenerates when grounding is weak. Reports baseline vs corrective faithfulness.

Runs in three phases (retrieve, generate, score) loading one model at a time,
so it fits on a CPU laptop.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from nexusrag.agents.grounding import GroundingVerifier
from nexusrag.eval import datasets as D
from nexusrag.eval.indexes import ExactDenseRetriever, corpus_to_chunks
from nexusrag.ingestion import Embedder
from nexusrag.retrieval import AdaptiveHybridRetriever, BM25Retriever
from nexusrag.retrieval.stopwords import STOP_WORDS

RESULTS_DIR = Path("benchmarks/results")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = """You are a precise scientific assistant. Using ONLY the numbered \
sources, answer the question in two or three sentences and cite with [1], [2].

Question: {question}

Sources:
{sources}

Answer:"""


class LocalGenerator:
    """Small CPU instruct model (no API keys, no Ollama)."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from transformers import pipeline

        self.model_name = model_name
        self.pipe = pipeline("text-generation", model=model_name, dtype="auto")

    def generate(self, prompt: str, max_new_tokens: int = 130) -> str:
        out = self.pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return str(out[0]["generated_text"][-1]["content"]).strip()

    def close(self) -> None:
        del self.pipe
        gc.collect()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_sources(passages: list[str]) -> str:
    return "\n".join(f"[{i}] {p[:400]}" for i, p in enumerate(passages, 1))


def _reformulate(query: str) -> str:
    """Keyword-only rewrite for the corrective re-retrieval."""
    words = [w for w in query.lower().split() if len(w) > 2 and w not in STOP_WORDS]
    return " ".join(words) if words else query


def evaluate(
    n: int = 15, top_k: int = 4, tau: float = 0.5, model_name: str | None = None
) -> dict[str, Any]:
    ds = D.load("scifact", prefer_vendored=True)
    qids = list(ds.queries)[:n]

    # phase 1: retrieve (embedder only)
    chunks = corpus_to_chunks({d: ds.doc_text(d) for d in ds.corpus})
    embedder = Embedder()
    dense = ExactDenseRetriever(embedder, chunks)
    bm25 = BM25Retriever()
    bm25.add(chunks)
    hybrid = AdaptiveHybridRetriever(dense, bm25, 0.7, 0.3, use_mmr=True, use_keyword_boost=True)

    initial: dict[str, list[str]] = {}
    reformed: dict[str, list[str]] = {}
    for qid in qids:
        q = ds.queries[qid]
        initial[qid] = [r.chunk.content for r in hybrid.retrieve(q, top_k=top_k)]
        reformed[qid] = [r.chunk.content for r in hybrid.retrieve(_reformulate(q), top_k=top_k)]
    del embedder, dense, bm25, hybrid
    gc.collect()

    # phase 2: generate (generator only)
    generator = LocalGenerator(model_name or DEFAULT_MODEL)
    model_used = generator.model_name
    base_answer: dict[str, str] = {}
    corr_answer: dict[str, str] = {}
    for qid in qids:
        q = ds.queries[qid]
        base_answer[qid] = generator.generate(
            PROMPT.format(question=q, sources=_format_sources(initial[qid]))
        )
        corr_answer[qid] = generator.generate(
            PROMPT.format(question=q, sources=_format_sources(reformed[qid]))
        )
    generator.close()
    del generator
    gc.collect()

    # phase 3: score grounding (NLI verifier only); apply faithfulness gate
    verifier = GroundingVerifier()
    baseline, corrective = [], []
    corrected = 0
    for qid in qids:
        f0 = verifier.verify(base_answer[qid], initial[qid]).faithfulness
        best = f0
        if f0 < tau:
            f1 = verifier.verify(corr_answer[qid], reformed[qid]).faithfulness
            if f1 > f0:
                corrected += 1
            best = max(f0, f1)
        baseline.append(f0)
        corrective.append(best)

    return {
        "dataset": "scifact-claims",
        "split": "sample",
        "generator": model_used,
        "metric": "NLI sentence-grounding faithfulness",
        "num_queries": len(qids),
        "top_k": top_k,
        "gate_tau": tau,
        "num_corrected": corrected,
        "baseline_faithfulness": _mean(baseline),
        "corrective_faithfulness": _mean(corrective),
        "improvement": _mean(corrective) - _mean(baseline),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Corrective-loop generation faithfulness")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = evaluate(n=args.n, top_k=args.top_k, tau=args.tau, model_name=args.model)
    print(f"generator={res['generator']}  n={res['num_queries']}  corrected={res['num_corrected']}")
    print(f"baseline faithfulness   = {res['baseline_faithfulness']:.3f}")
    print(f"corrective faithfulness = {res['corrective_faithfulness']:.3f}")
    print(f"improvement             = {res['improvement']:+.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / "generation_sample.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
