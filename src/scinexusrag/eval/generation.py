"""Generation evaluation."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from scinexusrag.eval import datasets
from scinexusrag.eval.indexes import ExactDenseRetriever, corpus_to_chunks
from scinexusrag.eval.provenance import evaluation_provenance
from scinexusrag.eval.result_files import write_result
from scinexusrag.generation.grounding import GroundingVerifier
from scinexusrag.ingestion.embedder import Embedder
from scinexusrag.retrieval.corrective import CorrectiveRetriever
from scinexusrag.retrieval.hybrid import AdaptiveHybridRetriever, rrf_fuse
from scinexusrag.retrieval.sparse import BM25Retriever

RESULTS_DIR = Path("outputs/generated")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SOURCE_CHAR_LIMIT = 400

PROMPT = """You are a precise scientific assistant. Using ONLY the numbered \
sources, answer the question in two or three sentences and cite with [1], [2].

Question: {question}

Sources:
{sources}

Answer:"""


class LocalGenerator:
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


def _format_sources(passages: list[str]) -> str:
    return "\n".join(
        f"[{index}] {text[:SOURCE_CHAR_LIMIT]}" for index, text in enumerate(passages, 1)
    )


def evaluate(
    n: int = 15, top_k: int = 4, tau: float = 0.5, model_name: str | None = None
) -> dict[str, Any]:
    if n <= 0 or top_k <= 0 or not 0 <= tau <= 1:
        raise ValueError("n and top_k must be positive and tau must be in [0, 1]")
    provenance = evaluation_provenance()
    ds = datasets.load("scifact", prefer_vendored=True)
    qids = list(ds.queries)[:n]
    if not ds.corpus or not qids:
        raise ValueError("generation evaluation requires nonempty corpus and queries")

    chunks = corpus_to_chunks({d: ds.doc_text(d) for d in ds.corpus})
    embedder = Embedder(device="cpu")
    dense = ExactDenseRetriever(embedder, chunks)
    bm25 = BM25Retriever()
    bm25.add(chunks)
    hybrid = AdaptiveHybridRetriever(dense, bm25, 0.5, 0.5)
    corrective = CorrectiveRetriever(hybrid, tau=tau)

    initial: dict[str, list[str]] = {}
    reformed: dict[str, list[str]] = {}
    depth = max(top_k, 50)
    for qid in qids:
        q = ds.queries[qid]
        first = hybrid.retrieve(q, top_k=depth, depth=depth)
        initial[qid] = [result.chunk.content[:SOURCE_CHAR_LIMIT] for result in first[:top_k]]

        expanded = corrective.expand(q, first)
        second = hybrid.retrieve(expanded, top_k=depth, depth=depth)
        fused = rrf_fuse([first, second], [1.0, 1.0], hybrid.rrf_k, top_k)
        reformed[qid] = [result.chunk.content[:SOURCE_CHAR_LIMIT] for result in fused]
    del embedder, dense, bm25, hybrid, corrective
    gc.collect()

    generator = LocalGenerator(model_name or DEFAULT_MODEL)
    model_used = generator.model_name
    base_answer: dict[str, str] = {}
    corr_answer: dict[str, str] = {}
    try:
        for qid in qids:
            q = ds.queries[qid]
            base_answer[qid] = generator.generate(
                PROMPT.format(question=q, sources=_format_sources(initial[qid]))
            )
            corr_answer[qid] = generator.generate(
                PROMPT.format(question=q, sources=_format_sources(reformed[qid]))
            )
    finally:
        generator.close()
    del generator
    gc.collect()

    verifier = GroundingVerifier(device="cpu")
    base_scores: list[float] = []
    corr_scores: list[float] = []
    corrected = 0
    for qid in qids:
        f0 = verifier.verify(base_answer[qid], initial[qid]).faithfulness
        gated = f0
        if f0 < tau:
            gated = verifier.verify(corr_answer[qid], reformed[qid]).faithfulness
            corrected += 1
        base_scores.append(f0)
        corr_scores.append(gated)

    baseline_faithfulness = sum(base_scores) / len(base_scores)
    corrective_faithfulness = sum(corr_scores) / len(corr_scores)
    return {
        "provenance": provenance,
        "dataset": "scifact",
        "split": "sample",
        "generator": model_used,
        "metric": "NLI sentence-grounding faithfulness",
        "num_queries": len(qids),
        "query_ids": qids,
        "top_k": top_k,
        "gate_tau": tau,
        "correction_strategy": "forced_prf_on_answer_gate",
        "nli_model": verifier.model_name,
        "source_char_limit": SOURCE_CHAR_LIMIT,
        "num_corrected": corrected,
        "baseline_faithfulness": baseline_faithfulness,
        "corrective_faithfulness": corrective_faithfulness,
        "improvement": corrective_faithfulness - baseline_faithfulness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrective-loop generation faithfulness")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate(n=args.n, top_k=args.top_k, tau=args.tau, model_name=args.model)
    print(
        f"generator={result['generator']}  n={result['num_queries']}  corrected={result['num_corrected']}"
    )
    print(f"baseline faithfulness   = {result['baseline_faithfulness']:.3f}")
    print(f"corrective faithfulness = {result['corrective_faithfulness']:.3f}")
    print(f"improvement             = {result['improvement']:+.3f}")

    out = Path(args.out) if args.out else RESULTS_DIR / "generation_sample.json"
    write_result(result, out, trailing_newline=True)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
