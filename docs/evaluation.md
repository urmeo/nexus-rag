# Evaluation

The [README results](../README.md#results) measure retrieval and evidence-sentence detection, not generated-answer accuracy.

## Protocols

### Retrieval

| Dataset | Full BEIR test queries | Documents |
| --- | ---: | ---: |
| SciFact | 300 | 5,183 abstracts |
| NFCorpus | 323 | 3,633 |

Each benchmark document is one passage; the application chunks uploads. Benchmarks use exact cosine search over in-memory NumPy vectors on CPU; the app uses LanceDB. Retrieval depth is **50**, RRF **k=60**, seed **0**, and nDCG uses linear gains.

The comparison covers BM25, `BAAI/bge-small-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`, equal-weight RRF, adaptive weights and corrective pseudo-relevance feedback (PRF). PRF adds passage terms for one further search at threshold **0.55**. Reranking and SPLADE are off. [retrieval_metrics.csv](../outputs/results/retrieval_metrics.csv) retains every mean and interval, including the MiniLM hybrid, adaptive and corrective results.

Combining methods did not consistently improve retrieval. BGE-small corrective scores match the adaptive baseline to three decimals on both datasets. SciFact records no retries at 0.55; thresholds **0.65** and **0.70** trigger retries and reduce test nDCG.

95% intervals use **10,000 query bootstraps**. Paired tests use **10,000 randomizations**, with Holm correction within each dataset. [Result tables](../outputs/results) preserve individual scores, query IDs, model and dataset revisions, runtime versions and source hashes. They identify commit `c3a4875`; their source SHA-256 matches the **49 Python files** at [b515aac](https://github.com/urmeo/nexus-rag/commit/b515aac7dbe6a2fc686c25f16b3f03420f624335).

### Retrieval time

[Timing measurements](../outputs/results/corrective_cost_quality.csv) use the first **120 SciFact test queries** in dataset order, one CPU pass and no dedicated warmup. Indexing and embedding-model loading precede timing; initial cross-encoder loading is included. Answer generation is excluded. There are no repeated-run intervals; timings do not predict another machine's performance.

The [threshold sweep](../outputs/results/corrective_sweeps.csv) selects **0.40** on the first **250 SciFact training queries**, unlike the main comparison's 0.55. It triggers no test retries. Reranking uses adaptive-retriever candidates. Sweep lines connect measured settings; the quality axis is zoomed and the marked threshold was selected on training data.

### Evidence detection

The task uses **188 annotated SciFact development claims** and **2,031 candidate sentences**, **18.0% positive**. Candidates come only from abstracts named in evidence annotations; positives are rationales supporting or contradicting the claim. This task does not search the full corpus.

| Scorer | Score | F1 threshold |
| --- | --- | ---: |
| Lexical overlap | Jaccard similarity after tokenization and stop-word removal | 0.30 |
| NLI | Entailment + contradiction probabilities from `cross-encoder/nli-deberta-v3-small` | 0.80 |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.40 |

F1 thresholds use the first **120 annotated training claims** and a **0.30–0.95** grid in **0.05** steps. ROC-AUC intervals use **2,000 claim bootstraps**, keeping each claim's sentences together. [Candidate scores](../outputs/results/faithfulness_candidates.csv) and [method results](../outputs/results/faithfulness_methods.csv) retain the measurements; `pr_auc` stores average precision.

Plots group ties at distinct thresholds. ROC-AUC uses rank-based area; the precision–recall step area equals average precision, not trapezoidal PR area. Optional application grounding takes each answer sentence's highest entailment score across all supplied sources; it does not check that sentence's particular inline citation.

## Reproduce

Install the [development dependencies](../CONTRIBUTING.md#setup), then run from the repository root:

```bash
export PYTHONPATH="$PWD/src"
python -m scinexusrag.eval --sample
```

Sample mode checks the evaluation path and skips corrective-threshold tuning because its subset has no separate tuning split. It does not replace full benchmarks. Uncached embedding, NLI and reranking models need downloads.

For full benchmarks and CSV/chart export:

```bash
python -m scinexusrag.eval
python -m scinexusrag.eval.report --results outputs/generated --csv outputs/generated/csv --figures outputs/generated/csv
```

Full retrieval requires pinned BEIR revisions and models; unavailable inputs cause failure. Evidence detection reads SciFact's mutable `release/latest` archive or an existing local copy. Corpus, training-claim and development-claim SHA-256 hashes are recorded; no dataset revision is enforced.

Fresh JSON goes to `outputs/generated/`; paper reports default to `outputs/generated/paper/`. Review scores and provenance before publishing CSVs and charts in `outputs/results/`. Timing and floating-point results can differ across environments.

Render the saved CSV measurements without loading models:

```bash
python -m scinexusrag.eval.report --results outputs/results --figures outputs/results
```

This writes five figures from the saved measurements:

| Figure | Data |
| --- | --- |
| `retrieval-comparison.svg` | Full-test nDCG@10 means and 95% query-bootstrap intervals |
| `retrieval-query-changes.svg` | Paired query scores: adaptive hybrid minus BGE-small dense retrieval |
| `retrieval-cost.svg` | Quality and CPU time on the first 120 SciFact test queries |
| `corrective-sweep.svg` | SciFact test quality and retry counts at each measured threshold |
| `evidence-curves.svg` | Development-candidate ROC and precision–recall curves |

The query-change figure pairs scores by query ID; zero means the ranking score did not change. Timing uses a separate 120-query subset and must not be compared directly with full-test means. Identical inputs produce identical SVGs in the same plotting environment; fonts and layout can differ elsewhere. [Pipeline diagrams](pipeline.md#implementation-and-diagrams) are maintained separately.

### Regression gate

The gate uses sample data and pinned BGE-small, NLI and reranking models. Once cached, it runs offline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scinexusrag.eval.gate
```

## Answer experiments

`python -m scinexusrag.eval.generation` compares NLI grounding on a SciFact sample using local Transformers model `Qwen/Qwen2.5-0.5B-Instruct`, rather than the app's Ollama model. Defaults supply up to **four passages**, each truncated to **400 characters**. An NLI gate chooses whether to use the PRF-retry answer. Answers are not assessed by humans; the generator revision is unpinned.

`python -m scinexusrag.eval.ragtruth` scores existing `wandb/RAGTruth-processed` responses, not NexusRAG answers. Each response uses at most **12 output sentences** and **50 context sentences**. Its dataset revision is pinned and recorded. No generation or RAGTruth scores are claimed in the README.

See the [README references](../README.md#references) for the retrieval, fusion and dataset papers.
