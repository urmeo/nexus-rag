# Pipeline

NexusRAG indexes documents, retrieves passages and sends them to Ollama for an answer with numbered sources. The app's **Pipeline** page at `/pipeline` explains the defaults; it is not a live progress view.

![Document indexing, passage retrieval, answer generation and source checks.](../assets/figures/pipeline.svg)

## Documents

| Stage | Behavior |
| --- | --- |
| Read | Extract text from PDF, DOCX, Markdown or UTF-8 text. Save parsed JSON, not the original upload. |
| Split | Pack sections and paragraphs into passages: target 1,200 characters, maximum 1,500, paragraph overlap up to 300. Keep character offsets in normalized text and attach up to 150 characters of nearby context on each side where offsets are available. |
| Index | Encode passage text with BGE-small, store vectors in LanceDB and update in-memory BM25. |

PDFs need embedded text; encrypted files are rejected and there is no OCR. DOCX extraction reads body paragraphs and tables, including nested table text, in document order. Cells become pipe-separated text; table layout, headers, footers and image text are not preserved.

Duplicate documents and empty extracted text fail ingestion. Each index records the embedding model, revision, prefixes, normalization and dimension; incompatible settings block access. See [configuration](../CONTRIBUTING.md#configuration) before changing models.

Writes share a process lock, check index counts and attempt rollback on failure. BM25 rebuilds from stored passages on first use; run **one worker**. If rollback fails, the document stays listed. After storage recovers, delete that record before uploading the same file again.

Deletion paginates all matching chunk IDs, removes vectors and BM25 entries, then removes document metadata. A failed step attempts to rebuild BM25 from remaining vectors; retry deletion to finish cleanup.

## Questions

![Dense and keyword search, weighted fusion and one optional feedback search.](../assets/figures/retrieval-flow.svg)

| Stage | Behavior |
| --- | --- |
| Prepare | Trim and validate the question. Fixed rules choose an answer-format hint and rewrite six known vague questions. |
| Search | Dense cosine search in LanceDB and BM25 keyword search each return up to 50 candidates. Exact vector search scales linearly with passage count. |
| Fuse | Weighted reciprocal rank fusion uses `k=60`; the top fused score is normalized to 1. Query-shape rules choose the weights below. |
| Retry | Enabled by default. If results exist and the best dense similarity is below 0.55, add up to ten frequent non-query terms from the first five passages. When new terms exist, search once more and merge both rankings with equal-weight RRF. |
| Select | Keep up to eight passages for source construction. An empty index or no matches returns a message without calling Ollama. |

| Query shape | Dense weight | BM25 weight |
| --- | ---: | ---: |
| At most four words, or fewer than 12 with a technical token | 0.3 | 0.7 |
| At least 12 words without a technical token | 0.7 | 0.3 |
| Other queries | 0.5 | 0.5 |

A technical token contains a digit or underscore, is uppercase, or exceeds 14 characters after punctuation trimming. Dense similarity controls the retry gate; it differs from fused scores and answer confidence.

![Numbered passages, Ollama generation, citation validation and optional NLI grounding.](../assets/figures/answer-checks.svg)

Remove duplicate passage IDs and retain up to **five complete sources** within **6,000 characters**, including headers and separators. Stop at the first block that does not fit; omit it and all lower-ranked passages. Sources may come from the same document. Question and instruction text sit outside this limit. If no source fits, return a context-limit error without calling Ollama or checking the answer.

Ollama defaults to `llama3.2:3b`. Its response budget is `min(LLM_MAX_TOKENS, 256 + 100 × source_count)`; the configured ceiling defaults to **768 tokens**.

Citation validation removes numeric `[n]` references outside the supplied source indices and warns about invalid or missing citations. It validates numbers, not factual support.

| Grounding | Confidence |
| --- | --- |
| Off by default | Half the mean source score clipped to [0, 1], plus half the fraction of distinct supplied sources cited by valid indices. |
| `SELF_CORRECTION_GROUNDING_ENABLED=true` | Fraction of answer sentences whose best entailment score across all supplied sources is at least 0.5. Unsupported sentences add warnings. |

NLI receives the same complete sources as generation, including nearby context. It checks each sentence against all sources, not its particular inline citation. Checks do not rewrite or regenerate answers, and neither confidence score guarantees correctness. Inspect the source passages.

A caught generation failure, including blank output, returns an error answer, retained sources, zero confidence and a warning; checks are skipped. Other retrieval, storage or NLI failures can return an HTTP error. Vector-index read failures return **503**, rather than an empty-corpus message; Python raises `VectorStoreError` with the original cause.

The interface calls `POST /api/query` and displays the answer, up to five source excerpts, confidence, processing time and warnings. Excerpts contain the first 500 characters plus `...` if truncated. Internal correction state and reasoning traces are omitted.

Python's `Orchestrator.query_streaming()` and `NexusRAG.query_streaming()` emit provisional text followed by:

| Event | Content |
| --- | --- |
| `verification` | Valid citation indices, faithfulness and warnings |
| `verified_answer` | Complete answer with invalid citations removed |

Consumers must replace provisional output with the final `answer` from the `data: {JSON}` event. Caught generation or verification failures replace it with an error message. Empty retrieval and context-limit failures also finish with an error answer and skip checks. The web interface uses the non-streaming route.

Cross-encoder reranking and SPLADE are library or evaluation options, not default app stages. [Retrieval and evidence benchmarks](evaluation.md) do not establish generated-answer accuracy.

## Implementation and diagrams

| Area | Tools |
| --- | --- |
| Documents and retrieval | Python, PyTorch, NumPy, pypdf, python-docx, Sentence Transformers (BGE-small), rank-bm25, LanceDB |
| Answers and interface | Ollama, citation validation, optional DeBERTa NLI, FastAPI, Uvicorn, HTML, CSS, JavaScript |
| Evaluation and checks | Hugging Face Datasets, Matplotlib, pytest, mypy (strict), Ruff, Node.js UI tests |

Approximate model weight sizes on disk:

| Model | Size |
| --- | --- |
| [BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/model.safetensors) | 133 MB, FP32 |
| [MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/blob/c5ee24cb16019beea0893ab7796b1df96625c6b8/model.safetensors) (library and evaluation) | 91 MB, FP32 |
| [DeBERTa NLI](https://huggingface.co/cross-encoder/nli-deberta-v3-small/blob/fa2804872c3b4bd748f38c0185cc85775361e735/model.safetensors) (optional) | 568 MB, FP32 |
| [Ollama `llama3.2:3b`](https://ollama.com/library/llama3.2:3b) | 2.0 GB, Q4_K_M |

Tokenizers, dependencies, documents and indexes need additional space. Peak RAM and full evaluation runtime have not been measured. Ollama model tags can change.

| Location | Contents |
| --- | --- |
| `src/scinexusrag/` | API, document workflow, ingestion, retrieval, generation, storage, utilities and web interface |
| `src/scinexusrag/eval/` | Datasets, metrics, reports, sample data and regression thresholds |
| `tests/` | Python unit/integration and JavaScript interface tests |
| `assets/` | Pipeline diagrams, Figma snapshot and upload recording |
| `outputs/results/` | Published CSV measurements, provenance and charts |
| `outputs/generated/` | Fresh JSON runs and reports to review before publication |
| `data/` | Default application storage; [settings](../CONTRIBUTING.md#configuration) control document and index paths |

Run commands from the repository root. Keep scratch work outside the repository.

| Diagram | Figma | SVG |
| --- | --- | --- |
| Pipeline | [Source](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=29-2) | [pipeline.svg](../assets/figures/pipeline.svg) |
| Retrieval | [Source](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=12-2) | [retrieval-flow.svg](../assets/figures/retrieval-flow.svg) |
| Answer checks | [Source](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=13-20) | [answer-checks.svg](../assets/figures/answer-checks.svg) |

Keep the three frames on Figma's **NexusRAG** page and their SVGs in sync. The [app SVG](../src/scinexusrag/web/images/pipeline.svg) must match the README diagram. Update the [editable layout snapshot](../assets/figma/pipeline.json) when the pipeline frame changes. SVGs include accessible titles, descriptions and font fallbacks.

The [21.1-second upload recording](../assets/media/interface-walkthrough.mp4) used Chrome on 12 October 2026 at [609c0c8](https://github.com/urmeo/nexus-rag/commit/609c0c8472baffc65982af3c4f2021b7fa5c2f75). The real local app and API used isolated data, without mocked responses or altered interface content. A 301-word fictional document produced six BGE-small passages. It shows indexing, desktop/mobile views and a question typed without submission. Ollama was unavailable; no answer is shown. The H.264 video has no audio, uses a 1440 × 1000 canvas and preserves capture timing and viewport proportions.

Planned work:

- Compare FiQA and SCIDOCS retrieval, SPECTER2 and SciNCL embeddings, and ColBERTv2 and monoT5 alongside SPLADE.
- Measure how full-paper chunk size and overlap affect retrieval; assess answers with human annotations, RAGAs and LLM judges.
- Persist BM25 and evaluate approximate vector search for larger collections.
