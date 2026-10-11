# Pipeline

NexusRAG builds passage indexes from uploaded documents. A question retrieves passages, Ollama generates an answer, and checks run before the standard API response is returned. The diagrams below describe the application with its default settings.

![Document indexing followed by retrieval, generation, checks and a response with sources.](../assets/figures/pipeline.svg)

## Index documents

| Step | Work | Stored or returned data |
| --- | --- | --- |
| Parse | Read PDF, DOCX, Markdown or UTF-8 text | Text, sections and available page metadata |
| Split | Pack sections and paragraphs into overlapping passages | Passage IDs, document names, sections, pages and nearby context |
| Embed | Encode each passage with BGE-small | Passage vectors |
| Store and index | Save parsed document JSON and passage vectors; update BM25 | Persistent LanceDB passages and an in-memory keyword index |

The default passage target is **1,200 characters**, with a **1,500-character maximum** and up to **300 characters of paragraph overlap**. Where text can be located, context windows retain up to **150 nearby characters** on each side. Embeddings use the main passage text.

PDF parsing reads text; it has no OCR step. Encrypted PDFs are rejected. The document store keeps parsed JSON rather than the original uploaded file.

Duplicate documents and empty extracted content produce failed ingestion results. Writes update the document store, vector store and BM25 under a process lock, then check that index counts match. Partial failures attempt rollback. BM25 is rebuilt from stored passages when initialized, so run one application worker.

## Retrieve passages

![Separate dense and BM25 rankings, adaptive fusion, a conditional feedback search, fusion of both passes and selection of up to eight passages. Empty indexes and matches return without Ollama.](../assets/figures/retrieval-flow.svg)

The API trims and validates the question. If the index is empty, it returns an upload-needed message. Otherwise, pattern matching chooses an answer-format hint, and a fixed map rewrites six known vague questions.

Dense search embeds the question and searches LanceDB by cosine similarity. BM25 ranks stored passage text by keyword matches. Each branch returns up to **50 candidates**. Weighted reciprocal rank fusion combines the rankings with **k=60**; the highest fused score is normalized to 1.

Fusion weights follow these rules:

| Query shape | Dense weight | BM25 weight |
| --- | ---: | ---: |
| At most four words, or fewer than 12 words with a technical token | 0.3 | 0.7 |
| At least 12 words with no technical token | 0.7 | 0.3 |
| Other queries | 0.5 | 0.5 |

A technical token contains a digit or underscore, is all uppercase, or exceeds 14 characters after trimming punctuation. These are fixed rules, not a learned routing model.

One feedback pass runs when correction is enabled, first-pass results exist, and the **best dense cosine score is below 0.55**. It takes up to **ten frequent non-query terms** from the **first five passages**, appends them to the question and repeats both searches. Equal-weight RRF then merges the first and feedback rankings. The result is capped at **eight passages** by default.

The gate uses dense similarity, which is separate from the final answer confidence. Empty retrieval results return a no-results message without calling Ollama.

## Generate and check the answer

![Select up to five sources, generate once with Ollama, remove invalid citation indices, then choose optional NLI grounding or the default heuristic before returning the response.](../assets/figures/answer-checks.svg)

The synthesizer removes duplicate passage IDs and keeps at most **five sources**. Sources retain document names, available sections and pages, and nearby context. It numbers them for `[n]` citations, clips formatted source context at **6,000 characters** when needed, and adds a truncation marker. The question, instructions and answer-format hint sit outside that source-context limit.

Ollama uses `llama3.2:3b` by default. The response budget is `min(LLM_MAX_TOKENS, 256 + 100 × source_count)`; `LLM_MAX_TOKENS` defaults to **768**. A generation failure returns an error message with the retrieved sources.

Citation validation removes `[n]` references outside the supplied source range. It adds warnings for invalid or missing citations. It does not check whether a source supports the claim.

NLI grounding is **off by default**. Enable it with `SELF_CORRECTION_GROUNDING_ENABLED=true`. When enabled, it compares every answer sentence with every supplied source, using available full context. A sentence counts as grounded when its **best entailment score is at least 0.5**. NLI reports a score and warnings; it does not rewrite the answer or trigger another generation call. It also does not verify a sentence's particular inline citation.

| Grounding setting | Final confidence |
| --- | --- |
| Disabled | Half the mean normalized source score plus half the fraction of sources cited by valid indices |
| Enabled | Fraction of answer sentences that meet the grounding threshold |

Read the returned passages when checking an answer. Neither confidence calculation establishes factual correctness.

## Response and implementation

The standard interface uses `POST /api/query`. Its response contains the answer, up to five source excerpts, confidence, processing time and warnings. Each excerpt shows the first **500 characters**, followed by `...` when truncated. Internal correction status and the reasoning trace are not included in this API response.

Python streaming helpers emit text before the later verification event. The diagrams describe the standard API response, which completes the checks before returning.

| Code | Responsibility |
| --- | --- |
| [pipeline.py](../src/scinexusrag/pipeline.py) | Initialize components, ingest documents, keep indexes synchronized and handle an empty index |
| [hybrid.py](../src/scinexusrag/retrieval/hybrid.py) | Dense/BM25 rank fusion and query-shape weights |
| [corrective.py](../src/scinexusrag/retrieval/corrective.py) | Dense-score gate, feedback terms, one search retry and fusion of both passes |
| [synthesizer.py](../src/scinexusrag/generation/synthesizer.py) | Numbered sources, prompt limits and Ollama generation |
| [orchestrator.py](../src/scinexusrag/generation/orchestrator.py) | Citation checks, optional grounding and final confidence |
| [config.py](../src/scinexusrag/config.py) | Application defaults and environment settings |

Cross-encoder reranking and SPLADE are available in library or evaluation tools; they are not stages in this application path. The separate evidence benchmark counts both supporting and contradicting rationale sentences. [Evaluation conditions](evaluation.md).
