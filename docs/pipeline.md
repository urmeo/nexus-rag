# Pipeline

NexusRAG turns uploaded documents into searchable passages. For each question, it retrieves passages, asks Ollama for an answer and returns source excerpts, confidence and warnings.

![Index documents on upload. For each question, retrieve passages, generate an answer and check it before returning sources.](../assets/figures/pipeline.svg)

Open **Pipeline** in the app header, or visit `/pipeline`, for the responsive guide. It explains the default behavior; it does not show live progress.

## On upload

| Step | What happens |
| --- | --- |
| Read documents | Extract text and available sections or pages from PDF, DOCX, Markdown or UTF-8 text. |
| Split into passages | Pack sections and paragraphs into overlapping passages. Keep document names and nearby context. |
| Index passages | Encode passage text with BGE-small, store vectors in LanceDB and update the in-memory BM25 keyword index. Save parsed document JSON. |

| Passage setting | Default |
| --- | ---: |
| Target length | 1,200 characters |
| Maximum length | 1,500 characters |
| Paragraph overlap | Up to 300 characters |
| Nearby context, where text can be located | Up to 150 characters on each side |

The document store saves parsed JSON, not the original upload. Duplicate documents and empty extracted text fail ingestion. Writes run under a process lock, check index counts and attempt rollback on partial failure. BM25 rebuilds from stored passages at initialization; run one application worker.

## For each question

![Meaning and keyword rankings feed fusion. A low dense-similarity score can trigger one expanded search; both rankings merge before selecting passages.](../assets/figures/retrieval-flow.svg)

1. **Prepare the question.** The API trims and validates it. Fixed rules choose an answer-format hint and rewrite six known vague questions.
2. **Search two ways.** Dense search embeds the question and searches LanceDB by cosine similarity. BM25 ranks passage text by keyword matches. Each returns up to 50 candidates.
3. **Combine the rankings.** Weighted reciprocal rank fusion uses `k=60`. Fixed query rules choose the weights below. The top fused score is normalized to 1.
4. **Search once more if needed.** When feedback is enabled (the default), results exist and the best dense similarity is below **0.55**, add up to ten frequent non-query terms from the first five passages. Repeat retrieval once, then merge the original and feedback rankings with equal-weight RRF. The merged results bypass the gate.
5. **Select passages.** Keep up to eight passages for source construction. An empty index or no matches returns a message without calling Ollama.

| Query shape | Dense weight | BM25 weight |
| --- | ---: | ---: |
| At most four words, or fewer than 12 words with a technical token | 0.3 | 0.7 |
| At least 12 words without a technical token | 0.7 | 0.3 |
| Other queries | 0.5 | 0.5 |

A technical token contains a digit or underscore, is uppercase, or exceeds 14 characters after punctuation trimming. These are fixed rules. Dense similarity controls the feedback gate; it is separate from fused ranking scores and answer confidence. Exact vector search scales linearly with stored passages.

## Generate and check

![Number source passages, generate with Ollama, check citation numbers and calculate confidence. Optional NLI scores sentence support; generation failures skip checks.](../assets/figures/answer-checks.svg)

| Step | What happens |
| --- | --- |
| Build sources | Remove duplicate passage IDs and keep up to five numbered passages. These may come from the same document. |
| Generate | Send the question, source text and answer-format hint to Ollama. The default model is `llama3.2:3b`. |
| Check citations | Keep numeric `[n]` citations within the supplied source range, remove out-of-range indices and warn about invalid or absent citations. |
| Score and return | Use the default heuristic or optional sentence grounding to calculate confidence, then return the answer and sources. Checks do not trigger answer regeneration. |

Formatted source text is clipped at **6,000 characters** when needed, followed by a truncation marker. The question and instructions sit outside that limit. The response budget is `min(LLM_MAX_TOKENS, 256 + 100 × source_count)`; the configured ceiling defaults to **768 tokens**.

Citation checks validate source numbers, not factual support. Optional NLI grounding is **off by default**; enable it with `SELF_CORRECTION_GROUNDING_ENABLED=true`.

| Grounding | Confidence calculation |
| --- | --- |
| Off | Half the mean normalized source score plus half the fraction of distinct sources cited by valid indices. |
| On | Fraction of answer sentences whose best entailment score across all supplied sources is at least 0.5. Unsupported sentences add warnings. |

NLI uses available full source context, which can exceed the clipped prompt. It checks each sentence against all sources, not its particular inline citation. It does not rewrite the answer. Neither confidence calculation establishes factual correctness; inspect the returned passages.

A caught generation failure or blank output retains sources and returns an error answer, zero confidence and a warning. Citation and grounding checks are skipped. Other retrieval, storage or NLI errors can produce an HTTP error.

## Response and limits

The interface uses `POST /api/query`. It returns the answer, up to five source excerpts, confidence, processing time and warnings. Excerpts contain the first **500 characters**, plus `...` when truncated. Internal correction status and reasoning traces are not exposed by this route. Python streaming helpers emit text before their later verification event.

- PDF extraction needs embedded text. There is no OCR, and encrypted PDFs are rejected.
- DOCX extraction reads paragraphs, not tables, headers or footers.
- Cross-encoder reranking and SPLADE are available in library or evaluation tools; they are not stages in this default application path.
- Retrieval and evidence benchmarks do not establish generated-answer accuracy. See the [evaluation conditions](evaluation.md) and [setup and operating limits](../CONTRIBUTING.md).

## Implementation and diagrams

| Code | Responsibility |
| --- | --- |
| [pipeline.py](../src/scinexusrag/pipeline.py) | Ingestion, synchronized indexes and empty-index handling |
| [hybrid.py](../src/scinexusrag/retrieval/hybrid.py) | Rank fusion and query-shape weights |
| [corrective.py](../src/scinexusrag/retrieval/corrective.py) | Feedback gate, expanded search and merge |
| [synthesizer.py](../src/scinexusrag/generation/synthesizer.py) | Numbered sources, prompt limits and generation |
| [orchestrator.py](../src/scinexusrag/generation/orchestrator.py) | Citation checks, optional grounding and confidence |
| [config.py](../src/scinexusrag/config.py) | Defaults and environment settings |

| Diagram | Figma source | Editable SVG |
| --- | --- | --- |
| Overview | [Pipeline](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=8-2) | [pipeline.svg](../assets/figures/pipeline.svg) |
| Retrieval | [Retrieval](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=12-2) | [retrieval-flow.svg](../assets/figures/retrieval-flow.svg) |
| Answer checks | [Answer checks](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=13-20) | [answer-checks.svg](../assets/figures/answer-checks.svg) |

The linked Figma frames match the SVG diagrams. The responsive app guide describes the same flow. Keep the Figma frames and SVG files in sync when editing; the SVGs include accessible titles, descriptions and font fallbacks.
