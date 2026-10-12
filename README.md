![Upload documents, parse overlapping passages and build dense and BM25 indexes. Each question retrieves passages, optionally searches once more, then generates an answer with checked citation indices and source excerpts.](assets/figures/pipeline.svg)

![Dense and BM25 rankings are fused; a conditional feedback search is merged with the first pass before selecting passages.](assets/figures/retrieval-flow.svg)

![Build numbered sources, generate an answer, check citation indices and compute confidence. Generation failures retain sources with zero confidence and skip checks. Optional NLI is off by default.](assets/figures/answer-checks.svg)

![Recorded NexusRAG interface with a fictional document indexed into six passages.](assets/figures/interface-desktop.png)

| System | SciFact nDCG@10 | NFCorpus nDCG@10 |
| --- | ---: | ---: |
| Dense · MiniLM | 0.648 | 0.318 |
| BM25 | 0.666 | 0.311 |
| Dense · BGE-small | 0.708 | 0.341 |
| Hybrid · RRF | 0.704 | 0.352 |
| Adaptive weights | 0.703 | 0.346 |
| Corrective PRF | 0.703 | 0.346 |

![Recorded retrieval scores on SciFact and NFCorpus, with 95% query-bootstrap intervals.](assets/figures/retrieval-comparison.svg)

| System | nDCG@10 | Milliseconds per query |
| --- | ---: | ---: |
| Adaptive hybrid | 0.734 | 16.9 |
| Corrective PRF | 0.734 | 17.3 |
| Cross-encoder reranking | 0.702 | 1318.9 |

![Recorded SciFact threshold sweep: retrieval quality and the percentage of queries that triggered a feedback search.](assets/figures/corrective-sweep.svg)

| Scorer | ROC-AUC | Average precision | F1 |
| --- | ---: | ---: | ---: |
| Lexical overlap | 0.686 | 0.365 | 0.112 |
| NLI | 0.688 | 0.331 | 0.368 |
| Cross-encoder | 0.755 | 0.476 | 0.469 |

![ROC and precision–recall curves computed from the recorded evidence-sentence scores, with random baselines.](assets/figures/evidence-curves.svg)
