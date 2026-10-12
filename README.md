# NexusRAG

Search research papers with dense embeddings and BM25. Ask a local Ollama model for answers with cited passages, then inspect the sources. Built with Python, PyTorch, sentence-transformers, NumPy, LanceDB and FastAPI.

## Pipeline

![Index research papers, retrieve passages with dense and BM25 search, then generate an answer with sources. Recorded retrieval results appear below.](assets/figures/pipeline.svg)

Uploads build the indexes; questions reuse them. The results strip covers retrieval benchmarks. [Editable Figma source](https://www.figma.com/design/bRUVlmkQKApYB2bWdHiblv?node-id=2-2) · [Evaluation conditions](docs/evaluation.md).

<details>
<summary>Retrieval and answer checks</summary>

![Dense and BM25 rankings are fused; a conditional feedback search is merged with the first pass before selecting passages.](assets/figures/retrieval-flow.svg)

![Build numbered sources, generate once, check citation indices and optionally score sentence grounding before returning the response.](assets/figures/answer-checks.svg)

[Pipeline rules, defaults and source code](docs/pipeline.md).

</details>

## Interface

![Recorded NexusRAG interface with the sample document indexed into six passages.](assets/figures/interface-desktop.png)

**Upload walkthrough · 21 seconds.** Upload the [fictional sample paper](examples/sample_paper.md), index six passages and open the mobile document drawer. Recorded with BGE-small embeddings. Ollama was unavailable, so answer generation is not shown.

https://github.com/user-attachments/assets/aea90800-ffe2-4820-bb18-d0527570c0bb

[Download the walkthrough (MP4)](assets/media/interface-walkthrough.mp4).

<details>
<summary>Mobile document drawer</summary>

![Recorded mobile document drawer showing the indexed sample and its six passages.](assets/figures/interface-mobile.png)

</details>

<details>
<summary>Historical Q&A recording · 92 seconds</summary>

Upload a PDF and ask two questions. This older recording uses MiniLM embeddings; its interface and defaults differ from the current version. Responses are examples, not validated results.

https://github.com/user-attachments/assets/190555b4-b444-4626-ab0e-3ef5bc31bf92

[Download the historical recording (MP4)](assets/media/historical-qa.mp4).

</details>

[Recording sources and conditions](docs/media.md).

## Run locally

Use Python **3.11+**. Install and start [Ollama](https://ollama.com/), then run:

```bash
git clone https://github.com/urmeo/nexus-rag.git
cd nexus-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ollama pull llama3.2:3b
scinexusrag
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload PDF, DOCX, Markdown or UTF-8 text and ask a question. [Try the fictional sample paper](examples/sample_paper.md).

Open **Pipeline** in the app header to see the document and question stages, default limits and recorded retrieval results.

Models need downloads on first use. Documents and indexes default to `data/`. Run **one application worker** because BM25 indexes and rate-limit counters are per process.

Copy [.env.example](.env.example) to `.env` to change settings. Generation sends retrieved text to `OLLAMA_BASE_URL`, which defaults to local Ollama. Use `ollama serve` if the server is not already running. [Setup, Docker and checks](CONTRIBUTING.md).

## Results

### Retrieval

Full BEIR test splits: **300 SciFact queries** over **5,183 abstracts** and **323 NFCorpus queries** over **3,633 documents**. Higher nDCG@10 is better. Hybrid, adaptive and corrective rows use BGE-small embeddings.

| System | SciFact nDCG@10 | NFCorpus nDCG@10 |
| --- | ---: | ---: |
| Dense · MiniLM | 0.648 | 0.318 |
| BM25 | 0.666 | 0.311 |
| Dense · BGE-small | 0.708 | 0.341 |
| Hybrid · RRF | 0.704 | 0.352 |
| Adaptive weights | 0.703 | 0.346 |
| Corrective PRF | 0.703 | 0.346 |

![Recorded retrieval scores on SciFact and NFCorpus, with 95% query-bootstrap intervals.](assets/figures/retrieval-comparison.svg)

Combining methods did not consistently improve the scores. Corrective retrieval triggered no retries on SciFact at the headline threshold of **0.55**. Reranking was disabled in these evaluations; the library and evaluation tools provide a cross-encoder reranker.

### Retrieval time and corrective threshold

First **120 SciFact test queries**, one CPU pass without warmup. Indexing and answer generation are excluded; initial reranker loading is included.

| System | nDCG@10 | Milliseconds per query |
| --- | ---: | ---: |
| Adaptive hybrid | 0.734 | 16.9 |
| Corrective PRF | 0.734 | 17.3 |
| Cross-encoder reranking | 0.702 | 1318.9 |

This experiment selected a corrective threshold of **0.40** on **250 training queries**, separate from the headline threshold of **0.55**. Neither threshold triggered retries on the SciFact test set. Timings have no repeated-run intervals.

![Recorded SciFact threshold sweep: retrieval quality and the percentage of queries that triggered a feedback search.](assets/figures/corrective-sweep.svg)

The points are measured settings; connecting lines guide the eye. Thresholds of **0.65** and **0.70** triggered retries and reduced test nDCG in this sweep.

### Evidence-sentence detection

**188 SciFact development claims**, **2,031 candidate sentences**, **18.0% positive**. Candidates come from annotated abstracts. Both supporting and contradicting rationale sentences count as evidence.

| Scorer | ROC-AUC | Average precision | F1 |
| --- | ---: | ---: | ---: |
| Lexical overlap | 0.686 | 0.365 | 0.112 |
| NLI | 0.688 | 0.331 | 0.368 |
| Cross-encoder | 0.755 | 0.476 | 0.469 |

![ROC and precision–recall curves computed from the recorded evidence-sentence scores, with random baselines.](assets/figures/evidence-curves.svg)

F1 thresholds were selected on **120 training claims**. This task measures evidence detection, not generated-answer accuracy.

<details>
<summary>Evaluation protocol and source results</summary>

Retrieval uses exact CPU search over NumPy vectors, seed **0**, depth **50**, RRF **k=60** and linear nDCG gains. Each benchmark document is one passage; the application chunks documents and uses LanceDB.

Retrieval intervals use **10,000 query bootstrap samples**. Paired tests use **10,000 randomizations**, with Holm correction within each dataset. Evidence ROC-AUC intervals use **2,000 claim bootstraps**. [Raw results](outputs/results) retain individual scores, revisions, source hashes and runtime versions. [Evaluation details and reproduction commands](docs/evaluation.md).

</details>

## Limitations

- Abstract-level retrieval and evidence benchmarks do not establish full-paper answer accuracy. Check answers against the retrieved passages.
- PDF extraction requires embedded text; there is no OCR. DOCX extraction reads paragraphs, not tables, headers or footers.
- Citation validation checks source indices. Optional NLI scores each answer sentence against all supplied sources; it does not verify that a particular inline citation supports the sentence.
- Confidence is a heuristic unless NLI grounding is enabled. Neither score guarantees correctness.
- Exact vector search scales linearly. BM25 rebuilds in memory and requires one application worker.
- Corrective retrieval can reduce ranking quality. Ollama model tags can change.

[Folder guide](docs/layout.md) · [Notebook](notebooks/01_quickstart.ipynb) · [Examples](examples) · [Regenerate charts](docs/evaluation.md#regenerate-the-charts) · [References](docs/evaluation.md#references) · [MIT license](LICENSE)

Sole contributor: [urme](https://github.com/urmeo).
