# NexusRAG

Search research papers and ask a local Ollama model for answers with cited passages. Built with Python, PyTorch, sentence-transformers, NumPy, LanceDB and FastAPI.

## Pipeline

![Documents are parsed and indexed; questions retrieve passages for Ollama to generate a cited answer.](assets/figures/pipeline.svg)

[Pipeline details](docs/pipeline.md) · [Folder guide](docs/layout.md)

## Recorded results

Full BEIR test splits: **300 SciFact queries** and **323 NFCorpus queries**. Higher nDCG@10 is better; error bars show 95% query-bootstrap intervals.

![Dense, keyword and hybrid retrieval scores on SciFact and NFCorpus.](assets/figures/retrieval-comparison.svg)

Combining methods did not consistently improve the scores. At the headline threshold, corrective retrieval triggered no retries on SciFact.

Evidence detection uses **188 SciFact development claims** and **2,031 candidate sentences**. Both supporting and contradicting rationale sentences count as evidence.

![ROC and precision-recall curves computed from the recorded evidence-sentence scores.](assets/figures/evidence-curves.svg)

These tasks measure retrieval and evidence detection. They do not establish generated-answer accuracy. [Tables, threshold curve and evaluation conditions](docs/evaluation.md).

## Run locally

Use Python 3.11 or later. Install and start [Ollama](https://ollama.com/), then run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ollama pull llama3.2:3b
scinexusrag
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload PDF, DOCX, Markdown or UTF-8 text and ask a question. [Try the fictional sample paper](examples/sample_paper.md).

- Models need downloads on first use. Documents and indexes default to `data/`.
- Run one application worker. BM25 indexes and rate-limit counters are per process.
- Check the cited passages. Confidence uses a heuristic by default; optional NLI grounding uses a model score.

Copy [.env.example](.env.example) to `.env` to change settings. Generation sends retrieved text to `OLLAMA_BASE_URL`, which defaults to local Ollama. Use `ollama serve` if the server is not already running. [Setup, Docker and checks](CONTRIBUTING.md).

## Project files

| Location | Contents |
| --- | --- |
| [`src/scinexusrag/`](src/scinexusrag) | Application, API, web interface and evaluation tools |
| [`assets/figures/`](assets/figures) | Pipeline diagram and charts |
| [`outputs/results/`](outputs/results) | Recorded measurements and provenance |
| [`docs/`](docs) | Pipeline, folder layout and evaluation guides |
| [`tests/`](tests) | Python and JavaScript checks |

[Notebook](notebooks/01_quickstart.ipynb) · [Regenerate charts](docs/evaluation.md#regenerate-the-charts) · [MIT license](LICENSE)

Sole contributor: [urmeo](https://github.com/urmeo).
