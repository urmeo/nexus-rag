# Contributing

## Setup

Python 3.11+ and Node 22; run from repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,eval]"
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov=scinexusrag --cov-branch --cov-fail-under=60
node --test tests/test_web_ui.js
python -m scinexusrag.eval.gate
```

Test changed behavior; keep commit titles short.

## App and configuration

Start Ollama and pull `llama3.2:3b`, then:

```bash
python -m uvicorn scinexusrag.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Upload PDF, DOCX, Markdown or UTF-8 text; enter `NEXUSRAG_API_KEY` through **API key**. Models may download initially; storage defaults to `data/`.

Copy [.env.example](.env.example) to `.env`; never commit secrets. Runtime uses environment variables; `configs/default.yaml` is reference-only. `NEXUSRAG_FRONTEND_DIR` overrides the UI.

`docker compose up --build` binds to loopback. Configure both services with `LLM_MODEL`; record Ollama digests. Bind-mounted `data/` must be writable by container UID 1000.

For network access, set `NEXUSRAG_API_KEY`, use TLS, restrict `API_CORS_ORIGINS`, and trust forwarded headers only from your proxy. Use one worker: BM25 and rate counters are per process. Default limits: 50 MiB uploads, 200 MiB decompressed DOCX, 60 queries/10 uploads per minute per IP.

## Evaluation

```bash
python -m scinexusrag.eval --sample
python -m scinexusrag.eval
python -m scinexusrag.eval.report --results outputs/generated
```

Samples use packaged data; uncached models need downloads. Full runs require pinned datasets/models and fail if unavailable. New results: `outputs/generated/`; published results: `outputs/results/` (the report's default input).

Retrieval/evidence scores do not measure generated answers; generation/RAGTruth evaluations are separate. Preserve query IDs, revisions, hashes, environments and raw scores; review before publishing. Never replace published results with samples or lower CI floors. Respect dataset/model licenses.

## Security

Report vulnerabilities through [private advisories](https://github.com/urmeo/nexus-rag/security/advisories/new) with version, reproduction and impact. Omit secrets/personal data; keep details private until fixed. Treat contributors respectfully.
