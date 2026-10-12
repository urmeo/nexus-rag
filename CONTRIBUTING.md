# Contributing

[urme](https://github.com/urmeo) is the sole contributor.

## Development setup

Use Python 3.11 or later. Node.js 22 is used for the JavaScript tests. Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,eval]"
```

Check changes with:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov=scinexusrag --cov-branch --cov-fail-under=60
node --test tests/test_web_ui.js
python -m scinexusrag.eval.gate
```

The evaluation gate uses packaged data and cached models. Download the pinned BGE-small, NLI and reranking models before running it offline. Add tests for changed behavior. Use short commit messages such as `updated readme`, `updated results`, `updated pipeline` or `updated visuals`.

## Run the application

Start Ollama and download the default model:

```bash
ollama pull llama3.2:3b
scinexusrag
```

Use `ollama serve` if Ollama is not already running. Open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload a PDF, DOCX, Markdown or UTF-8 text file, and ask a question. Embedding models may download on first use. Storage defaults to `data/`.

You can also start the server directly:

```bash
python -m uvicorn scinexusrag.api:app --host 127.0.0.1 --port 8000
```

## Configuration

Copy [.env.example](.env.example) to `.env` and edit the settings you need. The application reads environment variables and `.env`; [configs/default.yaml](configs/default.yaml) is a reference and is not loaded. Never commit secrets.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama address; retrieved passages are sent here |
| `LLM_MODEL` | `llama3.2:3b` | Ollama model tag |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda` or `mps` |
| `DATA_DIR` | `./data` | Document metadata directory |
| `LANCEDB_PATH` | `./data/lancedb` | Vector index directory; set separately from `DATA_DIR` |
| `SELF_CORRECTION_ENABLED` | `true` | Allow one search using feedback terms when confidence is low |
| `SELF_CORRECTION_GROUNDING_ENABLED` | `false` | Check answer sentences with an NLI model |
| `NEXUSRAG_API_KEY` | Empty | Require an API key on `/api` routes |
| `NEXUSRAG_FRONTEND_DIR` | Packaged `web/` directory | Serve a different interface directory |

When a server key is configured, enter it through **API key** in the interface. Interactive API documentation is disabled when a key is set. Without a key, `/docs` is available unless `API_DOCS_ENABLED=false`.

The defaults allow 50 MiB per upload, 200 MiB of declared uncompressed DOCX content, and 60 queries or 10 uploads per minute per IP. Optional MIME detection needs the system `libmagic` library; signature and UTF-8 checks also run. See [.env.example](.env.example) for the request-limit settings.

For network access, configure `NEXUSRAG_API_KEY`, TLS and `API_CORS_ORIGINS`. Trust forwarded headers only from your proxy. Run one worker because BM25 indexes and rate counters are per process.

## Docker

```bash
mkdir -p data
docker compose up --build
```

Compose binds the application to loopback and downloads the Ollama model before starting it. The bind-mounted `data/` directory must be writable by container UID 1000. Set `LLM_MODEL` for both services through the Compose environment or `.env`. Ollama model tags can change; record the model digest for comparisons.

The app service passes `OLLAMA_BASE_URL`, `LLM_MODEL` and `NEXUSRAG_API_KEY`. Add other application settings to its `environment` section in [docker-compose.yml](docker-compose.yml); the host `.env` is not mounted inside the container.

## Evaluation and results

See [docs/evaluation.md](docs/evaluation.md) for protocols, commands and result files. Fresh runs write to `outputs/generated/`; published results are in `outputs/results/`.

Keep query IDs, model and dataset revisions, source hashes, runtime versions and raw scores. Review new runs before publishing them. Samples do not replace full runs, and retrieval or evidence scores do not measure generated-answer accuracy. Do not lower regression thresholds to hide a failure. Follow dataset and model licenses.

## Security reports

Use [private security advisories](https://github.com/urmeo/nexus-rag/security/advisories/new) to report vulnerabilities. Include the version, steps to reproduce and impact. Omit secrets and personal data. Keep details private until a fix is available.
