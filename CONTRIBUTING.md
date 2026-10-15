# Contributing

[urme](https://github.com/urmeo) is the sole contributor.

## Development setup

Use Python 3.11.4 or later. Node.js 22 is used for the JavaScript tests. Run commands from the repository root:

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

Set environment variables or create `.env` in the repository root with the settings you need. [configs/default.yaml](configs/default.yaml) is a reference and is not loaded by the application. Never commit secrets.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama address; retrieved passages are sent here |
| `LLM_MODEL` | `llama3.2:3b` | Ollama model tag |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `EMBEDDING_REVISION` | Pinned for known models | Full model commit SHA for persistent indexes |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda` or `mps` |
| `DATA_DIR` | `./data` | Document metadata directory |
| `LANCEDB_PATH` | `./data/lancedb` | Vector index directory; set separately from `DATA_DIR` |
| `SELF_CORRECTION_ENABLED` | `true` | Allow one feedback search when dense similarity is low |
| `SELF_CORRECTION_GROUNDING_ENABLED` | `false` | Check answer sentences with an NLI model |
| `NEXUSRAG_API_KEY` | Empty | Require an API key on `/api` routes |
| `NEXUSRAG_FRONTEND_DIR` | Packaged `web/` directory | Serve a different interface directory |

Each persistent table binds a versioned embedding configuration: model, revision, exact query and document prefixes, normalization, and vector dimension. Legacy tables without this binding and any configuration mismatch are rejected, including empty tables. Custom Hugging Face models require a full 40-character commit SHA in `EMBEDDING_REVISION`. Local model paths are not accepted for persistent indexes.

To keep an existing collection, restore its original embedding settings. To use a different configuration, choose new `DATA_DIR` and `LANCEDB_PATH` directories and re-upload the original documents. `NexusRAG.clear_all()` or `DELETE /api/documents` explicitly deletes the current collection; use it only for an intentional reset before reingestion. Existing data is not automatically deleted, relabeled, or rebuilt.

When a server key is configured, enter it through **API key** in the interface. Interactive API documentation is disabled when a key is set. Without a key, `/docs` is available unless `API_DOCS_ENABLED=false`.

The defaults allow 50 MiB per upload, 200 MiB of declared uncompressed DOCX content, and 60 queries or 10 uploads per minute per IP. Change these with `API_MAX_UPLOAD_MB`, `API_MAX_UNCOMPRESSED_MB`, `API_QUERY_RATE_PER_MINUTE` and `API_UPLOAD_RATE_PER_MINUTE`. Optional MIME detection needs the system `libmagic` library; signature and UTF-8 checks also run.

For network access, configure `NEXUSRAG_API_KEY`, TLS and `API_CORS_ORIGINS`. Trust forwarded headers only from your proxy. Run one worker because BM25 indexes and rate counters are per process.

## Docker

```bash
docker build -t nexus-rag .
```

The image starts one application worker on port 8000. Set `OLLAMA_BASE_URL` to an Ollama server reachable from the container and download `LLM_MODEL` on that server. For local use, publish port 8000 on `127.0.0.1`. Mount persistent storage at `/app/data`; it must be writable by container UID 1000. Pass settings as container environment variables. Ollama model tags can change; record the model digest for comparisons.

## Evaluation and results

See [docs/evaluation.md](docs/evaluation.md) for protocols, commands and result files. Fresh runs write to `outputs/generated/`; published results are in `outputs/results/`.

Keep query IDs, model and dataset revisions, source hashes, runtime versions and raw scores. Review new runs before publishing them. Samples do not replace full runs, and retrieval or evidence scores do not measure generated-answer accuracy. Do not lower regression thresholds to hide a failure. Follow dataset and model licenses.

## Security reports

Use [private security advisories](https://github.com/urmeo/nexus-rag/security/advisories/new) to report vulnerabilities. Include the version, steps to reproduce and impact. Omit secrets and personal data. Keep details private until a fix is available.
