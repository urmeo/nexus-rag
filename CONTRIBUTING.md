# Contributing

## Setup

Use Python 3.11.4+. Interface tests also need Node.js. Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt \
  "datasets>=5.0.1" "matplotlib>=3.8.0" "scipy>=1.11.0" \
  "aiohttp>=3.14.4" "multidict>=6.9.1" "pillow>=13.0.0" \
  "pytest>=9.0.3" "pytest-asyncio>=0.25.0" "pytest-cov>=6.0.0" \
  "ruff==0.15.17" "mypy>=1.13.0"
export PYTHONPATH="$PWD/src"
```

Start `ollama serve` in another terminal if needed, then run:

```bash
ollama pull llama3.2:3b
python -m scinexusrag.api
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload PDF, DOCX, Markdown or UTF-8 text. Uncached embedding models download on first use. Run **one worker**; BM25 indexes and rate counters are per process.

Build the Docker image from committed source files, excluding local data and credentials:

```bash
git archive --format=tar HEAD Dockerfile requirements.txt src LICENSE \
  | docker build -t nexus-rag -
```

| Container setting | Value |
| --- | --- |
| Process | UID 1000, one worker at `0.0.0.0:8000` |
| Local port publishing | `-p 127.0.0.1:8000:8000` |
| Persistent storage | Mount `/app/data`, writable by UID 1000 |
| Ollama | Set `OLLAMA_BASE_URL` to a reachable server with `LLM_MODEL` downloaded |

Pass other settings as container environment variables. Override the container command to change its listening address or port.

## Configuration

Set environment variables or use an ignored root `.env` file. Never commit secrets. [settings.py](src/scinexusrag/settings.py) defines the application defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server; receives retrieved passages |
| `LLM_MODEL` | `llama3.2:3b` | Ollama model tag |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Passage and query embeddings |
| `EMBEDDING_REVISION` | Pinned for known models | Full model commit SHA |
| `EMBEDDING_DEVICE` | `cpu` | `cpu`, `cuda` or `mps` |
| `DATA_DIR` | `./data` | Parsed documents and metadata in `DATA_DIR/documents/` |
| `LANCEDB_PATH` | `./data/lancedb` | Vector index; independent of `DATA_DIR` |
| `SELF_CORRECTION_ENABLED` | `true` | One feedback search when dense similarity is low |
| `SELF_CORRECTION_GROUNDING_ENABLED` | `false` | Optional NLI sentence checks |
| `API_HOST` | `127.0.0.1` | Listening address |
| `NEXUSRAG_API_KEY` | Empty | Require a key on `/api` routes |
| `NEXUSRAG_FRONTEND_DIR` | Bundled `web/` | Custom interface directory |

Indexes bind to the embedding model, revision, query and document prefixes, normalization and dimension. Missing or mismatched settings block access, including empty indexes. Custom models require a 40-character commit SHA in `EMBEDDING_REVISION`; local model paths are unsupported.

Keep a collection's original embedding settings. To change them, choose new `DATA_DIR` and `LANCEDB_PATH` directories and re-upload the originals. Alternatively, clear the collection with `NexusRAG.clear_all()` or `DELETE /api/documents`, restart with the new settings and re-upload.

Enter the server key through **API key** in the interface; it stays in memory until reload. A configured key disables API docs. Otherwise, `/docs` is available unless `API_DOCS_ENABLED=false`.

| Limit | Default | Setting |
| --- | --- | --- |
| Upload size | 50 MiB | `API_MAX_UPLOAD_MB` |
| Declared uncompressed DOCX content | 200 MiB | `API_MAX_UNCOMPRESSED_MB` |
| PDF pages | 1,000 | Fixed parser limit |
| Extracted text | 2,000,000 characters | Fixed parser limit |
| Passages per document | 5,000 | `INGESTION_MAX_CHUNKS` |
| Queries per IP | 60/minute | `API_QUERY_RATE_PER_MINUTE` |
| Uploads per IP | 10/minute | `API_UPLOAD_RATE_PER_MINUTE` |

Upload authentication, same-site checks and rate limits run before multipart parsing. PDF text is counted after each page is extracted; the limits do not cap a single page's peak extraction memory. Other formats are checked after parsing. The passage limit applies before embedding and index initialization.

MIME detection uses system `libmagic`. Signature and UTF-8 checks apply without it. The default host with no API key accepts only `localhost`, `127.0.0.1` or `[::1]` requests.

Serve the interface and API from the same origin. For network access, set `API_HOST`, configure `NEXUSRAG_API_KEY` and use a TLS reverse proxy. `API_CORS_ORIGINS` takes a JSON array, for example `export API_CORS_ORIGINS='["https://research.example"]'`. Same-site checks also apply to allowed origins. Trust forwarded headers only from your proxy.

## Checks

```bash
python -m ruff check src tests
python -m ruff format --check --line-length 100 --target-version py311 src tests
MYPYPATH=src python -m mypy --strict --explicit-package-bases --ignore-missing-imports src
python -m pytest tests --asyncio-mode=auto -o asyncio_default_fixture_loop_scope=function \
  --cov=scinexusrag --cov-branch --cov-fail-under=60
node --test tests/test_web_ui.js
python -m scinexusrag.eval.gate
```

The gate uses sample data and pinned BGE-small, NLI and reranking models. It needs cached models or network access; Ollama is not required. Test changed behavior. Do not lower thresholds to hide failures.

Follow [evaluation protocols](docs/evaluation.md). Review fresh JSON runs in `outputs/generated/` before publishing CSV files and charts in `outputs/results/`. Preserve raw scores, query IDs, model and dataset revisions, source hashes and runtime versions. Record Ollama model digests when comparing results; tags can change. Sample runs do not replace full benchmarks. Retrieval and evidence scores do not establish answer accuracy. Follow dataset and model licenses.

Use short lowercase commit messages: `updated readme`, `updated results`, `updated tests` or `updated retrieval`. For `LICENSE`, use `update license`.

Report vulnerabilities through [private security advisories](https://github.com/urmeo/nexus-rag/security/advisories/new). Include the version, reproduction steps and impact, without secrets or personal data. Keep reports private until a fix is available.
