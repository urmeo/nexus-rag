# Architecture

## 1. Overview

NexusRAG is a local hybrid-retrieval RAG system for scientific papers. It ingests
documents, retrieves with fused dense and sparse signals, runs an optional
confidence-gated corrective pass, generates cited answers with a local LLM, and
verifies citations and grounding before returning a response. The whole stack runs
on a single machine against a local Ollama server.

Data flow:

- **Ingest** -> parse document -> semantic chunk -> embed chunks.
- **Store** -> write vectors + metadata to LanceDB; add chunks to an in-memory BM25 index.
- **Query** -> dense + BM25 retrieval -> reciprocal rank fusion (RRF) -> optional
  corrective pseudo-relevance-feedback (PRF) re-retrieval -> LLM synthesis ->
  citation verification + optional NLI grounding.

The pipeline is wired in `src/scinexusrag/pipeline.py`; components are lazy-loaded and
the instance is a process singleton.

## 2. Retrieval

### Dense

- The product path uses LanceDB. Search calls `.metric("cosine")`, so the stored
  `_distance` is cosine distance and the returned score is `max(0, 1 - distance)`,
  clamped to `[0, 1]`. See `VectorStore.search` in
  `src/scinexusrag/storage/vector_store.py`.
- LanceDB does **exact brute-force** search; no ANN index is built, so production
  retrieval is exact. This is fine at the current corpus scale but means query cost
  grows linearly with corpus size.
- The reproducible benchmark uses a separate in-memory `ExactDenseRetriever`
  (`src/scinexusrag/eval/indexes.py`) that does brute-force cosine over a precomputed
  embedding matrix, for determinism independent of the LanceDB store.

### Sparse

- BM25 via `rank-bm25` (`BM25Okapi`), in `src/scinexusrag/retrieval/sparse.py`.
- The index is **in-memory and rebuilt per process**. Incremental add and remove
  both rebuild the whole index internally.
- This is acceptable for the small benchmark corpora but does not scale.
  **Limitation:** for large corpora a persistent sparse index (Tantivy, Pyserini)
  is needed; nothing here is durable or incremental at scale.

### Fusion

- Weighted reciprocal rank fusion with `k = 60` (`rrf_fuse` in
  `src/scinexusrag/retrieval/hybrid.py`).
- `AdaptiveHybridRetriever` shifts the dense/sparse weight split by query shape:
  short or notation-heavy queries lean lexical, long natural-language queries lean
  dense, with a base 0.5/0.5 split otherwise.

### Corrective

- A **single** confidence-gated PRF re-retrieval pass — not iterative
  (`CorrectiveRetriever` in `src/scinexusrag/retrieval/corrective.py`).
- Confidence is the top dense cosine similarity. If it is at or above `tau`
  (default 0.55), the first pass is returned unchanged.
- Below `tau`, the query is expanded with frequent terms from the first-pass
  documents, a second retrieval runs, and the two passes are RRF-fused. The extra
  cost is paid only on low-confidence queries.

## 3. Generation

- `LLMClient` (`src/scinexusrag/generation/llm.py`) targets Ollama's HTTP API:
  `/api/generate` (sync + streaming) and `/api/chat`.
- Retries with exponential backoff on transient and 5xx errors; no retry on 4xx.
  Streaming is not retried because a partially consumed stream cannot be safely
  replayed. Timeout and connection errors are classified and wrapped as `LLMError`.
- The model is pinned to `llama3.2:3b` by default.
- **Limitation:** the client is Ollama-specific. There is no provider abstraction,
  so swapping to another backend (OpenAI-compatible, vLLM, etc.) requires a new
  client implementation, not just config.

## 4. Verification / Grounding

- `AnswerVerifier` (`src/scinexusrag/generation/verifier.py`) strips any `[n]`
  citation whose index falls outside the source range and records the removals as
  warnings.
- `GroundingVerifier` (`src/scinexusrag/generation/grounding.py`) runs per-sentence NLI
  entailment of each answer sentence against each source, using a cross-encoder NLI
  model (`cross-encoder/nli-deberta-v3-small`). A sentence is grounded when some
  source entails it above `threshold`; faithfulness is the grounded fraction.
- Grounding is **off by default** (`SelfCorrectionSettings.grounding_enabled = False`).
- **Cost / limitation:** grounding is O(sentences x sources) NLI calls and is
  currently **not batched or cached**. The RAGTruth evaluation
  (`src/scinexusrag/eval/ragtruth.py`) caps work explicitly to keep it tractable:
  `MAX_OUT_SENTS = 12` output sentences and `MAX_CTX_SENTS = 50` context sentences.
  Batching and caching are open performance work.

## 5. Serving

- Single-process FastAPI app served by uvicorn (`src/scinexusrag/api/__init__.py`,
  `src/scinexusrag/api/routes.py`).
- Blocking work (parsing, embedding, query, stats) is offloaded with
  `asyncio.to_thread`, so the event loop is **not** blocked during ingestion or
  query. Verify in `routes.py` (`ingest_document`, `query_documents`, etc.).
- **Concurrency limits (single-user scope):** `asyncio.to_thread` shares
  asyncio's default executor, capped at `min(32, cpu + 4)` threads. The slowapi
  limits are per-minute windows, not concurrency caps, so a burst beyond the
  pool queues in-process with no 503 backpressure. A hung Ollama holds a thread
  for up to one request timeout: the read-timeout path fails fast after a
  single attempt (the retry policy covers only connection/5xx errors, not
  timeouts, since retrying a slow model only multiplies the wall-clock hang).
  Acceptable for localhost use; if exposed beyond
  localhost, bound query/ingest with an `asyncio.Semaphore` that returns 503
  when full. The app-level `/health` probe bypasses the executor and stays
  responsive either way.
- **Streaming caveat:** `query_streaming` yields raw tokens as they are
  generated; citation verification and grounding run only after the stream
  completes and are surfaced as a trailing `data: {...}` verification event.
  Streamed text can therefore momentarily show citations that the final
  verification reports (and the non-streaming path would strip). No HTTP
  route currently exposes streaming — it is a library-level API.
- **Scaling out is not just adding workers:** the BM25 index and rate-limit
  counters are per-process, so multiple uvicorn workers would each hold their
  own diverging index. Scaling beyond one process requires a shared sparse
  index and rate-limit store first.
- Health endpoints: `/health` (app-level liveness, no pipeline init) and
  `/api/health` (reports `llm_available`, i.e. Ollama readiness, plus model and
  corpus stats).
- Security controls live in `src/scinexusrag/api/security.py`: optional API-key auth
  (default-deny when a key is set, open in local no-key mode), per-IP slowapi rate
  limits, and upload validation (content-type, magic bytes, libmagic sniff, UTF-8
  and zip-bomb checks). See `SECURITY.md` for details — not duplicated here.

## 6. Configuration

- Configuration is `pydantic-settings`, sourced **only from environment variables**
  (`src/scinexusrag/config.py`). Each section has its own env prefix or explicit
  validation aliases (e.g. `OLLAMA_BASE_URL`, `LLM_MODEL`, `LANCEDB_PATH`,
  `NEXUSRAG_API_KEY`).
- `configs/default.yaml` is an annotated **reference** of these settings, validated
  by `tests/unit/test_config.py`. It is **not** a second runtime config source.
- Because only environment variables are read at runtime, there is **no precedence
  ambiguity** between YAML and env.

## 7. Model Footprint

Approximate on-disk sizes (rough; for capacity planning, not exact):

| Model | Role | Approx. size |
| --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | embeddings | ~130 MB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker (cross-encoder) | ~90 MB |
| `cross-encoder/nli-deberta-v3-small` | grounding NLI | ~280 MB |
| `llama3.2:3b` (Ollama) | generation | ~2 GB |

Combined disk for all weights is roughly 2.5 GB. Running the full stack (embedder,
reranker, NLI, and the LLM resident in Ollama) needs on the order of several GB of
RAM in addition to the OS and Python process. "Runs on a laptop" assumes ~8 GB+ RAM
and that the models are already cached locally; on tight machines `unload_models()`
frees the ML models between tasks.

## 8. Frontend

- `frontend/` is a static single-page UI: `index.html` plus `css/` and `js/`,
  served by FastAPI (`api/__init__.py` mounts `/css`, `/js`, and serves
  `index.html` at `/`).
- It is for local interactive use only. It is **optional** — neither the benchmark
  nor the API depends on it, and the app starts fine without the directory present.
  Documented here so it is not mistaken for undocumented dead weight.
