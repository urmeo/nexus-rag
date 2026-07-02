# Security & Privacy Policy

## Reporting a vulnerability

Please report security issues **privately**, not in public issues:

- Preferred: GitHub → repository **Security** tab → **Report a vulnerability**
  (private Security Advisory).
- Or email **urme.bose1@gmail.com** with steps to reproduce and impact.

We aim to acknowledge reports within **5 business days** and to ship a fix or
mitigation for confirmed high-severity issues within **30 days**. Please do not
disclose publicly until a fix is available.

## Supported versions

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ |
| `0.1.x` | ✅ |
| < 0.1 | ❌ |

This is a research artifact; security fixes land on the latest `main` and the
most recent tagged release.

## Privacy posture

NexusRAG is **local-first** and keeps your data on your machine.

- **No telemetry, no analytics, no tracking.**
- Uploaded documents and embeddings stay local under `./data/` (git-ignored)
  and are never sent to a third party.
- The only outbound traffic is downloading open model weights from Hugging Face
  on first run, and requests to your **local** Ollama instance for generation.
- Benchmark data in `benchmarks/` is **public scientific data** (SciFact,
  NFCorpus via BEIR) and contains no personal data.

## Security controls (enforced in code)

- **Authentication** — when `NEXUSRAG_API_KEY` is set, every `/api` route
  requires a matching `X-API-Key` header (constant-time comparison). Empty in
  local mode; **required for any network-exposed deployment**.
- **Rate limiting** — per-client sliding-window limits on `/api/query` and
  `/api/ingest` (`src/nexusrag/api/security.py`).
- **Upload validation** — extension allowlist, content-type allowlist,
  magic-byte sniffing (PDF/DOCX), UTF-8 check for text, a max upload size, and a
  decompressed-size (zip-bomb) cap on DOCX.
- **Filename sanitization** — upload filenames are stripped of path components
  (`../../etc/passwd` → `passwd`).
- **Safe identifiers** — vector-store IDs are validated against an allowlist
  pattern before use in queries.
- **CORS** — restricted origins by default; credentials disabled under wildcard.
- **No secrets in the repo** — configuration uses environment variables with
  safe local defaults (`.env.example`); `.env` is git-ignored.

## Deployment hardening

- Docker binds the app to host loopback (`127.0.0.1:8000`); Ollama has **no
  host port** and is reachable only on the internal compose network.
- Container CPU/memory limits are set in `docker-compose.yml`.
- The image installs **hash-pinned** dependencies (`requirements-runtime.lock`,
  `pip install --require-hashes`).

## Secret scanning

- `gitleaks` runs in CI on every push/PR and as a local pre-commit hook
  (`.gitleaks.toml`, `.pre-commit-config.yaml`).
- The full commit history has been scanned; no secrets are present.
- `pip-audit` runs in CI against the hash-pinned runtime lock.

## Known advisories

- **CVE-2025-3000** (`torch`, transitive via `sentence-transformers`): memory
  corruption in `torch.jit.script`. NexusRAG never calls `torch.jit.script`, and
  no fixed `torch` release exists yet, so it is accepted and ignored in the CI
  audit. The pinned lockfiles are re-audited by `pip-audit` on every CI run, and
  the ignore is revisited when a fixed `torch` release ships.
