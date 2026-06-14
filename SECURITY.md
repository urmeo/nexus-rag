# Security & Privacy Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's **"Report a vulnerability"**
button under the repository's **Security** tab (Security Advisories), or open a
minimal private report there. Do not file public issues for undisclosed
vulnerabilities. We aim to acknowledge reports within a reasonable timeframe.

## Privacy posture

NexusRAG is **local-first** and designed to keep your data on your machine.

- **No telemetry, no analytics, no tracking.**
- **No API keys or credentials** are required or stored.
- **Uploaded documents and embeddings stay local** under `./data/` (git-ignored)
  and are never sent to a third party.
- The only outbound network traffic is:
  - downloading open model weights from Hugging Face on first run, and
  - requests to your **local** Ollama instance (`localhost`) for generation.
- The benchmark data shipped in `benchmarks/` is **public scientific data**
  (SciFact, NFCorpus via BEIR) used for evaluation — it contains no personal data.

## Security measures

- **Input validation** — uploads are restricted to `.pdf/.docx/.txt/.md`;
  unsupported types are rejected with a clear error.
- **Filename sanitization** — upload filenames are sanitized against path
  traversal (e.g. `../../etc/passwd` → `passwd`).
- **Safe identifiers** — vector-store IDs are validated against an allowlist
  pattern before use in queries.
- **CORS** — restricted origins by default; credentials are disabled when a
  wildcard origin is configured.
- **No secrets in the repo** — configuration uses environment variables with safe
  local defaults; no secrets are committed. Defaults are documented in
  `src/nexusrag/config.py` and `configs/default.yaml`.

## Supported versions

This is a research artifact; security fixes are applied to the latest `main`.
