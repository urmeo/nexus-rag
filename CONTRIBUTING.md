# Contributing to NexusRAG

Thanks for your interest in improving NexusRAG. This guide covers local setup,
the checks we run, and how to reproduce the benchmark.

## Development setup

NexusRAG targets Python 3.11+ and uses a src-layout package.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

The `dev` extra brings in the test and lint toolchain; `eval` adds the
benchmark dependencies (datasets, scipy, matplotlib).

## Checks

Run these before opening a pull request. CI runs the same tools.

```bash
make lint        # ruff check src/ tests/
make type-check  # mypy src/
make test        # pytest tests/
```

All three must pass. `make format` will auto-fix style issues.

## Reproducing the benchmark

```bash
make eval-sample   # offline vendored subset, no downloads
make eval          # full SciFact + NFCorpus retrieval ablation
```

`make eval-sample` runs the small offline sample used in CI. `make eval` runs
the full study and downloads the SciFact and NFCorpus splits on first use.

## Commit style

Commit messages are short, lowercase, and human, describing the change in a few
words (for example `updated paper` or `fix nfcorpus ndcg`). Keep one logical
change per commit.

## Branch and PR flow

1. Branch off `main`.
2. Make your change and keep commits focused.
3. Run `make lint type-check test` locally.
4. Open a pull request against `main` and fill in the template.
5. CI must be green before merge.

By contributing you agree your work is released under the project's MIT license.
