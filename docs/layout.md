# Folder layout

```text
nexus-rag/
├── assets/
│   ├── figma/              Figma pipeline snapshot
│   ├── figures/            Diagrams, charts and interface screenshots
│   └── media/              Interface recordings
├── docs/                   Pipeline, evaluation, media and layout guides
├── notebooks/              Python quickstart
├── outputs/
│   ├── results/            Committed measurements and provenance
│   └── generated/          Fresh runs and reports; ignored by Git
├── src/scinexusrag/
│   ├── api/                HTTP routes, request limits and authentication
│   ├── eval/               Dataset loading, metrics and report generation
│   │   └── data/           Packaged samples and regression thresholds
│   ├── generation/         Ollama client, synthesis and answer checks
│   ├── ingestion/          Parsing, chunking and embeddings
│   ├── retrieval/          Dense, BM25, fusion and corrective search
│   ├── storage/            Document metadata and vectors
│   ├── utils/              Shared filename and text helpers
│   └── web/                Packaged HTML, CSS and JavaScript
└── tests/                  Unit, integration and JavaScript UI tests
```

| Keep here | Purpose |
| --- | --- |
| `assets/figures/` | Images referenced by documentation; new charts are SVG for readable text and export |
| `assets/media/` | Saved MP4 recordings; sources and conditions are in [media.md](media.md) |
| `outputs/results/` | Original benchmark JSON and CSV; retain query IDs, scores and provenance |
| `outputs/generated/` | New results and reports for review before publication |
| `src/scinexusrag/eval/data/` | Small datasets and thresholds that must be included in the installed package |
| `src/scinexusrag/web/` | Interface files that must be included in the installed package |

Root files contain package and check settings, the license, citation information and Docker setup. Application data defaults to `data/`, which is ignored by Git. Settings come from environment variables and `.env`; defaults are in [config.py](../src/scinexusrag/config.py).

Run commands from the repository root so relative data and result paths resolve consistently. Store published figures in `assets/figures/`; keep scratch work outside the repository. [Regenerate the charts](evaluation.md#regenerate-the-charts) from the committed measurements.
