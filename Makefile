.PHONY: install install-dev test test-unit test-integration test-cov lint format type-check eval eval-sample eval-gate corrective faithfulness ragtruth generation reproduce paper run clean help docker-build docker-up docker-down docker-logs

PYTHON := python3
PIP := $(PYTHON) -m pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src/scinexusrag --cov-report=html --cov-report=term

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

type-check:
	$(PYTHON) -m mypy src/

eval:
	$(PYTHON) -m scinexusrag.eval.run --dataset scifact --split test
	$(PYTHON) -m scinexusrag.eval.run --dataset nfcorpus --split test

eval-sample:
	$(PYTHON) -m scinexusrag.eval.run --sample

eval-gate:
	$(PYTHON) -m scinexusrag.eval.gate

corrective:
	$(PYTHON) -m scinexusrag.eval.corrective --dataset scifact --split test
	$(PYTHON) -m scinexusrag.eval.corrective --dataset nfcorpus --split test

faithfulness:
	$(PYTHON) -m scinexusrag.eval.faithfulness

ragtruth:
	$(PYTHON) -m scinexusrag.eval.ragtruth

generation:
	$(PYTHON) -m scinexusrag.eval.generation

reproduce:
	$(PYTHON) -m scinexusrag.eval

paper:
	$(PYTHON) -m scinexusrag.eval.report --paper paper
	cd paper && tectonic main.tex

run:
	$(PYTHON) -m uvicorn scinexusrag.api:app --reload --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

help:
	@echo "Available commands:"
	@echo "  install        Install package"
	@echo "  install-dev    Install with dev dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Check code style"
	@echo "  format         Format code"
	@echo "  type-check     Run type checking"
	@echo "  run            Start FastAPI server + web UI"
	@echo "  eval           Run the SciFact + NFCorpus retrieval ablation"
	@echo "  eval-sample    Run the offline vendored subset (no download)"
	@echo "  eval-gate      Run the eval regression gate (CI parity)"
	@echo "  reproduce      Regenerate every committed README benchmark number"
	@echo "  corrective     Corrective-loop trigger and cost/quality analysis"
	@echo "  faithfulness   Run the evidence-detection evaluation"
	@echo "  generation     Run the generation-quality evaluation"
	@echo "  ragtruth       Run the RAGTruth hallucination-detection evaluation"
	@echo "  paper          Regenerate tables/figures and build the PDF"
	@echo "  clean          Remove build artifacts"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-up      Start services with Docker Compose"
	@echo "  docker-down    Stop Docker Compose services"
	@echo "  docker-logs    Tail Docker Compose logs"
