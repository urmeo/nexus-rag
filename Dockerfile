FROM python:3.11-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy application files
COPY configs/ configs/

# Copy frontend if present
COPY frontend/ frontend/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "nexusrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
