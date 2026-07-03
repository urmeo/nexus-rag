# ---- builder: compile dependencies into a venv (needs a toolchain) ----
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Pinned, hash-verified dependencies first (reproducible, supply-chain safe)
COPY requirements-runtime.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements-runtime.lock

# The application itself (deps already satisfied above)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# ---- runtime: slim image, no compiler shipped ----
FROM python:3.11-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libmagic1 && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=builder "$VIRTUAL_ENV" "$VIRTUAL_ENV"

WORKDIR /app
COPY configs/ configs/
COPY frontend/ frontend/

# Run as a non-root user
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "nexusrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
