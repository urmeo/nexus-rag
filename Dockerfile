FROM python:3.11-slim-trixie AS builder

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --only-binary=:all: --no-cache-dir -r requirements.txt

FROM python:3.11-slim-trixie AS runtime

ENV API_HOST=0.0.0.0
ENV PYTHONPATH=/app/src

RUN apt-get update && \
    apt-get install -y --no-install-recommends libmagic1t64 && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=builder "$VIRTUAL_ENV" "$VIRTUAL_ENV"

WORKDIR /app

COPY src/ src/
COPY LICENSE ./

RUN useradd --create-home --uid 1000 app && \
    mkdir -p /app/data && \
    chown app:app /app /app/data
USER 1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "from urllib.request import ProxyHandler, build_opener; build_opener(ProxyHandler({})).open('http://127.0.0.1:8000/health', timeout=3).close()"]

CMD ["uvicorn", "scinexusrag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
