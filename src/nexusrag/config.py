"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# One load covers every settings section below: each BaseSettings subclass
# reads the process environment independently, so .env must be in os.environ
# before any of them is instantiated.
load_dotenv()

# Pinned HF revisions for every model the project loads (supply chain): the
# loaders resolve through this map, so reported numbers reference fixed weights.
HF_REVISIONS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    "cross-encoder/nli-deberta-v3-small": "fa2804872c3b4bd748f38c0185cc85775361e735",
}


class LLMSettings(BaseSettings):
    """LLM configuration."""

    # LLM_ prefix so temperature/timeout read LLM_TEMPERATURE/LLM_TIMEOUT, not
    # bare TEMPERATURE/TIMEOUT (which would collide with unrelated env vars).
    model_config = SettingsConfigDict(env_prefix="LLM_", populate_by_name=True)

    # base_url keeps the conventional OLLAMA_BASE_URL name.
    base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    model: str = "llama3.2:3b"
    temperature: float = 0.1
    # Upper bound on generated tokens; the synthesizer scales its budget with
    # source count up to this cap.
    max_tokens: int = 768
    timeout: int = 60


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model: str = Field(default="BAAI/bge-small-en-v1.5")
    revision: str = Field(default=HF_REVISIONS["BAAI/bge-small-en-v1.5"])
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = 32


class IngestionSettings(BaseSettings):
    """Document ingestion configuration."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    chunk_size: int = 1200
    chunk_overlap: int = 300
    min_chunk_size: int = 200


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    top_k: int = 8
    # Hard cap on accepted question length (chars), enforced at the API boundary.
    max_query_length: int = 2000


class SelfCorrectionSettings(BaseSettings):
    """Confidence-gated corrective re-retrieval settings."""

    model_config = SettingsConfigDict(env_prefix="SELF_CORRECTION_")

    enabled: bool = True
    confidence_tau: float = 0.55
    feedback_docs: int = 5
    feedback_terms: int = 10

    grounding_enabled: bool = False
    grounding_model: str = "cross-encoder/nli-deberta-v3-small"
    grounding_threshold: float = 0.5


class StorageSettings(BaseSettings):
    """Storage configuration."""

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)

    lancedb_path: Path = Field(
        default=Path("./data/lancedb"),
        validation_alias="LANCEDB_PATH",
    )
    table_name: str = "chunks"


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # When set, every /api route requires the X-API-Key header. Empty keeps the
    # local-first UX open; any network-exposed deployment MUST set this.
    api_key: str = Field(default="", validation_alias="NEXUSRAG_API_KEY")

    # Per-client (IP) fixed-window rate limits.
    query_rate_per_minute: int = 60
    upload_rate_per_minute: int = 10

    # Upload guards: max raw bytes and max decompressed bytes (zip-bomb cap).
    max_upload_mb: int = 50
    max_uncompressed_mb: int = 200

    # Interactive API docs (/docs, /redoc). Auto-disabled when api_key is set.
    docs_enabled: bool = True


class Settings(BaseSettings):
    """Main application settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    self_correction: SelfCorrectionSettings = Field(default_factory=SelfCorrectionSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: APISettings = Field(default_factory=APISettings)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        validation_alias="DATA_DIR",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
