"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


HF_REVISIONS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
    "cross-encoder/nli-deberta-v3-small": "fa2804872c3b4bd748f38c0185cc85775361e735",
}


class LLMSettings(BaseSettings):
    """LLM configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_", populate_by_name=True)

    base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    model: str = "llama3.2:3b"
    temperature: float = Field(default=0.1, ge=0, allow_inf_nan=False)

    max_tokens: int = Field(default=768, gt=0)
    timeout: int = Field(default=60, gt=0)


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model: str = Field(default="BAAI/bge-small-en-v1.5")
    revision: str | None = None
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def resolve_revision(self) -> Self:
        if self.revision is None:
            self.revision = HF_REVISIONS.get(self.model)
        return self


class IngestionSettings(BaseSettings):
    """Document ingestion configuration."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    chunk_size: int = Field(default=1200, gt=0)
    chunk_overlap: int = Field(default=300, ge=0)
    min_chunk_size: int = Field(default=200, gt=0)

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must not exceed chunk_size")
        return self


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    top_k: int = Field(default=8, gt=0)

    max_query_length: int = Field(default=2000, gt=0)


class SelfCorrectionSettings(BaseSettings):
    """Confidence-gated corrective re-retrieval settings."""

    model_config = SettingsConfigDict(env_prefix="SELF_CORRECTION_")

    enabled: bool = True
    confidence_tau: float = Field(default=0.55, ge=0, le=1)
    feedback_docs: int = Field(default=5, gt=0)
    feedback_terms: int = Field(default=10, gt=0)

    grounding_enabled: bool = False
    grounding_model: str = "cross-encoder/nli-deberta-v3-small"
    grounding_threshold: float = Field(default=0.5, ge=0, le=1)


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

    api_key: str = Field(default="", validation_alias="NEXUSRAG_API_KEY")

    query_rate_per_minute: int = Field(default=60, gt=0)
    upload_rate_per_minute: int = Field(default=10, gt=0)

    max_upload_mb: int = Field(default=50, gt=0)
    max_uncompressed_mb: int = Field(default=200, gt=0)

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
