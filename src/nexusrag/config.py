"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM configuration."""

    model_config = SettingsConfigDict(env_prefix="", populate_by_name=True)

    base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    model: str = Field(default="llama3.2:3b", validation_alias="LLM_MODEL")
    temperature: float = 0.1
    max_tokens: int = 256
    timeout: int = 60


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model: str = Field(default="BAAI/bge-small-en-v1.5")
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
    rerank_top_k: int = 4
    similarity_threshold: float = 0.2
    max_query_length: int = 512


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
