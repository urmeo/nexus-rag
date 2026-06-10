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
    model: str = Field(
        default="llama3.2:3b",  # Smaller model for 8GB RAM systems
        validation_alias="LLM_MODEL",
    )
    temperature: float = 0.1
    max_tokens: int = 256  # Reduced for faster responses
    timeout: int = 60


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model: str = Field(default="all-MiniLM-L6-v2")
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_size: int = 16  # Reduced default for 8GB RAM systems


class IngestionSettings(BaseSettings):
    """Document ingestion configuration."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    chunk_size: int = 1200  # Large chunks for better context
    chunk_overlap: int = 300  # Significant overlap for continuity
    min_chunk_size: int = 200  # Ensure meaningful chunks


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    top_k: int = 8  # Retrieve more initially
    rerank_top_k: int = 4  # Keep top 4 after reranking
    similarity_threshold: float = 0.2  # Lower threshold for better recall
    max_query_length: int = 512


class SelfCorrectionSettings(BaseSettings):
    """Self-correction loop configuration."""

    model_config = SettingsConfigDict(env_prefix="SELF_CORRECTION_")

    enabled: bool = True
    max_iterations: int = 1  # Single pass with batched grading
    relevance_threshold: float = 0.4  # Slightly lower threshold
    reformulation_enabled: bool = True

    # NLI grounding (opt-in; loads a small cross-encoder)
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
    table_name: str = "documents"


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
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
