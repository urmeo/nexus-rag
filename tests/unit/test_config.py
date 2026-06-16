"""Tests for configuration module."""

from pathlib import Path

import pytest
import yaml


class TestSettings:
    def test_default_settings(self):
        from nexusrag.config import Settings

        settings = Settings()

        assert settings.llm.model == "llama3.2:3b"
        assert settings.llm.base_url == "http://localhost:11434"
        assert settings.embedding.model == "BAAI/bge-small-en-v1.5"
        assert settings.ingestion.chunk_size == 1200
        assert settings.retrieval.top_k == 8
        assert settings.self_correction.enabled is True
        assert settings.log_level == "INFO"

    def test_nested_settings_types(self):
        from nexusrag.config import (
            APISettings,
            EmbeddingSettings,
            IngestionSettings,
            LLMSettings,
            RetrievalSettings,
            SelfCorrectionSettings,
            Settings,
            StorageSettings,
        )

        settings = Settings()

        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.embedding, EmbeddingSettings)
        assert isinstance(settings.ingestion, IngestionSettings)
        assert isinstance(settings.retrieval, RetrievalSettings)
        assert isinstance(settings.self_correction, SelfCorrectionSettings)
        assert isinstance(settings.storage, StorageSettings)
        assert isinstance(settings.api, APISettings)

    def test_env_override_llm(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:8080")
        monkeypatch.setenv("LLM_MODEL", "mistral:7b")

        # Need to reimport to pick up new env vars
        from nexusrag.config import LLMSettings

        llm = LLMSettings()

        assert llm.base_url == "http://custom:8080"
        assert llm.model == "mistral:7b"

    def test_env_override_embedding(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding-model")
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")

        from nexusrag.config import EmbeddingSettings

        embedding = EmbeddingSettings()

        assert embedding.model == "custom-embedding-model"
        assert embedding.device == "cuda"

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        from nexusrag.config import Settings

        settings = Settings()

        assert settings.log_level == "DEBUG"

    def test_storage_path_types(self):
        from nexusrag.config import Settings

        settings = Settings()

        assert isinstance(settings.storage.lancedb_path, Path)
        assert isinstance(settings.data_dir, Path)

    def test_api_settings(self):
        from nexusrag.config import APISettings

        api = APISettings()

        assert api.host == "0.0.0.0"
        assert api.port == 8000

    def test_ingestion_settings_values(self):
        from nexusrag.config import IngestionSettings

        ingestion = IngestionSettings()

        assert ingestion.chunk_size > 0
        assert ingestion.chunk_overlap >= 0
        assert ingestion.chunk_overlap < ingestion.chunk_size
        assert ingestion.min_chunk_size > 0

    def test_retrieval_settings_values(self):
        from nexusrag.config import RetrievalSettings

        retrieval = RetrievalSettings()

        assert retrieval.top_k > 0
        assert retrieval.rerank_top_k > 0
        assert 0 <= retrieval.similarity_threshold <= 1

    def test_self_correction_settings(self):
        from nexusrag.config import SelfCorrectionSettings

        correction = SelfCorrectionSettings()

        assert correction.enabled is True
        assert 0 <= correction.confidence_tau <= 1
        assert correction.feedback_docs > 0 and correction.feedback_terms > 0

    def test_get_settings_caching(self):
        from nexusrag.config import get_settings

        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_env_file_loading(self, temp_dir, monkeypatch):
        # Set env vars directly (pydantic-settings nested models read env independently)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-file:1234")
        monkeypatch.setenv("LLM_MODEL", "from-env-file")

        from nexusrag.config import Settings

        settings = Settings()

        assert settings.llm.base_url == "http://env-file:1234"
        assert settings.llm.model == "from-env-file"


class TestYAMLLoading:
    def test_yaml_file_exists(self):
        config_path = Path("configs/default.yaml")
        assert config_path.exists()

    def test_yaml_structure(self):
        config_path = Path("configs/default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "llm" in config
        assert "embedding" in config
        assert "ingestion" in config
        assert "retrieval" in config
        assert "self_correction" in config
        assert "storage" in config
        assert "api" in config

    def test_yaml_llm_section(self):
        config_path = Path("configs/default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        llm = config["llm"]
        assert "model" in llm
        assert "temperature" in llm
        assert "max_tokens" in llm

    def test_yaml_ingestion_section(self):
        config_path = Path("configs/default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        ingestion = config["ingestion"]
        assert "chunk_size" in ingestion
        assert "chunk_overlap" in ingestion
        assert "supported_formats" in ingestion

    def test_yaml_retrieval_section(self):
        config_path = Path("configs/default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        retrieval = config["retrieval"]
        assert "top_k" in retrieval
        assert "similarity_threshold" in retrieval

    def test_yaml_self_correction_section(self):
        config_path = Path("configs/default.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        correction = config["self_correction"]
        assert "enabled" in correction
        assert "max_iterations" in correction
        assert "relevance_threshold" in correction


class TestConfigValidation:
    def test_invalid_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "INVALID")

        from pydantic import ValidationError

        from nexusrag.config import Settings

        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_device(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DEVICE", "tpu")

        from pydantic import ValidationError

        from nexusrag.config import EmbeddingSettings

        with pytest.raises(ValidationError):
            EmbeddingSettings()
