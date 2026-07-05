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

        assert api.host == "127.0.0.1"
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
        assert retrieval.max_query_length > 0

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


class TestConfigPrecedence:
    """Config is environment-variable driven; configs/default.yaml is a reference."""

    def test_env_var_overrides_default(self, monkeypatch):
        from nexusrag.config import Settings

        monkeypatch.setenv("LLM_MODEL", "env-wins")
        assert Settings().llm.model == "env-wins"

    def test_llm_settings_ignore_bare_env_names(self, monkeypatch):
        # temperature/timeout must read LLM_TEMPERATURE/LLM_TIMEOUT, never bare
        # TEMPERATURE/TIMEOUT which collide with unrelated environment vars.
        from nexusrag.config import LLMSettings

        monkeypatch.setenv("TEMPERATURE", "0.99")
        monkeypatch.setenv("TIMEOUT", "999")
        s = LLMSettings()
        assert s.temperature == 0.1
        assert s.timeout == 60

        monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("LLM_TIMEOUT", "33")
        s = LLMSettings()
        assert s.temperature == 0.7
        assert s.timeout == 33

    def test_yaml_is_not_a_runtime_source(self, monkeypatch):
        # default.yaml is documentation only: settings come from env + code
        # defaults, never from the YAML file, so there is no precedence ambiguity.
        from nexusrag.config import Settings

        monkeypatch.delenv("LLM_MODEL", raising=False)
        assert Settings().llm.model == "llama3.2:3b"  # the code default, not YAML


class TestYAMLDocMatchesSettings:
    """default.yaml is a reference doc, not loaded at runtime; guard only that
    it does not silently drift from the real Settings model."""

    def test_documented_sections_match_settings_submodels(self):
        from pydantic import BaseModel

        from nexusrag.config import Settings

        with open("configs/default.yaml") as f:
            documented = set(yaml.safe_load(f))

        real = {
            name
            for name, field in Settings.model_fields.items()
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)
        }
        # 'logging' documents the flat log_level/data_dir fields.
        assert documented - {"logging"} == real


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


class TestNoDeadConfig:
    def test_every_settings_field_is_read_in_src(self) -> None:
        # Guards the recurring bug class: a config knob that nothing consumes.
        import re

        from pydantic import BaseModel

        import nexusrag
        from nexusrag.config import Settings

        src = Path(nexusrag.__file__).parent
        code = "\n".join(p.read_text() for p in src.rglob("*.py") if p.name != "config.py")

        dead: list[str] = []
        for name, field in Settings.model_fields.items():
            ann = field.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                dead += [
                    f"{name}.{f}" for f in ann.model_fields if not re.search(rf"\b{f}\b", code)
                ]
            elif not re.search(rf"\b{name}\b", code):
                dead.append(name)
        assert not dead, f"config fields never read in src/: {dead}"
