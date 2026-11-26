"""Settings checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from scinexusrag.settings import (
    HF_REVISIONS,
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    SelfCorrectionSettings,
    Settings,
)


@pytest.fixture
def source_environment():
    source_dir = Path(__file__).resolve().parents[1] / "src"
    python_path = os.pathsep.join([str(source_dir), os.environ.get("PYTHONPATH", "")])
    return os.environ | {"PYTHONPATH": python_path}


class TestSettingsPrecedence:
    def test_env_var_overrides_default(self, monkeypatch):

        monkeypatch.setenv("LLM_MODEL", "env-wins")
        assert Settings().llm.model == "env-wins"

    def test_llm_settings_ignore_bare_env_names(self, monkeypatch):

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


class TestSettingsValidation:
    @pytest.mark.parametrize(
        "env_name",
        [
            "LLM_MAX_TOKENS",
            "LLM_TIMEOUT",
            "EMBEDDING_BATCH_SIZE",
            "INGESTION_CHUNK_SIZE",
            "INGESTION_MIN_CHUNK_SIZE",
            "INGESTION_MAX_CHUNKS",
            "RETRIEVAL_TOP_K",
            "API_QUERY_RATE_PER_MINUTE",
            "API_UPLOAD_RATE_PER_MINUTE",
        ],
    )
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_invalid_runtime_limits_rejected_from_environment(self, monkeypatch, env_name, value):

        monkeypatch.setenv(env_name, value)
        with pytest.raises(ValidationError):
            Settings()

    @pytest.mark.parametrize(
        "overrides",
        [{"chunk_overlap": -1}, {"chunk_overlap": 100}, {"min_chunk_size": 101}],
    )
    def test_invalid_chunk_relationships_rejected(self, overrides):

        kwargs = {"chunk_size": 100, "chunk_overlap": 0, "min_chunk_size": 20, **overrides}
        with pytest.raises(ValidationError):
            IngestionSettings(**kwargs)

    def test_zero_overlap_and_equal_minimum_are_valid(self):

        assert IngestionSettings(chunk_size=100, chunk_overlap=0, min_chunk_size=100)
        assert LLMSettings(temperature=0).temperature == 0

    @pytest.mark.parametrize("field", ["confidence_tau", "grounding_threshold"])
    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_invalid_probability_thresholds_rejected(self, field, value):

        with pytest.raises(ValidationError):
            SelfCorrectionSettings(**{field: value})

    def test_embedding_revision_matches_selected_model(self, monkeypatch):

        model = "sentence-transformers/all-MiniLM-L6-v2"
        monkeypatch.setenv("EMBEDDING_MODEL", model)
        monkeypatch.delenv("EMBEDDING_REVISION", raising=False)

        assert EmbeddingSettings().revision == HF_REVISIONS[model]
        assert EmbeddingSettings(model="custom-model").revision is None

    def test_explicit_embedding_revision_is_preserved(self, monkeypatch):

        monkeypatch.setenv("EMBEDDING_REVISION", "custom-sha")

        assert EmbeddingSettings(model="custom-model").revision == "custom-sha"

    def test_invalid_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "INVALID")

        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_device(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DEVICE", "tpu")

        with pytest.raises(ValidationError):
            EmbeddingSettings()


class TestSettingsLoading:
    def test_env_file_in_cwd_is_loaded(self, tmp_path, source_environment):
        (tmp_path / ".env").write_text("LLM_MODEL=dotenv-model\n")
        env = {k: v for k, v in source_environment.items() if k != "LLM_MODEL"}
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scinexusrag.settings import Settings; print(Settings().llm.model)",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "dotenv-model"

    def test_model_revisions_do_not_validate_unrelated_settings(self, tmp_path, source_environment):

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from scinexusrag.settings import HF_REVISIONS; "
                "print(HF_REVISIONS['BAAI/bge-small-en-v1.5'])",
            ],
            cwd=tmp_path,
            env=source_environment | {"LLM_MAX_TOKENS": "0"},
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == HF_REVISIONS["BAAI/bge-small-en-v1.5"]
