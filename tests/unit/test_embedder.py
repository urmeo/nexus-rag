"""Tests for Embedder."""

import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Barrier, Event, Lock, get_ident
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scinexusrag.config import HF_REVISIONS
from scinexusrag.ingestion import Embedder
from scinexusrag.ingestion.embedder import EmbeddingConfig, EmbeddingConfigError


class _ObservedLock:
    def __init__(self):
        self.lock = Lock()
        self.attempts = Queue()

    def __enter__(self):
        self.attempts.put(get_ident())
        if not self.lock.acquire(timeout=5):
            raise TimeoutError("Model lock was not released")
        return self

    def __exit__(self, *_args):
        self.lock.release()

    def wait_for_attempts(self, count):
        return [self.attempts.get(timeout=5) for _ in range(count)]


class TestEmbedder:
    @pytest.fixture(autouse=True)
    def sentence_transformers_module(self, monkeypatch):
        module = ModuleType("sentence_transformers")
        module.SentenceTransformer = MagicMock()
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_model = MagicMock()
            mock_model.get_embedding_dimension.return_value = 384
            mock_model.get_sentence_embedding_dimension.return_value = 384

            def mock_encode(texts, **kwargs):
                if isinstance(texts, str):
                    result = np.random.rand(384).astype(np.float32)
                else:
                    result = np.random.rand(len(texts), 384).astype(np.float32)
                return result

            mock_model.encode.side_effect = mock_encode
            mock_cls.return_value = mock_model
            yield mock_model

    def test_embed_single_text(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = ["This is a test sentence."]

        embeddings = embedder.embed(texts)

        assert embeddings.shape == (1, 384)
        assert embeddings.dtype == np.float32

    def test_embed_batch(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]

        embeddings = embedder.embed(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
        mock_sentence_transformer.encode.assert_called_once()

    def test_embed_query(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        query = "What is the meaning of life?"

        embedding = embedder.embed_query(query)

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embedding_dimension(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")

        assert embedder.dimension == 384
        mock_sentence_transformer.get_embedding_dimension.assert_called()

    def test_empty_list(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")

        embeddings = embedder.embed([])

        assert embeddings.shape == (0, 384)

    def test_batch_size_passed(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = ["text"] * 10

        embedder.embed(texts, batch_size=16)

        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["batch_size"] == 16

    def test_normalization_default(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model", normalize=True)
        texts = ["Test text"]

        embedder.embed(texts)

        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is True

    def test_normalization_disabled(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model", normalize=False)
        texts = ["Test text"]

        embedder.embed(texts)

        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is False

    def test_lazy_model_loading(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            embedder = Embedder(model_name="test-model")

            mock_cls.assert_not_called()

            _ = embedder.model

            mock_cls.assert_called_once_with("test-model", device=None, revision=None)

    def test_device_configuration(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = MagicMock()
            embedder = Embedder(model_name="test-model", device="cuda")

            _ = embedder.model

            mock_cls.assert_called_once_with("test-model", device="cuda", revision=None)

    def test_concurrent_model_first_access_constructs_once(self):
        embedder = Embedder(model_name="test-model", revision="a" * 40)
        lock = _ObservedLock()
        embedder._model_lock = lock
        start = Barrier(4)
        constructing = Event()
        release = Event()
        model = object()

        def construct(*_args, **_kwargs):
            constructing.set()
            assert release.wait(timeout=5)
            return model

        def load():
            start.wait(timeout=5)
            return embedder.model

        with (
            patch("sentence_transformers.SentenceTransformer", side_effect=construct) as factory,
            ThreadPoolExecutor(max_workers=4) as workers,
        ):
            futures = [workers.submit(load) for _ in range(4)]
            try:
                assert constructing.wait(timeout=5)
                assert len(set(lock.wait_for_attempts(4))) == 4
            finally:
                release.set()
            assert all(future.result(timeout=5) is model for future in futures)

        factory.assert_called_once_with("test-model", device=None, revision="a" * 40)
        assert embedder._model is model
        assert embedder._loaded_model_source == ("test-model", "a" * 40)

    def test_failed_constructor_can_retry_without_deadlock(self):
        embedder = Embedder(model_name="test-model", revision="a" * 40)
        embedder._model_lock = _ObservedLock()
        model = object()
        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=[RuntimeError("Constructor failed"), model],
        ) as factory:
            with pytest.raises(RuntimeError, match="Constructor failed"):
                _ = embedder.model
            assert embedder._model is None
            assert embedder._loaded_model_source is None

            with ThreadPoolExecutor(max_workers=1) as workers:
                assert workers.submit(lambda: embedder.model).result(timeout=5) is model
            assert embedder.model is model

        assert factory.call_count == 2
        assert embedder._loaded_model_source == ("test-model", "a" * 40)

    def test_unload_waits_for_initial_load_and_next_access_loads_fresh_model(self):
        embedder = Embedder(model_name="test-model", revision="a" * 40)
        lock = _ObservedLock()
        embedder._model_lock = lock
        constructing = Event()
        release = Event()
        original_model, fresh_model = object(), object()

        def construct(*_args, **_kwargs):
            constructing.set()
            assert release.wait(timeout=5)
            return original_model

        with (
            patch("sentence_transformers.SentenceTransformer", side_effect=construct) as factory,
            ThreadPoolExecutor(max_workers=2) as workers,
        ):
            loading = workers.submit(lambda: embedder.model)
            try:
                assert constructing.wait(timeout=5)
                lock.wait_for_attempts(1)
                unloading = workers.submit(embedder.unload)
                lock.wait_for_attempts(1)
                assert not unloading.done()
            finally:
                release.set()
            assert loading.result(timeout=5) is original_model
            unloading.result(timeout=5)
            assert embedder._model is None
            assert embedder._loaded_model_source is None

            factory.side_effect = None
            factory.return_value = fresh_model
            assert embedder.model is fresh_model
            assert embedder.model is fresh_model

        assert factory.call_count == 2
        assert embedder._loaded_model_source == ("test-model", "a" * 40)

    def test_similarity_normalized(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model", normalize=True)

        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        doc_embs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )

        doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

        similarities = embedder.similarity(query_emb, doc_embs)

        assert similarities.shape == (3,)
        assert similarities[0] == pytest.approx(1.0, abs=0.01)
        assert similarities[1] == pytest.approx(0.0, abs=0.01)

    def test_similarity_unnormalized(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model", normalize=False)

        query_emb = np.array([2.0, 0.0], dtype=np.float32)
        doc_embs = np.array(
            [
                [3.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        similarities = embedder.similarity(query_emb, doc_embs)

        assert similarities.shape == (2,)
        assert similarities[0] == pytest.approx(1.0, abs=0.01)
        assert similarities[1] == pytest.approx(0.0, abs=0.01)

    def test_show_progress_flag(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = ["text"] * 5

        embedder.embed(texts, show_progress=True)

        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["show_progress_bar"] is True

    def test_embedding_configuration_uses_effective_defaults(self, mock_sentence_transformer):
        embedder = Embedder()

        assert embedder.embedding_config == EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
            revision=HF_REVISIONS["BAAI/bge-small-en-v1.5"],
            query_prefix="Represent this sentence for searching relevant passages: ",
            doc_prefix="",
            normalize=True,
            dimension=384,
        )
        mock_sentence_transformer.encode.assert_not_called()

    def test_embedding_configuration_preserves_exact_overrides(self, mock_sentence_transformer):
        embedder = Embedder(
            model_name="test-model",
            revision="a" * 40,
            query_prefix="  query: ",
            doc_prefix="passage:  ",
            normalize=False,
        )

        assert embedder.embedding_config == EmbeddingConfig(
            "test-model", "a" * 40, "  query: ", "passage:  ", False, 384
        )

    @pytest.mark.parametrize("revision", [None, "main", "v1.0", "a" * 39, "g" * 40])
    def test_unknown_revision_rejects_before_model_loading(self, revision):
        with patch("sentence_transformers.SentenceTransformer") as model:
            embedder = Embedder(model_name="test-model", revision=revision)

            with pytest.raises(EmbeddingConfigError):
                _ = embedder.embedding_config

            model.assert_not_called()

    def test_local_model_path_rejects_even_with_commit_shaped_revision(self, tmp_path):
        with patch("sentence_transformers.SentenceTransformer") as model:
            embedder = Embedder(model_name=str(tmp_path), revision="a" * 40)

            with pytest.raises(EmbeddingConfigError):
                _ = embedder.embedding_config

            model.assert_not_called()

    @pytest.mark.parametrize("changed_field", ["model_name", "revision"])
    def test_loaded_weights_cannot_be_relabelled_before_unload(
        self, mock_sentence_transformer, changed_field
    ):
        embedder = Embedder(model_name="test-model", revision="a" * 40)
        original = embedder.embedding_config
        changed = "another-model" if changed_field == "model_name" else "b" * 40
        setattr(embedder, changed_field, changed)

        with pytest.raises(EmbeddingConfigError):
            _ = embedder.embedding_config
        mock_sentence_transformer.encode.assert_not_called()

        embedder.unload()
        reloaded = embedder.embedding_config
        assert getattr(reloaded, changed_field) == changed
        assert reloaded != original
