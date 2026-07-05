"""Tests for Embedder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nexusrag.ingestion import Embedder


class TestEmbedder:
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

            # Model should not be loaded yet
            mock_cls.assert_not_called()

            # Access model property triggers loading
            _ = embedder.model

            mock_cls.assert_called_once_with("test-model", device=None, revision=None)

    def test_device_configuration(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_cls.return_value = MagicMock()
            embedder = Embedder(model_name="test-model", device="cuda")

            _ = embedder.model

            mock_cls.assert_called_once_with("test-model", device="cuda", revision=None)

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

        # Manually normalize for test
        doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

        similarities = embedder.similarity(query_emb, doc_embs)

        assert similarities.shape == (3,)
        assert similarities[0] == pytest.approx(1.0, abs=0.01)  # Identical
        assert similarities[1] == pytest.approx(0.0, abs=0.01)  # Orthogonal

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
        assert similarities[0] == pytest.approx(1.0, abs=0.01)  # Same direction
        assert similarities[1] == pytest.approx(0.0, abs=0.01)  # Orthogonal

    def test_show_progress_flag(self, mock_sentence_transformer):
        embedder = Embedder(model_name="test-model")
        texts = ["text"] * 5

        embedder.embed(texts, show_progress=True)

        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["show_progress_bar"] is True
