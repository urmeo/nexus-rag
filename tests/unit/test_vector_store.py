"""VectorStore search uses cosine similarity, not raw L2 distance."""

import numpy as np

from nexusrag.ingestion import Chunk
from nexusrag.storage.vector_store import VectorStore


def _store(tmp_path, dim: int = 4) -> VectorStore:
    return VectorStore(path=tmp_path / "lancedb", embedding_dim=dim)


def test_search_scores_are_cosine_similarity(temp_dir) -> None:
    # b points the same direction as the query but is 3x longer: cosine treats it
    # as identical (sim 1.0), L2 would not. This is the regression guard for the
    # old `1 - l2_distance` bug that mis-fed the corrective confidence gate.
    store = _store(temp_dir)
    chunks = [
        Chunk(id="a", content="aligned unit", document_id="a"),
        Chunk(id="b", content="aligned scaled", document_id="b"),
        Chunk(id="c", content="orthogonal", document_id="c"),
    ]
    vectors = np.array(
        [[1, 0, 0, 0], [3, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.float32,
    )
    store.add(chunks, vectors)

    results = {
        r.chunk.id: r.score for r in store.search(np.array([1, 0, 0, 0], np.float32), top_k=3)
    }

    assert results["a"] > 0.99
    assert results["b"] > 0.99  # same direction, larger magnitude -> still cosine 1.0
    assert abs(results["c"]) < 0.01  # orthogonal -> cosine 0


def test_search_ranks_by_similarity(temp_dir) -> None:
    store = _store(temp_dir)
    chunks = [
        Chunk(id="near", content="near", document_id="near"),
        Chunk(id="far", content="far", document_id="far"),
    ]
    vectors = np.array([[1, 0.1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    store.add(chunks, vectors)

    ranked = store.search(np.array([1, 0, 0, 0], np.float32), top_k=2)

    assert ranked[0].chunk.id == "near"
    assert ranked[0].score > ranked[1].score
