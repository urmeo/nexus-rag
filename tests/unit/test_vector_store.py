"""VectorStore search uses cosine similarity, not raw L2 distance."""

import numpy as np
import pytest

from scinexusrag.ingestion import Chunk
from scinexusrag.storage.vector_store import VectorStore


def _store(tmp_path, dim: int = 4) -> VectorStore:
    return VectorStore(path=tmp_path / "lancedb", embedding_dim=dim)


def test_search_scores_are_cosine_similarity(temp_dir) -> None:

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
    assert results["b"] > 0.99
    assert abs(results["c"]) < 0.01


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


def test_search_score_clamped_to_unit_interval(temp_dir) -> None:

    store = _store(temp_dir)
    chunks = [Chunk(id="opp", content="opposite", document_id="opp")]
    store.add(chunks, np.array([[-1, 0, 0, 0]], dtype=np.float32))

    results = store.search(np.array([1, 0, 0, 0], np.float32), top_k=1)

    assert results[0].score == 0.0


@pytest.mark.parametrize("access", ["search", "get_all_chunks", "get_chunks_by_document"])
def test_chunk_context_and_citation_metadata_survive_reload(temp_dir, access) -> None:
    store = _store(temp_dir)
    chunk = Chunk(
        id="context",
        content="The sample was measured.",
        document_id="paper",
        metadata={"file_type": "pdf"},
        document_name="Paper.pdf",
        section_title="Methods",
        page_number=3,
        chunk_index=7,
        context_before="The sample contained twelve participants.",
        context_after="All measurements were blinded.",
    )
    store.add([chunk], np.array([[1, 0, 0, 0]], dtype=np.float32))

    reloaded = _store(temp_dir)
    if access == "search":
        restored = reloaded.search(np.array([1, 0, 0, 0], np.float32))[0].chunk
    elif access == "get_all_chunks":
        restored = reloaded.get_all_chunks()[0]
    else:
        restored = reloaded.get_chunks_by_document("paper")[0]

    assert restored.full_context == chunk.full_context
    assert restored.document_name == "Paper.pdf"
    assert restored.section_title == "Methods"
    assert restored.page_number == 3
    assert restored.chunk_index == 7
    assert restored.metadata["file_type"] == "pdf"
    assert reloaded.list_documents() == ["paper"]
