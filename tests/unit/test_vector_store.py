"""VectorStore search uses cosine similarity, not raw L2 distance."""

from types import SimpleNamespace
from unittest.mock import MagicMock

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


@pytest.mark.parametrize("operation", ["delete", "delete_by_document", "search_by_document"])
def test_trailing_newline_id_is_rejected_before_database_access(temp_dir, operation) -> None:
    store = _store(temp_dir)

    with pytest.raises(ValueError, match="Invalid ID format"):
        if operation == "delete":
            store.delete(["paper\n"])
        elif operation == "delete_by_document":
            store.delete_by_document("paper\n")
        else:
            store.search_by_document(np.ones(4, dtype=np.float32), "paper\n")

    assert store._db is None


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


@pytest.mark.parametrize("existing", [False, True])
def test_legacy_connection_can_create_reopen_and_clear_tables(temp_dir, existing) -> None:
    store = _store(temp_dir)
    vectors = np.array([[1, 0, 0, 0]], dtype=np.float32)
    if existing:
        store.add([Chunk(id="old", content="Original passage", document_id="paper")], vectors)

    class LegacyConnection:
        """Expose the table_names interface used by LanceDB 0.17."""

        def __init__(self, connection):
            self.connection = connection

        def table_names(self):
            return self.connection.list_tables().tables

        def __getattr__(self, name):
            if name == "list_tables":
                raise AttributeError(name)
            return getattr(self.connection, name)

    reloaded = _store(temp_dir)
    reloaded._db = LegacyConnection(store.db)
    reloaded.add([Chunk(id="new", content="New passage", document_id="paper")], vectors)

    assert reloaded.count() == 1 + existing
    assert {result.chunk.id for result in reloaded.search(vectors[0])} == (
        {"old", "new"} if existing else {"new"}
    )
    reloaded.clear()
    assert "chunks" not in store.db.list_tables().tables
    assert reloaded.count() == 0


@pytest.mark.parametrize("operation", ["open", "clear"])
def test_current_connection_finds_tables_on_later_pages(temp_dir, operation) -> None:
    store = _store(temp_dir)
    database = MagicMock(spec=["list_tables", "open_table", "create_table", "drop_table"])
    pages = {
        None: SimpleNamespace(tables=["another_table"], page_token="next_page"),
        "next_page": SimpleNamespace(tables=["chunks"], page_token=None),
    }
    database.list_tables.side_effect = lambda *, page_token: pages[page_token]
    store._db = database

    if operation == "open":
        assert store.table is database.open_table.return_value
        database.open_table.assert_called_once_with("chunks")
        database.create_table.assert_not_called()
    else:
        store.clear()
        database.drop_table.assert_called_once_with("chunks")
