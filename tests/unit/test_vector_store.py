"""Vector storage, cosine search and persisted embedding configuration."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import MagicMock

import lancedb
import numpy as np
import pyarrow as pa
import pytest

from scinexusrag.ingestion import Chunk
from scinexusrag.ingestion.embedder import EmbeddingConfig, EmbeddingConfigError
from scinexusrag.retrieval.dense import DenseRetriever
from scinexusrag.storage.vector_store import VectorStore, VectorStoreError


def _config(dim: int = 4, **changes) -> EmbeddingConfig:
    return replace(EmbeddingConfig("test-model", "a" * 40, "", "", True, dim), **changes)


def _store(tmp_path, dim: int = 4, **changes) -> VectorStore:
    return VectorStore(path=tmp_path / "lancedb", embedding_config=_config(dim, **changes))


class _ObservedLock:
    """Signal a contender's attempt before it waits on the real lock."""

    def __init__(self, lock):
        self.lock = lock
        self.contender = None
        self.attempted = Event()

    def __enter__(self):
        if get_ident() == self.contender:
            self.attempted.set()
        if not self.lock.acquire(timeout=5):
            raise TimeoutError("Vector store lock was not released")
        return self

    def __exit__(self, *_args):
        self.lock.release()

    def run_contender(self, operation):
        self.contender = get_ident()
        return operation()


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


@pytest.mark.parametrize(
    "operation",
    [
        "delete",
        "delete_by_document",
        "search_by_document",
        "get_chunks_by_document",
        "get_chunk_ids_by_document",
    ],
)
def test_trailing_newline_id_is_rejected_before_database_access(temp_dir, operation) -> None:
    store = _store(temp_dir)

    with pytest.raises(ValueError, match="Invalid ID format"):
        if operation == "delete":
            store.delete(["paper\n"])
        elif operation == "delete_by_document":
            store.delete_by_document("paper\n")
        elif operation == "search_by_document":
            store.search_by_document(np.ones(4, dtype=np.float32), "paper\n")
        else:
            getattr(store, operation)("paper\n")

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
        start_offset=120,
        end_offset=144,
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
    assert (restored.start_offset, restored.end_offset) == (chunk.start_offset, chunk.end_offset)
    assert restored.metadata["start_offset"] == chunk.start_offset
    assert restored.metadata["end_offset"] == chunk.end_offset
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
    schema = store.table.schema
    store._table = None
    database = MagicMock(spec=["list_tables", "open_table", "create_table", "drop_table"])
    database.open_table.return_value.schema = schema
    database.open_table.return_value.count_rows.return_value = 0
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


def test_embedding_configuration_is_persisted_and_reopened(temp_dir) -> None:
    config = _config(query_prefix="query: ", doc_prefix="passage: ", normalize=False)
    store = VectorStore(path=temp_dir / "lancedb", embedding_config=config)
    chunk = Chunk(id="stored", content="Stored passage", document_id="paper")
    store.add([chunk], np.array([[1, 0, 0, 0]], dtype=np.float32))

    reloaded = VectorStore(path=store.path, embedding_config=config)
    metadata = reloaded.table.schema.metadata

    assert json.loads(metadata[b"nexusrag.embedding_config"]) == {"version": 1, **asdict(config)}
    assert reloaded.embedding_dim == config.dimension
    assert reloaded.count() == 1
    assert reloaded.search(np.array([1, 0, 0, 0], dtype=np.float32))[0].chunk.id == "stored"


@pytest.mark.parametrize(
    "change",
    [
        {"model_name": "another-model"},
        {"revision": "b" * 40},
        {"query_prefix": "query: "},
        {"doc_prefix": "passage: "},
        {"normalize": False},
        {"dimension": 3},
    ],
    ids=["model", "revision", "query-prefix", "document-prefix", "normalization", "dimension"],
)
@pytest.mark.parametrize("operation", ["search", "add"])
def test_changed_configuration_rejects_before_vector_query_or_write(
    temp_dir, monkeypatch, change, operation
) -> None:
    original = _store(temp_dir)
    original.add(
        [Chunk(id="original", content="Original passage", document_id="paper")],
        np.ones((1, 4), dtype=np.float32),
    )
    table = original.table
    original_metadata = table.schema.metadata
    search = MagicMock(side_effect=AssertionError("queried incompatible vectors"))
    add = MagicMock(side_effect=AssertionError("wrote incompatible vectors"))
    monkeypatch.setattr(table, "search", search)
    monkeypatch.setattr(table, "add", add)
    reloaded = VectorStore(path=original.path, embedding_config=replace(_config(), **change))
    reloaded._db = MagicMock(wraps=original.db)
    reloaded._db.open_table.return_value = table

    for _ in range(2):
        with pytest.raises(EmbeddingConfigError):
            if operation == "search":
                reloaded.search(np.ones(reloaded.embedding_dim, dtype=np.float32))
            else:
                reloaded.add(
                    [Chunk(id="new", content="New passage", document_id="another")],
                    np.ones((1, reloaded.embedding_dim), dtype=np.float32),
                )
        assert reloaded._table is None

    search.assert_not_called()
    add.assert_not_called()
    assert table.count_rows() == 1
    assert table.schema.metadata == original_metadata


@pytest.mark.parametrize(
    "operation",
    [
        "count",
        "list_documents",
        "get_all_chunks",
        "get_chunks_by_document",
        "get_chunk_ids_by_document",
    ],
)
def test_read_helpers_propagate_configuration_errors(temp_dir, operation) -> None:
    original = _store(temp_dir)
    original.add(
        [Chunk(id="original", content="Original passage", document_id="paper")],
        np.ones((1, 4), dtype=np.float32),
    )
    changed = _store(temp_dir, revision="b" * 40)

    with pytest.raises(EmbeddingConfigError):
        if operation in {"get_chunks_by_document", "get_chunk_ids_by_document"}:
            getattr(changed, operation)("paper")
        else:
            getattr(changed, operation)()

    assert original.count() == 1


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_count_propagates_storage_failure_instead_of_reporting_empty(
    temp_dir, monkeypatch, error_type
) -> None:
    store = _store(temp_dir)
    table = store.table
    failure = error_type("Cannot read row count")
    monkeypatch.setattr(table, "count_rows", MagicMock(side_effect=failure))

    with pytest.raises(VectorStoreError, match="row count") as raised:
        store.count()

    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("chunk_count", [0, 3, 4, 5, 8])
def test_document_scans_cross_page_boundaries_without_losing_rows(
    temp_dir, monkeypatch, chunk_count
) -> None:
    monkeypatch.setattr("scinexusrag.storage.vector_store.SCAN_PAGE_SIZE", 4)
    store = _store(temp_dir)
    chunks = [
        Chunk(id=f"paper-{i}", content=f"Measurement {i}", document_id="paper")
        for i in range(chunk_count)
    ]
    unrelated = Chunk(id="other", content="Different document", document_id="other")
    store.add([unrelated, *chunks], np.ones((chunk_count + 1, 4), dtype=np.float32))

    assert {chunk.id for chunk in store.get_chunks_by_document("paper")} == {
        chunk.id for chunk in chunks
    }
    assert store.get_chunk_ids_by_document("paper") == {chunk.id for chunk in chunks}
    assert {chunk.id for chunk in store.get_all_chunks()} == {"other", *(c.id for c in chunks)}
    assert set(store.list_documents()) == ({"other", "paper"} if chunks else {"other"})
    assert store.get_chunks_by_document("missing") == []
    assert store.get_chunk_ids_by_document("missing") == set()


def test_document_scans_include_chunks_beyond_the_previous_100000_limit(temp_dir) -> None:
    store = _store(temp_dir)
    count = 100_001
    ids = [f"chunk-{i}" for i in range(count)]
    store.table.add(
        pa.table(
            {
                "id": ids,
                "content": ["Measured current is 42 mA."] * count,
                "document_id": ["large-paper"] * count,
                "metadata": ["{}"] * count,
                "vector": [[1.0, 0.0, 0.0, 0.0]] * count,
            },
            schema=store.table.schema,
        )
    )

    assert store.get_chunk_ids_by_document("large-paper") == set(ids)
    assert {chunk.id for chunk in store.get_chunks_by_document("large-paper")} == set(ids)


@pytest.mark.parametrize(
    "operation",
    ["list_documents", "get_all_chunks", "get_chunks_by_document", "get_chunk_ids_by_document"],
)
def test_failed_scan_does_not_report_an_empty_or_partial_corpus(
    temp_dir, monkeypatch, operation
) -> None:
    store = _store(temp_dir)
    table = store.table
    failure = OSError("Storage is unavailable")
    monkeypatch.setattr(table, "search", MagicMock(side_effect=failure))

    with pytest.raises(VectorStoreError, match="read passages") as raised:
        if operation in {"get_chunks_by_document", "get_chunk_ids_by_document"}:
            getattr(store, operation)("paper")
        else:
            getattr(store, operation)()

    assert raised.value.__cause__ is failure


def test_failure_on_a_later_scan_page_does_not_return_a_partial_document(
    temp_dir, monkeypatch
) -> None:
    store = _store(temp_dir)
    monkeypatch.setattr("scinexusrag.storage.vector_store.SCAN_PAGE_SIZE", 1)
    store.add(
        [Chunk(id="a", content="First measurement", document_id="paper")],
        np.ones((1, 4), dtype=np.float32),
    )
    table = store.table
    search = table.search
    calls = 0

    def fail_on_second_page(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Cannot read the next page")
        return search(*args, **kwargs)

    monkeypatch.setattr(table, "search", fail_on_second_page)
    with pytest.raises(VectorStoreError, match="read passages"):
        store.get_chunks_by_document("paper")

    assert calls == 2


def test_vector_search_failure_is_distinct_from_empty_results(temp_dir, monkeypatch) -> None:
    store = _store(temp_dir)
    failure = OSError("Storage is unavailable")
    monkeypatch.setattr(store.table, "search", MagicMock(side_effect=failure))

    with pytest.raises(VectorStoreError, match="search the vector index") as raised:
        store.search(np.ones(4, dtype=np.float32))

    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("metadata", ["not JSON", "[]", "null"])
@pytest.mark.parametrize("operation", ["get_all_chunks", "get_chunks_by_document", "search"])
def test_corrupt_passage_metadata_is_reported_as_a_storage_error(
    temp_dir, metadata, operation
) -> None:
    store = _store(temp_dir)
    store.table.add(
        [
            {
                "id": "corrupt",
                "content": "Stored measurement",
                "document_id": "paper",
                "metadata": metadata,
                "vector": [1, 0, 0, 0],
            }
        ]
    )

    with pytest.raises(VectorStoreError, match="decode passage metadata"):
        if operation == "get_chunks_by_document":
            store.get_chunks_by_document("paper")
        elif operation == "search":
            store.search(np.ones(4, dtype=np.float32))
        else:
            store.get_all_chunks()


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize(
    "metadata_kind", ["missing", "incomplete", "corrupt", "unsupported-version"]
)
def test_unbound_or_invalid_tables_are_never_adopted(temp_dir, populated, metadata_kind) -> None:
    store = _store(temp_dir)
    schema = store.table.schema.remove_metadata()
    store.clear()
    config_data = {"version": 1, **asdict(_config())}
    if metadata_kind == "incomplete":
        del config_data["doc_prefix"]
    elif metadata_kind == "unsupported-version":
        config_data["version"] = 2
    if metadata_kind != "missing":
        encoded = b"not JSON" if metadata_kind == "corrupt" else json.dumps(config_data).encode()
        schema = schema.with_metadata({b"nexusrag.embedding_config": encoded})
    raw_table = store.db.create_table("chunks", schema=schema)
    if populated:
        raw_table.add(
            [
                {
                    "id": "old",
                    "content": "Old passage",
                    "document_id": "paper",
                    "metadata": "{}",
                    "vector": [1, 0, 0, 0],
                }
            ]
        )
    reloaded = _store(temp_dir)

    with pytest.raises(EmbeddingConfigError):
        reloaded.count()

    assert raw_table.count_rows() == int(populated)
    assert raw_table.schema.metadata == schema.metadata
    assert reloaded._table is None


def test_cached_table_configuration_is_checked_on_every_access(temp_dir) -> None:
    store = _store(temp_dir)
    table = store.table
    changed_schema = table.schema.with_metadata(
        {
            b"nexusrag.embedding_config": json.dumps(
                {"version": 1, **asdict(_config(revision="b" * 40))}
            ).encode()
        }
    )
    cached = MagicMock(wraps=table)
    cached.schema = changed_schema
    store._table = cached

    with pytest.raises(EmbeddingConfigError):
        store.search(np.ones(4, dtype=np.float32))

    cached.search.assert_not_called()


def test_vector_schema_dimension_must_match_verified_metadata(temp_dir) -> None:
    store = _store(temp_dir)
    schema = store.table.schema
    store.clear()
    incompatible = schema.set(
        schema.get_field_index("vector"), pa.field("vector", pa.list_(pa.float32(), 2))
    )
    store.db.create_table("chunks", schema=incompatible)

    with pytest.raises(EmbeddingConfigError):
        store.count()

    assert store._table is None


def test_active_embedder_configuration_must_match_store(temp_dir) -> None:
    store = _store(temp_dir)
    store.validate_embedding_config(_config())

    with pytest.raises(EmbeddingConfigError):
        store.validate_embedding_config(_config(query_prefix="different: "))

    assert store.count() == 0


@pytest.mark.parametrize("document_id", [None, "paper"])
def test_dense_retrieval_checks_active_configuration_before_embedding(
    temp_dir, document_id
) -> None:
    store = _store(temp_dir)
    embedder = MagicMock()
    embedder.embedding_config = _config(query_prefix="another query prefix: ")
    retriever = DenseRetriever(embedder, store)

    with pytest.raises(EmbeddingConfigError):
        retriever.retrieve("Which measurement?", document_id=document_id)

    embedder.embed_query.assert_not_called()
    assert store._db is None


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("populated", [False, True])
def test_explicit_clear_recovers_rejected_table_with_a_new_dimension(
    temp_dir, legacy, populated
) -> None:
    original = _store(temp_dir)
    schema = original.table.schema
    if legacy:
        original.clear()
        raw_table = original.db.create_table("chunks", schema=schema.remove_metadata())
    else:
        raw_table = original.table
    if populated:
        raw_table.add(
            [
                {
                    "id": "old",
                    "content": "Old passage",
                    "document_id": "paper",
                    "metadata": "{}",
                    "vector": [1, 0, 0, 0],
                }
            ]
        )
    changed = _store(temp_dir, dim=2, model_name="new-model")

    with pytest.raises(EmbeddingConfigError):
        changed.count()

    changed.clear()
    changed.add(
        [Chunk(id="new", content="New passage", document_id="new-paper")],
        np.array([[1, 0]], dtype=np.float32),
    )
    reopened = _store(temp_dir, dim=2, model_name="new-model")
    assert reopened.count() == 1
    assert reopened.table.schema.field("vector").type.list_size == 2
    assert reopened.list_documents() == ["new-paper"]
    assert reopened.search(np.array([1, 0], dtype=np.float32))[0].chunk.id == "new"


@pytest.mark.parametrize("operation", ["count", "search", "add"])
@pytest.mark.parametrize("extra_old_write", [False, True], ids=["equal-version", "lower-version"])
def test_cached_handle_rejects_a_recreated_collection(temp_dir, operation, extra_old_write) -> None:
    original = _store(temp_dir)
    vector = np.array([[1, 0, 0, 0]], dtype=np.float32)
    original.add([Chunk(id="old", content="Old passage", document_id="old-paper")], vector)
    if extra_old_write:
        original.add(
            [Chunk(id="old-more", content="Another old passage", document_id="old-paper")], vector
        )
    original_version = original.table.version
    replacement = _store(temp_dir, model_name="another-model")
    replacement.clear()
    replacement.add([Chunk(id="new", content="New passage", document_id="new-paper")], vector)
    assert replacement.table.version <= original_version

    with pytest.raises(EmbeddingConfigError):
        if operation == "count":
            original.count()
        elif operation == "search":
            original.search(vector[0])
        else:
            original.add(
                [Chunk(id="stale", content="Stale writer", document_id="stale-paper")], vector
            )

    assert replacement.count() == 1
    assert replacement.list_documents() == ["new-paper"]


@pytest.mark.parametrize("bootstrap_stage", ["connect", "create_table"])
def test_concurrent_first_access_connects_and_creates_once(temp_dir, monkeypatch, bootstrap_stage):
    store = _store(temp_dir)
    observed = _ObservedLock(store._init_lock)
    store._init_lock = observed
    entered = Event()
    release = Event()
    original = lancedb.connect if bootstrap_stage == "connect" else store.db.create_table

    def held_bootstrap(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Bootstrap was not released")
        return original(*args, **kwargs)

    bootstrap = MagicMock(side_effect=held_bootstrap)
    if bootstrap_stage == "connect":
        monkeypatch.setattr(lancedb, "connect", bootstrap)
    else:
        monkeypatch.setattr(store.db, "create_table", bootstrap)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(store.count)
        try:
            assert entered.wait(timeout=5)
            second = workers.submit(observed.run_contender, store.count)
            assert observed.attempted.wait(timeout=5)
            assert bootstrap.call_count == 1
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5) == 0

    assert bootstrap.call_count == 1
    assert store.db.list_tables().tables == ["chunks"]


@pytest.mark.parametrize("bootstrap_stage", ["connect", "create_table"])
def test_failed_bootstrap_releases_lock_and_can_retry(temp_dir, monkeypatch, bootstrap_stage):
    store = _store(temp_dir)
    original = lancedb.connect if bootstrap_stage == "connect" else store.db.create_table

    def fail_once(*args, **kwargs):
        if bootstrap.call_count == 1:
            raise OSError("Storage temporarily unavailable")
        return original(*args, **kwargs)

    bootstrap = MagicMock(side_effect=fail_once)
    if bootstrap_stage == "connect":
        monkeypatch.setattr(lancedb, "connect", bootstrap)
    else:
        monkeypatch.setattr(store.db, "create_table", bootstrap)

    with pytest.raises(VectorStoreError) as raised:
        store.count()

    assert isinstance(raised.value.__cause__, OSError)
    assert store._table is None
    if bootstrap_stage == "connect":
        assert store._db is None
    assert store.count() == 0
    assert bootstrap.call_count == 2
    assert json.loads(store.table.schema.metadata[b"nexusrag.embedding_config"])["version"] == 1


def test_clear_waits_for_table_bootstrap_then_removes_the_published_table(temp_dir, monkeypatch):
    store = _store(temp_dir)
    observed = _ObservedLock(store._init_lock)
    store._init_lock = observed
    entered = Event()
    release = Event()
    original_create = store.db.create_table

    def held_create(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Table creation was not released")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(store.db, "create_table", held_create)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(lambda: store.table)
        try:
            assert entered.wait(timeout=5)
            cleared = workers.submit(observed.run_contender, store.clear)
            assert observed.attempted.wait(timeout=5)
        finally:
            release.set()
        assert first.result(timeout=5) is not None
        assert cleared.result(timeout=5) is None

    assert store._table is None
    assert "chunks" not in store.db.list_tables().tables
    assert store.count() == 0


def test_failed_clear_keeps_cached_data_and_allows_retry(temp_dir, monkeypatch):
    store = _store(temp_dir)
    store.add(
        [Chunk(id="kept", content="Stored passage", document_id="paper")],
        np.ones((1, 4), dtype=np.float32),
    )
    original_drop = store.db.drop_table
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("Cannot remove table")
        return original_drop(*args, **kwargs)

    monkeypatch.setattr(store.db, "drop_table", fail_once)
    with pytest.raises(OSError):
        store.clear()

    assert store.count() == 1
    store.clear()
    assert store._table is None
    assert "chunks" not in store.db.list_tables().tables
