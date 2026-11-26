"""Storage checks."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import lancedb
import numpy as np
import pyarrow as pa
import pytest
from support import document, make_config, make_store

from scinexusrag.ingestion.chunker import Chunk
from scinexusrag.ingestion.embedder import EmbeddingConfig, EmbeddingConfigError
from scinexusrag.ingestion.parser import ParsedDocument
from scinexusrag.storage.document_store import (
    DocumentExistsError,
    DocumentStore,
    DocumentStoreError,
)
from scinexusrag.storage.vector_store import VectorStore, VectorStoreError


@pytest.fixture
def document_store(tmp_path):
    return DocumentStore(tmp_path)


def _doc(doc_id: str = "abc123def456", name: str = "paper.pdf") -> ParsedDocument:
    return ParsedDocument(
        id=doc_id,
        content="Some parsed content with several words.",
        metadata={"filename": name, "file_type": "pdf"},
    )


def test_orphaned_doc_file_recovered_on_load(tmp_path, document_store) -> None:

    document_store.add(_doc())
    (tmp_path / "_index.json").unlink()

    reloaded = DocumentStore(path=tmp_path)

    assert reloaded.exists("abc123def456")
    entry = reloaded.list_with_metadata()["abc123def456"]
    assert entry["filename"] == "paper.pdf"
    assert entry["word_count"] > 0
    assert (tmp_path / "_index.json").exists()


def test_dangling_index_entry_dropped_on_load(tmp_path, document_store) -> None:

    document_store.add(_doc())
    (tmp_path / "abc123def456.json").unlink()

    reloaded = DocumentStore(path=tmp_path)

    assert not reloaded.exists("abc123def456")
    assert reloaded.list_with_metadata() == {}


def test_delete_survives_reload(tmp_path, document_store) -> None:
    document_store.add(_doc())
    assert document_store.delete("abc123def456") is True

    reloaded = DocumentStore(path=tmp_path)

    assert not reloaded.exists("abc123def456")
    assert reloaded.list_with_metadata() == {}


def test_reserved_index_id_is_rejected(tmp_path, document_store) -> None:

    with pytest.raises(ValueError, match="reserved"):
        document_store.add(_doc(doc_id="_index"))


def test_trailing_newline_id_is_rejected_before_writing(tmp_path, document_store) -> None:

    with pytest.raises(ValueError, match="invalid characters"):
        document_store.add(_doc(doc_id="paper\n"))

    assert list(tmp_path.iterdir()) == []


def test_interrupted_index_update_cannot_restore_a_deleted_document(
    tmp_path, monkeypatch, document_store
) -> None:
    document_store.add(_doc())
    monkeypatch.setattr(
        document_store, "_save_index", Mock(side_effect=OSError("interrupted index write"))
    )

    with pytest.raises(OSError, match="interrupted index write"):
        document_store.delete("abc123def456")

    reloaded = DocumentStore(path=tmp_path)
    assert not reloaded.exists("abc123def456")
    assert reloaded.list_with_metadata() == {}


def test_failed_file_deletion_keeps_the_index_entry(tmp_path, monkeypatch, document_store) -> None:
    document_store.add(_doc())
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=OSError("file is unavailable")))

    with pytest.raises(OSError, match="file is unavailable"):
        document_store.delete("abc123def456")

    assert document_store.exists("abc123def456")
    assert DocumentStore(path=tmp_path).exists("abc123def456")


@pytest.mark.parametrize("operation", ["add", "delete", "update"])
def test_stats_use_one_snapshot_during_concurrent_writes(
    tmp_path, operation, document_store
) -> None:
    reading = Event()
    written = Event()

    class PausingCount(int):
        def __radd__(self, other):
            reading.set()
            assert written.wait(5), "Writer could not finish while statistics were calculated"
            return other + int(self)

    document_store.add(_doc("first"))
    document_store.add(_doc("second"))
    document_store.update_metadata("first", "word_count", PausingCount(7))
    document_store.update_metadata("first", "section_count", 1)
    document_store.update_metadata("second", "word_count", 2)
    document_store.update_metadata("second", "section_count", 2)

    def write():
        try:
            assert reading.wait(5), "Statistics never reached aggregation"
            if operation == "add":
                document_store.add(_doc("third"))
            elif operation == "delete":
                document_store.delete("second")
            else:
                document_store.update_metadata("second", "word_count", 99)
                document_store.update_metadata("second", "section_count", 99)
        finally:
            written.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(write)
        stats = document_store.get_stats()
        writer.result(timeout=5)

    assert stats == {"document_count": 2, "total_words": 9}


def test_metadata_snapshot_does_not_share_mutable_entries(tmp_path, document_store) -> None:
    document_store.add(_doc("first"))
    snapshot = document_store.list_with_metadata()
    initial_count = snapshot["first"]["word_count"]

    document_store.update_metadata("first", "word_count", 99)
    snapshot["first"]["filename"] = "changed.txt"

    assert snapshot["first"]["word_count"] == initial_count
    assert document_store.list_with_metadata()["first"]["filename"] == "paper.pdf"
    assert document_store.get_stats()["total_words"] == 99


@pytest.mark.parametrize("name", [" \t", {"invalid": "name"}])
@pytest.mark.parametrize("recover", [False, True])
def test_invalid_document_names_fall_back_when_added_or_recovered(
    tmp_path, name, recover, document_store
):
    parsed = document()
    parsed.metadata["filename"] = name
    if recover:
        (tmp_path / "paper.json").write_text(
            json.dumps({"id": parsed.id, "content": parsed.content, "metadata": parsed.metadata}),
            encoding="utf-8",
        )
    else:
        document_store.add(parsed)

    assert document_store.list_with_metadata()["paper"]["filename"] == "unknown"


def test_public_metadata_snapshots_do_not_expose_nested_store_state(tmp_path, document_store):
    document_store.add(document())
    document_store.update_metadata("paper", "notes", {"tags": ["original"]})

    snapshot = document_store.list_with_metadata()
    snapshot["paper"]["notes"]["tags"].append("changed")
    snapshot["paper"]["word_count"] = 99
    snapshot.pop("paper")

    assert len(document_store.list_with_metadata()) == 1
    assert document_store.list_with_metadata()["paper"]["notes"] == {"tags": ["original"]}
    assert document_store.get_stats()["total_words"] == 2
    assert DocumentStore(tmp_path).list_with_metadata() == document_store.list_with_metadata()


def test_metadata_updates_copy_caller_owned_values(tmp_path, document_store):
    document_store.add(document())
    value = {"tags": ["original"]}
    document_store.update_metadata("paper", "notes", value)
    value["tags"].append("changed")

    assert document_store.list_with_metadata()["paper"]["notes"] == {"tags": ["original"]}
    assert DocumentStore(tmp_path).list_with_metadata() == document_store.list_with_metadata()


@pytest.mark.parametrize("doc_id", ["_index", "_INDEX", "_Index"])
def test_reserved_index_names_are_rejected_before_any_write(tmp_path, doc_id, document_store):

    with pytest.raises(ValueError, match="reserved"):
        document_store.add(document(doc_id))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["add", "delete"])
def test_case_colliding_document_ids_cannot_overwrite_or_delete_a_document(
    tmp_path, operation, document_store
):
    document_store.add(document("paper"))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(ValueError, match="letter case"):
        if operation == "add":
            document_store.add(document("PAPER"))
        else:
            document_store.delete("PAPER")

    assert len(document_store.list_with_metadata()) == 1
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("indexed", [False, True])
def test_recovery_rejects_document_symlinks_without_reading_or_writing_targets(tmp_path, indexed):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"id": "linked", "content": "Private body.", "metadata": {}}))
    directory = tmp_path / "documents"
    directory.mkdir()
    (directory / "linked.json").symlink_to(outside)
    if indexed:
        (directory / "_index.json").write_text(json.dumps({"linked": {"word_count": 2}}))
    before = outside.read_bytes()

    with pytest.raises(DocumentStoreError, match="Unsafe stored"):
        len(DocumentStore(directory).list_with_metadata())

    assert outside.read_bytes() == before
    assert (directory / "linked.json").is_symlink()
    assert (directory / "_index.json").exists() is indexed


def test_index_symlinks_are_rejected_without_touching_the_target(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    directory = tmp_path / "documents"
    directory.mkdir()
    (directory / "_index.json").symlink_to(outside)

    with pytest.raises(DocumentStoreError, match="index cannot be a symbolic link"):
        len(DocumentStore(directory).list_with_metadata())

    assert outside.read_text() == "{}"
    assert (directory / "_index.json").is_symlink()


def test_document_symlinks_inside_the_store_cannot_be_overwritten(tmp_path, document_store):
    document_store.add(document("original"))
    (tmp_path / "alias.json").symlink_to(tmp_path / "original.json")
    before = (tmp_path / "original.json").read_bytes()

    with pytest.raises(ValueError, match="symbolic link"):
        document_store.add(document("alias"))

    assert (tmp_path / "original.json").read_bytes() == before


def test_recovery_rejects_case_colliding_index_entries_without_rewriting_files(tmp_path):
    index = {"paper": {}, "PAPER": {}}
    (tmp_path / "_index.json").write_text(json.dumps(index))
    before = (tmp_path / "_index.json").read_bytes()

    with pytest.raises(DocumentStoreError, match="letter case"):
        len(DocumentStore(tmp_path).list_with_metadata())

    assert (tmp_path / "_index.json").read_bytes() == before


@pytest.mark.parametrize("key,value", [("word_count", 99), ("notes", {"tags": ["new"]})])
def test_failed_metadata_updates_restore_the_previous_memory_and_disk_state(
    tmp_path, monkeypatch, key, value, document_store
):
    document_store.add(document())
    before = document_store.list_with_metadata()
    disk_before = (tmp_path / "_index.json").read_bytes()

    def fail():
        raise OSError("index unavailable")

    monkeypatch.setattr(document_store, "_save_index", fail)

    with pytest.raises(OSError, match="index unavailable"):
        document_store.update_metadata("paper", key, value)

    assert document_store.list_with_metadata() == before
    assert (tmp_path / "_index.json").read_bytes() == disk_before
    assert DocumentStore(tmp_path).list_with_metadata() == before


def test_failed_add_index_write_removes_only_the_new_document_file(
    tmp_path, monkeypatch, document_store
):
    document_store.add(document("original"))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def fail():
        raise OSError("index unavailable")

    monkeypatch.setattr(document_store, "_save_index", fail)

    with pytest.raises(OSError, match="index unavailable"):
        document_store.add(document("new"))

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert set(document_store.list_with_metadata()) == {"original"}
    assert DocumentStore(tmp_path).list_with_metadata() == document_store.list_with_metadata()


def test_failed_add_cleanup_preserves_original_error_and_recovers_the_orphan(
    tmp_path, monkeypatch, document_store
):
    document_store.add(document("original"))
    failure = OSError("index unavailable")
    unlink = type(tmp_path).unlink

    def fail_index():
        raise failure

    def fail_new_unlink(path, *args, **kwargs):
        if path.name == "new.json":
            raise PermissionError("cleanup unavailable")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(document_store, "_save_index", fail_index)
    monkeypatch.setattr(type(tmp_path), "unlink", fail_new_unlink)

    with pytest.raises(OSError, match="index unavailable") as raised:
        document_store.add(document("new"))

    assert raised.value is failure
    assert json.loads((tmp_path / "new.json").read_text())["id"] == "new"
    assert set(document_store.list_with_metadata()) == {"original", "new"}
    assert DocumentStore(tmp_path).list_with_metadata() == document_store.list_with_metadata()
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_file_creation_cannot_be_overwritten_or_removed(
    tmp_path, monkeypatch, document_store
):
    assert len(document_store.list_with_metadata()) == 0
    writer = DocumentStore(tmp_path)
    link = os.link
    existing_files = {}
    published = False

    def create_before_publication(source, target):
        nonlocal published
        if not published:
            published = True
            writer.add(ParsedDocument("paper", "Original evidence.", {"filename": "original.txt"}))
            writer.update_metadata("paper", "notes", "Preserved")
            existing_files.update(
                {path.name: path.read_bytes() for path in tmp_path.glob("*.json")}
            )
        return link(source, target)

    monkeypatch.setattr(os, "link", create_before_publication)

    with pytest.raises(DocumentExistsError, match="already exists"):
        document_store.add(document())

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == existing_files
    assert DocumentStore(tmp_path).list_with_metadata() == writer.list_with_metadata()


def test_interrupted_publication_recovers_the_completed_document_file(
    tmp_path, monkeypatch, document_store
):
    assert len(document_store.list_with_metadata()) == 0
    link = os.link

    def interrupt_after_publication(source, target):
        link(source, target)
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", interrupt_after_publication)

    with pytest.raises(KeyboardInterrupt):
        document_store.add(document())

    assert document_store._index is None
    assert json.loads((tmp_path / "paper.json").read_text())["content"] == "Two words."
    assert set(document_store.list_with_metadata()) == {"paper"}
    assert DocumentStore(tmp_path).list_with_metadata() == document_store.list_with_metadata()
    assert not list(tmp_path.glob("*.tmp"))


def test_duplicate_add_preserves_the_original_document_and_metadata(tmp_path, document_store):
    document_store.add(document())
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    replacement = ParsedDocument("paper", "Different document body.", {"filename": "other.txt"})

    with pytest.raises(ValueError, match="already exists"):
        document_store.add(replacement)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_add_never_overwrites_a_file_created_after_the_index_was_cached(tmp_path, document_store):
    assert len(document_store.list_with_metadata()) == 0
    payload = {"id": "paper", "content": "Existing text.", "metadata": {}, "sections": []}
    (tmp_path / "paper.json").write_text(json.dumps(payload))
    before = (tmp_path / "paper.json").read_bytes()

    with pytest.raises(ValueError, match="already exists"):
        document_store.add(document())

    assert (tmp_path / "paper.json").read_bytes() == before
    assert not (tmp_path / "_index.json").exists()


@pytest.mark.parametrize("operation", ["delete", "clear"])
def test_missing_document_files_are_removed_from_cached_metadata(
    tmp_path, operation, document_store
):
    document_store.add(document())
    (tmp_path / "paper.json").unlink()

    if operation == "delete":
        assert document_store.delete("paper") is True
    else:
        assert document_store.clear() == 1
    assert len(document_store.list_with_metadata()) == 0
    assert not document_store.exists("paper")
    assert len(DocumentStore(tmp_path).list_with_metadata()) == 0


def test_clear_rewrites_the_index_once(tmp_path, monkeypatch, document_store):
    for doc_id in ("first", "second", "third"):
        document_store.add(_doc(doc_id))
    write_json = document_store._write_json
    targets = []

    def record(target, value, **options):
        targets.append(target)
        write_json(target, value, **options)

    monkeypatch.setattr(document_store, "_write_json", record)

    assert document_store.clear() == 3
    assert targets == [tmp_path / "_index.json"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["_index.json"]
    assert len(DocumentStore(tmp_path).list_with_metadata()) == 0


def test_interrupted_clear_reloads_the_remaining_documents(tmp_path, monkeypatch, document_store):
    for doc_id in ("first", "second"):
        document_store.add(_doc(doc_id))
    unlink = Path.unlink

    def fail_on_second(path, missing_ok=False):
        if path.stem == "second":
            raise PermissionError("file is locked")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_on_second)
    with pytest.raises(PermissionError):
        document_store.clear()
    monkeypatch.setattr(Path, "unlink", unlink)

    assert set(document_store.list_with_metadata()) == {"second"}
    assert set(DocumentStore(tmp_path).list_with_metadata()) == {"second"}


def test_failed_missing_file_cleanup_reconciles_on_retry(tmp_path, monkeypatch, document_store):
    document_store.add(document())
    (tmp_path / "paper.json").unlink()

    def fail():
        raise OSError("index unavailable")

    monkeypatch.setattr(document_store, "_save_index", fail)

    with pytest.raises(OSError, match="index unavailable"):
        document_store.delete("paper")

    assert len(document_store.list_with_metadata()) == 0
    assert len(DocumentStore(tmp_path).list_with_metadata()) == 0


@pytest.mark.parametrize("data", [b"{broken", b"\xff", b"[]", b"null", b'{"paper": []}'])
def test_corrupt_index_is_reported_without_overwriting_it_or_adding_documents(
    tmp_path, data, document_store
):
    (tmp_path / "_index.json").write_bytes(data)

    with pytest.raises(DocumentStoreError):
        document_store.add(document())

    assert (tmp_path / "_index.json").read_bytes() == data
    assert not (tmp_path / "paper.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "entry",
    [
        {"word_count": "2"},
        {"word_count": True},
        {"section_count": -1},
        {"chunk_count": None},
        {"filename": ["paper.txt"]},
        {"file_type": None},
    ],
)
def test_invalid_index_metadata_is_reported_before_reconciliation(tmp_path, entry):
    data = json.dumps({"paper": entry})
    (tmp_path / "_index.json").write_text(data)

    with pytest.raises(DocumentStoreError, match="Invalid document index entry"):
        DocumentStore(tmp_path).get_stats()

    assert (tmp_path / "_index.json").read_text() == data


@pytest.mark.parametrize(
    "data",
    [
        b"{broken",
        b"\xff",
        b"[]",
        b'{"id":"other","content":"Text."}',
        b'{"id":"paper","content":null}',
        b'{"id":"paper","content":"Text.","metadata":null}',
        b'{"id":"paper","content":"Text.","sections":{}}',
        b'{"id":"paper","content":"Text.","sections":["invalid"]}',
    ],
)
def test_invalid_orphan_documents_are_preserved_and_never_reported_as_empty(tmp_path, data):
    (tmp_path / "paper.json").write_bytes(data)

    with pytest.raises(DocumentStoreError):
        len(DocumentStore(tmp_path).list_with_metadata())

    assert (tmp_path / "paper.json").read_bytes() == data
    assert not (tmp_path / "_index.json").exists()


def test_recovery_read_errors_propagate_without_rewriting_files(tmp_path, monkeypatch):
    (tmp_path / "paper.json").write_text("{}")
    read_text = type(tmp_path).read_text

    def fail(path, *args, **kwargs):
        if path.name == "paper.json":
            raise PermissionError("document unavailable")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "read_text", fail)

    with pytest.raises(PermissionError, match="document unavailable"):
        len(DocumentStore(tmp_path).list_with_metadata())

    assert not (tmp_path / "_index.json").exists()


@pytest.mark.parametrize(
    "key,value", [(None, 2), (1, 2), ("", 2), ("word_count", -1), ("notes", float("nan"))]
)
def test_invalid_metadata_updates_leave_memory_and_disk_unchanged(
    tmp_path, key, value, document_store
):
    document_store.add(document())
    before = document_store.list_with_metadata()
    disk_before = (tmp_path / "_index.json").read_bytes()

    with pytest.raises(ValueError):
        document_store.update_metadata("paper", key, value)

    assert document_store.list_with_metadata() == before
    assert (tmp_path / "_index.json").read_bytes() == disk_before


@pytest.mark.parametrize("content", [None, 42, ["words"]])
def test_nontext_documents_are_rejected_before_creating_files(tmp_path, content, document_store):

    with pytest.raises(ValueError, match="content must be text"):
        document_store.add(ParsedDocument("paper", content, {}))

    assert list(tmp_path.iterdir()) == []


def test_nonfinite_document_metadata_is_rejected_before_writing_a_document(
    tmp_path, document_store
):

    with pytest.raises(ValueError):
        document_store.add(ParsedDocument("paper", "Some text.", {"score": float("inf")}))

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", ["_index.json", "paper.json"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_values_are_rejected_when_reading_stored_data(tmp_path, filename, value):
    payload = (
        {"paper": {"notes": value}}
        if filename == "_index.json"
        else {"id": "paper", "content": "Words.", "metadata": {"score": value}}
    )
    data = json.dumps(payload)
    (tmp_path / filename).write_text(data)

    with pytest.raises(DocumentStoreError, match="Cannot read stored JSON"):
        len(DocumentStore(tmp_path).list_with_metadata())

    assert (tmp_path / filename).read_text() == data


def test_index_symlink_created_after_loading_is_never_replaced_by_a_write(tmp_path):
    directory = tmp_path / "documents"
    document_store = DocumentStore(directory)
    document_store.add(document())
    before = document_store.list_with_metadata()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (directory / "_index.json").unlink()
    (directory / "_index.json").symlink_to(outside)

    with pytest.raises(DocumentStoreError, match="symbolic link"):
        document_store.update_metadata("paper", "notes", "Changed")

    assert document_store.list_with_metadata() == before
    assert outside.read_text() == "{}"
    assert (directory / "_index.json").is_symlink()


@pytest.mark.parametrize("operation", ["exists", "update"])
@pytest.mark.parametrize("doc_id", [None, [], "", "../paper", "paper\n", "_INDEX"])
def test_metadata_access_rejects_invalid_document_ids_without_changing_state(
    tmp_path, operation, doc_id, document_store
):
    document_store.add(document())
    before = document_store.list_with_metadata()

    with pytest.raises(ValueError, match="Document ID"):
        if operation == "exists":
            document_store.exists(doc_id)
        else:
            document_store.update_metadata(doc_id, "notes", "changed")

    assert document_store.list_with_metadata() == before


@pytest.mark.parametrize("filename", ["_index.json", "paper.json"])
def test_overflowing_json_numbers_are_rejected_without_rewriting_files(tmp_path, filename):
    data = (
        '{"paper":{"notes":{"ratio":1e999}}}'
        if filename == "_index.json"
        else '{"id":"paper","content":"Words.","metadata":{"notes":{"ratio":1e999}}}'
    )
    (tmp_path / filename).write_text(data)

    with pytest.raises(DocumentStoreError, match="Cannot read stored JSON"):
        len(DocumentStore(tmp_path).list_with_metadata())

    assert (tmp_path / filename).read_text() == data


def test_finite_custom_metadata_numbers_survive_a_reload(tmp_path, document_store):
    document_store.add(document())
    document_store.update_metadata("paper", "notes", {"ratio": 0.125, "exponent": 1.25e25})

    assert DocumentStore(tmp_path).list_with_metadata() == document_store.list_with_metadata()


class TestIdSanitizer:
    def test_path_traversal_and_injection_rejected(self) -> None:
        from scinexusrag.storage.vector_store import _sanitize_id

        for bad in ["../../etc/passwd", "a/b", "a\\b", "id; DROP TABLE x", "a' OR '1'='1", "a b"]:
            with pytest.raises(ValueError):
                _sanitize_id(bad)

    def test_safe_ids_accepted(self) -> None:
        from scinexusrag.storage.vector_store import _sanitize_id

        for ok in ["doc_123", "chunk-abc", "AaZz09_-"]:
            assert _sanitize_id(ok) == ok

    @pytest.mark.parametrize("length", [129, 200_001])
    def test_oversized_ids_are_rejected_before_pattern_matching(self, monkeypatch, length):
        from scinexusrag.storage import vector_store

        def unexpected_match(_value):
            pytest.fail("Oversized IDs must be rejected before pattern matching.")

        monkeypatch.setattr(
            vector_store, "SAFE_ID_PATTERN", SimpleNamespace(fullmatch=unexpected_match)
        )

        with pytest.raises(ValueError, match="ID too long"):
            vector_store._sanitize_id("a" * (length - 1) + "!")


class _ObservedLock:
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


def test_search_scores_are_cosine_similarity(vector_store) -> None:

    chunks = [
        Chunk(id="a", content="aligned unit", document_id="a"),
        Chunk(id="b", content="aligned scaled", document_id="b"),
        Chunk(id="c", content="orthogonal", document_id="c"),
    ]
    vectors = np.array(
        [[1, 0, 0, 0], [3, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.float32,
    )
    vector_store.add(chunks, vectors)

    results = {
        r.chunk.id: r.score
        for r in vector_store.search(np.array([1, 0, 0, 0], np.float32), top_k=3)
    }

    assert results["a"] > 0.99
    assert results["b"] > 0.99
    assert abs(results["c"]) < 0.01


def test_search_ranks_by_similarity(vector_store) -> None:
    chunks = [
        Chunk(id="near", content="near", document_id="near"),
        Chunk(id="far", content="far", document_id="far"),
    ]
    vectors = np.array([[1, 0.1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    vector_store.add(chunks, vectors)

    ranked = vector_store.search(np.array([1, 0, 0, 0], np.float32), top_k=2)

    assert ranked[0].chunk.id == "near"
    assert ranked[0].score > ranked[1].score


def test_search_score_clamped_to_unit_interval(vector_store) -> None:

    chunks = [Chunk(id="opp", content="opposite", document_id="opp")]
    vector_store.add(chunks, np.array([[-1, 0, 0, 0]], dtype=np.float32))

    results = vector_store.search(np.array([1, 0, 0, 0], np.float32), top_k=1)

    assert results[0].score == 0.0


@pytest.mark.parametrize("operation", ["delete_by_document", "get_chunk_ids_by_document"])
def test_trailing_newline_id_is_rejected_before_database_access(vector_store, operation) -> None:

    with pytest.raises(ValueError, match="Invalid ID format"):
        getattr(vector_store, operation)("paper\n")

    assert vector_store._db is None


@pytest.mark.parametrize("access", ["search", "get_all_chunks"])
def test_chunk_context_and_citation_metadata_survive_reload(tmp_path, access, vector_store) -> None:
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
    vector_store.add([chunk], np.array([[1, 0, 0, 0]], dtype=np.float32))

    reloaded = make_store(tmp_path)
    if access == "search":
        restored = reloaded.search(np.array([1, 0, 0, 0], np.float32))[0].chunk
    else:
        restored = reloaded.get_all_chunks()[0]

    assert restored.full_context == chunk.full_context
    assert restored.document_name == "Paper.pdf"
    assert restored.section_title == "Methods"
    assert restored.page_number == 3
    assert restored.chunk_index == 7
    assert (restored.start_offset, restored.end_offset) == (chunk.start_offset, chunk.end_offset)
    assert restored.metadata["start_offset"] == chunk.start_offset
    assert restored.metadata["end_offset"] == chunk.end_offset
    assert restored.metadata["file_type"] == "pdf"
    assert reloaded.get_chunk_ids_by_document("paper") == {"context"}


@pytest.mark.parametrize("existing", [False, True])
def test_legacy_connection_can_create_reopen_and_clear_tables(
    tmp_path, existing, vector_store
) -> None:
    vectors = np.array([[1, 0, 0, 0]], dtype=np.float32)
    if existing:
        vector_store.add(
            [Chunk(id="old", content="Original passage", document_id="paper")], vectors
        )

    class LegacyConnection:
        """Legacy interface."""

        def __init__(self, connection):
            self.connection = connection

        def table_names(self):
            return self.connection.list_tables().tables

        def __getattr__(self, name):
            if name == "list_tables":
                raise AttributeError(name)
            return getattr(self.connection, name)

    reloaded = make_store(tmp_path)
    reloaded._db = LegacyConnection(vector_store.db)
    reloaded.add([Chunk(id="new", content="New passage", document_id="paper")], vectors)

    assert reloaded.count() == 1 + existing
    assert {result.chunk.id for result in reloaded.search(vectors[0])} == (
        {"old", "new"} if existing else {"new"}
    )
    reloaded.clear()
    assert "chunks" not in vector_store.db.list_tables().tables
    assert reloaded.count() == 0


@pytest.mark.parametrize("operation", ["open", "clear"])
def test_current_connection_finds_tables_on_later_pages(vector_store, operation) -> None:
    schema = vector_store.table.schema
    vector_store._table = None
    database = MagicMock(spec=["list_tables", "open_table", "create_table", "drop_table"])
    database.open_table.return_value.schema = schema
    database.open_table.return_value.count_rows.return_value = 0
    pages = {
        None: SimpleNamespace(tables=["another_table"], page_token="next_page"),
        "next_page": SimpleNamespace(tables=["chunks"], page_token=None),
    }
    database.list_tables.side_effect = lambda *, page_token: pages[page_token]
    vector_store._db = database

    if operation == "open":
        assert vector_store.table is database.open_table.return_value
        database.open_table.assert_called_once_with("chunks")
        database.create_table.assert_not_called()
    else:
        vector_store.clear()
        database.drop_table.assert_called_once_with("chunks")


def test_embedding_configuration_is_persisted_and_reopened(tmp_path) -> None:
    config = make_config(query_prefix="query: ", doc_prefix="passage: ", normalize=False)
    vector_store = VectorStore(path=tmp_path / "lancedb", embedding_config=config)
    chunk = Chunk(id="stored", content="Stored passage", document_id="paper")
    vector_store.add([chunk], np.array([[1, 0, 0, 0]], dtype=np.float32))

    reloaded = VectorStore(path=vector_store.path, embedding_config=config)
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
    tmp_path, monkeypatch, change, operation
) -> None:
    original = make_store(tmp_path)
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
    reloaded = VectorStore(path=original.path, embedding_config=replace(make_config(), **change))
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


@pytest.mark.parametrize("operation", ["count", "get_all_chunks", "get_chunk_ids_by_document"])
def test_read_helpers_propagate_configuration_errors(tmp_path, operation) -> None:
    original = make_store(tmp_path)
    original.add(
        [Chunk(id="original", content="Original passage", document_id="paper")],
        np.ones((1, 4), dtype=np.float32),
    )
    changed = make_store(tmp_path, revision="b" * 40)

    with pytest.raises(EmbeddingConfigError):
        if operation == "get_chunk_ids_by_document":
            getattr(changed, operation)("paper")
        else:
            getattr(changed, operation)()

    assert original.count() == 1


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_count_propagates_storage_failure_instead_of_reporting_empty(
    vector_store, monkeypatch, error_type
) -> None:
    table = vector_store.table
    failure = error_type("Cannot read row count")
    monkeypatch.setattr(table, "count_rows", MagicMock(side_effect=failure))

    with pytest.raises(VectorStoreError, match="row count") as raised:
        vector_store.count()

    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("chunk_count", [0, 3, 4, 5, 8])
def test_document_scans_cross_page_boundaries_without_losing_rows(
    vector_store, monkeypatch, chunk_count
) -> None:
    monkeypatch.setattr("scinexusrag.storage.vector_store.SCAN_PAGE_SIZE", 4)
    chunks = [
        Chunk(id=f"paper-{i}", content=f"Measurement {i}", document_id="paper")
        for i in range(chunk_count)
    ]
    unrelated = Chunk(id="other", content="Different document", document_id="other")
    vector_store.add([unrelated, *chunks], np.ones((chunk_count + 1, 4), dtype=np.float32))

    assert vector_store.get_chunk_ids_by_document("paper") == {chunk.id for chunk in chunks}
    assert {chunk.id for chunk in vector_store.get_all_chunks()} == {
        "other",
        *(c.id for c in chunks),
    }
    assert vector_store.get_chunk_ids_by_document("missing") == set()


def test_document_scans_include_chunks_beyond_the_previous_100000_limit(vector_store) -> None:
    count = 100_001
    ids = [f"chunk-{i}" for i in range(count)]
    vector_store.table.add(
        pa.table(
            {
                "id": ids,
                "content": ["Measured current is 42 mA."] * count,
                "document_id": ["large-paper"] * count,
                "metadata": ["{}"] * count,
                "vector": [[1.0, 0.0, 0.0, 0.0]] * count,
            },
            schema=vector_store.table.schema,
        )
    )

    assert vector_store.get_chunk_ids_by_document("large-paper") == set(ids)
    assert {chunk.id for chunk in vector_store.get_all_chunks()} == set(ids)


@pytest.mark.parametrize("operation", ["get_all_chunks", "get_chunk_ids_by_document"])
def test_failed_scan_does_not_report_an_empty_or_partial_corpus(
    vector_store, monkeypatch, operation
) -> None:
    table = vector_store.table
    failure = OSError("Storage is unavailable")
    monkeypatch.setattr(table, "search", MagicMock(side_effect=failure))

    with pytest.raises(VectorStoreError, match="read passages") as raised:
        if operation == "get_chunk_ids_by_document":
            getattr(vector_store, operation)("paper")
        else:
            getattr(vector_store, operation)()

    assert raised.value.__cause__ is failure


def test_failure_on_a_later_scan_page_does_not_return_a_partial_document(
    vector_store, monkeypatch
) -> None:
    monkeypatch.setattr("scinexusrag.storage.vector_store.SCAN_PAGE_SIZE", 1)
    vector_store.add(
        [Chunk(id="a", content="First measurement", document_id="paper")],
        np.ones((1, 4), dtype=np.float32),
    )
    table = vector_store.table
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
        vector_store.get_chunk_ids_by_document("paper")

    assert calls == 2


def test_vector_search_failure_is_distinct_from_empty_results(vector_store, monkeypatch) -> None:
    failure = OSError("Storage is unavailable")
    monkeypatch.setattr(vector_store.table, "search", MagicMock(side_effect=failure))

    with pytest.raises(VectorStoreError, match="search the vector index") as raised:
        vector_store.search(np.ones(4, dtype=np.float32))

    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("metadata", ["not JSON", "[]", "null"])
@pytest.mark.parametrize("operation", ["get_all_chunks", "search"])
def test_corrupt_passage_metadata_is_reported_as_a_storage_error(
    vector_store, metadata, operation
) -> None:
    vector_store.table.add(
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
        if operation == "search":
            vector_store.search(np.ones(4, dtype=np.float32))
        else:
            vector_store.get_all_chunks()


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize(
    "metadata_kind", ["missing", "incomplete", "corrupt", "unsupported-version"]
)
def test_unbound_or_invalid_tables_are_never_adopted(
    tmp_path, populated, metadata_kind, vector_store
) -> None:
    schema = vector_store.table.schema.remove_metadata()
    vector_store.clear()
    config_data = {"version": 1, **asdict(make_config())}
    if metadata_kind == "incomplete":
        del config_data["doc_prefix"]
    elif metadata_kind == "unsupported-version":
        config_data["version"] = 2
    if metadata_kind != "missing":
        encoded = b"not JSON" if metadata_kind == "corrupt" else json.dumps(config_data).encode()
        schema = schema.with_metadata({b"nexusrag.embedding_config": encoded})
    raw_table = vector_store.db.create_table("chunks", schema=schema)
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
    reloaded = make_store(tmp_path)

    with pytest.raises(EmbeddingConfigError):
        reloaded.count()

    assert raw_table.count_rows() == int(populated)
    assert raw_table.schema.metadata == schema.metadata
    assert reloaded._table is None


def test_cached_table_configuration_is_checked_on_every_access(vector_store) -> None:
    table = vector_store.table
    changed_schema = table.schema.with_metadata(
        {
            b"nexusrag.embedding_config": json.dumps(
                {"version": 1, **asdict(make_config(revision="b" * 40))}
            ).encode()
        }
    )
    cached = MagicMock(wraps=table)
    cached.schema = changed_schema
    vector_store._table = cached

    with pytest.raises(EmbeddingConfigError):
        vector_store.search(np.ones(4, dtype=np.float32))

    cached.search.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("populated", [False, True])
def test_explicit_clear_recovers_rejected_table_with_a_new_dimension(
    tmp_path, legacy, populated
) -> None:
    original = make_store(tmp_path)
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
    changed = make_store(tmp_path, dim=2, model_name="new-model")

    with pytest.raises(EmbeddingConfigError):
        changed.count()

    changed.clear()
    changed.add(
        [Chunk(id="new", content="New passage", document_id="new-paper")],
        np.array([[1, 0]], dtype=np.float32),
    )
    reopened = make_store(tmp_path, dim=2, model_name="new-model")
    assert reopened.count() == 1
    assert reopened.table.schema.field("vector").type.list_size == 2
    assert reopened.get_chunk_ids_by_document("new-paper") == {"new"}
    assert reopened.search(np.array([1, 0], dtype=np.float32))[0].chunk.id == "new"


@pytest.mark.parametrize("operation", ["count", "search", "add"])
@pytest.mark.parametrize("extra_old_write", [False, True], ids=["equal-version", "lower-version"])
def test_cached_handle_rejects_a_recreated_collection(tmp_path, operation, extra_old_write) -> None:
    original = make_store(tmp_path)
    vector = np.array([[1, 0, 0, 0]], dtype=np.float32)
    original.add([Chunk(id="old", content="Old passage", document_id="old-paper")], vector)
    if extra_old_write:
        original.add(
            [Chunk(id="old-more", content="Another old passage", document_id="old-paper")], vector
        )
    original_version = original.table.version
    replacement = make_store(tmp_path, model_name="another-model")
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
    assert replacement.get_chunk_ids_by_document("new-paper") == {"new"}


@pytest.mark.parametrize("bootstrap_stage", ["connect", "create_table"])
def test_concurrent_first_access_connects_and_creates_once(
    vector_store, monkeypatch, bootstrap_stage
):
    observed = _ObservedLock(vector_store._init_lock)
    vector_store._init_lock = observed
    entered = Event()
    release = Event()
    original = lancedb.connect if bootstrap_stage == "connect" else vector_store.db.create_table

    def held_bootstrap(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Bootstrap was not released")
        return original(*args, **kwargs)

    bootstrap = MagicMock(side_effect=held_bootstrap)
    if bootstrap_stage == "connect":
        monkeypatch.setattr(lancedb, "connect", bootstrap)
    else:
        monkeypatch.setattr(vector_store.db, "create_table", bootstrap)

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(vector_store.count)
        try:
            assert entered.wait(timeout=5)
            second = workers.submit(observed.run_contender, vector_store.count)
            assert observed.attempted.wait(timeout=5)
            assert bootstrap.call_count == 1
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5) == 0

    assert bootstrap.call_count == 1
    assert vector_store.db.list_tables().tables == ["chunks"]


@pytest.mark.parametrize("bootstrap_stage", ["connect", "create_table"])
def test_failed_bootstrap_releases_lock_and_can_retry(vector_store, monkeypatch, bootstrap_stage):
    original = lancedb.connect if bootstrap_stage == "connect" else vector_store.db.create_table

    def fail_once(*args, **kwargs):
        if bootstrap.call_count == 1:
            raise OSError("Storage temporarily unavailable")
        return original(*args, **kwargs)

    bootstrap = MagicMock(side_effect=fail_once)
    if bootstrap_stage == "connect":
        monkeypatch.setattr(lancedb, "connect", bootstrap)
    else:
        monkeypatch.setattr(vector_store.db, "create_table", bootstrap)

    with pytest.raises(VectorStoreError) as raised:
        vector_store.count()

    assert isinstance(raised.value.__cause__, OSError)
    assert vector_store._table is None
    if bootstrap_stage == "connect":
        assert vector_store._db is None
    assert vector_store.count() == 0
    assert bootstrap.call_count == 2
    assert (
        json.loads(vector_store.table.schema.metadata[b"nexusrag.embedding_config"])["version"] == 1
    )


def test_clear_waits_for_table_bootstrap_then_removes_the_published_table(
    vector_store, monkeypatch
):
    observed = _ObservedLock(vector_store._init_lock)
    vector_store._init_lock = observed
    entered = Event()
    release = Event()
    original_create = vector_store.db.create_table

    def held_create(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("Table creation was not released")
        return original_create(*args, **kwargs)

    monkeypatch.setattr(vector_store.db, "create_table", held_create)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(lambda: vector_store.table)
        try:
            assert entered.wait(timeout=5)
            cleared = workers.submit(observed.run_contender, vector_store.clear)
            assert observed.attempted.wait(timeout=5)
        finally:
            release.set()
        assert first.result(timeout=5) is not None
        assert cleared.result(timeout=5) is None

    assert vector_store._table is None
    assert "chunks" not in vector_store.db.list_tables().tables
    assert vector_store.count() == 0


def test_failed_clear_keeps_cached_data_and_allows_retry(vector_store, monkeypatch):
    vector_store.add(
        [Chunk(id="kept", content="Stored passage", document_id="paper")],
        np.ones((1, 4), dtype=np.float32),
    )
    original_drop = vector_store.db.drop_table
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("Cannot remove table")
        return original_drop(*args, **kwargs)

    monkeypatch.setattr(vector_store.db, "drop_table", fail_once)
    with pytest.raises(OSError):
        vector_store.clear()

    assert vector_store.count() == 1
    vector_store.clear()
    assert vector_store._table is None
    assert "chunks" not in vector_store.db.list_tables().tables


@pytest.fixture
def contract_vectors(tmp_path):
    return VectorStore(
        tmp_path / "vectors", embedding_config=EmbeddingConfig("model", "a" * 40, "", "", True, 2)
    )


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones(2),
        np.ones((1, 3)),
        np.ones((2, 2)),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
        np.array([[1.0, 1e100]]),
        np.zeros((1, 2)),
        np.array([[True, False]]),
        np.array([["1", "2"]]),
        np.ones((1, 2), dtype=complex),
    ],
)
def test_invalid_vectors_fail_before_opening_database(contract_vectors, vectors):
    with pytest.raises(ValueError):
        contract_vectors.add([Chunk("chunk", "passage", "paper")], vectors)

    assert contract_vectors._db is None


@pytest.mark.parametrize("field", ["id", "document_id"])
@pytest.mark.parametrize("value", [None, 1, "", "paper\n", "paper'", "a" * 129])
def test_invalid_chunk_ids_fail_before_opening_database(contract_vectors, field, value):
    chunk = Chunk("chunk", "passage", "paper")
    setattr(chunk, field, value)

    with pytest.raises(ValueError):
        contract_vectors.add([chunk], np.ones((1, 2), dtype=np.float32))

    assert contract_vectors._db is None


def test_duplicate_batch_ids_reject_entire_write(contract_vectors):
    chunks = [Chunk("same", "first", "paper"), Chunk("same", "second", "paper")]

    with pytest.raises(ValueError, match="Duplicate"):
        contract_vectors.add(chunks, np.ones((2, 2), dtype=np.float32))

    assert contract_vectors._db is None


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_metadata_is_not_persisted(contract_vectors, value):
    chunk = Chunk("chunk", "passage", "paper", metadata={"score": value})

    with pytest.raises(ValueError):
        contract_vectors.add([chunk], np.ones((1, 2), dtype=np.float32))

    assert contract_vectors._db is None


def test_empty_batch_does_not_open_database(contract_vectors):
    assert contract_vectors.add([], np.empty((0, 2), dtype=np.float32)) == 0
    assert contract_vectors._db is None


def test_vectors_without_chunks_are_rejected(contract_vectors):
    with pytest.raises(ValueError, match="shape"):
        contract_vectors.add([], np.ones((1, 2), dtype=np.float32))

    assert contract_vectors._db is None


def test_failed_vector_write_exposes_storage_error_with_original_cause(
    contract_vectors, monkeypatch
):
    failure = OSError("disk write failed")
    monkeypatch.setattr(contract_vectors.table, "add", Mock(side_effect=failure))

    with pytest.raises(VectorStoreError, match="write passages") as raised:
        contract_vectors.add(
            [Chunk("chunk", "passage", "paper")], np.ones((1, 2), dtype=np.float32)
        )

    assert raised.value.__cause__ is failure
    assert contract_vectors.count() == 0


@pytest.mark.parametrize(
    "vector",
    [
        np.ones((1, 2)),
        np.ones(3),
        np.array([np.nan, 1]),
        np.array([np.inf, 1]),
        np.array([1e100, 1]),
        np.zeros(2),
        np.array(["1", "2"]),
        np.array([True, False]),
    ],
)
def test_invalid_query_vectors_fail_before_opening_database(contract_vectors, vector):
    with pytest.raises(ValueError):
        contract_vectors.search(vector)

    assert contract_vectors._db is None


@pytest.mark.parametrize("top_k", [True, False, 1.5, "1", None])
def test_invalid_search_depth_fails_before_opening_database(contract_vectors, top_k):
    with pytest.raises(ValueError, match="top_k"):
        contract_vectors.search(np.ones(2, dtype=np.float32), top_k=top_k)

    assert contract_vectors._db is None


@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_nonpositive_search_depth_does_not_open_database(contract_vectors, top_k):
    assert contract_vectors.search(np.ones(2, dtype=np.float32), top_k=top_k) == []
    assert contract_vectors._db is None


def search_with_row(contract_vectors, monkeypatch, row):
    query = Mock()
    query.metric.return_value = query
    query.limit.return_value = query
    query.to_list.return_value = [row]
    monkeypatch.setattr(contract_vectors.table, "search", Mock(return_value=query))
    return contract_vectors.search(np.ones(2, dtype=np.float32))


@pytest.mark.parametrize("distance", [None, True, "0.1", np.nan, np.inf, -np.inf])
def test_invalid_cosine_distance_is_not_a_confident_result(contract_vectors, monkeypatch, distance):
    row = {
        "id": "chunk",
        "content": "passage",
        "document_id": "paper",
        "metadata": "{}",
        "_distance": distance,
    }

    with pytest.raises(VectorStoreError, match="cosine distance"):
        search_with_row(contract_vectors, monkeypatch, row)


def test_missing_cosine_distance_is_a_storage_error(contract_vectors, monkeypatch):
    row = {"id": "chunk", "content": "passage", "document_id": "paper", "metadata": "{}"}

    with pytest.raises(VectorStoreError, match="cosine distance"):
        search_with_row(contract_vectors, monkeypatch, row)


@pytest.mark.parametrize("distance,expected", [(-1e-6, 1.0), (2.0, 0.0), (0.25, 0.75)])
def test_search_scores_stay_in_unit_interval(contract_vectors, monkeypatch, distance, expected):
    row = {
        "id": "chunk",
        "content": "passage",
        "document_id": "paper",
        "metadata": "{}",
        "_distance": distance,
    }

    results = search_with_row(contract_vectors, monkeypatch, row)

    assert results[0].score == expected
    assert type(results[0].score) is float


@pytest.mark.parametrize(
    "field,value", [("id", None), ("document_id", "invalid'"), ("content", None)]
)
def test_corrupt_stored_passage_fields_are_storage_errors(
    contract_vectors, monkeypatch, field, value
):
    row = {
        "id": "chunk",
        "content": "passage",
        "document_id": "paper",
        "metadata": "{}",
        "_distance": 0.0,
    }
    row[field] = value

    with pytest.raises(VectorStoreError, match="passage fields"):
        search_with_row(contract_vectors, monkeypatch, row)


def test_existing_ids_reject_entire_batch_without_overwriting(contract_vectors):
    contract_vectors.add(
        [Chunk("existing", "original passage", "original")], np.ones((1, 2), np.float32)
    )
    incoming = [Chunk("new", "new passage", "new"), Chunk("existing", "replacement", "new")]

    with pytest.raises(ValueError, match="already exists"):
        contract_vectors.add(incoming, np.ones((2, 2), np.float32))

    assert contract_vectors.count() == 1
    assert contract_vectors.get_all_chunks()[0].content == "original passage"


def test_existing_ids_are_checked_across_bounded_batches(contract_vectors, monkeypatch):
    monkeypatch.setattr("scinexusrag.storage.vector_store.SCAN_PAGE_SIZE", 2)
    contract_vectors.add(
        [Chunk("existing", "original passage", "original")], np.ones((1, 2), np.float32)
    )
    incoming = [Chunk(str(index), "new passage", "new") for index in range(4)]
    incoming.append(Chunk("existing", "replacement", "new"))

    with pytest.raises(ValueError, match="already exists"):
        contract_vectors.add(incoming, np.ones((5, 2), np.float32))

    assert contract_vectors.count() == 1
    assert contract_vectors.get_chunk_ids_by_document("original") == {"existing"}


def test_failed_duplicate_check_never_writes_vectors(contract_vectors, monkeypatch):
    table = contract_vectors.table
    failure = OSError("cannot inspect existing ids")
    writer = Mock()
    monkeypatch.setattr(table, "search", Mock(side_effect=failure))
    monkeypatch.setattr(table, "add", writer)

    with pytest.raises(VectorStoreError, match="check existing IDs") as raised:
        contract_vectors.add([Chunk("new", "passage", "paper")], np.ones((1, 2), np.float32))

    assert raised.value.__cause__ is failure
    writer.assert_not_called()


def test_deletion_count_ignores_unrelated_concurrent_writes(contract_vectors, monkeypatch):
    contract_vectors.add(
        [Chunk("target", "target passage", "paper"), Chunk("other", "other passage", "other")],
        np.ones((2, 2), dtype=np.float32),
    )
    writer = VectorStore(contract_vectors.path, embedding_config=contract_vectors.embedding_config)
    original_delete = contract_vectors.table.delete

    def delete_after_other_write(predicate):
        writer.add(
            [Chunk(f"new-{index}", "new passage", "other") for index in range(3)],
            np.ones((3, 2), dtype=np.float32),
        )
        return original_delete(predicate)

    monkeypatch.setattr(contract_vectors.table, "delete", delete_after_other_write)

    assert contract_vectors.delete_by_document("paper") == 1
    assert contract_vectors.count() == 4
    assert contract_vectors.get_chunk_ids_by_document("paper") == set()


def test_partial_failed_deletion_reports_error_and_can_retry(contract_vectors, monkeypatch):
    contract_vectors.add(
        [Chunk("first", "first passage", "paper"), Chunk("second", "second passage", "paper")],
        np.ones((2, 2), dtype=np.float32),
    )
    table = contract_vectors.table
    original_delete = table.delete
    failure = OSError("delete interrupted")

    def partial_delete(_predicate):
        original_delete("id = 'first'")
        raise failure

    monkeypatch.setattr(table, "delete", partial_delete)
    with pytest.raises(VectorStoreError, match="delete passages") as raised:
        contract_vectors.delete_by_document("paper")

    assert raised.value.__cause__ is failure
    assert contract_vectors.get_chunk_ids_by_document("paper") == {"second"}
    monkeypatch.setattr(table, "delete", original_delete)
    assert contract_vectors.delete_by_document("paper") == 1
    assert contract_vectors.count() == 0


@pytest.mark.parametrize("mutation", ["delete", "add"])
def test_paginated_scans_keep_one_version_across_other_handles(
    contract_vectors, monkeypatch, mutation
):
    monkeypatch.setattr("scinexusrag.storage.vector_store.SCAN_PAGE_SIZE", 2)
    chunks = [Chunk(letter, f"passage {letter}", "paper") for letter in "abcd"]
    contract_vectors.add(chunks, np.ones((4, 2), dtype=np.float32))
    writer = VectorStore(contract_vectors.path, embedding_config=contract_vectors.embedding_config)
    scan = contract_vectors._read_rows(["id"])

    first = next(scan)["id"]
    if mutation == "delete":
        writer.table.delete(f"id = '{first}'")
    else:
        writer.add([Chunk("new", "new passage", "paper")], np.ones((1, 2), np.float32))
    remaining = [row["id"] for row in scan]

    assert [first, *remaining] == list("abcd")
    assert contract_vectors.count() == (3 if mutation == "delete" else 5)
    assert contract_vectors.get_chunk_ids_by_document("paper") == (
        set("bcd") if mutation == "delete" else {*"abcd", "new"}
    )


def test_closing_a_partial_scan_restores_writes(contract_vectors, monkeypatch):
    monkeypatch.setattr("scinexusrag.storage.vector_store.SCAN_PAGE_SIZE", 1)
    contract_vectors.add([Chunk("old", "old passage", "paper")], np.ones((1, 2), np.float32))
    scan = contract_vectors._read_rows(["id"])
    assert next(scan)["id"] == "old"

    scan.close()
    contract_vectors.add([Chunk("new", "new passage", "paper")], np.ones((1, 2), np.float32))

    assert contract_vectors.count() == 2


def test_snapshot_cleanup_failure_preserves_original_scan_error(contract_vectors, monkeypatch):
    table = contract_vectors.table
    original = OSError("scan failed")
    cleanup = OSError("restore failed")
    monkeypatch.setattr(table, "search", Mock(side_effect=original))
    monkeypatch.setattr(table, "checkout_latest", Mock(side_effect=cleanup))

    with pytest.raises(VectorStoreError, match="read passages") as raised:
        contract_vectors.get_all_chunks()

    assert raised.value.__cause__ is original
    assert contract_vectors._table is None
    assert contract_vectors.count() == 0


def test_snapshot_cleanup_failure_does_not_return_partial_success(contract_vectors, monkeypatch):
    table = contract_vectors.table
    cleanup = OSError("restore failed")
    monkeypatch.setattr(table, "checkout_latest", Mock(side_effect=cleanup))

    with pytest.raises(VectorStoreError, match="read passages") as raised:
        contract_vectors.get_all_chunks()

    assert raised.value.__cause__ is cleanup
    assert contract_vectors._table is None
    contract_vectors.add([Chunk("new", "new passage", "paper")], np.ones((1, 2), np.float32))
    assert contract_vectors.count() == 1


@pytest.mark.parametrize(
    "metadata",
    [
        {"context_before": None},
        {"context_after": ["passage"]},
        {"document_name": 7},
        {"section_title": False},
        {"page_number": "3"},
        {"page_number": -1},
        {"chunk_index": True},
        {"chunk_index": None},
        {"start_offset": -1},
        {"end_offset": 1.5},
        {"start_offset": 20, "end_offset": 10},
        {"nested": {"score": np.nan}},
        {"nested": {"score": np.inf}},
    ],
)
def test_invalid_persisted_chunk_metadata_is_a_storage_error(contract_vectors, metadata):
    contract_vectors.table.add(
        [
            {
                "id": "corrupt",
                "content": "passage",
                "document_id": "paper",
                "metadata": json.dumps(metadata),
                "vector": [1.0, 0.0],
            }
        ]
    )

    with pytest.raises(VectorStoreError, match="decode passage metadata"):
        contract_vectors.get_all_chunks()


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_before", None),
        ("document_name", 3),
        ("page_number", "3"),
        ("chunk_index", True),
        ("start_offset", -1),
    ],
)
def test_invalid_chunk_metadata_never_opens_database(contract_vectors, field, value):
    chunk = Chunk("chunk", "passage", "paper")
    setattr(chunk, field, value)

    with pytest.raises(ValueError):
        contract_vectors.add([chunk], np.ones((1, 2), np.float32))

    assert contract_vectors._db is None


def test_zero_page_and_offsets_survive_round_trip(contract_vectors):
    chunk = Chunk("chunk", "passage", "paper", page_number=0, start_offset=0, end_offset=7)
    contract_vectors.add([chunk], np.ones((1, 2), np.float32))

    restored = contract_vectors.get_all_chunks()[0]

    assert restored.page_number == 0
    assert (restored.start_offset, restored.end_offset) == (0, 7)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("stored_scale", [1e-30, 1.0, 1e20, 1e38])
@pytest.mark.parametrize("query_scale", [1e-30, 1.0, 1e20, 1e38])
def test_cosine_scores_do_not_depend_on_finite_vector_scale(
    contract_vectors, normalize, stored_scale, query_scale
):
    config = replace(contract_vectors.embedding_config, normalize=normalize)
    contract_vectors = VectorStore(contract_vectors.path, embedding_config=config)
    vectors = np.array([[stored_scale, 0], [0, stored_scale]], dtype=np.float32)
    query = np.array([query_scale, 0], dtype=np.float32)
    original_vectors, original_query = vectors.copy(), query.copy()
    chunks = [
        Chunk("aligned", "aligned passage", "paper"),
        Chunk("orthogonal", "other passage", "paper"),
    ]

    contract_vectors.add(chunks, vectors)
    results = {result.chunk.id: result.score for result in contract_vectors.search(query, top_k=2)}

    assert results["aligned"] == pytest.approx(1.0, abs=1e-6)
    assert results["orthogonal"] == pytest.approx(0.0, abs=1e-6)
    assert contract_vectors.embedding_config == config
    np.testing.assert_array_equal(vectors, original_vectors)
    np.testing.assert_array_equal(query, original_query)
