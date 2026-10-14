"""DocumentStore crash recovery: index and doc files reconcile on load."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

from scinexusrag.ingestion import ParsedDocument
from scinexusrag.storage.document_store import DocumentStore


def _doc(doc_id: str = "abc123def456", name: str = "paper.pdf") -> ParsedDocument:
    return ParsedDocument(
        id=doc_id,
        content="Some parsed content with several words.",
        metadata={"original_filename": name, "file_type": "pdf"},
    )


def test_orphaned_doc_file_recovered_on_load(temp_dir) -> None:

    store = DocumentStore(path=temp_dir)
    store.add(_doc())
    (temp_dir / "_index.json").unlink()

    reloaded = DocumentStore(path=temp_dir)

    assert reloaded.exists("abc123def456")
    assert reloaded.index["abc123def456"]["filename"] == "paper.pdf"
    assert reloaded.index["abc123def456"]["word_count"] > 0
    assert (temp_dir / "_index.json").exists()


def test_dangling_index_entry_dropped_on_load(temp_dir) -> None:

    store = DocumentStore(path=temp_dir)
    store.add(_doc())
    (temp_dir / "abc123def456.json").unlink()

    reloaded = DocumentStore(path=temp_dir)

    assert not reloaded.exists("abc123def456")
    assert reloaded.count() == 0


def test_delete_survives_reload(temp_dir) -> None:
    store = DocumentStore(path=temp_dir)
    store.add(_doc())
    assert store.delete("abc123def456") is True

    reloaded = DocumentStore(path=temp_dir)

    assert not reloaded.exists("abc123def456")
    assert reloaded.count() == 0


def test_reserved_index_id_is_rejected(temp_dir) -> None:

    store = DocumentStore(path=temp_dir)
    with pytest.raises(ValueError, match="reserved"):
        store.add(_doc(doc_id="_index"))


def test_trailing_newline_id_is_rejected_before_writing(temp_dir) -> None:
    store = DocumentStore(path=temp_dir)

    with pytest.raises(ValueError, match="invalid characters"):
        store.add(_doc(doc_id="paper\n"))

    assert list(temp_dir.iterdir()) == []


def test_interrupted_index_update_cannot_restore_a_deleted_document(temp_dir, monkeypatch) -> None:
    store = DocumentStore(path=temp_dir)
    store.add(_doc())
    monkeypatch.setattr(store, "_save_index", Mock(side_effect=OSError("interrupted index write")))

    with pytest.raises(OSError, match="interrupted index write"):
        store.delete("abc123def456")

    reloaded = DocumentStore(path=temp_dir)
    assert not reloaded.exists("abc123def456")
    assert reloaded.count() == 0


def test_failed_file_deletion_keeps_the_index_entry(temp_dir, monkeypatch) -> None:
    store = DocumentStore(path=temp_dir)
    store.add(_doc())
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=OSError("file is unavailable")))

    with pytest.raises(OSError, match="file is unavailable"):
        store.delete("abc123def456")

    assert store.exists("abc123def456")
    assert DocumentStore(path=temp_dir).exists("abc123def456")


@pytest.mark.parametrize("operation", ["add", "delete", "update"])
def test_stats_use_one_snapshot_during_concurrent_writes(temp_dir, operation) -> None:
    reading = Event()
    written = Event()

    class PausingCount(int):
        def __radd__(self, other):
            reading.set()
            assert written.wait(5), "Writer could not finish while statistics were calculated"
            return other + int(self)

    store = DocumentStore(path=temp_dir)
    store.add(_doc("first"))
    store.add(_doc("second"))
    store.update_metadata("first", "word_count", PausingCount(7))
    store.update_metadata("first", "section_count", 1)
    store.update_metadata("second", "word_count", 2)
    store.update_metadata("second", "section_count", 2)

    def write():
        try:
            assert reading.wait(5), "Statistics never reached aggregation"
            if operation == "add":
                store.add(_doc("third"))
            elif operation == "delete":
                store.delete("second")
            else:
                store.update_metadata("second", "word_count", 99)
                store.update_metadata("second", "section_count", 99)
        finally:
            written.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(write)
        stats = store.get_stats()
        writer.result(timeout=5)

    assert stats == {
        "document_count": 2,
        "total_words": 9,
        "total_sections": 3,
        "storage_path": str(temp_dir.resolve()),
    }


def test_metadata_snapshot_does_not_share_mutable_entries(temp_dir) -> None:
    store = DocumentStore(path=temp_dir)
    store.add(_doc("first"))
    snapshot = store.list_with_metadata()
    initial_count = snapshot["first"]["word_count"]

    store.update_metadata("first", "word_count", 99)
    snapshot["first"]["filename"] = "changed.txt"

    assert snapshot["first"]["word_count"] == initial_count
    assert store.list_with_metadata()["first"]["filename"] == "paper.pdf"
    assert store.get_stats()["total_words"] == 99
