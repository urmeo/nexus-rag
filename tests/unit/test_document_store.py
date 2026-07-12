"""DocumentStore crash recovery: index and doc files reconcile on load."""

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
    # Crash after the doc write but before the index write leaves an
    # invisible doc file; a fresh load must adopt it back into the index.
    store = DocumentStore(path=temp_dir)
    store.add(_doc())
    (temp_dir / "_index.json").unlink()

    reloaded = DocumentStore(path=temp_dir)

    assert reloaded.exists("abc123def456")
    assert reloaded.index["abc123def456"]["filename"] == "paper.pdf"
    assert reloaded.index["abc123def456"]["word_count"] > 0
    assert (temp_dir / "_index.json").exists()  # repaired index persisted


def test_dangling_index_entry_dropped_on_load(temp_dir) -> None:
    # The reverse crash leaves an index entry with no doc file — a phantom
    # that would block re-ingestion; a fresh load must drop it.
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
    # A document whose id is "_index" would resolve to the same file as the
    # store's index and clobber it on add(); the id must be rejected.
    store = DocumentStore(path=temp_dir)
    with pytest.raises(ValueError, match="reserved"):
        store.add(_doc(doc_id="_index"))
