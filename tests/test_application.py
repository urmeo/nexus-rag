"""Application checks."""

from __future__ import annotations

import json
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Barrier, Event, RLock, get_ident
from unittest.mock import MagicMock, Mock, PropertyMock

import numpy as np
import pytest
from docx import Document as DocxDocument

from scinexusrag.application import DocumentInventory, IngestResult, NexusRAG
from scinexusrag.ingestion.chunker import Chunk
from scinexusrag.ingestion.embedder import EmbeddingConfig, EmbeddingConfigError
from scinexusrag.ingestion.parser import ParsedDocument
from scinexusrag.settings import Settings, StorageSettings
from scinexusrag.storage.document_store import DocumentStore
from scinexusrag.storage.vector_store import VectorStore, VectorStoreError


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.dimension = 384
    embedder.embedding_config = EmbeddingConfig("test-model", "a" * 40, "", "", True, 384)

    def mock_embed(texts, batch_size=None, show_progress=False):
        if isinstance(texts, str):
            texts = [texts]
        rng = np.random.default_rng(zlib.crc32("".join(texts).encode()))
        return rng.random((len(texts), 384), dtype=np.float32)

    embedder.embed = mock_embed
    return embedder


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model = "test-llm"

    def mock_generate(prompt, **kwargs):
        if "What is" in prompt or "?" in prompt:
            return "The fixture records temperature at thirty-second intervals."
        return "Sensor A is the reference."

    llm.generate = mock_generate
    llm.is_available = MagicMock(return_value=True)
    return llm


@pytest.fixture
def pipeline_rag(test_settings: Settings, mock_embedder, mock_llm) -> NexusRAG:
    pipeline_rag = NexusRAG(settings=test_settings)

    pipeline_rag._embedder = mock_embedder
    pipeline_rag._llm = mock_llm

    return pipeline_rag


@pytest.fixture
def pipeline_text_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "test_document.txt"
    content = """INTRODUCTION

This research document is a software test fixture about temperature sensors.
The reference sensor is labeled A, and the comparison sensor is labeled B.

Each reading has a timestamp, a sensor label and a temperature in degrees Celsius.

METHODS

The methodology reads both sensors at thirty-second intervals.
Readings with the same timestamp belong to the same measurement pair.

Store each measurement pair on one row, with separate columns for sensors A and B.

RESULTS

The fixture contains paired readings with matching timestamps.
Its values are synthetic test data, not measured research results.

Missing readings are represented by an empty sensor column.

DISCUSSION

Compare sensor readings only when both values are present.
Keep missing values separate from numeric temperature differences.

The reference sensor provides the label used in retrieval queries.

CONCLUSION

Both sensors use the same units and sampling interval.

"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def pipeline_markdown_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "test_paper.md"
    content = """# Scientific Paper on Testing

## Abstract

This is a Markdown fixture for document ingestion and retrieval tests.

## Introduction

### Background

The fixture uses nested headings to test section extraction.
Each section contains text that can be indexed as a passage.

### Objectives

1. Parse headings and paragraphs.
2. Index the resulting passages.
3. Retrieve a passage by its text.

## Methodology

### Data Preparation

Save the fixture as UTF-8 Markdown.
Keep headings on separate lines from their paragraphs.

### Testing Framework

The tests use real parsing and storage with fixed model responses.

## Results

Test assertions check stored passages, query sources and deletion.

## Conclusion

This fixture contains no benchmark measurements.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def docs_directory(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    (docs_dir / "doc1.txt").write_text(
        "Document 1 content. This is the first test document.\n"
        "It contains multiple sentences for proper chunking.\n"
        "Testing the ingestion of text files.\n"
    )

    (docs_dir / "doc2.txt").write_text(
        "Document 2 content. This is the second test document.\n"
        "It also has enough content for chunking.\n"
        "Multiple ingestion operations are tested.\n"
    )

    (docs_dir / "paper.md").write_text(
        "# Test Paper\n\n"
        "## Introduction\n"
        "This is a markdown document for testing.\n\n"
        "## Methods\n"
        "Testing with multiple formats.\n\n"
        "## Results\n"
        "The fixture has one Methods section.\n"
    )

    return docs_dir


class TestBasicPipelineWorkflow:
    def test_uppercase_evidence_is_ingested_with_exact_source_offsets(self, pipeline_rag: NexusRAG):
        content = "THE MEASURED CURRENT IS 42 MA.\n\nCALIBRATION PASSED."

        result = pipeline_rag.ingest_bytes(content.encode(), "measurements.txt", ".txt")

        assert result.success, result.error
        assert result.chunk_count > 0
        chunks = pipeline_rag.vector_store.get_all_chunks()
        assert all("[" not in chunk.content for chunk in chunks)
        for line in content.splitlines():
            if line:
                assert sum(line in chunk.content for chunk in chunks) == 1
        for chunk in chunks:
            assert chunk.start_offset is not None and chunk.end_offset is not None
            assert content[chunk.start_offset : chunk.end_offset] == chunk.content
        sources = pipeline_rag.bm25.retrieve("42 MA", top_k=1)
        assert len(sources) == 1
        assert "THE MEASURED CURRENT IS 42 MA." in sources[0].chunk.content

    def test_ingest_single_text_file(self, pipeline_rag: NexusRAG, pipeline_text_file: Path):
        result = pipeline_rag.ingest(pipeline_text_file)

        assert result.success is True
        assert result.filename == "test_document.txt"
        assert result.chunk_count > 0
        assert result.word_count > 0
        assert result.document_id
        assert result.error is None

    def test_ingest_markdown_file(self, pipeline_rag: NexusRAG, pipeline_markdown_file: Path):
        result = pipeline_rag.ingest(pipeline_markdown_file)

        assert result.success is True
        assert result.filename == "test_paper.md"
        assert result.chunk_count > 0
        assert result.word_count > 0

    @pytest.mark.parametrize("interleaved", [False, True])
    def test_docx_table_rows_remain_retrievable_after_reload(
        self, pipeline_rag: NexusRAG, interleaved: bool
    ):
        document = DocxDocument()
        before = "A paragraph before the measurement table."
        after = "A paragraph after the measurement table."
        if interleaved:
            document.add_heading("Measurements", level=1)
            document.add_paragraph(before)
        rows = [
            ["Sensor", "Reading", "Run"],
            ["calibration", "42.50", "007"],
            ["comparison", "", "-3.25"],
        ]
        table = document.add_table(rows=3, cols=3)
        for row, values in zip(table.rows, rows, strict=True):
            for cell, value in zip(row.cells, values, strict=True):
                cell.text = value
        if interleaved:
            document.add_paragraph(after)
        upload = BytesIO()
        document.save(upload)

        result = pipeline_rag.ingest_bytes(upload.getvalue(), "measurements.docx", ".docx")

        assert result.success, result.error
        assert result.chunk_count > 0
        stored = json.loads(pipeline_rag.document_store._doc_path(result.document_id).read_text())
        content = stored["content"]
        numeric_row = "calibration | 42.50 | 007"
        assert numeric_row in content
        empty_cell_row = next(
            line for line in content.splitlines() if line.startswith("comparison")
        )
        assert [cell.strip() for cell in empty_cell_row.split("|")] == ["comparison", "", "-3.25"]
        if interleaved:
            assert content.index(before) < content.index(numeric_row) < content.index(after)
        assert stored["sections"]
        for section in stored["sections"]:
            assert content[section["start_offset"] : section["end_offset"]] == section["content"]

        restarted = NexusRAG(settings=pipeline_rag.settings)
        restarted._embedder = pipeline_rag.embedder
        restarted._llm = pipeline_rag.llm
        matches = restarted.bm25.retrieve("42.50", top_k=1)
        assert len(matches) == 1
        match = matches[0].chunk
        assert numeric_row in match.content
        persisted = {chunk.id: chunk for chunk in restarted.vector_store.get_all_chunks()}[match.id]
        assert persisted.full_context == match.full_context
        start, end = persisted.start_offset, persisted.end_offset
        assert start is not None and end is not None
        assert numeric_row in content[start:end]
        assert persisted.context_before == content[max(0, start - 150) : start].strip()
        assert persisted.context_after == content[end : end + 150].strip()

        prepared = restarted.orchestrator.synthesizer.prepare("What reading was recorded?", matches)
        assert prepared.error is None
        assert prepared.sources[0].document_name == "measurements.docx"
        assert prepared.sources[0].chunk_id == match.id
        assert (
            prepared.sources[0].full_content or prepared.sources[0].content
        ) == match.full_context
        assert numeric_row in prepared.prompt

    def test_ingest_updates_inventory_and_statistics(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):

        ingest_result = pipeline_rag.ingest(pipeline_text_file)
        assert ingest_result.success is True
        document_id = ingest_result.document_id

        docs = pipeline_rag.list_documents()
        assert len(docs) == 1
        assert docs[0]["id"] == document_id

        stats = pipeline_rag.get_stats()
        assert stats.total_documents == 1
        assert stats.total_chunks == ingest_result.chunk_count

    def test_delete_document_workflow(self, pipeline_rag: NexusRAG, pipeline_text_file: Path):

        ingest_result = pipeline_rag.ingest(pipeline_text_file)
        document_id = ingest_result.document_id

        assert len(pipeline_rag.list_documents()) == 1

        deleted = pipeline_rag.delete_document(document_id)
        assert deleted is True

        assert len(pipeline_rag.list_documents()) == 0

        stats = pipeline_rag.get_stats()
        assert stats.total_documents == 0
        assert stats.total_chunks == 0

    def test_delete_nonexistent_document(self, pipeline_rag: NexusRAG):
        deleted = pipeline_rag.delete_document("nonexistent_id")
        assert deleted is False

    def test_query_propagates_failed_storage_reads_before_returning_an_answer(
        self, pipeline_rag: NexusRAG, monkeypatch
    ):
        failure = OSError("Index read failed")
        monkeypatch.setattr(
            pipeline_rag.vector_store.table, "count_rows", MagicMock(side_effect=failure)
        )
        pipeline_rag._orchestrator = MagicMock()

        with pytest.raises(VectorStoreError, match="row count") as raised:
            pipeline_rag.query("What was measured?")

        assert raised.value.__cause__ is failure
        pipeline_rag._orchestrator.query.assert_not_called()

    @pytest.mark.parametrize("lazy", [False, True])
    def test_failed_sparse_read_can_retry_without_publishing_an_empty_index(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch, lazy
    ):
        original = pipeline_rag
        result = original.ingest(pipeline_text_file)
        assert result.success
        pipeline_rag = NexusRAG(settings=original.settings)
        pipeline_rag._embedder = original.embedder
        get_all_chunks = pipeline_rag.vector_store.get_all_chunks
        monkeypatch.setattr(
            pipeline_rag.vector_store,
            "get_all_chunks",
            MagicMock(side_effect=VectorStoreError("Cannot read passages from the vector index.")),
        )

        with pytest.raises(VectorStoreError, match="read passages"):
            if lazy:
                _ = pipeline_rag.bm25
            else:
                pipeline_rag._rebuild_bm25_index()

        assert pipeline_rag._bm25 is None
        monkeypatch.setattr(pipeline_rag.vector_store, "get_all_chunks", get_all_chunks)
        assert pipeline_rag.bm25.count() == result.chunk_count

    @pytest.mark.parametrize("stage", ["enumeration", "dense_before", "dense_after", "sparse"])
    def test_partial_deletion_reconciles_indexes_and_can_be_retried(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch, stage
    ):
        result = pipeline_rag.ingest(pipeline_text_file)
        other = pipeline_rag.ingest_bytes(
            b"A separate document records voltage readings for sensor C.", "voltage.txt", ".txt"
        )
        assert result.success and other.success
        other_ids = pipeline_rag.vector_store.get_chunk_ids_by_document(other.document_id)
        failure = OSError(f"Failed during {stage}")
        if stage == "enumeration":
            owner, method = pipeline_rag.vector_store, "get_chunk_ids_by_document"
        elif stage.startswith("dense"):
            owner, method = pipeline_rag.vector_store, "delete_by_document"
        else:
            owner, method = pipeline_rag.bm25, "remove"
        operation = getattr(owner, method)

        def fail(*args, **kwargs):
            if stage == "dense_after":
                operation(*args, **kwargs)
            raise failure

        monkeypatch.setattr(owner, method, fail)
        with pytest.raises(OSError, match=f"Failed during {stage}"):
            pipeline_rag.delete_document(result.document_id)
        monkeypatch.setattr(owner, method, operation)

        assert pipeline_rag.document_store.exists(result.document_id)
        assert pipeline_rag.document_store.exists(other.document_id)
        assert pipeline_rag.vector_store.get_chunk_ids_by_document(other.document_id) == other_ids
        assert pipeline_rag.bm25.count() == pipeline_rag.vector_store.count()
        if stage in {"dense_after", "sparse"}:
            assert pipeline_rag.vector_store.get_chunk_ids_by_document(result.document_id) == set()
            assert pipeline_rag.bm25.retrieve("thirty-second") == []
        else:
            assert pipeline_rag.bm25.count() == result.chunk_count + other.chunk_count

        assert pipeline_rag.delete_document(result.document_id)
        assert not pipeline_rag.document_store.exists(result.document_id)
        assert pipeline_rag.bm25.count() == pipeline_rag.vector_store.count() == other.chunk_count
        assert {chunk.id for chunk in pipeline_rag.vector_store.get_all_chunks()} == other_ids
        restarted = NexusRAG(settings=pipeline_rag.settings)
        restarted._embedder = pipeline_rag.embedder
        assert restarted.bm25.count() == other.chunk_count

    def test_failed_deletion_recovery_discards_stale_sparse_handles(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch
    ):
        result = pipeline_rag.ingest(pipeline_text_file)
        assert result.success
        sparse = pipeline_rag.bm25
        pipeline_rag._orchestrator = MagicMock()
        get_all_chunks = pipeline_rag.vector_store.get_all_chunks
        monkeypatch.setattr(
            sparse, "remove", MagicMock(side_effect=OSError("Sparse deletion failed"))
        )
        monkeypatch.setattr(
            pipeline_rag.vector_store,
            "get_all_chunks",
            MagicMock(side_effect=VectorStoreError("Storage remains unavailable")),
        )

        with pytest.raises(OSError, match="Sparse deletion failed"):
            pipeline_rag.delete_document(result.document_id)

        assert pipeline_rag._bm25 is None
        assert pipeline_rag._orchestrator is None
        assert pipeline_rag.document_store.exists(result.document_id)
        assert pipeline_rag.vector_store.count() == 0
        monkeypatch.setattr(pipeline_rag.vector_store, "get_all_chunks", get_all_chunks)
        assert pipeline_rag.bm25 is not sparse
        assert pipeline_rag.bm25.count() == 0
        assert pipeline_rag.delete_document(result.document_id)
        assert len(pipeline_rag.document_store.list_with_metadata()) == 0

    def test_delete_document_with_more_than_100000_chunks_removes_all_sparse_entries(
        self, pipeline_rag: NexusRAG
    ):
        pipeline_rag._embedder.embedding_config = replace(
            pipeline_rag.embedder.embedding_config, dimension=4
        )
        document = pipeline_rag.parser.parse_bytes(b"Current is 42 mA.", "large.txt", ".txt")
        count = 100_001
        chunks = [
            Chunk(id=f"large-{i}", content="Current is 42 mA.", document_id=document.id)
            for i in range(count)
        ]
        pipeline_rag.document_store.add(document)
        pipeline_rag.vector_store.add(chunks, np.ones((count, 4), dtype=np.float32))
        assert pipeline_rag.bm25.count() == count

        assert pipeline_rag.delete_document(document.id)

        assert pipeline_rag.vector_store.count() == pipeline_rag.bm25.count() == 0
        assert pipeline_rag.bm25.retrieve("42 mA") == []
        assert len(pipeline_rag.document_store.list_with_metadata()) == 0

    @pytest.mark.parametrize("failure", ["unlink", "index_save"])
    def test_delete_metadata_failure_keeps_retrieval_indexes_synced(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch, failure: str
    ):
        result = pipeline_rag.ingest(pipeline_text_file)
        assert result.success
        store = pipeline_rag.document_store
        error = OSError("document metadata unavailable")
        if failure == "unlink":
            monkeypatch.setattr(Path, "unlink", MagicMock(side_effect=error))
        else:
            monkeypatch.setattr(store, "_save_index", MagicMock(side_effect=error))

        with pytest.raises(OSError, match="document metadata unavailable"):
            pipeline_rag.delete_document(result.document_id)

        assert pipeline_rag.vector_store.count() == pipeline_rag.bm25.count() == 0
        assert store.exists(result.document_id) is (failure == "unlink")
        assert pipeline_rag.query("What methods were used?").sources == []


class TestMultiFileIngestion:
    def test_ingesting_several_files_updates_stats(
        self, pipeline_rag: NexusRAG, docs_directory: Path
    ):
        results = [pipeline_rag.ingest(path) for path in sorted(docs_directory.iterdir())]

        assert len(results) == 3
        assert all(result.success and result.chunk_count > 0 for result in results)
        assert len(pipeline_rag.list_documents()) == 3
        total_chunks = sum(r.chunk_count for r in results)
        total_words = sum(r.word_count for r in results)

        stats = pipeline_rag.get_stats()
        assert stats.total_documents == 3
        assert stats.total_chunks == total_chunks
        assert stats.total_words == total_words


class TestConcurrentWrites:
    @pytest.mark.parametrize("blocked_stage", ["vector_count", "llm_check"])
    def test_stats_snapshot_serializes_stores_without_locking_network_checks(
        self, pipeline_rag: NexusRAG, monkeypatch, blocked_stage
    ):
        store = pipeline_rag.vector_store
        count_rows = store.count
        entered, release = Event(), Event()
        lock = _ObservedWriteLock()
        pipeline_rag._write_lock = lock

        def hold(operation):
            entered.set()
            assert release.wait(timeout=5)
            return operation()

        if blocked_stage == "vector_count":
            monkeypatch.setattr(store, "count", lambda: hold(count_rows))
        else:
            monkeypatch.setattr(pipeline_rag.llm, "is_available", lambda: hold(lambda: True))

        with ThreadPoolExecutor(max_workers=2) as workers:
            stats_result = workers.submit(pipeline_rag.get_stats)
            try:
                assert entered.wait(timeout=5)
                ingest_result = workers.submit(
                    pipeline_rag.ingest_bytes,
                    b"A new document contains current readings from sensor A.",
                    "current.txt",
                    ".txt",
                )
                if blocked_stage == "vector_count":
                    lock.wait_for_threads(2)
                    assert not ingest_result.done()
                else:
                    assert ingest_result.result(timeout=5).success
            finally:
                release.set()
            snapshot = stats_result.result(timeout=5)
            assert ingest_result.result(timeout=5).success

        assert snapshot.total_documents == snapshot.total_chunks == snapshot.total_words == 0
        assert len(pipeline_rag.document_store.list_with_metadata()) == 1
        assert pipeline_rag.vector_store.count() > 0

    @pytest.mark.parametrize("first_operation", ["ingest", "list_documents"])
    def test_ingest_during_cold_index_load(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch, first_operation
    ):
        pipeline_rag.vector_store.count()
        store = pipeline_rag.document_store
        load_index = store._load_index
        index_loaded = Event()
        release_index = Event()
        second_started = Event()

        def delayed_load():
            index = load_index()
            if not index_loaded.is_set():
                index_loaded.set()
                assert release_index.wait(timeout=10)
            return index

        def second_ingest():
            second_started.set()
            return pipeline_rag.ingest(pipeline_text_file)

        monkeypatch.setattr(store, "_load_index", delayed_load)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = (
                pool.submit(pipeline_rag.ingest, pipeline_text_file)
                if first_operation == "ingest"
                else pool.submit(pipeline_rag.list_documents)
            )
            try:
                assert index_loaded.wait(timeout=5)
                second = pool.submit(second_ingest)
                assert second_started.wait(timeout=5)
                with pytest.raises(TimeoutError):
                    second.result(timeout=1)
            finally:
                release_index.set()
            first_result = first.result(timeout=10)
            results = [second.result(timeout=10)]
            if first_operation == "ingest":
                results.append(first_result)

        successful = [result for result in results if result.success]
        rejected = [result for result in results if not result.success]
        assert len(successful) == 1
        if first_operation == "ingest":
            assert len(rejected) == 1
            assert rejected[0].error == "Document already exists"
        else:
            assert not rejected
        assert len(store.list_with_metadata()) == 1
        assert pipeline_rag.vector_store.count() == successful[0].chunk_count
        assert pipeline_rag.bm25.count() == successful[0].chunk_count

    def test_parallel_duplicate_ingest_preserves_document(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch
    ):
        pipeline_rag.vector_store.count()
        len(pipeline_rag.document_store.list_with_metadata())
        embed = pipeline_rag.embedder.embed
        barrier = Barrier(2)

        def simultaneous_embed(*args, **kwargs):
            result = embed(*args, **kwargs)
            barrier.wait(timeout=10)
            return result

        monkeypatch.setattr(pipeline_rag.embedder, "embed", simultaneous_embed)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(pipeline_rag.ingest, pipeline_text_file) for _ in range(2)]
            results = [future.result(timeout=20) for future in futures]

        successful = [result for result in results if result.success]
        rejected = [result for result in results if not result.success]
        assert len(successful) == len(rejected) == 1
        assert rejected[0].error == "Document already exists"
        assert len(pipeline_rag.document_store.list_with_metadata()) == 1
        assert pipeline_rag.vector_store.count() == successful[0].chunk_count
        assert pipeline_rag.bm25.count() == successful[0].chunk_count

    def test_parallel_ingest_keeps_indexes_consistent(self, pipeline_rag: NexusRAG, tmp_path: Path):
        files = []
        for i in range(4):
            p = tmp_path / f"doc_{i}.txt"
            p.write_text(f"Document number {i}.\n" + f"Unique sentence {i} repeated often.\n" * 30)
            files.append(p)

        barrier = Barrier(len(files))

        def worker(fp: Path) -> IngestResult:
            barrier.wait(timeout=5)
            return pipeline_rag.ingest(fp)

        with ThreadPoolExecutor(max_workers=len(files)) as workers:
            futures = [workers.submit(worker, path) for path in files]
            results = [future.result(timeout=20) for future in futures]

        assert len(results) == len(files)
        assert all(r.success for r in results)
        total_chunks = sum(r.chunk_count for r in results)
        assert pipeline_rag.vector_store.count() == total_chunks
        assert pipeline_rag.bm25.count() == total_chunks


class TestDuplicateDetection:
    def test_duplicate_file_detection(self, pipeline_rag: NexusRAG, pipeline_text_file: Path):

        result1 = pipeline_rag.ingest(pipeline_text_file)
        assert result1.success is True

        result2 = pipeline_rag.ingest(pipeline_text_file)
        assert result2.success is False
        assert result2.error == "Document already exists"
        assert result2.chunk_count == 0

        docs = pipeline_rag.list_documents()
        assert len(docs) == 1

    def test_same_file_in_two_folders_is_ingested_once(
        self, pipeline_rag: NexusRAG, tmp_path: Path
    ):
        content = "Test content for duplicate detection.\n" * 5
        paths = []
        for name in ("first", "second"):
            directory = tmp_path / name
            directory.mkdir()
            paths.append(directory / "paper.txt")
            paths[-1].write_text(content)

        results = [pipeline_rag.ingest(path) for path in paths]

        assert len(results) == 2
        successful = [r for r in results if r.success]
        rejected = [r for r in results if not r.success]
        assert len(successful) == len(rejected) == 1
        assert rejected[0].error == "Document already exists"
        assert rejected[0].document_id == successful[0].document_id
        assert len(pipeline_rag.document_store.list_with_metadata()) == 1
        assert pipeline_rag.vector_store.count() == successful[0].chunk_count
        assert pipeline_rag.bm25.count() == successful[0].chunk_count


class TestClearOperations:
    def test_clear_metadata_failure_keeps_retrieval_indexes_synced(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch
    ):
        assert pipeline_rag.ingest(pipeline_text_file).success
        monkeypatch.setattr(
            pipeline_rag.document_store,
            "clear",
            MagicMock(side_effect=OSError("document metadata unavailable")),
        )

        with pytest.raises(OSError, match="document metadata unavailable"):
            pipeline_rag.clear_all()

        assert pipeline_rag.vector_store.count() == pipeline_rag.bm25.count() == 0
        assert pipeline_rag.query("What methods were used?").sources == []

    def test_clear_then_reingest_updates_existing_retriever(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):
        assert pipeline_rag.ingest(pipeline_text_file).success
        sparse = pipeline_rag.orchestrator.retriever.base.sparse
        assert sparse.retrieve("research")

        pipeline_rag.clear_all()
        assert not sparse.retrieve("research")
        result = pipeline_rag.ingest(pipeline_text_file)

        assert result.success
        assert sparse.count() == result.chunk_count
        assert sparse.retrieve("research")

    def test_clear_all_documents(self, pipeline_rag: NexusRAG, docs_directory: Path):

        for path in sorted(docs_directory.iterdir()):
            pipeline_rag.ingest(path)
        assert pipeline_rag.get_stats().total_documents == 3

        pipeline_rag.clear_all()

        assert pipeline_rag.get_stats().total_documents == 0
        assert pipeline_rag.get_stats().total_chunks == 0
        assert len(pipeline_rag.list_documents()) == 0

    def test_clear_all_then_reingest(self, pipeline_rag: NexusRAG, pipeline_text_file: Path):

        result1 = pipeline_rag.ingest(pipeline_text_file)
        assert result1.success is True

        pipeline_rag.clear_all()
        assert pipeline_rag.get_stats().total_documents == 0

        result2 = pipeline_rag.ingest(pipeline_text_file)
        assert result2.success is True


class TestStatisticsAndMetadata:
    def test_get_stats_reports_an_unavailable_llm(self, pipeline_rag: NexusRAG):
        pipeline_rag.llm.is_available.return_value = False

        stats = pipeline_rag.get_stats()

        assert stats.llm_available is False

    def test_get_stats_empty_system(self, pipeline_rag: NexusRAG):
        stats = pipeline_rag.get_stats()

        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        assert stats.total_words == 0
        assert stats.llm_available is True
        assert stats.embedding_model == "test-model"
        assert stats.llm_model == "test-llm"

    def test_get_stats_after_ingestion(self, pipeline_rag: NexusRAG, pipeline_text_file: Path):
        result = pipeline_rag.ingest(pipeline_text_file)

        stats = pipeline_rag.get_stats()
        assert stats.total_documents == 1
        assert stats.total_chunks == result.chunk_count
        assert stats.total_words > 0

    def test_list_documents_empty(self, pipeline_rag: NexusRAG):
        docs = pipeline_rag.list_documents()
        assert docs == []

    def test_list_documents_with_metadata(self, pipeline_rag: NexusRAG, pipeline_text_file: Path):
        ingest_result = pipeline_rag.ingest(pipeline_text_file)

        docs = pipeline_rag.list_documents()
        assert len(docs) == 1

        doc = docs[0]
        assert doc["id"] == ingest_result.document_id
        assert doc["filename"] == pipeline_text_file.name

    def test_stats_with_multiple_documents(self, pipeline_rag: NexusRAG, docs_directory: Path):
        results = [pipeline_rag.ingest(path) for path in sorted(docs_directory.iterdir())]

        stats = pipeline_rag.get_stats()
        assert stats.total_documents == len(results)

        total_chunks_expected = sum(r.chunk_count for r in results)
        assert stats.total_chunks == total_chunks_expected


class TestPipelineErrors:
    def test_ingest_empty_file(self, pipeline_rag: NexusRAG, tmp_path: Path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = pipeline_rag.ingest(empty_file)

        assert result.success is False
        assert result.error == "No content extracted"
        assert result.chunk_count == 0

    def test_ingest_nonexistent_file(self, pipeline_rag: NexusRAG):
        result = pipeline_rag.ingest("/nonexistent/path/file.txt")

        assert result.success is False
        assert result.error == "File not found: file.txt"

    def test_ingest_unsupported_format(self, pipeline_rag: NexusRAG, tmp_path: Path):
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("Some content")

        result = pipeline_rag.ingest(unsupported)

        assert result.success is False
        assert result.error == "Unsupported format: .xyz"

    def test_ingest_bytes_empty(self, pipeline_rag: NexusRAG):
        result = pipeline_rag.ingest_bytes(b"", "empty.txt", ".txt")

        assert result.success is False
        assert result.chunk_count == 0

    def test_ingest_bytes_valid(self, pipeline_rag: NexusRAG):
        content = b"This is test content for bytes ingestion.\n" * 5
        result = pipeline_rag.ingest_bytes(content, "test.txt", ".txt")

        assert result.success is True
        assert result.filename == "test.txt"
        assert result.chunk_count > 0

    def test_ingest_bytes_without_extension(self, pipeline_rag: NexusRAG):
        content = b"Test content.\n" * 5

        result = pipeline_rag.ingest_bytes(content, "test.txt", "txt")

        assert result.success is True


class TestLazyLoading:
    def test_components_not_loaded_initially(self, pipeline_rag: NexusRAG):

        assert pipeline_rag._parser is None
        assert pipeline_rag._chunker is None
        assert pipeline_rag._vector_store is None
        assert pipeline_rag._document_store is None
        assert pipeline_rag._bm25 is None

    @pytest.mark.parametrize("component", ["parser", "chunker", "document_store"])
    def test_component_lazy_loads_on_access(self, pipeline_rag, component):
        assert getattr(pipeline_rag, "_" + component) is None
        loaded = getattr(pipeline_rag, component)
        assert loaded is not None
        assert getattr(pipeline_rag, "_" + component) is loaded


class TestIngestBytes:
    def test_ingest_bytes_text(self, pipeline_rag: NexusRAG):
        content = b"""
        # Test Document

        Sensor A records temperature in degrees Celsius.
        Each reading includes a timestamp.

        Sensor B uses the same sampling interval.
        Readings with matching timestamps form a measurement pair.
        """

        result = pipeline_rag.ingest_bytes(content, "bytes_doc.txt", ".txt")

        assert result.success is True
        assert result.filename == "bytes_doc.txt"
        assert result.chunk_count > 0

    def test_ingest_bytes_md_format(self, pipeline_rag: NexusRAG):
        content = b"""# Markdown Test

## Section 1
Content here for testing.

## Section 2
More content for proper chunking.
"""

        result = pipeline_rag.ingest_bytes(content, "test.md", ".md")

        assert result.success is True
        assert result.chunk_count > 0

    def test_ingest_bytes_duplicate_detection(self, pipeline_rag: NexusRAG):
        content = b"Unique test content for duplicate detection.\n" * 5

        result1 = pipeline_rag.ingest_bytes(content, "dup.txt", ".txt")
        assert result1.success is True

        result2 = pipeline_rag.ingest_bytes(content, "dup.txt", ".txt")
        assert result2.success is False
        assert result2.document_id == result1.document_id
        assert "already exists" in (result2.error or "")


class TestEdgeCases:
    def test_ingest_very_short_document(self, pipeline_rag: NexusRAG, tmp_path: Path):
        short_file = tmp_path / "short.txt"
        short_file.write_text("Short content.")

        result = pipeline_rag.ingest(short_file)

        assert result.success, result.error
        assert result.chunk_count == 1
        assert result.word_count == 2

    def test_ingest_large_document(self, pipeline_rag: NexusRAG, tmp_path: Path):
        large_file = tmp_path / "large.txt"

        large_content = "This is a test sentence that will be repeated many times. " * 200
        large_file.write_text(large_content)

        result = pipeline_rag.ingest(large_file)

        assert result.success is True
        assert result.chunk_count > 1

    def test_concurrent_operations_sequence(self, pipeline_rag: NexusRAG, tmp_path: Path):
        file1 = tmp_path / "doc1.txt"
        file2 = tmp_path / "doc2.txt"

        content1 = "Document 1 content. " * 20
        content2 = "Document 2 content. " * 20

        file1.write_text(content1)
        file2.write_text(content2)

        result1 = pipeline_rag.ingest(file1)
        result2 = pipeline_rag.ingest(file2)

        assert result1.success is True
        assert result2.success is True

        docs = pipeline_rag.list_documents()
        assert len(docs) == 2

        deleted = pipeline_rag.delete_document(result1.document_id)
        assert deleted is True

        docs = pipeline_rag.list_documents()
        assert len(docs) == 1


class TestIntegrationWithRealComponents:
    def test_restart_rebuilds_sparse_index_from_persisted_chunks(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):
        result = pipeline_rag.ingest(pipeline_text_file)
        assert result.success

        restarted = NexusRAG(settings=pipeline_rag.settings)
        restarted._embedder = pipeline_rag.embedder
        restarted._llm = pipeline_rag.llm

        assert restarted.bm25.count() == result.chunk_count
        assert restarted.bm25.retrieve("research")
        assert {chunk.document_id for chunk in restarted.vector_store.get_all_chunks()} == {
            result.document_id
        }

    def test_with_real_parser_and_chunker(self, pipeline_rag: NexusRAG, pipeline_text_file: Path):
        result = pipeline_rag.ingest(pipeline_text_file)

        assert result.success is True
        assert result.chunk_count > 0

        stats = pipeline_rag.get_stats()
        assert stats.total_chunks == result.chunk_count

    def test_document_store_persistence(self, pipeline_rag: NexusRAG, pipeline_text_file: Path):
        ingest_result = pipeline_rag.ingest(pipeline_text_file)
        assert ingest_result.success is True

        docs = pipeline_rag.list_documents()
        assert len(docs) == 1

        doc = docs[0]
        assert doc["id"] == ingest_result.document_id


class TestParametrized:
    @pytest.mark.parametrize(
        "filename,content",
        [
            ("test1.txt", "Test content one.\n" * 10),
            ("test2.txt", "Test content two.\n" * 10),
            ("test3.txt", "Different content.\n" * 10),
        ],
    )
    def test_ingest_multiple_variations(
        self, pipeline_rag: NexusRAG, tmp_path: Path, filename: str, content: str
    ):
        file_path = tmp_path / filename
        file_path.write_text(content)

        result = pipeline_rag.ingest(file_path)

        assert result.success is True
        assert result.filename == filename

    @pytest.mark.parametrize(
        "format_ext,format_name",
        [
            (".txt", "text"),
            (".md", "markdown"),
        ],
    )
    def test_various_formats(
        self, pipeline_rag: NexusRAG, tmp_path: Path, format_ext: str, format_name: str
    ):
        file_path = tmp_path / f"doc{format_ext}"

        if format_ext == ".txt":
            content = "Test content for text format.\n" * 10
        else:
            content = "# Test\n\nContent for markdown format.\n" * 5

        file_path.write_text(content)

        result = pipeline_rag.ingest(file_path)

        assert result.success is True


class TestIngestRollback:
    @pytest.mark.parametrize("invalid_field", ["content", "metadata"])
    def test_prewrite_validation_preserves_preexisting_document_and_vectors(
        self, pipeline_rag: NexusRAG, invalid_field
    ):
        document = ParsedDocument("existing", "Original evidence.", {"filename": "current.txt"})
        chunks = pipeline_rag.chunker.chunk(document)
        embeddings = pipeline_rag.embedder.embed([chunk.content for chunk in chunks])
        assert len(pipeline_rag.document_store.list_with_metadata()) == 0
        writer = DocumentStore(pipeline_rag.document_store.path)
        writer.add(document)
        writer.update_metadata(document.id, "notes", {"tags": ["original"]})
        pipeline_rag.vector_store.add(chunks, embeddings)
        files_before = {path.name: path.read_bytes() for path in writer.path.iterdir()}
        vectors_before = pipeline_rag.vector_store.table.to_arrow()
        invalid = replace(document, **{invalid_field: 42})

        with pytest.raises(ValueError):
            pipeline_rag._persist(invalid, chunks, embeddings)

        assert {path.name: path.read_bytes() for path in writer.path.iterdir()} == files_before
        assert pipeline_rag.vector_store.table.to_arrow().equals(vectors_before)
        assert DocumentStore(writer.path).list_with_metadata() == writer.list_with_metadata()
        assert pipeline_rag.bm25.count() == pipeline_rag.vector_store.count() == len(chunks)

    @pytest.mark.parametrize("entry_point", ["ingest", "ingest_bytes"])
    def test_duplicate_rejected_after_cached_inventory_preserves_existing_stores(
        self, pipeline_rag: NexusRAG, tmp_path: Path, entry_point
    ):
        source = tmp_path / "current.txt"
        source.write_text("The measured current is forty-two milliamps.", encoding="utf-8")
        document = pipeline_rag.parser.parse(source)
        chunks = pipeline_rag.chunker.chunk(document)
        assert len(pipeline_rag.document_store.list_with_metadata()) == 0

        writer = DocumentStore(pipeline_rag.document_store.path)
        writer.add(document)
        writer.update_metadata(document.id, "notes", {"tags": ["original"]})
        writer.update_metadata(document.id, "chunk_count", len(chunks))
        pipeline_rag.vector_store.add(
            chunks, pipeline_rag.embedder.embed([chunk.content for chunk in chunks])
        )
        existing_metadata = writer.list_with_metadata()
        existing_files = {path.name: path.read_bytes() for path in writer.path.iterdir()}
        existing_vectors = pipeline_rag.vector_store.table.to_arrow()
        assert len(pipeline_rag.document_store.list_with_metadata()) == 0

        if entry_point == "ingest":
            result = pipeline_rag.ingest(source)
        else:
            result = pipeline_rag.ingest_bytes(source.read_bytes(), source.name, ".txt")

        assert not result.success
        assert result.error == "Document already exists"
        assert result.document_id == document.id
        assert {path.name: path.read_bytes() for path in writer.path.iterdir()} == existing_files
        assert DocumentStore(writer.path).list_with_metadata() == existing_metadata
        assert pipeline_rag.document_store.list_with_metadata() == existing_metadata
        assert pipeline_rag.vector_store.table.to_arrow().equals(existing_vectors)
        assert pipeline_rag.bm25.count() == pipeline_rag.vector_store.count() == len(chunks)

    def test_failed_count_during_ingest_keeps_dense_rows_attached_to_a_document(
        self, pipeline_rag: NexusRAG, monkeypatch
    ):
        table = pipeline_rag.vector_store.table
        count_rows = table.count_rows
        failure = OSError("Row count temporarily unavailable")
        monkeypatch.setattr(table, "count_rows", MagicMock(side_effect=failure))

        with pytest.raises(VectorStoreError, match="row count") as raised:
            pipeline_rag.ingest_bytes(
                b"The measured current is forty-two milliamps.", "current.txt", ".txt"
            )

        assert raised.value.__cause__ is failure
        monkeypatch.setattr(table, "count_rows", count_rows)
        chunks = pipeline_rag.vector_store.get_all_chunks()
        documents = pipeline_rag.document_store.list_with_metadata()
        assert chunks and len(documents) == 1
        assert {chunk.document_id for chunk in chunks} == set(documents)
        assert pipeline_rag.bm25.count() == pipeline_rag.vector_store.count() == len(chunks)
        document_id = chunks[0].document_id
        restarted = NexusRAG(settings=pipeline_rag.settings)
        restarted._embedder = pipeline_rag.embedder
        assert restarted.document_store.exists(document_id)
        assert restarted.bm25.count() == len(chunks)
        assert restarted.delete_document(document_id)
        assert len(restarted.document_store.list_with_metadata()) == 0
        assert restarted.bm25.count() == restarted.vector_store.count() == 0

    @pytest.mark.parametrize(
        "stage", ["dense_before", "dense_partial", "dense_after", "sparse_read", "document_file"]
    )
    def test_failed_ingest_rollback_preserves_the_original_error_and_can_be_recovered(
        self, pipeline_rag: NexusRAG, monkeypatch, stage
    ):
        other = pipeline_rag.ingest_bytes(
            b"A separate document records voltage for sensor B.", "voltage.txt", ".txt"
        )
        assert other.success
        document = pipeline_rag.parser.parse_bytes(
            b"The measured current is forty-two milliamps.", "current.txt", ".txt"
        )
        chunks = [
            Chunk(
                id=f"current-{index}",
                content=f"Current measurement {index}",
                document_id=document.id,
            )
            for index in range(3)
        ]
        original_error = OSError("Document metadata update failed")
        monkeypatch.setattr(
            pipeline_rag.document_store, "update_metadata", MagicMock(side_effect=original_error)
        )
        if stage.startswith("dense"):
            owner, method = pipeline_rag.vector_store, "delete_by_document"
        elif stage == "sparse_read":
            owner, method = pipeline_rag.vector_store, "get_all_chunks"
        else:
            owner, method = pipeline_rag.document_store, "delete"
        cleanup = getattr(owner, method)

        def fail_cleanup(*args, **kwargs):
            if stage == "dense_partial":
                pipeline_rag.vector_store.table.delete(f"id = '{chunks[0].id}'")
            elif stage == "dense_after":
                cleanup(*args, **kwargs)
            raise OSError(f"Rollback failed during {stage}")

        monkeypatch.setattr(owner, method, fail_cleanup)
        with pytest.raises(OSError, match="Document metadata update failed") as raised:
            pipeline_rag._persist(document, chunks, np.ones((3, 384), dtype=np.float32))

        assert raised.value is original_error
        assert pipeline_rag.document_store.exists(document.id)
        assert pipeline_rag.document_store.exists(other.document_id)
        if stage == "sparse_read":
            assert pipeline_rag._bm25 is None
            assert pipeline_rag._orchestrator is None
        monkeypatch.setattr(owner, method, cleanup)
        expected_remaining = 3 if stage == "dense_before" else 2 if stage == "dense_partial" else 0
        assert len(pipeline_rag.vector_store.get_chunk_ids_by_document(document.id)) == (
            expected_remaining
        )
        assert (
            pipeline_rag.bm25.count()
            == pipeline_rag.vector_store.count()
            == other.chunk_count + expected_remaining
        )
        assert all(
            pipeline_rag.document_store.exists(chunk.document_id)
            for chunk in pipeline_rag.vector_store.get_all_chunks()
        )
        restarted = NexusRAG(settings=pipeline_rag.settings)
        restarted._embedder = pipeline_rag.embedder
        assert restarted.bm25.count() == other.chunk_count + expected_remaining
        assert restarted.delete_document(document.id)
        assert len(restarted.document_store.list_with_metadata()) == 1
        assert restarted.document_store.exists(other.document_id)
        assert restarted.bm25.count() == restarted.vector_store.count() == other.chunk_count

    def test_document_index_failure_rolls_back_document_file(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch
    ):
        store = pipeline_rag.document_store
        store.list_with_metadata()
        save_index = store._save_index
        failed = False

        def fail_first_save():
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("simulated index write failure")
            save_index()

        monkeypatch.setattr(store, "_save_index", fail_first_save)
        result = pipeline_rag.ingest(pipeline_text_file)

        assert not result.success
        assert store.list_with_metadata() == {}
        assert pipeline_rag.vector_store.count() == 0
        assert pipeline_rag.bm25.count() == 0
        assert all(path == store._index_path for path in store.path.glob("*.json"))
        assert pipeline_rag.ingest(pipeline_text_file).success

    def test_vector_failure_rolls_back_document_store(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch
    ):
        def boom(chunks, embeddings):
            raise RuntimeError("simulated storage failure")

        monkeypatch.setattr(pipeline_rag.vector_store, "add", boom)

        result = pipeline_rag.ingest(pipeline_text_file)

        assert result.success is False
        assert len(pipeline_rag.document_store.list_with_metadata()) == 0
        assert pipeline_rag.bm25.count() == 0

    def test_bm25_failure_rolls_back_all_stores(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch
    ):
        def boom(chunks):
            raise RuntimeError("simulated index failure")

        monkeypatch.setattr(pipeline_rag.bm25, "add_incremental", boom)

        result = pipeline_rag.ingest(pipeline_text_file)

        assert result.success is False
        assert len(pipeline_rag.document_store.list_with_metadata()) == 0
        assert pipeline_rag.vector_store.count() == 0


class _ObservedWriteLock:
    def __init__(self):
        self.lock = RLock()
        self.attempts = Queue()

    def __enter__(self):
        self.attempts.put(get_ident())
        if not self.lock.acquire(timeout=5):
            raise TimeoutError("Pipeline lock was not released")
        return self

    def __exit__(self, *_args):
        self.lock.release()

    def wait_for_attempts(self, count):
        return [self.attempts.get(timeout=5) for _ in range(count)]

    def wait_for_threads(self, count):
        threads = set()
        while len(threads) < count:
            threads.add(self.attempts.get(timeout=5))
        return threads


class TestLazyResourceLifecycle:
    @pytest.mark.parametrize("resource", ["bm25", "orchestrator"])
    @pytest.mark.parametrize("fail_first", [False, True])
    def test_lazy_retrieval_handles_wait_for_a_complete_sparse_index(
        self, pipeline_rag: NexusRAG, monkeypatch, resource, fail_first
    ):
        store = pipeline_rag.vector_store
        get_all_chunks = store.get_all_chunks
        lock = _ObservedWriteLock()
        pipeline_rag._write_lock = lock
        entered, release = Event(), Event()
        reads = 0

        def hold_first_read():
            nonlocal reads
            reads += 1
            if reads == 1:
                entered.set()
                assert release.wait(timeout=5)
                if fail_first:
                    raise VectorStoreError("Sparse initialization failed")
            return get_all_chunks()

        monkeypatch.setattr(store, "get_all_chunks", hold_first_read)
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(lambda: getattr(pipeline_rag, resource))
            try:
                assert entered.wait(timeout=5)
                assert pipeline_rag._bm25 is None
                assert pipeline_rag._orchestrator is None
                second = workers.submit(lambda: getattr(pipeline_rag, resource))
                lock.wait_for_threads(2)
                assert not second.done()
            finally:
                release.set()
            if fail_first:
                with pytest.raises(VectorStoreError, match="Sparse initialization failed"):
                    first.result(timeout=5)
                assert second.result(timeout=5) is getattr(pipeline_rag, resource)
                assert reads == 2
            else:
                assert first.result(timeout=5) is second.result(timeout=5)
                assert reads == 1
        assert pipeline_rag.bm25.count() == 0

    @pytest.mark.parametrize(
        "resource,factory_name",
        [("embedder", "Embedder"), ("vector_store", "VectorStore"), ("llm", "LLMClient")],
    )
    def test_concurrent_first_access_constructs_one_handle(
        self, test_settings, mock_embedder, monkeypatch, resource, factory_name
    ):
        pipeline_rag = NexusRAG(test_settings)
        if resource == "vector_store":
            pipeline_rag._embedder = mock_embedder
        lock = _ObservedWriteLock()
        pipeline_rag._write_lock = lock
        start = Barrier(4)
        constructing = Event()
        release = Event()
        handle = MagicMock()

        def construct(**_kwargs):
            constructing.set()
            assert release.wait(timeout=5)
            return handle

        def load():
            start.wait(timeout=5)
            return getattr(pipeline_rag, resource)

        factory = MagicMock(side_effect=construct)
        monkeypatch.setattr("scinexusrag.application." + factory_name, factory)
        with ThreadPoolExecutor(max_workers=4) as workers:
            futures = [workers.submit(load) for _ in range(4)]
            try:
                assert constructing.wait(timeout=5)
                assert len(lock.wait_for_threads(4)) == 4
            finally:
                release.set()
            assert all(future.result(timeout=5) is handle for future in futures)

        if resource == "embedder":
            factory.assert_called_once_with(
                model_name="test-model", device="cpu", revision="a" * 40
            )
        elif resource == "vector_store":
            factory.assert_called_once_with(
                path=test_settings.storage.lancedb_path,
                table_name=test_settings.storage.table_name,
                embedding_config=mock_embedder.embedding_config,
            )
        else:
            factory.assert_called_once_with(
                model=test_settings.llm.model,
                base_url=test_settings.llm.base_url,
                timeout=test_settings.llm.timeout,
            )

    @pytest.mark.parametrize(
        "resource,factory_name",
        [("embedder", "Embedder"), ("vector_store", "VectorStore"), ("llm", "LLMClient")],
    )
    def test_failed_constructor_can_retry(
        self, test_settings, mock_embedder, monkeypatch, resource, factory_name
    ):
        pipeline_rag = NexusRAG(test_settings)
        if resource == "vector_store":
            pipeline_rag._embedder = mock_embedder
        pipeline_rag._write_lock = _ObservedWriteLock()
        handle = MagicMock()
        factory = MagicMock(side_effect=[RuntimeError("Constructor failed"), handle])
        monkeypatch.setattr("scinexusrag.application." + factory_name, factory)

        with pytest.raises(RuntimeError, match="Constructor failed"):
            getattr(pipeline_rag, resource)
        assert getattr(pipeline_rag, "_" + resource) is None
        with ThreadPoolExecutor(max_workers=1) as workers:
            assert (
                workers.submit(lambda: getattr(pipeline_rag, resource)).result(timeout=5) is handle
            )
        assert getattr(pipeline_rag, resource) is handle
        assert factory.call_count == 2


class TestStoredEmbeddingConfiguration:
    def test_explicit_reset_does_not_require_a_valid_embedder(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):
        original = pipeline_rag
        assert original.ingest(pipeline_text_file).success
        restarted = NexusRAG(settings=original.settings)
        invalid_embedder = MagicMock()
        config = PropertyMock(side_effect=EmbeddingConfigError("Unpinned model revision"))
        type(invalid_embedder).embedding_config = config
        restarted._embedder = invalid_embedder

        restarted.clear_all()

        config.assert_not_called()
        invalid_embedder.embed.assert_not_called()
        invalid_embedder.embed_query.assert_not_called()
        assert len(restarted.document_store.list_with_metadata()) == 0
        assert (
            restarted.settings.storage.table_name
            not in original.vector_store.db.list_tables().tables
        )
        assert restarted._vector_store is None
        assert restarted._bm25 is None

    def test_changed_model_rejects_query_before_embedding_or_generation(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):
        original = pipeline_rag
        ingested = original.ingest(pipeline_text_file)
        assert ingested.success
        changed_embedder = MagicMock()
        changed_embedder.dimension = 384
        changed_embedder.embedding_config = replace(
            original.embedder.embedding_config, model_name="another-model"
        )
        restarted = NexusRAG(settings=original.settings)
        restarted._embedder = changed_embedder
        restarted._llm = MagicMock()

        with pytest.raises(EmbeddingConfigError):
            restarted.query("What measurements were recorded?")

        changed_embedder.embed_query.assert_not_called()
        restarted._llm.generate.assert_not_called()
        assert original.vector_store.count() == ingested.chunk_count
        assert len(original.document_store.list_with_metadata()) == 1

    def test_changed_revision_rejects_ingest_before_embedding_writes_or_rollback(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path, monkeypatch
    ):
        original = pipeline_rag
        ingested = original.ingest(pipeline_text_file)
        assert ingested.success
        changed_embedder = MagicMock()
        changed_embedder.dimension = 384
        changed_embedder.embedding_config = replace(
            original.embedder.embedding_config, revision="b" * 40
        )
        restarted = NexusRAG(settings=original.settings)
        restarted._embedder = changed_embedder
        document = restarted.parser.parse_bytes(
            b"A distinct upload records pressure readings from sensor C.", "new.txt", ".txt"
        )
        writes = MagicMock()
        deletes = MagicMock()
        monkeypatch.setattr(restarted.document_store, "add", writes)
        monkeypatch.setattr(restarted.document_store, "delete", deletes)

        with pytest.raises(EmbeddingConfigError):
            restarted._ingest_document(document, "new.txt", show_progress=False)
        with pytest.raises(EmbeddingConfigError):
            restarted._persist(document, [], np.empty((0, 384), dtype=np.float32))
        new_path = pipeline_text_file.with_name("new.txt")
        new_path.write_text(document.content)
        with pytest.raises(EmbeddingConfigError):
            restarted.ingest(new_path)
        with pytest.raises(EmbeddingConfigError):
            restarted.ingest_bytes(document.content.encode(), "new.txt", ".txt")

        changed_embedder.embed.assert_not_called()
        writes.assert_not_called()
        deletes.assert_not_called()
        assert original.vector_store.count() == ingested.chunk_count
        assert len(original.document_store.list_with_metadata()) == 1

    def test_explicit_reset_clears_documents_and_allows_reingestion(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):
        assert pipeline_rag.ingest(pipeline_text_file).success
        changed_config = replace(pipeline_rag.embedder.embedding_config, model_name="another-model")
        original = NexusRAG(settings=pipeline_rag.settings)
        original._llm = pipeline_rag.llm
        original._embedder = MagicMock()
        original._embedder.dimension = 384
        original._embedder.embedding_config = changed_config
        original._embedder.embed.return_value = np.ones((1, 384), dtype=np.float32)

        with pytest.raises(EmbeddingConfigError):
            original.query("What measurements were recorded?")

        original.clear_all()
        assert len(original.document_store.list_with_metadata()) == 0
        assert original.vector_store.count() == 0
        assert original.bm25.count() == 0
        original._embedder.embed.side_effect = lambda texts, **kwargs: np.ones(
            (len(texts), 384), dtype=np.float32
        )
        result = original.ingest(pipeline_text_file)

        assert result.success
        assert len(original.document_store.list_with_metadata()) == 1
        assert original.vector_store.count() == original.bm25.count() == result.chunk_count


class _ImmediateVisibilityStore:
    def __init__(self, real):
        self._real = real
        self._seen: list = []

    def add(self, chunks, embeddings):
        n = self._real.add(chunks, embeddings)
        self._seen.extend(chunks)
        return n

    def get_all_chunks(self):
        return list(self._seen)

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestBM25NoDoubleCount:
    def test_ingest_does_not_double_count_bm25_on_lazy_init(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):

        pipeline_rag._vector_store = _ImmediateVisibilityStore(pipeline_rag.vector_store)

        result = pipeline_rag.ingest(pipeline_text_file)

        assert result.success
        assert pipeline_rag.bm25.count() == result.chunk_count

    def test_first_ingest_keeps_bm25_and_vector_in_sync(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):

        r = pipeline_rag.ingest(pipeline_text_file)
        assert r.success
        assert pipeline_rag.bm25.count() == r.chunk_count
        assert pipeline_rag.vector_store.count() == r.chunk_count


class TestEndToEnd:
    def test_ingest_query_cites_a_real_source(
        self, pipeline_rag: NexusRAG, pipeline_text_file: Path
    ):

        pipeline_rag._llm.generate = lambda prompt, **kw: (
            "The sensors are read every thirty seconds [1]."
        )
        query_embedding = np.random.default_rng(0).random(384, dtype=np.float32)
        pipeline_rag._embedder.embed_query = lambda q: query_embedding.copy()

        assert pipeline_rag.ingest(pipeline_text_file).success
        resp = pipeline_rag.query("What methodology was used?")

        assert resp.sources, "query returned no sources"
        assert "[1]" in resp.answer, "answer did not cite a source"
        assert resp.confidence > 0

        assert resp.sources[0].chunk_id


@pytest.fixture
def inventory_rag(tmp_path):
    instance = NexusRAG(
        Settings(
            data_dir=tmp_path,
            storage=StorageSettings(lancedb_path=tmp_path / "lancedb"),
        )
    )
    instance._vector_store = MagicMock(spec=VectorStore)
    instance._vector_store.count.return_value = 1
    instance.document_store.add(
        ParsedDocument("original", "Sensor A reads temperature.", {"filename": "original.txt"})
    )
    instance.document_store.update_metadata("original", "chunk_count", 1)
    return instance


def test_inventory_uses_vector_count_when_metadata_has_no_chunk_count(inventory_rag):
    inventory_rag.document_store.add(
        ParsedDocument("recovered", "Sensor B reads pressure.", {"filename": "recovered.txt"})
    )
    inventory_rag._vector_store.count.return_value = 7

    inventory = inventory_rag.get_document_inventory()

    assert inventory.total_documents == 2
    assert inventory.total_chunks == 7
    assert "chunk_count" not in inventory.documents[1]
    assert inventory_rag._llm is None
    assert inventory_rag._embedder is None


def test_inventory_metadata_is_detached_from_later_store_updates(inventory_rag):
    inventory = inventory_rag.get_document_inventory()
    inventory_rag.document_store.update_metadata("original", "chunk_count", 4)

    assert inventory.documents[0]["chunk_count"] == 1
    assert inventory_rag.get_document_inventory().documents[0]["chunk_count"] == 4
    with pytest.raises(FrozenInstanceError):
        inventory.total_chunks = 4


def test_inventory_uses_stored_identity_when_metadata_contains_an_id(inventory_rag):
    inventory_rag.document_store.update_metadata("original", "id", "different")

    inventory = inventory_rag.get_document_inventory()

    assert inventory.documents[0]["id"] == "original"
    assert inventory_rag.document_store.exists("original")
    assert not inventory_rag.document_store.exists("different")


@pytest.mark.parametrize("method", ["list_documents", "get_document_inventory"])
def test_document_reads_wait_for_complete_writes(inventory_rag, method, observed_write_lock):
    lock = observed_write_lock
    inventory_rag._write_lock = lock
    inserted, finish_write = Event(), Event()

    def write_document():
        with inventory_rag._write_lock:
            inventory_rag.document_store.add(
                ParsedDocument("pending", "Sensor C reads humidity.", {"filename": "pending.txt"})
            )
            inserted.set()
            assert finish_write.wait(timeout=5)
            inventory_rag.document_store.update_metadata("pending", "chunk_count", 4)
            inventory_rag._vector_store.count.return_value = 5

    with ThreadPoolExecutor(max_workers=2) as workers:
        write = workers.submit(write_document)
        try:
            assert inserted.wait(timeout=5)
            read = workers.submit(getattr(inventory_rag, method))
            lock.wait_for_threads(2)
            assert not read.done()
        finally:
            finish_write.set()
        write.result(timeout=5)
        snapshot = read.result(timeout=5)

    documents = snapshot.documents if isinstance(snapshot, DocumentInventory) else snapshot
    assert [document["id"] for document in documents] == ["original", "pending"]
    assert documents[1]["chunk_count"] == 4
    if isinstance(snapshot, DocumentInventory):
        assert snapshot.total_documents == 2
        assert snapshot.total_chunks == 5


def test_inventory_holds_write_lock_until_vector_count_finishes(inventory_rag, observed_write_lock):
    lock = observed_write_lock
    inventory_rag._write_lock = lock
    counting, finish_count = Event(), Event()

    def count_vectors():
        counting.set()
        assert finish_count.wait(timeout=5)
        return 1

    inventory_rag._vector_store.count.side_effect = count_vectors

    def delete_document():
        with inventory_rag._write_lock:
            inventory_rag.document_store.delete("original")

    with ThreadPoolExecutor(max_workers=2) as workers:
        read = workers.submit(inventory_rag.get_document_inventory)
        try:
            assert counting.wait(timeout=5)
            write = workers.submit(delete_document)
            lock.wait_for_threads(2)
            assert not write.done()
        finally:
            finish_count.set()
        inventory = read.result(timeout=5)
        write.result(timeout=5)

    assert inventory.total_documents == inventory.total_chunks == 1
    assert inventory_rag.list_documents() == []


@pytest.mark.parametrize(
    "error",
    [VectorStoreError("Storage unavailable"), EmbeddingConfigError("Incompatible embeddings")],
)
def test_inventory_preserves_index_errors(inventory_rag, error):
    inventory_rag._vector_store.count.side_effect = error

    with pytest.raises(type(error), match=str(error)):
        inventory_rag.get_document_inventory()

    assert inventory_rag._llm is None


def test_inventory_preserves_metadata_read_errors(inventory_rag, monkeypatch):
    failure = MagicMock(side_effect=OSError("Metadata unavailable"))
    monkeypatch.setattr(inventory_rag.document_store, "list_with_metadata", failure)

    with pytest.raises(OSError, match="Metadata unavailable"):
        inventory_rag.get_document_inventory()

    inventory_rag._vector_store.count.assert_not_called()


def test_passage_limit_rejects_before_model_and_storage_initialization(test_settings):
    test_settings.ingestion.max_chunks = 1
    rag = NexusRAG(settings=test_settings)
    rag._chunker = Mock()
    rag._chunker.chunk.return_value = [Mock(), Mock()]

    result = rag.ingest_bytes(b"Evidence.", "paper.txt", ".txt")

    assert result.success is False
    assert "limit of 1 passages" in result.error
    assert rag._embedder is None
    assert rag._vector_store is None
    assert rag._document_store is None


def test_passage_limit_accepts_exact_boundary(pipeline_rag):
    pipeline_rag.settings.ingestion.max_chunks = 1
    result = pipeline_rag.ingest_bytes(b"Evidence.", "paper.txt", ".txt")
    assert result.success is True
    assert result.chunk_count == 1
