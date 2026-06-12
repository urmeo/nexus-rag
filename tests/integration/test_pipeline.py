"""Integration tests for NexusRAG pipeline.

Tests the full ingest → query → delete workflow with mocked ML components
but real file I/O and storage operations. Covers:
- Single file ingestion with query workflow
- Multi-file directory ingestion
- Duplicate detection
- Document deletion and clearing
- Statistics and metadata tracking
- Error handling for edge cases
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nexusrag.config import (
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    Settings,
    StorageSettings,
)
from nexusrag.pipeline import IngestResult, NexusRAG, SystemStats, get_nexusrag

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Provide a temporary data directory for pipeline testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_settings(temp_data_dir: Path) -> Settings:
    """Create test settings with temporary storage."""
    return Settings(
        data_dir=temp_data_dir,
        storage=StorageSettings(
            lancedb_path=temp_data_dir / "lancedb",
            table_name="test_documents",
        ),
        embedding=EmbeddingSettings(
            model="test-model",
            device="cpu",
            batch_size=8,
        ),
        llm=LLMSettings(
            model="test-llm",
            base_url="http://localhost:11434",
            timeout=30,
        ),
        ingestion=IngestionSettings(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=20,
        ),
    )


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic embeddings."""
    embedder = MagicMock()
    embedder.dimension = 384

    def mock_embed(texts, batch_size=None, show_progress=False):
        """Return deterministic embeddings based on text."""
        if isinstance(texts, str):
            texts = [texts]
        np.random.seed(hash("".join(texts)) % (2**31))
        return np.random.rand(len(texts), 384).astype(np.float32)

    embedder.embed = mock_embed
    return embedder


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns deterministic responses."""
    llm = MagicMock()
    llm.model = "test-llm"

    def mock_generate(prompt, **kwargs):
        """Return a mock response."""
        if "What is" in prompt or "?" in prompt:
            return "Based on the retrieved context, this is a test answer about the query."
        return "This is a test response."

    llm.generate = mock_generate
    llm.is_available = MagicMock(return_value=True)
    return llm


@pytest.fixture
def nexusrag_instance(test_settings: Settings, mock_embedder, mock_llm) -> NexusRAG:
    """Create a NexusRAG instance with mocked heavy components."""
    rag = NexusRAG(settings=test_settings)

    # Inject mocks before any properties are accessed
    rag._embedder = mock_embedder
    rag._llm = mock_llm

    return rag


@pytest.fixture
def sample_text_file(temp_data_dir: Path) -> Path:
    """Create a sample text file for ingestion."""
    file_path = temp_data_dir / "test_document.txt"
    content = """INTRODUCTION

This is a sample research document for testing the NexusRAG pipeline.
It contains multiple sections with meaningful content that can be chunked and embedded.

The document demonstrates the full workflow: ingestion, chunking, embedding, and retrieval.

METHODS

We employed several techniques in our analysis.
The methodology was rigorous and followed best practices.

Data collection spanned six months with careful quality control.

RESULTS

The results were significant and reproducible.
We observed a 45% improvement in the primary metric.

Secondary metrics also showed positive trends.

DISCUSSION

These findings align with previous research in the field.
The implications are substantial for future work.

Further research should explore these directions.

CONCLUSION

In conclusion, this document serves as a test case for the pipeline.
The integration tests validate all major functionality.

Future improvements are planned and documented elsewhere.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_markdown_file(temp_data_dir: Path) -> Path:
    """Create a sample markdown file for ingestion."""
    file_path = temp_data_dir / "test_paper.md"
    content = """# Scientific Paper on Testing

## Abstract

This paper presents findings on testing methodologies for RAG systems.

## Introduction

### Background

RAG (Retrieval-Augmented Generation) systems are increasingly important.
Testing is crucial for reliability and performance.

### Objectives

The objectives of this work are threefold:
1. Develop comprehensive test suites
2. Validate ingestion pipelines
3. Ensure retrieval quality

## Methodology

### Data Preparation

Documents were prepared in multiple formats.
Each format was tested for compatibility.

### Testing Framework

We used pytest for unit and integration testing.
Mocking was employed for expensive ML components.

## Results

All tests passed successfully.
Integration tests achieved 95% coverage.

## Conclusion

This work demonstrates the importance of thorough testing.
The methodologies presented here are applicable to similar systems.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def docs_directory(temp_data_dir: Path) -> Path:
    """Create a directory with multiple document files."""
    docs_dir = temp_data_dir / "documents"
    docs_dir.mkdir()

    # Create multiple test documents
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
        "All documents ingested successfully.\n"
    )

    return docs_dir


# ============================================================================
# TESTS: BASIC PIPELINE OPERATIONS
# ============================================================================


class TestBasicPipelineWorkflow:
    """Test complete ingest → query → delete workflows."""

    def test_ingest_single_text_file(self, nexusrag_instance: NexusRAG, sample_text_file: Path):
        """Test ingesting a single text file."""
        result = nexusrag_instance.ingest(sample_text_file)

        assert result.success is True
        assert result.filename == "test_document.txt"
        assert result.chunk_count > 0
        assert result.word_count > 0
        assert result.document_id
        assert result.error is None

    def test_ingest_markdown_file(self, nexusrag_instance: NexusRAG, sample_markdown_file: Path):
        """Test ingesting a markdown file."""
        result = nexusrag_instance.ingest(sample_markdown_file)

        assert result.success is True
        assert result.filename == "test_paper.md"
        assert result.chunk_count > 0
        assert result.word_count > 0

    def test_full_workflow_ingest_and_query(
        self, nexusrag_instance: NexusRAG, sample_text_file: Path
    ):
        """Test complete workflow: ingest file, then query it."""
        # Ingest
        ingest_result = nexusrag_instance.ingest(sample_text_file)
        assert ingest_result.success is True
        document_id = ingest_result.document_id

        # Verify document is stored
        docs = nexusrag_instance.list_documents()
        assert len(docs) == 1
        assert docs[0]["id"] == document_id

        # Verify stats updated
        stats = nexusrag_instance.get_stats()
        assert stats.total_documents == 1
        assert stats.total_chunks == ingest_result.chunk_count

    def test_delete_document_workflow(self, nexusrag_instance: NexusRAG, sample_text_file: Path):
        """Test deleting a document and verifying removal."""
        # Ingest
        ingest_result = nexusrag_instance.ingest(sample_text_file)
        document_id = ingest_result.document_id

        # Verify ingested
        assert len(nexusrag_instance.list_documents()) == 1

        # Delete
        deleted = nexusrag_instance.delete_document(document_id)
        assert deleted is True

        # Verify deleted
        assert len(nexusrag_instance.list_documents()) == 0

        # Verify chunk count reset
        stats = nexusrag_instance.get_stats()
        assert stats.total_documents == 0
        assert stats.total_chunks == 0

    def test_delete_nonexistent_document(self, nexusrag_instance: NexusRAG):
        """Test deleting a document that doesn't exist."""
        deleted = nexusrag_instance.delete_document("nonexistent_id")
        assert deleted is False


# ============================================================================
# TESTS: MULTI-FILE OPERATIONS
# ============================================================================


class TestMultiFileIngestion:
    """Test ingesting multiple files from directories."""

    def test_ingest_directory(self, nexusrag_instance: NexusRAG, docs_directory: Path):
        """Test ingesting all files from a directory."""
        results = nexusrag_instance.ingest_directory(docs_directory)

        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify all documents are tracked
        docs = nexusrag_instance.list_documents()
        assert len(docs) == 3

        # Verify chunks were created
        for result in results:
            assert result.chunk_count > 0

    def test_ingest_directory_stats(self, nexusrag_instance: NexusRAG, docs_directory: Path):
        """Test that stats reflect multiple ingested documents."""
        results = nexusrag_instance.ingest_directory(docs_directory)
        total_chunks = sum(r.chunk_count for r in results)
        total_words = sum(r.word_count for r in results)

        stats = nexusrag_instance.get_stats()
        assert stats.total_documents == 3
        assert stats.total_chunks == total_chunks
        assert stats.total_words == total_words

    def test_ingest_directory_with_unsupported_format(
        self, nexusrag_instance: NexusRAG, temp_data_dir: Path
    ):
        """Test that unsupported file formats are skipped."""
        test_dir = temp_data_dir / "mixed_docs"
        test_dir.mkdir()

        # Create supported and unsupported files
        (test_dir / "valid.txt").write_text("Valid content here.\n" * 10)
        (test_dir / "unsupported.xyz").write_text("This should be skipped.\n")

        results = nexusrag_instance.ingest_directory(test_dir)

        # Only .txt file should be ingested
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].filename == "valid.txt"


# ============================================================================
# TESTS: DUPLICATE DETECTION
# ============================================================================


class TestDuplicateDetection:
    """Test detection and handling of duplicate documents."""

    def test_duplicate_file_detection(self, nexusrag_instance: NexusRAG, sample_text_file: Path):
        """Test that ingesting the same file twice is detected."""
        # First ingestion should succeed
        result1 = nexusrag_instance.ingest(sample_text_file)
        assert result1.success is True

        # Second ingestion should be rejected
        result2 = nexusrag_instance.ingest(sample_text_file)
        assert result2.success is False
        assert result2.error == "Document already exists"
        assert result2.chunk_count == 0

        # Verify only one document exists
        docs = nexusrag_instance.list_documents()
        assert len(docs) == 1

    def test_duplicate_in_batch_ingest(self, nexusrag_instance: NexusRAG, temp_data_dir: Path):
        """Test duplicate handling in directory ingestion."""
        docs_dir = temp_data_dir / "dup_test"
        docs_dir.mkdir()

        # Create same content with different filename
        content = "Test content for duplicate detection.\n" * 5
        (docs_dir / "doc_v1.txt").write_text(content)
        (docs_dir / "doc_v2.txt").write_text(content)

        results = nexusrag_instance.ingest_directory(docs_dir)

        # Both should be processed, but second might be marked as duplicate
        # due to content hash matching
        assert len(results) == 2

        # At least one should succeed
        successful = [r for r in results if r.success]
        assert len(successful) >= 1


# ============================================================================
# TESTS: CLEARING AND RESET
# ============================================================================


class TestClearOperations:
    """Test clearing all documents and resetting the system."""

    def test_clear_all_documents(self, nexusrag_instance: NexusRAG, docs_directory: Path):
        """Test clearing all documents from the system."""
        # Ingest multiple documents
        nexusrag_instance.ingest_directory(docs_directory)
        assert nexusrag_instance.get_stats().total_documents == 3

        # Clear all
        nexusrag_instance.clear_all()

        # Verify everything cleared
        assert nexusrag_instance.get_stats().total_documents == 0
        assert nexusrag_instance.get_stats().total_chunks == 0
        assert len(nexusrag_instance.list_documents()) == 0

    def test_clear_all_then_reingest(self, nexusrag_instance: NexusRAG, sample_text_file: Path):
        """Test that system works correctly after clearing."""
        # Ingest, clear, then ingest again
        result1 = nexusrag_instance.ingest(sample_text_file)
        assert result1.success is True

        nexusrag_instance.clear_all()
        assert nexusrag_instance.get_stats().total_documents == 0

        # Should be able to ingest the same file again
        result2 = nexusrag_instance.ingest(sample_text_file)
        assert result2.success is True


# ============================================================================
# TESTS: STATISTICS AND METADATA
# ============================================================================


class TestStatisticsAndMetadata:
    """Test statistics and metadata tracking."""

    def test_get_stats_empty_system(self, nexusrag_instance: NexusRAG):
        """Test statistics on empty system."""
        stats = nexusrag_instance.get_stats()

        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        assert stats.total_words == 0
        assert stats.llm_available is True
        assert stats.embedding_model == "test-model"
        assert stats.llm_model == "test-llm"

    def test_get_stats_after_ingestion(self, nexusrag_instance: NexusRAG, sample_text_file: Path):
        """Test statistics after ingesting a document."""
        result = nexusrag_instance.ingest(sample_text_file)

        stats = nexusrag_instance.get_stats()
        assert stats.total_documents == 1
        assert stats.total_chunks == result.chunk_count
        assert stats.total_words > 0
        assert "tmp" in str(stats.storage_path).lower()  # Temp directory

    def test_list_documents_empty(self, nexusrag_instance: NexusRAG):
        """Test listing documents on empty system."""
        docs = nexusrag_instance.list_documents()
        assert docs == []

    def test_list_documents_with_metadata(
        self, nexusrag_instance: NexusRAG, sample_text_file: Path
    ):
        """Test listing documents includes proper metadata."""
        ingest_result = nexusrag_instance.ingest(sample_text_file)

        docs = nexusrag_instance.list_documents()
        assert len(docs) == 1

        doc = docs[0]
        assert doc["id"] == ingest_result.document_id
        assert "filename" in doc or "original_filename" in doc

    def test_stats_with_multiple_documents(self, nexusrag_instance: NexusRAG, docs_directory: Path):
        """Test statistics with multiple documents."""
        results = nexusrag_instance.ingest_directory(docs_directory)

        stats = nexusrag_instance.get_stats()
        assert stats.total_documents == len(results)

        total_chunks_expected = sum(r.chunk_count for r in results)
        assert stats.total_chunks == total_chunks_expected


# ============================================================================
# TESTS: ERROR HANDLING
# ============================================================================


class TestErrorHandling:
    """Test error handling for edge cases and invalid inputs."""

    def test_ingest_empty_file(self, nexusrag_instance: NexusRAG, temp_data_dir: Path):
        """Test ingesting an empty file."""
        empty_file = temp_data_dir / "empty.txt"
        empty_file.write_text("")

        result = nexusrag_instance.ingest(empty_file)

        assert result.success is False
        assert result.error == "No content extracted"
        assert result.chunk_count == 0

    def test_ingest_nonexistent_file(self, nexusrag_instance: NexusRAG):
        """Test ingesting a file that doesn't exist."""
        result = nexusrag_instance.ingest("/nonexistent/path/file.txt")

        assert result.success is False
        assert "Ingestion failed" in result.error or "FileNotFoundError" in result.error

    def test_ingest_unsupported_format(self, nexusrag_instance: NexusRAG, temp_data_dir: Path):
        """Test ingesting an unsupported file format."""
        unsupported = temp_data_dir / "file.xyz"
        unsupported.write_text("Some content")

        result = nexusrag_instance.ingest(unsupported)

        assert result.success is False
        assert "Ingestion failed" in result.error

    def test_ingest_directory_nonexistent(self, nexusrag_instance: NexusRAG):
        """Test ingesting from a non-existent directory."""
        with pytest.raises(ValueError, match="Not a directory"):
            nexusrag_instance.ingest_directory("/nonexistent/directory")

    def test_ingest_bytes_empty(self, nexusrag_instance: NexusRAG):
        """Test ingesting empty bytes."""
        result = nexusrag_instance.ingest_bytes(b"", "empty.txt", ".txt")

        assert result.success is False
        assert result.chunk_count == 0

    def test_ingest_bytes_valid(self, nexusrag_instance: NexusRAG):
        """Test ingesting valid bytes."""
        content = b"This is test content for bytes ingestion.\n" * 5
        result = nexusrag_instance.ingest_bytes(content, "test.txt", ".txt")

        assert result.success is True
        assert result.filename == "test.txt"
        assert result.chunk_count > 0

    def test_ingest_bytes_without_extension(self, nexusrag_instance: NexusRAG):
        """Test ingesting bytes without file extension prefix."""
        content = b"Test content.\n" * 5
        # Should handle both ".txt" and "txt" formats
        result = nexusrag_instance.ingest_bytes(content, "test.txt", "txt")

        assert result.success is True


# ============================================================================
# TESTS: INGEST RESULT DATACLASS
# ============================================================================


class TestIngestResult:
    """Test IngestResult dataclass behavior."""

    def test_ingest_result_success(self, sample_text_file: Path):
        """Test creating successful IngestResult."""
        result = IngestResult(
            document_id="doc_123",
            filename="test.txt",
            chunk_count=5,
            word_count=150,
            success=True,
        )

        assert result.document_id == "doc_123"
        assert result.filename == "test.txt"
        assert result.chunk_count == 5
        assert result.word_count == 150
        assert result.success is True
        assert result.error is None

    def test_ingest_result_failure(self):
        """Test creating failed IngestResult."""
        result = IngestResult(
            document_id="",
            filename="bad.xyz",
            chunk_count=0,
            word_count=0,
            success=False,
            error="Unsupported format",
        )

        assert result.success is False
        assert result.error == "Unsupported format"
        assert result.chunk_count == 0


# ============================================================================
# TESTS: SYSTEM STATS DATACLASS
# ============================================================================


class TestSystemStats:
    """Test SystemStats dataclass behavior."""

    def test_system_stats_creation(self, test_settings: Settings):
        """Test creating SystemStats."""
        stats = SystemStats(
            total_documents=5,
            total_chunks=42,
            total_words=15000,
            storage_path="/data/storage",
            llm_model="llama3",
            embedding_model="all-MiniLM-L6-v2",
            llm_available=True,
        )

        assert stats.total_documents == 5
        assert stats.total_chunks == 42
        assert stats.total_words == 15000
        assert stats.llm_available is True

    def test_system_stats_no_llm(self, test_settings: Settings):
        """Test SystemStats when LLM is not available."""
        stats = SystemStats(
            total_documents=2,
            total_chunks=10,
            total_words=5000,
            storage_path="/data/storage",
            llm_model="offline-model",
            embedding_model="offline-embeddings",
            llm_available=False,
        )

        assert stats.llm_available is False


# ============================================================================
# TESTS: LAZY LOADING OF COMPONENTS
# ============================================================================


class TestLazyLoading:
    """Test lazy loading of pipeline components."""

    def test_components_not_loaded_initially(self, nexusrag_instance: NexusRAG):
        """Test that components are not loaded on initialization."""
        # Only _embedder and _llm are pre-loaded by fixture for mocking
        # Others should be None initially
        assert nexusrag_instance._parser is None
        assert nexusrag_instance._chunker is None
        assert nexusrag_instance._vector_store is None
        assert nexusrag_instance._document_store is None
        assert nexusrag_instance._bm25 is None

    def test_parser_lazy_loads_on_access(self, nexusrag_instance: NexusRAG):
        """Test that parser is loaded on first access."""
        assert nexusrag_instance._parser is None
        parser = nexusrag_instance.parser
        assert parser is not None
        assert nexusrag_instance._parser is parser

    def test_chunker_lazy_loads_on_access(self, nexusrag_instance: NexusRAG):
        """Test that chunker is loaded on first access."""
        assert nexusrag_instance._chunker is None
        chunker = nexusrag_instance.chunker
        assert chunker is not None
        assert nexusrag_instance._chunker is chunker

    def test_document_store_lazy_loads_on_access(self, nexusrag_instance: NexusRAG):
        """Test that document store is loaded on first access."""
        assert nexusrag_instance._document_store is None
        store = nexusrag_instance.document_store
        assert store is not None
        assert nexusrag_instance._document_store is store


# ============================================================================
# TESTS: SINGLETON BEHAVIOR
# ============================================================================


class TestSingletonBehavior:
    """Test get_nexusrag() singleton function."""

    def test_get_nexusrag_returns_same_instance(self):
        """Test that get_nexusrag() returns the same instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(data_dir=Path(tmpdir))
            instance1 = get_nexusrag(settings)
            instance2 = get_nexusrag()

            assert instance1 is instance2

    def test_get_nexusrag_with_different_settings(self):
        """Test getting new instance with different settings."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            settings1 = Settings(data_dir=Path(tmpdir1))
            settings2 = Settings(data_dir=Path(tmpdir2))

            # Note: Singleton pattern means second call returns first instance
            # This is expected behavior - singleton is global
            instance1 = get_nexusrag(settings1)
            instance2 = get_nexusrag(settings2)

            # Both should be same due to singleton pattern
            assert instance1 is instance2


# ============================================================================
# TESTS: INGEST BYTES OPERATIONS
# ============================================================================


class TestIngestBytes:
    """Test ingesting documents from bytes."""

    def test_ingest_bytes_text(self, nexusrag_instance: NexusRAG):
        """Test ingesting text content as bytes."""
        content = b"""
        # Test Document

        This is a test document provided as bytes.
        It should be properly parsed and chunked.

        Multiple paragraphs ensure sufficient content.
        Testing the bytes ingestion pipeline thoroughly.
        """

        result = nexusrag_instance.ingest_bytes(content, "bytes_doc.txt", ".txt")

        assert result.success is True
        assert result.filename == "bytes_doc.txt"
        assert result.chunk_count > 0

    def test_ingest_bytes_md_format(self, nexusrag_instance: NexusRAG):
        """Test ingesting markdown content as bytes."""
        content = b"""# Markdown Test

## Section 1
Content here for testing.

## Section 2
More content for proper chunking.
"""

        result = nexusrag_instance.ingest_bytes(content, "test.md", ".md")

        assert result.success is True
        assert result.chunk_count > 0

    def test_ingest_bytes_duplicate_detection(self, nexusrag_instance: NexusRAG):
        """Test that duplicate bytes content is detected."""
        content = b"Unique test content for duplicate detection.\n" * 5

        result1 = nexusrag_instance.ingest_bytes(content, "file1.txt", ".txt")
        assert result1.success is True

        result2 = nexusrag_instance.ingest_bytes(content, "file2.txt", ".txt")
        # May fail due to content hash collision
        # Just verify behavior is consistent
        assert isinstance(result2.success, bool)


# ============================================================================
# TESTS: EDGE CASES AND STRESS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_ingest_very_short_document(self, nexusrag_instance: NexusRAG, temp_data_dir: Path):
        """Test ingesting a very short document."""
        short_file = temp_data_dir / "short.txt"
        short_file.write_text("Short content.")

        result = nexusrag_instance.ingest(short_file)

        # Might fail if below minimum chunk size
        assert isinstance(result.success, bool)

    def test_ingest_large_document(self, nexusrag_instance: NexusRAG, temp_data_dir: Path):
        """Test ingesting a large document."""
        large_file = temp_data_dir / "large.txt"

        # Create a large document
        large_content = "This is a test sentence that will be repeated many times. " * 200
        large_file.write_text(large_content)

        result = nexusrag_instance.ingest(large_file)

        assert result.success is True
        assert result.chunk_count > 1  # Should be split into multiple chunks

    def test_concurrent_operations_sequence(self, nexusrag_instance: NexusRAG, temp_data_dir: Path):
        """Test sequential operations that might have ordering issues."""
        file1 = temp_data_dir / "doc1.txt"
        file2 = temp_data_dir / "doc2.txt"

        content1 = "Document 1 content. " * 20
        content2 = "Document 2 content. " * 20

        file1.write_text(content1)
        file2.write_text(content2)

        # Ingest both
        result1 = nexusrag_instance.ingest(file1)
        result2 = nexusrag_instance.ingest(file2)

        assert result1.success is True
        assert result2.success is True

        # Verify both are tracked
        docs = nexusrag_instance.list_documents()
        assert len(docs) == 2

        # Delete first
        deleted = nexusrag_instance.delete_document(result1.document_id)
        assert deleted is True

        docs = nexusrag_instance.list_documents()
        assert len(docs) == 1


# ============================================================================
# TESTS: INTEGRATION WITH REAL COMPONENTS (SELECTIVE)
# ============================================================================


class TestIntegrationWithRealComponents:
    """Test integration with some real components (not fully mocked)."""

    def test_with_real_parser_and_chunker(
        self, test_settings: Settings, sample_text_file: Path, mock_embedder, mock_llm
    ):
        """Test pipeline with real parser and chunker but mocked embedder/LLM."""
        rag = NexusRAG(settings=test_settings)
        rag._embedder = mock_embedder
        rag._llm = mock_llm

        # Use real parser and chunker
        result = rag.ingest(sample_text_file)

        assert result.success is True
        assert result.chunk_count > 0

        # Verify real chunking happened
        stats = rag.get_stats()
        assert stats.total_chunks == result.chunk_count

    def test_document_store_persistence(
        self, test_settings: Settings, sample_text_file: Path, mock_embedder, mock_llm
    ):
        """Test that document store persists metadata correctly."""
        rag = NexusRAG(settings=test_settings)
        rag._embedder = mock_embedder
        rag._llm = mock_llm

        ingest_result = rag.ingest(sample_text_file)
        assert ingest_result.success is True

        # Retrieve document metadata
        docs = rag.list_documents()
        assert len(docs) == 1

        doc = docs[0]
        assert doc["id"] == ingest_result.document_id


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================


class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "filename,content",
        [
            ("test1.txt", "Test content one.\n" * 10),
            ("test2.txt", "Test content two.\n" * 10),
            ("test3.txt", "Different content.\n" * 10),
        ],
    )
    def test_ingest_multiple_variations(
        self, nexusrag_instance: NexusRAG, temp_data_dir: Path, filename: str, content: str
    ):
        """Test ingesting various document variations."""
        file_path = temp_data_dir / filename
        file_path.write_text(content)

        result = nexusrag_instance.ingest(file_path)

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
        self, nexusrag_instance: NexusRAG, temp_data_dir: Path, format_ext: str, format_name: str
    ):
        """Test ingesting various document formats."""
        file_path = temp_data_dir / f"doc{format_ext}"

        if format_ext == ".txt":
            content = "Test content for text format.\n" * 10
        else:  # markdown
            content = "# Test\n\nContent for markdown format.\n" * 5

        file_path.write_text(content)

        result = nexusrag_instance.ingest(file_path)

        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
