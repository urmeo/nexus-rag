"""
Comprehensive unit tests for NexusRAG API routes.

Tests all endpoints with proper mocking of the NexusRAG singleton
to avoid requiring real ML models or document stores.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nexusrag.agents import RAGResponse, Source
from nexusrag.api.routes import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_FILENAME_LENGTH,
    MAX_QUERY_LENGTH,
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
    UploadResponse,
    router,
)
from nexusrag.pipeline import IngestResult, NexusRAG, SystemStats


@pytest.fixture
def app():
    """Create a FastAPI app with the routes mounted for testing."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_nexusrag():
    """Create a mock NexusRAG instance with common defaults."""
    mock = MagicMock(spec=NexusRAG)

    # Default stats response
    mock.get_stats.return_value = SystemStats(
        total_documents=2,
        total_chunks=10,
        total_words=5000,
        storage_path="/tmp/nexusrag",
        llm_model="llama3.2:3b",
        embedding_model="all-MiniLM-L6-v2",
        llm_available=True,
    )

    # Default empty document list
    mock.list_documents.return_value = []

    return mock


@pytest.fixture
def patch_get_nexusrag(mock_nexusrag):
    """Patch get_nexusrag to return our mock."""
    with patch("nexusrag.api.routes.get_nexusrag", return_value=mock_nexusrag):
        yield mock_nexusrag


# ===========================
# Health Check Tests
# ===========================


class TestHealthCheck:
    """Tests for GET /api/health endpoint."""

    def test_health_check_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful health check."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert data["llm_available"] is True
        assert data["llm_model"] == "llama3.2:3b"
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        assert data["total_documents"] == 2
        assert data["total_chunks"] == 10

    def test_health_check_with_error(self, client):
        """Test health check when NexusRAG raises an exception."""
        with patch("nexusrag.api.routes.get_nexusrag", side_effect=Exception("Test error")):
            response = client.get("/api/health")

            assert response.status_code == 200
            data = response.json()

            # Should still return 200 but with error status
            assert data["status"] == "error"
            assert data["llm_available"] is False

    def test_health_check_response_model(self, client, patch_get_nexusrag):
        """Test that response conforms to HealthResponse model."""
        response = client.get("/api/health")
        data = response.json()

        # Validate against model
        health_response = HealthResponse(**data)
        assert health_response.status in ["ok", "error"]


# ===========================
# Document Ingestion Tests
# ===========================


class TestIngestDocument:
    """Tests for POST /api/ingest endpoint."""

    def test_ingest_pdf_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful PDF file ingestion."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_123",
            filename="test.pdf",
            chunk_count=5,
            word_count=1000,
            success=True,
        )

        # Create a simple PDF file-like object
        file_content = b"%PDF-1.4\ntest content"

        response = client.post(
            "/api/ingest",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["document_id"] == "doc_123"
        assert data["filename"] == "test.pdf"
        assert data["chunk_count"] == 5
        assert data["word_count"] == 1000
        assert data["error"] is None

    def test_ingest_txt_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful TXT file ingestion."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_456",
            filename="test.txt",
            chunk_count=3,
            word_count=500,
            success=True,
        )

        response = client.post(
            "/api/ingest",
            files={"file": ("test.txt", io.BytesIO(b"Some text content"), "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_ingest_docx_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful DOCX file ingestion."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_789",
            filename="test.docx",
            chunk_count=4,
            word_count=800,
            success=True,
        )

        response = client.post(
            "/api/ingest",
            files={
                "file": (
                    "test.docx",
                    io.BytesIO(b"docx content"),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_ingest_md_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful Markdown file ingestion."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_md",
            filename="test.md",
            chunk_count=2,
            word_count=300,
            success=True,
        )

        response = client.post(
            "/api/ingest",
            files={"file": ("test.md", io.BytesIO(b"# Test\nMarkdown content"), "text/markdown")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_ingest_no_filename(self, client, patch_get_nexusrag):
        """Test ingest with missing filename raises error."""
        response = client.post(
            "/api/ingest",
            files={"file": ("", io.BytesIO(b"content"), "text/plain")},
        )

        assert response.status_code in (400, 422)

    def test_ingest_filename_too_long(self, client, patch_get_nexusrag):
        """Test ingest with filename exceeding MAX_FILENAME_LENGTH."""
        long_filename = "a" * (MAX_FILENAME_LENGTH + 1) + ".txt"

        response = client.post(
            "/api/ingest",
            files={"file": (long_filename, io.BytesIO(b"content"), "text/plain")},
        )

        assert response.status_code == 400
        assert "Filename too long" in response.json()["detail"]

    def test_ingest_unsupported_file_type(self, client, patch_get_nexusrag):
        """Test ingest with unsupported file type raises 400."""
        response = client.post(
            "/api/ingest",
            files={"file": ("test.xyz", io.BytesIO(b"content"), "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_ingest_empty_file(self, client, patch_get_nexusrag):
        """Test ingest with empty file raises 400."""
        response = client.post(
            "/api/ingest",
            files={"file": ("test.txt", io.BytesIO(b""), "text/plain")},
        )

        assert response.status_code == 400
        assert "File is empty" in response.json()["detail"]

    def test_ingest_file_too_large(self, client, patch_get_nexusrag):
        """Test ingest with file exceeding MAX_FILE_SIZE_MB raises 413."""
        # Create content larger than MAX_FILE_SIZE_BYTES
        oversized_content = b"x" * (MAX_FILE_SIZE_BYTES + 1)

        response = client.post(
            "/api/ingest",
            files={"file": ("large.pdf", io.BytesIO(oversized_content), "application/pdf")},
        )

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]
        assert f"{MAX_FILE_SIZE_MB}MB" in response.json()["detail"]

    def test_ingest_file_at_size_limit(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test ingest with file exactly at size limit succeeds."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_large",
            filename="large.pdf",
            chunk_count=100,
            word_count=50000,
            success=True,
        )

        # Create content exactly at limit
        content_at_limit = b"x" * MAX_FILE_SIZE_BYTES

        response = client.post(
            "/api/ingest",
            files={"file": ("large.pdf", io.BytesIO(content_at_limit), "application/pdf")},
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_ingest_path_traversal_in_filename(self, client, patch_get_nexusrag):
        """Test that path traversal attempts in filename are sanitized."""
        mock_nexusrag = MagicMock(spec=NexusRAG)
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_traversal",
            filename="test.txt",
            chunk_count=1,
            word_count=100,
            success=True,
        )

        with patch("nexusrag.api.routes.get_nexusrag", return_value=mock_nexusrag):
            response = client.post(
                "/api/ingest",
                files={"file": ("../../etc/passwd.txt", io.BytesIO(b"content"), "text/plain")},
            )

        assert response.status_code == 200
        # Verify that the actual filename passed to ingest_bytes is sanitized
        call_args = mock_nexusrag.ingest_bytes.call_args
        assert call_args[0][1] == "passwd.txt"  # Should be sanitized

    def test_ingest_ingestion_failure(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test ingest when NexusRAG.ingest_bytes returns error."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="",
            filename="test.txt",
            chunk_count=0,
            word_count=0,
            success=False,
            error="Failed to parse document",
        )

        response = client.post(
            "/api/ingest",
            files={"file": ("test.txt", io.BytesIO(b"invalid content"), "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "Failed to parse document"

    def test_ingest_exception_returns_500(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that exceptions during ingestion return 500 error."""
        mock_nexusrag.ingest_bytes.side_effect = Exception("Database error")

        response = client.post(
            "/api/ingest",
            files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
        )

        assert response.status_code == 500
        assert "Failed to process document" in response.json()["detail"]

    def test_ingest_response_model(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that upload response conforms to UploadResponse model."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_123",
            filename="test.pdf",
            chunk_count=5,
            word_count=1000,
            success=True,
        )

        response = client.post(
            "/api/ingest",
            files={"file": ("test.pdf", io.BytesIO(b"content"), "application/pdf")},
        )

        data = response.json()
        upload_response = UploadResponse(**data)
        assert upload_response.success is True


# ===========================
# Query Tests
# ===========================


class TestQueryDocuments:
    """Tests for POST /api/query endpoint."""

    def test_query_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful query."""
        from nexusrag.agents import RAGResponse

        source = Source(
            index=1,
            chunk_id="chunk_1",
            document_id="doc_123",
            document_name="test.pdf",
            content="Relevant content about the topic",
            section_title="Introduction",
            page_number=1,
            score=0.95,
        )

        mock_response = RAGResponse(
            answer="The answer to your question is...",
            sources=[source],
            confidence=0.92,
            reasoning_trace=[],
            processing_time_ms=150.5,
        )
        mock_nexusrag.query.return_value = mock_response
        mock_nexusrag.list_documents.return_value = [
            {"id": "doc_123", "filename": "test.pdf", "original_filename": "test.pdf"}
        ]

        response = client.post(
            "/api/query",
            json={"question": "What is the capital of France?"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "The answer to your question is..."
        assert data["confidence"] == 0.92
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "test.pdf"
        assert data["processing_time_ms"] == 150.5

    def test_query_empty_question(self, client, patch_get_nexusrag):
        """Test query with empty question raises 422."""
        response = client.post(
            "/api/query",
            json={"question": ""},
        )

        assert response.status_code == 422  # Validation error
        assert "cannot be empty" in response.json().get("detail", [{}])[0].get("msg", "").lower()

    def test_query_whitespace_only_question(self, client, patch_get_nexusrag):
        """Test query with whitespace-only question raises 422."""
        response = client.post(
            "/api/query",
            json={"question": "   \n\t   "},
        )

        assert response.status_code == 422

    def test_query_question_too_long(self, client, patch_get_nexusrag):
        """Test query with question exceeding MAX_QUERY_LENGTH raises 422."""
        long_question = "a" * (MAX_QUERY_LENGTH + 1)

        response = client.post(
            "/api/query",
            json={"question": long_question},
        )

        assert response.status_code == 422
        assert "too long" in response.json().get("detail", [{}])[0].get("msg", "").lower()

    def test_query_at_max_length(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test query with question at exactly MAX_QUERY_LENGTH succeeds."""
        max_length_question = "a" * MAX_QUERY_LENGTH

        mock_response = RAGResponse(
            answer="Answer",
            sources=[],
            confidence=0.5,
            reasoning_trace=[],
            processing_time_ms=100.0,
        )
        mock_nexusrag.query.return_value = mock_response

        response = client.post(
            "/api/query",
            json={"question": max_length_question},
        )

        assert response.status_code == 200

    def test_query_missing_question_field(self, client, patch_get_nexusrag):
        """Test query without question field raises 422."""
        response = client.post(
            "/api/query",
            json={},
        )

        assert response.status_code == 422

    def test_query_no_documents(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test query when no documents are available."""
        from nexusrag.agents import RAGResponse

        mock_response = RAGResponse(
            answer="No documents have been uploaded yet.",
            sources=[],
            confidence=0.0,
            reasoning_trace=[],
            processing_time_ms=10.0,
        )
        mock_nexusrag.query.return_value = mock_response
        mock_nexusrag.list_documents.return_value = []

        response = client.post(
            "/api/query",
            json={"question": "What is the answer?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "No documents" in data["answer"] or data["confidence"] == 0.0

    def test_query_multiple_sources(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test query response with multiple sources."""
        from nexusrag.agents import RAGResponse

        sources = [
            Source(
                index=1,
                chunk_id="chunk_1",
                document_id="doc_1",
                document_name="paper1.pdf",
                content="Content from paper 1",
                section_title="Methods",
                page_number=5,
                score=0.95,
            ),
            Source(
                index=2,
                chunk_id="chunk_2",
                document_id="doc_2",
                document_name="paper2.pdf",
                content="Content from paper 2",
                section_title="Results",
                page_number=10,
                score=0.87,
            ),
        ]

        mock_response = RAGResponse(
            answer="Combined answer from both sources",
            sources=sources,
            confidence=0.91,
            reasoning_trace=[],
            processing_time_ms=200.0,
        )
        mock_nexusrag.query.return_value = mock_response
        mock_nexusrag.list_documents.return_value = [
            {"id": "doc_1", "filename": "paper1.pdf", "original_filename": "paper1.pdf"},
            {"id": "doc_2", "filename": "paper2.pdf", "original_filename": "paper2.pdf"},
        ]

        response = client.post(
            "/api/query",
            json={"question": "What did the research show?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["score"] == 0.95
        assert data["sources"][1]["score"] == 0.87

    def test_query_long_source_content_truncated(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that long source content is truncated to 500 chars."""
        from nexusrag.agents import RAGResponse

        long_content = "x" * 1000  # 1000 characters
        source = Source(
            index=1,
            chunk_id="chunk_1",
            document_id="doc_123",
            document_name="test.pdf",
            content=long_content,
            section_title="Section",
            page_number=1,
            score=0.9,
        )

        mock_response = RAGResponse(
            answer="Answer",
            sources=[source],
            confidence=0.9,
            reasoning_trace=[],
            processing_time_ms=100.0,
        )
        mock_nexusrag.query.return_value = mock_response
        mock_nexusrag.list_documents.return_value = []

        response = client.post(
            "/api/query",
            json={"question": "What is this?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"][0]["content"]) <= 503  # 500 + "..."
        assert data["sources"][0]["content"].endswith("...")

    def test_query_exception_returns_500(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that exceptions during query return 500 error."""
        mock_nexusrag.query.side_effect = Exception("LLM error")

        response = client.post(
            "/api/query",
            json={"question": "What is the answer?"},
        )

        assert response.status_code == 500
        assert "Failed to process query" in response.json()["detail"]

    def test_query_response_model(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that query response conforms to QueryResponse model."""
        from nexusrag.agents import RAGResponse

        mock_response = RAGResponse(
            answer="Test answer",
            sources=[],
            confidence=0.85,
            reasoning_trace=[],
            processing_time_ms=120.0,
        )
        mock_nexusrag.query.return_value = mock_response

        response = client.post(
            "/api/query",
            json={"question": "Test question?"},
        )

        data = response.json()
        query_response = QueryResponse(**data)
        assert query_response.confidence == 0.85


# ===========================
# Documents List Tests
# ===========================


class TestListDocuments:
    """Tests for GET /api/documents endpoint."""

    def test_list_documents_empty(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test listing documents when none exist."""
        mock_nexusrag.list_documents.return_value = []

        response = client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()

        assert data["documents"] == []
        assert data["total_documents"] == 2
        assert data["total_chunks"] == 10

    def test_list_documents_with_documents(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test listing documents when multiple exist."""
        mock_nexusrag.list_documents.return_value = [
            {
                "id": "doc_1",
                "filename": "paper1.pdf",
                "original_filename": "paper1.pdf",
                "word_count": 1000,
                "chunk_count": 5,
                "file_type": "pdf",
                "uploaded_at": "2024-01-15T10:30:00",
            },
            {
                "id": "doc_2",
                "filename": "paper2.pdf",
                "original_filename": "paper2.pdf",
                "word_count": 800,
                "chunk_count": 4,
                "file_type": "pdf",
                "uploaded_at": "2024-01-15T11:00:00",
            },
        ]

        response = client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()

        assert len(data["documents"]) == 2
        assert data["documents"][0]["id"] == "doc_1"
        assert data["documents"][0]["word_count"] == 1000
        assert data["documents"][1]["id"] == "doc_2"

    def test_list_documents_response_model(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that documents list response conforms to DocumentListResponse model."""
        mock_nexusrag.list_documents.return_value = [
            {
                "id": "doc_1",
                "filename": "test.pdf",
                "original_filename": "test.pdf",
                "word_count": 500,
                "chunk_count": 2,
                "file_type": "pdf",
                "uploaded_at": "2024-01-15T10:00:00",
            }
        ]

        response = client.get("/api/documents")
        data = response.json()

        doc_list_response = DocumentListResponse(**data)
        assert len(doc_list_response.documents) == 1
        assert doc_list_response.total_documents == 2

    def test_list_documents_exception_returns_500(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that exceptions during list return 500 error."""
        mock_nexusrag.list_documents.side_effect = Exception("Storage error")

        response = client.get("/api/documents")

        assert response.status_code == 500
        assert "Failed to retrieve documents" in response.json()["detail"]


# ===========================
# Status Tests
# ===========================


class TestGetStatus:
    """Tests for GET /api/status endpoint."""

    def test_get_status_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful status retrieval."""
        response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()

        assert data["total_documents"] == 2
        assert data["total_chunks"] == 10
        assert data["total_words"] == 5000
        assert data["llm_available"] is True
        assert data["llm_model"] == "llama3.2:3b"
        assert data["embedding_model"] == "all-MiniLM-L6-v2"

    def test_get_status_response_model(self, client, patch_get_nexusrag):
        """Test that status response conforms to StatusResponse model."""
        response = client.get("/api/status")
        data = response.json()

        status_response = StatusResponse(**data)
        assert status_response.total_documents == 2

    def test_get_status_exception_returns_500(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that exceptions during status retrieval return 500 error."""
        mock_nexusrag.get_stats.side_effect = Exception("Database error")

        response = client.get("/api/status")

        assert response.status_code == 500
        assert "Failed to retrieve status" in response.json()["detail"]


# ===========================
# Delete Tests
# ===========================


class TestDeleteDocument:
    """Tests for DELETE /api/documents/{document_id} endpoint."""

    def test_delete_document_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful document deletion."""
        mock_nexusrag.delete_document.return_value = True

        response = client.delete("/api/documents/doc_123")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["message"] == "Document deleted"
        mock_nexusrag.delete_document.assert_called_once_with("doc_123")

    def test_delete_document_not_found(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test deleting non-existent document returns 404."""
        mock_nexusrag.delete_document.return_value = False

        response = client.delete("/api/documents/nonexistent")

        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_delete_document_empty_id(self, client, patch_get_nexusrag):
        """Test delete with empty document_id hits the clear_all endpoint."""
        response = client.delete("/api/documents/")

        # FastAPI routing matches the clear_all endpoint (DELETE /api/documents)
        assert response.status_code == 200

    def test_delete_document_id_too_long(self, client, patch_get_nexusrag):
        """Test delete with document_id exceeding length limit raises 400."""
        long_id = "a" * 65  # Exceeds 64-char limit

        response = client.delete(f"/api/documents/{long_id}")

        assert response.status_code == 400
        assert "Invalid document ID" in response.json()["detail"]

    def test_delete_document_exception_returns_500(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that exceptions during deletion return 500 error."""
        mock_nexusrag.delete_document.side_effect = Exception("Deletion error")

        response = client.delete("/api/documents/doc_123")

        assert response.status_code == 500
        assert "Failed to delete document" in response.json()["detail"]

    def test_delete_document_validation_error(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test delete with invalid document_id format."""
        mock_nexusrag.delete_document.side_effect = ValueError("Invalid ID format")

        response = client.delete("/api/documents/invalid-id-format")

        assert response.status_code == 400
        assert "Invalid document ID format" in response.json()["detail"]

    def test_delete_document_response_model(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that delete response conforms to DeleteResponse model."""
        mock_nexusrag.delete_document.return_value = True

        response = client.delete("/api/documents/doc_123")
        data = response.json()

        delete_response = DeleteResponse(**data)
        assert delete_response.success is True


class TestClearAllDocuments:
    """Tests for DELETE /api/documents endpoint (clear all)."""

    def test_clear_all_documents_success(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test successful clearing of all documents."""
        response = client.delete("/api/documents")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["message"] == "All documents cleared"
        mock_nexusrag.clear_all.assert_called_once()

    def test_clear_all_documents_exception_returns_500(
        self, client, patch_get_nexusrag, mock_nexusrag
    ):
        """Test that exceptions during clear_all return 500 error."""
        mock_nexusrag.clear_all.side_effect = Exception("Clear error")

        response = client.delete("/api/documents")

        assert response.status_code == 500
        assert "Failed to clear documents" in response.json()["detail"]

    def test_clear_all_documents_response_model(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that clear response conforms to DeleteResponse model."""
        response = client.delete("/api/documents")
        data = response.json()

        delete_response = DeleteResponse(**data)
        assert delete_response.success is True


# ===========================
# Integration Tests
# ===========================


class TestIntegration:
    """Integration tests combining multiple endpoints."""

    def test_workflow_ingest_query_list(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test workflow: ingest document -> query -> list documents."""
        from nexusrag.agents import RAGResponse

        # Step 1: Ingest document
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_1",
            filename="research.pdf",
            chunk_count=5,
            word_count=1000,
            success=True,
        )

        ingest_response = client.post(
            "/api/ingest",
            files={"file": ("research.pdf", io.BytesIO(b"PDF content"), "application/pdf")},
        )
        assert ingest_response.status_code == 200

        # Step 2: Query the document
        source = Source(
            index=1,
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="research.pdf",
            content="Research findings",
            section_title="Results",
            page_number=5,
            score=0.95,
        )

        mock_response = RAGResponse(
            answer="The research shows...",
            sources=[source],
            confidence=0.90,
            reasoning_trace=[],
            processing_time_ms=150.0,
        )
        mock_nexusrag.query.return_value = mock_response
        mock_nexusrag.list_documents.return_value = [
            {
                "id": "doc_1",
                "filename": "research.pdf",
                "original_filename": "research.pdf",
                "word_count": 1000,
                "chunk_count": 5,
                "file_type": "pdf",
                "uploaded_at": "2024-01-15T10:00:00",
            }
        ]

        query_response = client.post(
            "/api/query",
            json={"question": "What are the findings?"},
        )
        assert query_response.status_code == 200
        assert query_response.json()["answer"] == "The research shows..."

        # Step 3: List documents
        list_response = client.get("/api/documents")
        assert list_response.status_code == 200
        assert len(list_response.json()["documents"]) == 1

    def test_workflow_ingest_delete(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test workflow: ingest document -> delete document."""
        # Ingest
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_to_delete",
            filename="temp.txt",
            chunk_count=2,
            word_count=100,
            success=True,
        )

        ingest_response = client.post(
            "/api/ingest",
            files={"file": ("temp.txt", io.BytesIO(b"content"), "text/plain")},
        )
        assert ingest_response.status_code == 200

        # Delete
        mock_nexusrag.delete_document.return_value = True

        delete_response = client.delete("/api/documents/doc_to_delete")
        assert delete_response.status_code == 200
        assert delete_response.json()["success"] is True


# ===========================
# Pydantic Validation Tests
# ===========================


class TestPydanticValidation:
    """Direct tests of Pydantic model validation."""

    def test_query_request_validation_empty_question(self):
        """Test QueryRequest validation rejects empty question."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="")

        assert "Question cannot be empty" in str(exc_info.value)

    def test_query_request_validation_whitespace_question(self):
        """Test QueryRequest validation rejects whitespace-only question."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="   \n\t   ")

        assert "Question cannot be empty" in str(exc_info.value)

    def test_query_request_validation_question_too_long(self):
        """Test QueryRequest validation rejects overly long question."""
        long_question = "a" * (MAX_QUERY_LENGTH + 1)

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question=long_question)

        assert "too long" in str(exc_info.value).lower()

    def test_query_request_valid_question(self):
        """Test QueryRequest accepts valid question."""
        request = QueryRequest(question="What is the meaning of life?")
        assert request.question == "What is the meaning of life?"

    def test_query_request_trims_whitespace(self):
        """Test QueryRequest trims leading/trailing whitespace."""
        request = QueryRequest(question="  What is this?  \n")
        assert request.question == "What is this?"


# ===========================
# Error Handling Tests
# ===========================


class TestErrorHandling:
    """Tests for proper error handling across endpoints."""

    def test_missing_content_type_header(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test ingest handles missing content type gracefully."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_1",
            filename="test.txt",
            chunk_count=1,
            word_count=100,
            success=True,
        )

        response = client.post(
            "/api/ingest",
            files={"file": ("test.txt", io.BytesIO(b"content"))},
        )

        assert response.status_code == 200

    def test_concurrent_ingest_requests(self, client, patch_get_nexusrag, mock_nexusrag):
        """Test that multiple concurrent ingests are handled properly."""
        mock_nexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_1",
            filename="test.txt",
            chunk_count=1,
            word_count=100,
            success=True,
        )

        responses = []
        for i in range(3):
            response = client.post(
                "/api/ingest",
                files={"file": (f"test{i}.txt", io.BytesIO(b"content"), "text/plain")},
            )
            responses.append(response)

        for response in responses:
            assert response.status_code == 200

    def test_invalid_json_in_query(self, client, patch_get_nexusrag):
        """Test that invalid JSON in query request returns 422."""
        response = client.post(
            "/api/query",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422
