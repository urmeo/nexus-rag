"""Tests for the API routes."""

import io
import zipfile
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scinexusrag.api import create_app
from scinexusrag.api.routes import (
    MAX_FILENAME_LENGTH,
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
    UploadResponse,
    router,
)
from scinexusrag.config import EmbeddingSettings, Settings, StorageSettings, get_settings
from scinexusrag.generation import RAGResponse, Source
from scinexusrag.ingestion.embedder import Embedder, EmbeddingConfig, EmbeddingConfigError
from scinexusrag.pipeline import IngestResult, NexusRAG, SystemStats

MAX_FILE_SIZE_MB = get_settings().api.max_upload_mb
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_QUERY_LENGTH = get_settings().retrieval.max_query_length


def _docx_bytes() -> bytes:
    """A minimal real .docx (zip) so magic-byte validation accepts it."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<?xml version='1.0'?><Types/>")
    return buf.getvalue()


PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\ntest content"
DOCX_BYTES = _docx_bytes()


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_scinexusrag():
    mock = MagicMock(spec=NexusRAG)

    mock.get_stats.return_value = SystemStats(
        total_documents=2,
        total_chunks=10,
        total_words=5000,
        storage_path="/tmp/scinexusrag",
        llm_model="llama3.2:3b",
        embedding_model="all-MiniLM-L6-v2",
        llm_available=True,
    )

    mock.list_documents.return_value = []

    return mock


@pytest.fixture
def patch_get_scinexusrag(mock_scinexusrag):
    with patch("scinexusrag.api.routes.get_scinexusrag", return_value=mock_scinexusrag):
        yield mock_scinexusrag


class TestHealthCheck:
    def test_health_check_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
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
        with patch("scinexusrag.api.routes.get_scinexusrag", side_effect=Exception("Test error")):
            response = client.get("/api/health")

            assert response.status_code == 200
            data = response.json()

            assert data["status"] == "error"
            assert data["llm_available"] is False

    def test_health_check_response_model(self, client, patch_get_scinexusrag):
        response = client.get("/api/health")
        data = response.json()

        health_response = HealthResponse(**data)
        assert health_response.status in ["ok", "error"]


class TestIngestDocument:
    def test_ingest_pdf_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_123",
            filename="test.pdf",
            chunk_count=5,
            word_count=1000,
            success=True,
        )

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

    def test_ingest_txt_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
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

    def test_ingest_docx_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
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
                    io.BytesIO(DOCX_BYTES),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_ingest_md_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
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

    def test_ingest_no_filename(self, client, patch_get_scinexusrag):
        response = client.post(
            "/api/ingest",
            files={"file": ("", io.BytesIO(b"content"), "text/plain")},
        )

        assert response.status_code in (400, 422)

    def test_ingest_filename_too_long(self, client, patch_get_scinexusrag):
        long_filename = "a" * (MAX_FILENAME_LENGTH + 1) + ".txt"

        response = client.post(
            "/api/ingest",
            files={"file": (long_filename, io.BytesIO(b"content"), "text/plain")},
        )

        assert response.status_code == 400
        assert "Filename too long" in response.json()["detail"]

    def test_ingest_unsupported_file_type(self, client, patch_get_scinexusrag):
        response = client.post(
            "/api/ingest",
            files={"file": ("test.xyz", io.BytesIO(b"content"), "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_ingest_empty_file(self, client, patch_get_scinexusrag):
        response = client.post(
            "/api/ingest",
            files={"file": ("test.txt", io.BytesIO(b""), "text/plain")},
        )

        assert response.status_code == 400
        assert "File is empty" in response.json()["detail"]

    def test_ingest_file_too_large(self, client, patch_get_scinexusrag):

        oversized_content = b"x" * (MAX_FILE_SIZE_BYTES + 1)

        response = client.post(
            "/api/ingest",
            files={"file": ("large.pdf", io.BytesIO(oversized_content), "application/pdf")},
        )

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]
        assert f"{MAX_FILE_SIZE_MB}MB" in response.json()["detail"]

    def test_ingest_file_at_size_limit(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_large",
            filename="large.pdf",
            chunk_count=100,
            word_count=50000,
            success=True,
        )

        content_at_limit = b"%PDF-1.4\n" + b"x" * (MAX_FILE_SIZE_BYTES - 9)

        response = client.post(
            "/api/ingest",
            files={"file": ("large.pdf", io.BytesIO(content_at_limit), "application/pdf")},
        )

        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_ingest_path_traversal_in_filename(self, client, patch_get_scinexusrag):
        mock_scinexusrag = MagicMock(spec=NexusRAG)
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_traversal",
            filename="test.txt",
            chunk_count=1,
            word_count=100,
            success=True,
        )

        with patch("scinexusrag.api.routes.get_scinexusrag", return_value=mock_scinexusrag):
            response = client.post(
                "/api/ingest",
                files={"file": ("../../etc/passwd.txt", io.BytesIO(b"content"), "text/plain")},
            )

        assert response.status_code == 200

        call_args = mock_scinexusrag.ingest_bytes.call_args
        assert call_args[0][1] == "passwd.txt"

    def test_ingest_ingestion_failure(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
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

    def test_ingest_exception_returns_500(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.side_effect = Exception("Database error")

        response = client.post(
            "/api/ingest",
            files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
        )

        assert response.status_code == 500
        assert "Failed to process document" in response.json()["detail"]

    def test_ingest_response_model(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_123",
            filename="test.pdf",
            chunk_count=5,
            word_count=1000,
            success=True,
        )

        response = client.post(
            "/api/ingest",
            files={"file": ("test.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        data = response.json()
        upload_response = UploadResponse(**data)
        assert upload_response.success is True


class TestQueryDocuments:
    def test_query_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
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
        mock_scinexusrag.query.return_value = mock_response
        mock_scinexusrag.list_documents.return_value = [
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

    def test_query_surfaces_warnings(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.query.return_value = RAGResponse(
            answer="Ungrounded answer.",
            sources=[],
            confidence=0.3,
            reasoning_trace=[],
            processing_time_ms=12.0,
            warnings=["answer cites no sources"],
        )
        mock_scinexusrag.list_documents.return_value = []

        data = client.post("/api/query", json={"question": "q?"}).json()

        assert data["warnings"] == ["answer cites no sources"]

    def test_query_empty_question(self, client, patch_get_scinexusrag):
        response = client.post(
            "/api/query",
            json={"question": ""},
        )

        assert response.status_code == 422
        assert "cannot be empty" in response.json().get("detail", [{}])[0].get("msg", "").lower()

    def test_query_whitespace_only_question(self, client, patch_get_scinexusrag):
        response = client.post(
            "/api/query",
            json={"question": "   \n\t   "},
        )

        assert response.status_code == 422

    def test_query_question_too_long(self, client, patch_get_scinexusrag):
        long_question = "a" * (MAX_QUERY_LENGTH + 1)

        response = client.post(
            "/api/query",
            json={"question": long_question},
        )

        assert response.status_code == 422
        assert "too long" in response.json().get("detail", [{}])[0].get("msg", "").lower()

    def test_query_at_max_length(self, client, patch_get_scinexusrag, mock_scinexusrag):
        max_length_question = "a" * MAX_QUERY_LENGTH

        mock_response = RAGResponse(
            answer="Answer",
            sources=[],
            confidence=0.5,
            reasoning_trace=[],
            processing_time_ms=100.0,
        )
        mock_scinexusrag.query.return_value = mock_response

        response = client.post(
            "/api/query",
            json={"question": max_length_question},
        )

        assert response.status_code == 200

    def test_query_missing_question_field(self, client, patch_get_scinexusrag):
        response = client.post(
            "/api/query",
            json={},
        )

        assert response.status_code == 422

    def test_query_no_documents(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_response = RAGResponse(
            answer="No documents have been uploaded yet.",
            sources=[],
            confidence=0.0,
            reasoning_trace=[],
            processing_time_ms=10.0,
        )
        mock_scinexusrag.query.return_value = mock_response
        mock_scinexusrag.list_documents.return_value = []

        response = client.post(
            "/api/query",
            json={"question": "What is the answer?"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["sources"] == []
        assert data["processing_time_ms"] == 10.0

    def test_query_multiple_sources(self, client, patch_get_scinexusrag, mock_scinexusrag):
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
        mock_scinexusrag.query.return_value = mock_response
        mock_scinexusrag.list_documents.return_value = [
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

    def test_query_preserves_source_indices(self, client, patch_get_scinexusrag, mock_scinexusrag):
        sources = [
            Source(index=index, chunk_id=f"chunk_{index}", document_id="doc", content="Evidence.")
            for index in (2, 4)
        ]
        mock_scinexusrag.query.return_value = RAGResponse("Answer [2] [4].", sources, 0.5)

        response = client.post("/api/query", json={"question": "What was measured?"})

        assert response.status_code == 200
        assert [source["index"] for source in response.json()["sources"]] == [2, 4]

    def test_query_long_source_content_truncated(
        self, client, patch_get_scinexusrag, mock_scinexusrag
    ):
        long_content = "x" * 1000
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
        mock_scinexusrag.query.return_value = mock_response
        mock_scinexusrag.list_documents.return_value = []

        response = client.post(
            "/api/query",
            json={"question": "What is this?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"][0]["content"]) <= 503
        assert data["sources"][0]["content"].endswith("...")

    def test_query_exception_returns_500(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.query.side_effect = Exception("LLM error")

        response = client.post(
            "/api/query",
            json={"question": "What is the answer?"},
        )

        assert response.status_code == 500
        assert "Failed to process query" in response.json()["detail"]

    def test_query_response_model(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_response = RAGResponse(
            answer="Test answer",
            sources=[],
            confidence=0.85,
            reasoning_trace=[],
            processing_time_ms=120.0,
        )
        mock_scinexusrag.query.return_value = mock_response

        response = client.post(
            "/api/query",
            json={"question": "Test question?"},
        )

        data = response.json()
        query_response = QueryResponse(**data)
        assert query_response.confidence == 0.85


class TestListDocuments:
    def test_list_documents_empty(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.list_documents.return_value = []

        response = client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()

        assert data["documents"] == []
        assert data["total_documents"] == 2
        assert data["total_chunks"] == 10

    def test_list_documents_with_documents(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.list_documents.return_value = [
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

    def test_list_documents_pagination(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.list_documents.return_value = [
            {"id": f"doc_{i}", "filename": f"p{i}.pdf", "word_count": 10} for i in range(5)
        ]

        page = client.get("/api/documents?limit=2&offset=2").json()

        assert [d["id"] for d in page["documents"]] == ["doc_2", "doc_3"]
        assert page["total_documents"] == 2

        assert client.get("/api/documents?limit=0").status_code == 422
        assert client.get("/api/documents?offset=-1").status_code == 422

    def test_list_documents_response_model(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.list_documents.return_value = [
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

    def test_list_documents_exception_returns_500(
        self, client, patch_get_scinexusrag, mock_scinexusrag
    ):
        mock_scinexusrag.list_documents.side_effect = Exception("Storage error")

        response = client.get("/api/documents")

        assert response.status_code == 500
        assert "Failed to retrieve documents" in response.json()["detail"]


class TestGetStatus:
    def test_get_status_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
        response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()

        assert data["total_documents"] == 2
        assert data["total_chunks"] == 10
        assert data["total_words"] == 5000
        assert data["llm_available"] is True
        assert data["llm_model"] == "llama3.2:3b"
        assert data["embedding_model"] == "all-MiniLM-L6-v2"

    def test_get_status_response_model(self, client, patch_get_scinexusrag):
        response = client.get("/api/status")
        data = response.json()

        status_response = StatusResponse(**data)
        assert status_response.total_documents == 2

    def test_get_status_exception_returns_500(
        self, client, patch_get_scinexusrag, mock_scinexusrag
    ):
        mock_scinexusrag.get_stats.side_effect = Exception("Database error")

        response = client.get("/api/status")

        assert response.status_code == 500
        assert "Failed to retrieve status" in response.json()["detail"]


class TestDeleteDocument:
    def test_delete_document_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.delete_document.return_value = True

        response = client.delete("/api/documents/doc_123")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["message"] == "Document deleted"
        mock_scinexusrag.delete_document.assert_called_once_with("doc_123")

    def test_delete_document_not_found(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.delete_document.return_value = False

        response = client.delete("/api/documents/nonexistent")

        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_trailing_slash_uses_clear_all_route(self, client, patch_get_scinexusrag):
        response = client.delete("/api/documents/")

        assert response.status_code == 200
        patch_get_scinexusrag.clear_all.assert_called_once()

    def test_delete_document_id_too_long(self, client, patch_get_scinexusrag):
        long_id = "a" * 65

        response = client.delete(f"/api/documents/{long_id}")

        assert response.status_code == 400
        assert "Invalid document ID" in response.json()["detail"]

    def test_delete_document_exception_returns_500(
        self, client, patch_get_scinexusrag, mock_scinexusrag
    ):
        mock_scinexusrag.delete_document.side_effect = Exception("Deletion error")

        response = client.delete("/api/documents/doc_123")

        assert response.status_code == 500
        assert "Failed to delete document" in response.json()["detail"]

    def test_delete_document_validation_error(
        self, client, patch_get_scinexusrag, mock_scinexusrag
    ):
        mock_scinexusrag.delete_document.side_effect = ValueError("Invalid ID format")

        response = client.delete("/api/documents/invalid-id-format")

        assert response.status_code == 400
        assert "Invalid document ID format" in response.json()["detail"]

    def test_delete_document_response_model(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.delete_document.return_value = True

        response = client.delete("/api/documents/doc_123")
        data = response.json()

        delete_response = DeleteResponse(**data)
        assert delete_response.success is True


class TestClearAllDocuments:
    def test_clear_all_documents_success(self, client, patch_get_scinexusrag, mock_scinexusrag):
        response = client.delete("/api/documents")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["message"] == "All documents cleared"
        mock_scinexusrag.clear_all.assert_called_once()

    def test_clear_all_documents_exception_returns_500(
        self, client, patch_get_scinexusrag, mock_scinexusrag
    ):
        mock_scinexusrag.clear_all.side_effect = Exception("Clear error")

        response = client.delete("/api/documents")

        assert response.status_code == 500
        assert "Failed to clear documents" in response.json()["detail"]

    def test_clear_all_documents_response_model(
        self, client, patch_get_scinexusrag, mock_scinexusrag
    ):
        response = client.delete("/api/documents")
        data = response.json()

        delete_response = DeleteResponse(**data)
        assert delete_response.success is True


class TestIntegration:
    def test_workflow_ingest_query_list(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
            document_id="doc_1",
            filename="research.pdf",
            chunk_count=5,
            word_count=1000,
            success=True,
        )

        ingest_response = client.post(
            "/api/ingest",
            files={"file": ("research.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert ingest_response.status_code == 200

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
        mock_scinexusrag.query.return_value = mock_response
        mock_scinexusrag.list_documents.return_value = [
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

        list_response = client.get("/api/documents")
        assert list_response.status_code == 200
        assert len(list_response.json()["documents"]) == 1

    def test_workflow_ingest_delete(self, client, patch_get_scinexusrag, mock_scinexusrag):

        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
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

        mock_scinexusrag.delete_document.return_value = True

        delete_response = client.delete("/api/documents/doc_to_delete")
        assert delete_response.status_code == 200
        assert delete_response.json()["success"] is True


class TestPydanticValidation:
    def test_query_request_validation_empty_question(self):
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="")

        assert "Question cannot be empty" in str(exc_info.value)

    def test_query_request_validation_whitespace_question(self):
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question="   \n\t   ")

        assert "Question cannot be empty" in str(exc_info.value)

    def test_query_request_validation_question_too_long(self):
        long_question = "a" * (MAX_QUERY_LENGTH + 1)

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(question=long_question)

        assert "too long" in str(exc_info.value).lower()

    def test_query_request_valid_question(self):
        request = QueryRequest(question="What is the meaning of life?")
        assert request.question == "What is the meaning of life?"

    def test_query_request_trims_whitespace(self):
        request = QueryRequest(question="  What is this?  \n")
        assert request.question == "What is this?"


class TestErrorHandling:
    def test_missing_content_type_header(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
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

    def test_repeated_ingest_requests(self, client, patch_get_scinexusrag, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.return_value = IngestResult(
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

    def test_invalid_json_in_query(self, client, patch_get_scinexusrag):
        response = client.post(
            "/api/query",
            content="{invalid json}",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


@pytest.fixture
def embedding_client(patch_get_scinexusrag):
    return TestClient(create_app())


class TestEmbeddingConfigErrors:
    @staticmethod
    def assert_conflict(response):
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "EMBEDDING_REVISION" in detail
        assert "DATA_DIR" in detail
        assert "LANCEDB_PATH" in detail
        assert "original documents" in detail
        assert "private embedding metadata" not in detail

    @pytest.mark.parametrize(
        "path", ["/api/health", "/api/status", "/api/documents", "/api/metrics"]
    )
    def test_count_conflict_is_not_reported_as_empty_or_healthy(
        self, embedding_client, mock_scinexusrag, path
    ):
        mock_scinexusrag.get_stats.side_effect = EmbeddingConfigError("private embedding metadata")

        self.assert_conflict(embedding_client.get(path))

    @pytest.mark.parametrize("failed_method", ["query", "list_documents"])
    def test_query_conflict_is_actionable_and_does_not_expose_metadata(
        self, embedding_client, mock_scinexusrag, failed_method
    ):
        mock_scinexusrag.query.return_value = RAGResponse("Answer.", [], 0.0)
        getattr(mock_scinexusrag, failed_method).side_effect = EmbeddingConfigError(
            "private embedding metadata"
        )

        response = embedding_client.post("/api/query", json={"question": "What was measured?"})

        self.assert_conflict(response)

    def test_ingest_conflict_returns_409(self, embedding_client, mock_scinexusrag):
        mock_scinexusrag.ingest_bytes.side_effect = EmbeddingConfigError(
            "private embedding metadata"
        )

        response = embedding_client.post(
            "/api/ingest", files={"file": ("paper.txt", b"Sensor readings.", "text/plain")}
        )

        self.assert_conflict(response)

    def test_real_ingest_rejects_same_dimension_model_change(self, temp_dir):
        settings = Settings(
            data_dir=temp_dir,
            storage=StorageSettings(lancedb_path=temp_dir / "lancedb"),
            embedding=EmbeddingSettings(model="test-model", revision="a" * 40),
        )
        config = EmbeddingConfig("test-model", "a" * 40, "", "", True, 4)
        original_embedder = MagicMock(spec=Embedder)
        original_embedder.embedding_config = config
        original_embedder.embed.return_value = np.array([[1, 0, 0, 0]], dtype=np.float32)
        original = NexusRAG(settings)
        original._embedder = original_embedder
        result = original.ingest_bytes(
            b"Sensor A records temperature every thirty seconds.", "original.txt", ".txt"
        )
        assert result.success
        assert original.vector_store.count() == 1

        changed = NexusRAG(settings)
        changed_embedder = MagicMock(spec=Embedder)
        changed_embedder.embedding_config = replace(config, model_name="different-model")
        changed._embedder = changed_embedder
        with (
            patch("scinexusrag.api.routes.get_scinexusrag", return_value=changed),
            TestClient(create_app()) as client,
        ):
            self.assert_conflict(
                client.post(
                    "/api/ingest",
                    files={"file": ("new.txt", b"Sensor B records pressure.", "text/plain")},
                )
            )
            self.assert_conflict(
                client.post("/api/query", json={"question": "What did the sensor record?"})
            )

        changed_embedder.embed.assert_not_called()
        changed_embedder.embed_query.assert_not_called()
        assert changed._llm is None
        reopened = NexusRAG(settings)
        reopened._embedder = original_embedder
        assert reopened.vector_store.count() == 1
        assert [document["id"] for document in reopened.list_documents()] == [result.document_id]

    def test_delete_conflict_returns_409(self, embedding_client, mock_scinexusrag):
        mock_scinexusrag.delete_document.side_effect = EmbeddingConfigError(
            "private embedding metadata"
        )

        self.assert_conflict(embedding_client.delete("/api/documents/paper"))

    def test_explicit_clear_does_not_require_compatible_stats(
        self, embedding_client, mock_scinexusrag
    ):
        mock_scinexusrag.get_stats.side_effect = EmbeddingConfigError("private embedding metadata")

        response = embedding_client.delete("/api/documents")

        assert response.status_code == 200
        mock_scinexusrag.clear_all.assert_called_once()
        mock_scinexusrag.get_stats.assert_not_called()

    def test_auth_runs_before_index_checks(self, embedding_client, mock_scinexusrag):
        mock_scinexusrag.get_stats.side_effect = EmbeddingConfigError("private embedding metadata")
        with patch(
            "scinexusrag.api.security.get_settings", return_value=_settings(api_key="secret")
        ):
            response = embedding_client.get("/api/health")

        assert response.status_code == 401
        mock_scinexusrag.get_stats.assert_not_called()


def _settings(**api_overrides) -> Settings:
    s = Settings()
    for key, value in api_overrides.items():
        setattr(s.api, key, value)
    return s


class TestApiKeyAuth:
    """Default-deny auth must be enforced end-to-end, not just unit-tested."""

    def test_missing_key_rejected(self, client, patch_get_scinexusrag):
        with patch(
            "scinexusrag.api.security.get_settings", return_value=_settings(api_key="secret")
        ):
            assert client.get("/api/status").status_code == 401

    def test_wrong_key_rejected(self, client, patch_get_scinexusrag):
        with patch(
            "scinexusrag.api.security.get_settings", return_value=_settings(api_key="secret")
        ):
            r = client.get("/api/status", headers={"X-API-Key": "wrong"})
            assert r.status_code == 401

    def test_correct_key_allowed(self, client, patch_get_scinexusrag):
        with patch(
            "scinexusrag.api.security.get_settings", return_value=_settings(api_key="secret")
        ):
            r = client.get("/api/status", headers={"X-API-Key": "secret"})
            assert r.status_code == 200


class TestUploadValidationThroughRoute:
    """The magic-byte / zip-bomb guard must fire via the HTTP layer."""

    def test_wrong_magic_bytes_rejected(self, client, patch_get_scinexusrag):
        files = {"file": ("bad.pdf", b"this is not a pdf at all", "application/pdf")}
        r = client.post("/api/ingest", files=files)
        assert r.status_code == 415

    def test_zip_bomb_rejected(self, client, patch_get_scinexusrag):

        with patch(
            "scinexusrag.api.security.get_settings", return_value=_settings(max_uncompressed_mb=0)
        ):
            files = {
                "file": (
                    "doc.docx",
                    DOCX_BYTES,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            }
            r = client.post("/api/ingest", files=files)
            assert r.status_code == 413
