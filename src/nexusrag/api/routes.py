"""FastAPI routes for NexusRAG API."""

import asyncio
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, field_validator

from nexusrag.api.metrics import get_metrics_collector
from nexusrag.pipeline import get_nexusrag
from nexusrag.utils.filenames import resolve_display_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Security limits
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_QUERY_LENGTH = 2000
MAX_FILENAME_LENGTH = 255


class QueryRequest(BaseModel):
    """Query request body."""

    question: str  # Frontend sends 'question' not 'query'

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question length and content."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Question too long (max {MAX_QUERY_LENGTH} characters)")
        return v


class QueryResponse(BaseModel):
    """Query response body."""

    answer: str
    confidence: float
    sources: list[dict[str, Any]]
    processing_time_ms: float


class UploadResponse(BaseModel):
    """Upload response body."""

    success: bool
    document_id: str
    filename: str
    chunk_count: int
    word_count: int
    error: str | None = None


class StatusResponse(BaseModel):
    """System status response."""

    total_documents: int
    total_chunks: int
    total_words: int
    llm_available: bool
    llm_model: str
    embedding_model: str


class DocumentInfo(BaseModel):
    """Document metadata."""

    id: str
    filename: str
    original_filename: str = ""
    word_count: int
    chunk_count: int = 0
    file_type: str = ""
    uploaded_at: str = ""


class DocumentListResponse(BaseModel):
    """Document list response."""

    documents: list[DocumentInfo]
    total_documents: int = 0
    total_chunks: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    llm_available: bool
    llm_model: str = ""
    embedding_model: str = ""
    total_documents: int = 0
    total_chunks: int = 0


class DeleteResponse(BaseModel):
    """Delete response body."""

    success: bool
    message: str


class MetricsResponse(BaseModel):
    """Operational metrics response."""

    uptime_seconds: float
    total_queries: int
    total_ingestions: int
    total_deletions: int
    avg_query_time_ms: float
    total_documents: int
    total_chunks: int
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for frontend connection status."""
    try:
        rag = get_nexusrag()
        stats = await asyncio.to_thread(rag.get_stats)
        return HealthResponse(
            status="ok",
            llm_available=stats.llm_available,
            llm_model=stats.llm_model,
            embedding_model=stats.embedding_model,
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
        )
    except Exception:
        return HealthResponse(status="error", llm_available=False)


@router.post("/ingest", response_model=UploadResponse)
async def ingest_document(
    file: Annotated[UploadFile, File(description="Document to upload")],
) -> UploadResponse:
    """
    Upload a document for ingestion.

    Accepts PDF, DOCX, TXT, or MD files. Max size: 50MB.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Validate filename length
    if len(file.filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Filename too long")

    # Sanitize filename - extract just the name, no paths
    filename = file.filename.split("/")[-1].split("\\")[-1]
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = "." + filename.split(".")[-1].lower()
    if ext not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: PDF, DOCX, TXT, MD",
        )

    try:
        # Streaming size check: read in chunks to avoid loading huge files into memory
        chunks: list[bytes] = []
        total_size = 0
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB at a time
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
                )
            chunks.append(chunk)
        content = b"".join(chunks)

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        rag = get_nexusrag()
        result = await asyncio.to_thread(rag.ingest_bytes, content, filename, ext)

        if result.success:
            get_metrics_collector().record_ingest()

        return UploadResponse(
            success=result.success,
            document_id=result.document_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            word_count=result.word_count,
            error=result.error,
        )
    except HTTPException:
        raise
    except Exception:
        # Log the actual error but return generic message
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail="Failed to process document") from None


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the knowledge base.

    Returns an answer with sources and confidence score.
    """
    # Validation is now handled by pydantic field_validator
    try:
        rag = get_nexusrag()
        t0 = time.monotonic()
        response = await asyncio.to_thread(rag.query, request.question)
        elapsed_ms = (time.monotonic() - t0) * 1000
        get_metrics_collector().record_query(elapsed_ms)

        # Build document name lookup (prefer original_filename)
        doc_names: dict[str, str] = {}
        try:
            docs = await asyncio.to_thread(rag.list_documents)
            for doc in docs:
                doc_names[doc.get("id", "")] = resolve_display_name(doc)
        except Exception:
            pass

        # Build clean sources response
        sources = []
        for idx, source in enumerate(response.sources):
            filename = doc_names.get(source.document_id) or source.document_name or "Unknown"
            if "/" in filename:
                filename = filename.split("/")[-1]

            # Truncate content for response
            content = source.content
            if len(content) > 500:
                content = content[:500] + "..."

            sources.append(
                {
                    "index": idx + 1,
                    "content": content,
                    "text": content,
                    "filename": filename,
                    "document_id": source.document_id,
                    "section_title": source.section_title or "",
                    "page": source.page_number,
                    "page_number": source.page_number,
                    "score": round(source.score, 3),
                }
            )

        return QueryResponse(
            answer=response.answer,
            confidence=response.confidence,
            sources=sources,
            processing_time_ms=round(response.processing_time_ms, 1),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail="Failed to process query") from None


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all ingested documents."""
    try:
        rag = get_nexusrag()
        docs = await asyncio.to_thread(rag.list_documents)
        stats = await asyncio.to_thread(rag.get_stats)

        doc_list = []
        for doc in docs:
            filename = resolve_display_name(doc)

            doc_list.append(
                DocumentInfo(
                    id=doc.get("id", ""),
                    filename=filename,
                    original_filename=doc.get("original_filename", filename),
                    word_count=doc.get("word_count", 0),
                    chunk_count=doc.get("chunk_count", 0),
                    file_type=doc.get("file_type", ""),
                    uploaded_at=doc.get("uploaded_at", ""),
                )
            )

        return DocumentListResponse(
            documents=doc_list,
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
        )
    except Exception:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents") from None


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get system status and statistics."""
    try:
        rag = get_nexusrag()
        stats = await asyncio.to_thread(rag.get_stats)

        return StatusResponse(
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
            total_words=stats.total_words,
            llm_available=stats.llm_available,
            llm_model=stats.llm_model,
            embedding_model=stats.embedding_model,
        )
    except Exception:
        logger.exception("Failed to get status")
        raise HTTPException(status_code=500, detail="Failed to retrieve status") from None


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str) -> DeleteResponse:
    """Delete a specific document."""
    # Basic validation of document_id format
    if not document_id or len(document_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        rag = get_nexusrag()
        success = await asyncio.to_thread(rag.delete_document, document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        get_metrics_collector().record_delete()
        return DeleteResponse(success=True, message="Document deleted")
    except HTTPException:
        raise
    except ValueError:
        # Validation errors from document store
        raise HTTPException(status_code=400, detail="Invalid document ID format") from None
    except Exception:
        logger.exception("Failed to delete document")
        raise HTTPException(status_code=500, detail="Failed to delete document") from None


@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents() -> DeleteResponse:
    """Delete all documents and reset the system."""
    try:
        rag = get_nexusrag()
        await asyncio.to_thread(rag.clear_all)

        return DeleteResponse(success=True, message="All documents cleared")
    except Exception:
        logger.exception("Failed to clear documents")
        raise HTTPException(status_code=500, detail="Failed to clear documents") from None


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get operational metrics."""
    try:
        collector = get_metrics_collector()
        stats_data = collector.snapshot()

        # Get pipeline stats for document/chunk counts
        total_documents = 0
        total_chunks = 0
        try:
            rag = get_nexusrag()
            pipeline_stats = await asyncio.to_thread(rag.get_stats)
            total_documents = pipeline_stats.total_documents
            total_chunks = pipeline_stats.total_chunks
        except Exception:
            pass

        return MetricsResponse(
            uptime_seconds=round(stats_data["uptime_seconds"], 1),
            total_queries=stats_data["total_queries"],
            total_ingestions=stats_data["total_ingestions"],
            total_deletions=stats_data["total_deletions"],
            avg_query_time_ms=round(stats_data["avg_query_time_ms"], 1),
            total_documents=total_documents,
            total_chunks=total_chunks,
            version="0.1.0",
        )
    except Exception:
        logger.exception("Failed to get metrics")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics") from None
