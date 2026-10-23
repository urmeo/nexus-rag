"""API routes."""

import asyncio
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import PurePosixPath
from typing import Annotated, Self

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from scinexusrag.api.dependencies import get_scinexusrag
from scinexusrag.api.metrics import get_metrics_collector
from scinexusrag.api.security import (
    limiter,
    query_limit,
    require_api_key,
    require_same_site,
    upload_limit,
    validate_upload,
)
from scinexusrag.generation.synthesizer import Source
from scinexusrag.ingestion.embedder import EmbeddingConfigError
from scinexusrag.settings import get_settings
from scinexusrag.storage.document_store import INDEX_STEM, MAX_ID_LENGTH, SAFE_ID_PATTERN
from scinexusrag.storage.vector_store import VectorStoreError
from scinexusrag.utils.filenames import resolve_display_name
from scinexusrag.version import __version__

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

MAX_FILENAME_LENGTH = 255
_SOURCE_PREVIEW_CHARS = 500


def _basename(filename: str) -> str:
    return filename.replace("\\", "/").rsplit("/", 1)[-1]


@contextmanager
def _api_errors(detail: str, *, keep_http: bool = False) -> Iterator[None]:
    try:
        yield
    except (EmbeddingConfigError, VectorStoreError):
        raise
    except Exception as error:
        if keep_http and isinstance(error, HTTPException):
            raise
        logger.exception(detail)
        raise HTTPException(status_code=500, detail=detail) from None


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        max_length = get_settings().retrieval.max_query_length
        if len(question) > max_length:
            raise ValueError(f"Question too long (max {max_length} characters)")
        return question


class SourceResponse(BaseModel):
    index: int
    content: str
    text: str
    truncated: bool
    filename: str
    document_id: str
    section_title: str
    page: int | None
    page_number: int | None
    score: float

    @classmethod
    def from_source(cls, source: Source, document_name: str = "") -> Self:
        filename = _basename(document_name) or _basename(source.document_name) or "Unknown"
        truncated = len(source.content) > _SOURCE_PREVIEW_CHARS
        content = source.content[:_SOURCE_PREVIEW_CHARS] + ("..." if truncated else "")
        return cls(
            index=source.index,
            content=content,
            text=content,
            truncated=truncated,
            filename=filename,
            document_id=source.document_id,
            section_title=source.section_title or "",
            page=source.page_number,
            page_number=source.page_number,
            score=round(source.score, 3),
        )


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[SourceResponse]
    processing_time_ms: float
    warnings: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    chunk_count: int
    word_count: int
    error: str | None = None


class _UploadRequest(BaseModel):
    file: UploadFile = Field(
        description="Document to upload", json_schema_extra={"format": "binary"}
    )


class StatusResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_words: int
    llm_available: bool
    llm_model: str
    embedding_model: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    original_filename: str = ""
    word_count: int
    chunk_count: int = 0
    file_type: str = ""
    uploaded_at: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int = 0
    total_chunks: int = 0


class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    llm_model: str = ""
    embedding_model: str = ""
    total_documents: int = 0
    total_chunks: int = 0


class DeleteResponse(BaseModel):
    success: bool
    message: str


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_queries: int
    total_ingestions: int
    total_deletions: int
    avg_query_time_ms: float
    total_documents: int
    total_chunks: int
    version: str


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def health_check(response: Response) -> HealthResponse:
    try:
        rag = get_scinexusrag()
        stats = await asyncio.to_thread(rag.get_stats)
        return HealthResponse(
            status="ok",
            llm_available=stats.llm_available,
            llm_model=stats.llm_model,
            embedding_model=stats.embedding_model,
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
        )
    except (EmbeddingConfigError, VectorStoreError):
        raise
    except Exception:
        logger.exception("Health check failed")
        response.status_code = 503
        return HealthResponse(status="error", llm_available=False)


@router.post(
    "/ingest",
    response_model=UploadResponse,
    dependencies=[Depends(require_same_site)],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"multipart/form-data": {"schema": _UploadRequest.model_json_schema()}},
        }
    },
)
@limiter.limit(upload_limit)
async def ingest_document(request: Request) -> UploadResponse:
    try:
        form = await request.form(max_files=1, max_fields=1, max_part_size=64 * 1024)
    except StarletteHTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="There was an error parsing the body") from None
    try:
        try:
            payload = _UploadRequest.model_validate(dict(form))
        except ValidationError as error:
            raise RequestValidationError(
                [
                    {**item, "loc": ("body", *item["loc"])}
                    for item in error.errors(include_input=False, include_context=False)
                ]
            ) from None
        return await _ingest_upload(payload.file)
    finally:
        await form.close()


async def _ingest_upload(file: UploadFile) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if len(file.filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Filename too long")

    filename = _basename(file.filename)
    if not filename or any(
        ord(character) < 32 or ord(character) == 127 for character in file.filename
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")

    extension = PurePosixPath(filename).suffix.lower()
    if extension not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension or 'no extension'}. Supported: PDF, DOCX, TXT, MD",
        )

    max_mb = get_settings().api.max_upload_mb
    max_bytes = max_mb * 1024 * 1024
    with _api_errors("Failed to process document", keep_http=True):
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {max_mb}MB")

        if not content:
            raise HTTPException(status_code=400, detail="File is empty")

        await asyncio.to_thread(validate_upload, extension, content, file.content_type)

        rag = get_scinexusrag()
        result = await asyncio.to_thread(rag.ingest_bytes, content, filename, extension)

        if result.success:
            get_metrics_collector().record_ingest()

        return UploadResponse.model_validate(result, from_attributes=True)


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_same_site)],
)
@limiter.limit(query_limit)
async def query_documents(request: Request, payload: QueryRequest) -> QueryResponse:
    with _api_errors("Failed to process query", keep_http=True):
        rag = get_scinexusrag()
        started = time.monotonic()
        response = await asyncio.to_thread(rag.query, payload.question)
        elapsed_ms = (time.monotonic() - started) * 1000
        get_metrics_collector().record_query(elapsed_ms)

        doc_names: dict[str, str] = {}
        try:
            docs = await asyncio.to_thread(rag.list_documents)
            for doc in docs:
                doc_names[doc.get("id", "")] = resolve_display_name(doc, fallback="")
        except (EmbeddingConfigError, VectorStoreError):
            raise
        except Exception:
            logger.debug("Could not load source filenames", exc_info=True)

        return QueryResponse(
            answer=response.answer,
            confidence=response.confidence,
            sources=[
                SourceResponse.from_source(source, doc_names.get(source.document_id, ""))
                for source in response.sources
            ],
            processing_time_ms=round(response.processing_time_ms, 1),
            warnings=response.warnings,
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    with _api_errors("Failed to retrieve documents"):
        rag = get_scinexusrag()
        inventory = await asyncio.to_thread(rag.get_document_inventory)

        doc_list = []
        for doc in inventory.documents[offset : offset + limit]:
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
            total_documents=inventory.total_documents,
            total_chunks=inventory.total_chunks,
        )


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    with _api_errors("Failed to retrieve status"):
        rag = get_scinexusrag()
        stats = await asyncio.to_thread(rag.get_stats)

        return StatusResponse.model_validate(stats, from_attributes=True)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_same_site)],
)
async def delete_document(document_id: str) -> DeleteResponse:
    if (
        not document_id
        or len(document_id) > MAX_ID_LENGTH
        or not SAFE_ID_PATTERN.fullmatch(document_id)
        or document_id.casefold() == INDEX_STEM
    ):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    with _api_errors("Failed to delete document", keep_http=True):
        rag = get_scinexusrag()
        success = await asyncio.to_thread(rag.delete_document, document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        get_metrics_collector().record_delete()
        return DeleteResponse(success=True, message="Document deleted")


@router.delete(
    "/documents",
    response_model=DeleteResponse,
    dependencies=[Depends(require_same_site)],
)
async def clear_all_documents() -> DeleteResponse:
    with _api_errors("Failed to clear documents"):
        rag = get_scinexusrag()
        await asyncio.to_thread(rag.clear_all)

        return DeleteResponse(success=True, message="All documents cleared")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    with _api_errors("Failed to retrieve metrics"):
        collector = get_metrics_collector()
        stats_data = collector.snapshot()

        rag = get_scinexusrag()
        pipeline_stats = await asyncio.to_thread(rag.get_stats)

        return MetricsResponse(
            uptime_seconds=round(stats_data["uptime_seconds"], 1),
            total_queries=stats_data["total_queries"],
            total_ingestions=stats_data["total_ingestions"],
            total_deletions=stats_data["total_deletions"],
            avg_query_time_ms=round(stats_data["avg_query_time_ms"], 1),
            total_documents=pipeline_stats.total_documents,
            total_chunks=pipeline_stats.total_chunks,
            version=__version__,
        )
