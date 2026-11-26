"""Request and response bodies of the HTTP API."""

from typing import Self

from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator

from scinexusrag.generation.synthesizer import Source
from scinexusrag.settings import get_settings
from scinexusrag.utils.filenames import basename

SOURCE_PREVIEW_CHARS = 500


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
    truncated: bool
    filename: str
    document_id: str
    section_title: str
    page_number: int | None
    score: float

    @classmethod
    def from_source(cls, source: Source) -> Self:
        truncated = len(source.content) > SOURCE_PREVIEW_CHARS
        return cls(
            index=source.index,
            content=source.content[:SOURCE_PREVIEW_CHARS] + ("..." if truncated else ""),
            truncated=truncated,
            filename=basename(source.document_name) or "Unknown",
            document_id=source.document_id,
            section_title=source.section_title or "",
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


class UploadRequest(BaseModel):
    file: UploadFile = Field(
        description="Document to upload", json_schema_extra={"format": "binary"}
    )


class DocumentInfo(BaseModel):
    id: str
    filename: str
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
    total_words: int = 0


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
