"""Main NexusRAG pipeline integrating all components."""

import contextlib
import gc
import logging
import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nexusrag.config import Settings, get_settings
from nexusrag.generation import LLMClient, Orchestrator, RAGResponse, ReasoningStep
from nexusrag.ingestion import (
    Chunk,
    DocumentParseError,
    DocumentParser,
    Embedder,
    ParsedDocument,
    SemanticChunker,
)
from nexusrag.retrieval import (
    AdaptiveHybridRetriever,
    BM25Retriever,
    CorrectiveRetriever,
    DenseRetriever,
)
from nexusrag.storage import DocumentStore, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of document ingestion."""

    document_id: str
    filename: str
    chunk_count: int
    word_count: int
    success: bool
    error: str | None = None


@dataclass
class SystemStats:
    """System statistics."""

    total_documents: int
    total_chunks: int
    total_words: int
    storage_path: str
    llm_model: str
    embedding_model: str
    llm_available: bool


class NexusRAG:
    """Document ingestion, corrective hybrid retrieval, and cited generation."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._initialized = False
        # Serializes all index/store mutations; concurrent API threads
        # otherwise interleave BM25 rebuilds and silently drop documents.
        self._write_lock = threading.RLock()

        # Lazy-loaded components
        self._parser: DocumentParser | None = None
        self._chunker: SemanticChunker | None = None
        self._embedder: Embedder | None = None
        self._vector_store: VectorStore | None = None
        self._document_store: DocumentStore | None = None
        self._bm25: BM25Retriever | None = None
        self._llm: LLMClient | None = None
        self._orchestrator: Orchestrator | None = None

    @property
    def parser(self) -> DocumentParser:
        if self._parser is None:
            self._parser = DocumentParser()
        return self._parser

    @property
    def chunker(self) -> SemanticChunker:
        if self._chunker is None:
            self._chunker = SemanticChunker(
                min_chunk_size=self.settings.ingestion.min_chunk_size,
                target_chunk_size=self.settings.ingestion.chunk_size,
                max_chunk_size=self.settings.ingestion.chunk_size + 300,
                overlap_size=self.settings.ingestion.chunk_overlap,
                include_context=True,
                context_chars=150,
            )
        return self._chunker

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(
                model_name=self.settings.embedding.model,
                device=self.settings.embedding.device,
                revision=self.settings.embedding.revision,
            )
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(
                path=self.settings.storage.lancedb_path,
                table_name=self.settings.storage.table_name,
                embedding_dim=self.embedder.dimension,
            )
        return self._vector_store

    @property
    def document_store(self) -> DocumentStore:
        if self._document_store is None:
            self._document_store = DocumentStore(path=self.settings.data_dir / "documents")
        return self._document_store

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            with self._write_lock:
                if self._bm25 is None:
                    self._bm25 = BM25Retriever()
                    self._rebuild_bm25_index()
        return self._bm25

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(
                model=self.settings.llm.model,
                base_url=self.settings.llm.base_url,
                timeout=self.settings.llm.timeout,
            )
        return self._llm

    @property
    def orchestrator(self) -> Orchestrator:
        if self._orchestrator is None:
            dense = DenseRetriever(self.embedder, self.vector_store)
            hybrid = AdaptiveHybridRetriever(dense, self.bm25, 0.5, 0.5)
            sc = self.settings.self_correction
            retriever = CorrectiveRetriever(
                hybrid,
                tau=sc.confidence_tau,
                feedback_docs=sc.feedback_docs,
                feedback_terms=sc.feedback_terms,
            )

            grounding_verifier = None
            if sc.grounding_enabled:
                from nexusrag.generation.grounding import GroundingVerifier

                grounding_verifier = GroundingVerifier(
                    model_name=sc.grounding_model,
                    threshold=sc.grounding_threshold,
                    device=self.settings.embedding.device,
                )

            self._orchestrator = Orchestrator(
                retriever=retriever,
                llm=self.llm,
                top_k=self.settings.retrieval.top_k,
                document_store=self.document_store,
                grounding_verifier=grounding_verifier,
                max_tokens=self.settings.llm.max_tokens,
                temperature=self.settings.llm.temperature,
            )
        return self._orchestrator

    def _persist(
        self, document: ParsedDocument, chunks: list[Chunk], embeddings: NDArray[np.float32]
    ) -> None:
        """Write document + chunks to all stores; roll back on partial failure."""
        with self._write_lock:
            self.document_store.add(document)
            try:
                self.vector_store.add(chunks, embeddings)
                self.bm25.add_incremental(chunks)
                self.document_store.update_metadata(document.id, "chunk_count", len(chunks))
            except Exception:
                with contextlib.suppress(Exception):
                    self.bm25.remove({c.id for c in chunks})
                with contextlib.suppress(Exception):
                    self.vector_store.delete_by_document(document.id)
                with contextlib.suppress(Exception):
                    self.document_store.delete(document.id)
                raise

    def ingest(self, file_path: str | Path) -> IngestResult:
        path = Path(file_path)

        try:
            document = self.parser.parse(path)
            if self.document_store.exists(document.id):
                return IngestResult(
                    document_id=document.id,
                    filename=path.name,
                    chunk_count=0,
                    word_count=document.word_count,
                    success=False,
                    error="Document already exists",
                )

            chunks = self.chunker.chunk(document)
            if not chunks:
                return IngestResult(
                    document_id=document.id,
                    filename=path.name,
                    chunk_count=0,
                    word_count=document.word_count,
                    success=False,
                    error="No content extracted",
                )

            embeddings = self.embedder.embed(
                [c.content for c in chunks],
                batch_size=self.settings.embedding.batch_size,
            )
            self._persist(document, chunks, embeddings)
            gc.collect()

            return IngestResult(
                document_id=document.id,
                filename=path.name,
                chunk_count=len(chunks),
                word_count=document.word_count,
                success=True,
            )

        except DocumentParseError as e:
            return IngestResult(
                document_id="",
                filename=path.name,
                chunk_count=0,
                word_count=0,
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception(f"Failed to ingest file: {path.name}")
            return IngestResult(
                document_id="",
                filename=path.name,
                chunk_count=0,
                word_count=0,
                success=False,
                error=f"Ingestion failed: {type(e).__name__}",
            )

    def ingest_bytes(self, data: bytes, filename: str, extension: str) -> IngestResult:
        """Ingest document from bytes (for file uploads)."""
        try:
            document = self.parser.parse_bytes(data, filename, extension)

            if self.document_store.exists(document.id):
                return IngestResult(
                    document_id=document.id,
                    filename=filename,
                    chunk_count=0,
                    word_count=document.word_count,
                    success=False,
                    error="Document already exists",
                )

            chunks = self.chunker.chunk(document)
            if not chunks:
                return IngestResult(
                    document_id=document.id,
                    filename=filename,
                    chunk_count=0,
                    word_count=document.word_count,
                    success=False,
                    error="No content extracted",
                )

            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.embed(
                texts,
                batch_size=self.settings.embedding.batch_size,
                show_progress=False,
            )

            self._persist(document, chunks, embeddings)
            gc.collect()

            return IngestResult(
                document_id=document.id,
                filename=filename,
                chunk_count=len(chunks),
                word_count=document.word_count,
                success=True,
            )

        except DocumentParseError as e:
            return IngestResult(
                document_id="",
                filename=filename,
                chunk_count=0,
                word_count=0,
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception(f"Failed to ingest bytes: {filename}")
            return IngestResult(
                document_id="",
                filename=filename,
                chunk_count=0,
                word_count=0,
                success=False,
                error=f"Ingestion failed: {type(e).__name__}",
            )

    def ingest_directory(self, dir_path: str | Path, recursive: bool = False) -> list[IngestResult]:
        """Ingest every supported file; unsupported ones appear as failed results."""
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        pattern = "**/*" if recursive else "*"
        results: list[IngestResult] = []
        skipped = 0

        for file_path in sorted(path.glob(pattern)):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() in self.parser.SUPPORTED_EXTENSIONS:
                results.append(self.ingest(file_path))
            else:
                skipped += 1
                results.append(
                    IngestResult(
                        document_id="",
                        filename=file_path.name,
                        chunk_count=0,
                        word_count=0,
                        success=False,
                        error=f"Unsupported file type: {file_path.suffix or 'no extension'}",
                    )
                )

        if skipped:
            logger.warning("Skipped %d unsupported file(s) in %s", skipped, path)
        return results

    def query(self, question: str) -> RAGResponse:
        if self.vector_store.count() == 0:
            return RAGResponse(
                answer="No documents have been uploaded yet. Please upload research papers first.",
                sources=[],
                confidence=0.0,
                reasoning_trace=[ReasoningStep("validation", "no documents in knowledge base")],
            )

        return self.orchestrator.query(question)

    def query_streaming(self, question: str) -> Generator[str, None, None]:
        """Stream query response token by token."""
        if self.vector_store.count() == 0:
            yield "No documents have been uploaded yet. Please upload research papers first."
            return

        yield from self.orchestrator.query_streaming(question)

    def delete_document(self, document_id: str) -> bool:
        """Remove a document and its chunks."""
        with self._write_lock:
            if not self.document_store.exists(document_id):
                return False

            # Get chunk IDs before deleting from vector store
            doc_chunks = self.vector_store.get_chunks_by_document(document_id)
            chunk_ids = {c.id for c in doc_chunks}

            self.vector_store.delete_by_document(document_id)
            self.document_store.delete(document_id)

            # Incrementally remove from BM25 instead of full rebuild
            if chunk_ids and self._bm25 is not None:
                self._bm25.remove(chunk_ids)

            return True

    def clear_all(self) -> None:
        """Remove all documents and reset the system."""
        with self._write_lock:
            self.vector_store.clear()
            self.document_store.clear()
            self.bm25.clear()

    def get_stats(self) -> SystemStats:
        """Get system statistics."""
        doc_stats = self.document_store.get_stats()

        return SystemStats(
            total_documents=doc_stats["document_count"],
            total_chunks=self.vector_store.count(),
            total_words=doc_stats["total_words"],
            storage_path=str(self.settings.data_dir),
            llm_model=self.settings.llm.model,
            embedding_model=self.settings.embedding.model,
            llm_available=self.llm.is_available(),
        )

    def list_documents(self) -> list[dict[str, Any]]:
        """List all ingested documents."""
        return [
            {"id": doc_id, **meta}
            for doc_id, meta in self.document_store.list_with_metadata().items()
        ]

    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index from vector store."""
        with self._write_lock:
            if self._bm25 is None:
                self._bm25 = BM25Retriever()

            # Get all chunks from vector store
            chunks = self.vector_store.get_all_chunks()

            # Rebuild BM25 index with all chunks
            self._bm25.clear()
            if chunks:
                self._bm25.add(chunks)

    def unload_models(self) -> None:
        """
        Unload ML models to free memory.

        Useful for 8GB RAM systems when switching between tasks.
        Models will be lazy-loaded again when needed.
        """
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None

        if self._orchestrator is not None:
            self._orchestrator = None

        if self._llm is not None:
            self._llm.close()
            self._llm = None

        # Force garbage collection
        gc.collect()


# Singleton instance
_instance: NexusRAG | None = None
_instance_lock = threading.Lock()


def get_nexusrag(settings: Settings | None = None) -> NexusRAG:
    """Get or create NexusRAG singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NexusRAG(settings)
    return _instance
