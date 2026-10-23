"""Document workflow."""

import gc
import json
import logging
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
from numpy.typing import NDArray

from scinexusrag.generation.llm import LLMClient
from scinexusrag.generation.orchestrator import Orchestrator, RAGResponse, ReasoningStep
from scinexusrag.ingestion.chunker import Chunk, SemanticChunker
from scinexusrag.ingestion.embedder import Embedder, EmbeddingConfigError
from scinexusrag.ingestion.parser import DocumentParseError, DocumentParser, ParsedDocument
from scinexusrag.retrieval.corrective import CorrectiveRetriever
from scinexusrag.retrieval.dense import DenseRetriever
from scinexusrag.retrieval.hybrid import AdaptiveHybridRetriever
from scinexusrag.retrieval.sparse import BM25Retriever
from scinexusrag.settings import Settings, get_settings
from scinexusrag.storage.document_store import DocumentExistsError, DocumentStore
from scinexusrag.storage.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: str
    filename: str
    chunk_count: int
    word_count: int
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class DocumentInventory:
    documents: list[dict[str, Any]]
    total_chunks: int

    @property
    def total_documents(self) -> int:
        return len(self.documents)


@dataclass
class SystemStats:
    total_documents: int
    total_chunks: int
    total_words: int
    storage_path: str
    llm_model: str
    embedding_model: str
    llm_available: bool


class NexusRAG:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        self._write_lock = threading.RLock()

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
        with self._write_lock:
            if self._embedder is None:
                self._embedder = Embedder(
                    model_name=self.settings.embedding.model,
                    device=self.settings.embedding.device,
                    revision=self.settings.embedding.revision,
                )
            return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        with self._write_lock:
            if self._vector_store is None:
                self._vector_store = VectorStore(
                    path=self.settings.storage.lancedb_path,
                    table_name=self.settings.storage.table_name,
                    embedding_config=self.embedder.embedding_config,
                )
            return self._vector_store

    @property
    def document_store(self) -> DocumentStore:
        if self._document_store is None:
            with self._write_lock:
                if self._document_store is None:
                    self._document_store = DocumentStore(path=self.settings.data_dir / "documents")
        return self._document_store

    @property
    def bm25(self) -> BM25Retriever:
        with self._write_lock:
            if self._bm25 is None:
                self._rebuild_bm25_index()
            assert self._bm25 is not None
            return self._bm25

    @property
    def llm(self) -> LLMClient:
        with self._write_lock:
            if self._llm is None:
                self._llm = LLMClient(
                    model=self.settings.llm.model,
                    base_url=self.settings.llm.base_url,
                    timeout=self.settings.llm.timeout,
                )
            return self._llm

    @property
    def orchestrator(self) -> Orchestrator:
        with self._write_lock:
            if self._orchestrator is None:
                dense = DenseRetriever(self.embedder, self.vector_store)
                hybrid = AdaptiveHybridRetriever(dense, self.bm25, 0.5, 0.5)
                sc = self.settings.self_correction
                retriever = CorrectiveRetriever(
                    hybrid,
                    tau=sc.confidence_tau,
                    feedback_docs=sc.feedback_docs,
                    feedback_terms=sc.feedback_terms,
                    enabled=sc.enabled,
                )

                grounding_verifier = None
                if sc.grounding_enabled:
                    from scinexusrag.generation.grounding import GroundingVerifier

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
        with self._write_lock:
            self.vector_store.validate_embedding_config(self.embedder.embedding_config)
            self.document_store.add(document)
            try:
                self.vector_store.add(chunks, embeddings)
                self.bm25.add_incremental(chunks)
                self.document_store.update_metadata(document.id, "chunk_count", len(chunks))
                self._assert_indexes_synced("persist")
            except Exception:
                try:
                    self.vector_store.delete_by_document(document.id)
                    self._rebuild_bm25_index()
                    self.document_store.delete(document.id)
                except Exception:
                    logger.warning("Ingestion rollback did not finish", exc_info=True)
                    self._recover_sparse_index("failed ingestion rollback")
                raise

    def _assert_indexes_synced(self, where: str) -> None:
        bm25_n, vector_n = self.bm25.count(), self.vector_store.count()
        if bm25_n != vector_n:
            raise RuntimeError(f"index desync after {where}: bm25={bm25_n} vector={vector_n}")

    def _ingest_document(
        self, document: ParsedDocument, filename: str, show_progress: bool
    ) -> IngestResult:
        chunks = self.chunker.chunk(document)
        if not chunks:
            return self._failed_ingest(filename, "No content extracted", document)
        if len(chunks) > self.settings.ingestion.max_chunks:
            return self._failed_ingest(
                filename,
                f"Document exceeds the limit of {self.settings.ingestion.max_chunks:,} passages",
                document,
            )

        with self._write_lock:
            self.vector_store.validate_embedding_config(self.embedder.embedding_config)
            if self.document_store.exists(document.id):
                return self._failed_ingest(filename, "Document already exists", document)

        embeddings = self.embedder.embed(
            [c.content for c in chunks],
            batch_size=self.settings.embedding.batch_size,
            show_progress=show_progress,
        )
        with self._write_lock:
            if self.document_store.exists(document.id):
                return self._failed_ingest(filename, "Document already exists", document)
            try:
                self._persist(document, chunks, embeddings)
            except DocumentExistsError:
                return self._failed_ingest(filename, "Document already exists", document)
        gc.collect()

        return IngestResult(
            document_id=document.id,
            filename=filename,
            chunk_count=len(chunks),
            word_count=document.word_count,
            success=True,
        )

    @staticmethod
    def _failed_ingest(
        filename: str, error: str, document: ParsedDocument | None = None
    ) -> IngestResult:
        return IngestResult(
            document_id=document.id if document is not None else "",
            filename=filename,
            chunk_count=0,
            word_count=document.word_count if document is not None else 0,
            success=False,
            error=error,
        )

    def ingest(self, file_path: str | Path) -> IngestResult:
        path = Path(file_path)
        return self._ingest(lambda: self.parser.parse(path), path.name, show_progress=True)

    def ingest_bytes(self, data: bytes, filename: str, extension: str) -> IngestResult:
        return self._ingest(
            lambda: self.parser.parse_bytes(data, filename, extension),
            filename,
            show_progress=False,
        )

    def _ingest(
        self, parse: Callable[[], ParsedDocument], filename: str, show_progress: bool
    ) -> IngestResult:
        try:
            document = parse()
            return self._ingest_document(document, filename, show_progress=show_progress)
        except DocumentParseError as e:
            return self._failed_ingest(filename, str(e))
        except (EmbeddingConfigError, VectorStoreError):
            raise
        except Exception as e:
            logger.exception("Failed to ingest document: %s", filename)
            return self._failed_ingest(filename, f"Ingestion failed: {type(e).__name__}")

    def ingest_directory(self, dir_path: str | Path, recursive: bool = False) -> list[IngestResult]:
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
                    self._failed_ingest(
                        file_path.name,
                        f"Unsupported file type: {file_path.suffix or 'no extension'}",
                    )
                )

        if skipped:
            logger.warning("Skipped %d unsupported file(s) in %s", skipped, path)
        return results

    def query(self, question: str) -> RAGResponse:
        self.vector_store.validate_embedding_config(self.embedder.embedding_config)
        if self.vector_store.count() == 0:
            return RAGResponse(
                answer="No documents have been uploaded yet. Please upload research papers first.",
                sources=[],
                confidence=0.0,
                reasoning_trace=[ReasoningStep("validation", "no documents in knowledge base")],
            )

        return self.orchestrator.query(question)

    def query_streaming(self, question: str) -> Generator[str, None, None]:
        self.vector_store.validate_embedding_config(self.embedder.embedding_config)
        if self.vector_store.count() == 0:
            error = "No documents have been uploaded yet. Please upload research papers first."
            yield error
            verification = {
                "event": "verification",
                "citations_valid": [],
                "faithfulness": None,
                "warnings": [error],
            }
            yield f"\ndata: {json.dumps(verification)}\n\n"
            yield f"\ndata: {json.dumps({'event': 'verified_answer', 'answer': error})}\n\n"
            return

        yield from self.orchestrator.query_streaming(question)

    def delete_document(self, document_id: str) -> bool:
        with self._write_lock:
            if not self.document_store.exists(document_id):
                return False

            sparse = self.bm25
            chunk_ids = self.vector_store.get_chunk_ids_by_document(document_id)
            try:
                self.vector_store.delete_by_document(document_id)
                if chunk_ids:
                    sparse.remove(chunk_ids)
                self._assert_indexes_synced("delete")
                self.document_store.delete(document_id)
            except Exception:
                self._recover_sparse_index("failed deletion")
                raise
            return True

    def clear_all(self) -> None:
        with self._write_lock:
            if self._vector_store is not None:
                self._vector_store.clear()
            else:
                lancedb.connect(str(self.settings.storage.lancedb_path)).drop_table(
                    self.settings.storage.table_name, ignore_missing=True
                )
            if self._bm25 is not None:
                self._bm25.clear()
            self.document_store.clear()

    def get_stats(self) -> SystemStats:
        with self._write_lock:
            doc_stats = self.document_store.get_stats()
            chunk_count = self.vector_store.count()

        return SystemStats(
            total_documents=doc_stats["document_count"],
            total_chunks=chunk_count,
            total_words=doc_stats["total_words"],
            storage_path=str(self.settings.data_dir),
            llm_model=self.settings.llm.model,
            embedding_model=self.settings.embedding.model,
            llm_available=self.llm.is_available(),
        )

    def list_documents(self) -> list[dict[str, Any]]:
        with self._write_lock:
            return [
                {**meta, "id": doc_id}
                for doc_id, meta in self.document_store.list_with_metadata().items()
            ]

    def get_document_inventory(self) -> DocumentInventory:
        with self._write_lock:
            return DocumentInventory(
                documents=self.list_documents(),
                total_chunks=self.vector_store.count(),
            )

    def _rebuild_bm25_index(self) -> None:
        with self._write_lock:
            chunks = self.vector_store.get_all_chunks()
            sparse = self._bm25 if self._bm25 is not None else BM25Retriever()
            sparse.add(chunks)
            self._bm25 = sparse

    def _recover_sparse_index(self, operation: str) -> None:
        try:
            self._rebuild_bm25_index()
        except Exception:
            self._bm25 = None
            self._orchestrator = None
            logger.exception("Could not rebuild the sparse index after %s", operation)

    def unload_models(self) -> None:
        with self._write_lock:
            if self._embedder is not None:
                self._embedder.unload()
                self._embedder = None

            self._orchestrator = None
            self._vector_store = None
            self._bm25 = None

            if self._llm is not None:
                self._llm.close()
                self._llm = None

        gc.collect()
