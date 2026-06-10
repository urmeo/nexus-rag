"""Document ingestion module for parsing, chunking, and embedding."""

from nexusrag.ingestion.chunker import (
    Chunk,
    FixedSizeChunker,
    HierarchicalChunker,
    SemanticChunker,  # Alias for backward compatibility
    get_chunker,
)
from nexusrag.ingestion.embedder import Embedder
from nexusrag.ingestion.parser import DocumentParser, ParsedDocument, Section

__all__ = [
    "Chunk",
    "DocumentParser",
    "Embedder",
    "FixedSizeChunker",
    "HierarchicalChunker",
    "ParsedDocument",
    "SemanticChunker",
    "Section",
    "get_chunker",
]
