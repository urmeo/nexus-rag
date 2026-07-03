"""Document ingestion module for parsing, chunking, and embedding."""

from nexusrag.ingestion.chunker import (
    Chunk,
    FixedSizeChunker,
    HierarchicalChunker,
    SemanticChunker,  # pipeline-facing name for HierarchicalChunker
    get_chunker,
)
from nexusrag.ingestion.embedder import Embedder
from nexusrag.ingestion.parser import (
    DocumentParseError,
    DocumentParser,
    ParsedDocument,
    Section,
)

__all__ = [
    "Chunk",
    "DocumentParseError",
    "DocumentParser",
    "Embedder",
    "FixedSizeChunker",
    "HierarchicalChunker",
    "ParsedDocument",
    "SemanticChunker",
    "Section",
    "get_chunker",
]
