"""Document ingestion module for parsing, chunking, and embedding."""

from scinexusrag.ingestion.chunker import (
    Chunk,
    FixedSizeChunker,
    HierarchicalChunker,
    SemanticChunker,  # pipeline-facing name for HierarchicalChunker
    get_chunker,
)
from scinexusrag.ingestion.embedder import Embedder
from scinexusrag.ingestion.parser import (
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
