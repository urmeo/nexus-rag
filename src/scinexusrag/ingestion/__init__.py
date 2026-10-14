"""Document ingestion module for parsing, chunking, and embedding."""

from scinexusrag.ingestion.chunker import (
    Chunk,
    FixedSizeChunker,
    HierarchicalChunker,
    SemanticChunker,
    get_chunker,
)
from scinexusrag.ingestion.embedder import Embedder, EmbeddingConfig, EmbeddingConfigError
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
    "EmbeddingConfig",
    "EmbeddingConfigError",
    "FixedSizeChunker",
    "HierarchicalChunker",
    "ParsedDocument",
    "SemanticChunker",
    "Section",
    "get_chunker",
]
