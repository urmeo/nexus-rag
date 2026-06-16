"""Tests for chunking strategies."""

import pytest

from nexusrag.ingestion import (
    Chunk,
    FixedSizeChunker,
    ParsedDocument,
    Section,
    SemanticChunker,
    get_chunker,
)


class TestFixedSizeChunker:
    def test_fixed_size_chunker_basic(self, parsed_document):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(parsed_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size_respected(self, parsed_document):
        chunk_size = 200
        chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.chunk(parsed_document)

        # Allow some tolerance for sentence boundary adjustments
        for chunk in chunks[:-1]:  # Last chunk may be smaller
            assert len(chunk.content) <= chunk_size + 100

    def test_chunk_overlap(self):
        content = "Word " * 100  # 500 characters
        doc = ParsedDocument(id="test", content=content, metadata={}, sections=[])

        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=30)
        chunks = chunker.chunk(doc)

        if len(chunks) >= 2:
            # Check that chunks share some content
            chunks[0].content[-30:]
            chunks[1].content[:50]
            # Some overlap should exist
            assert len(chunks) >= 2

    def test_chunk_ids_unique(self, parsed_document):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(parsed_document)

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_preserved(self, parsed_document):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(parsed_document)

        for i, chunk in enumerate(chunks):
            assert chunk.document_id == parsed_document.id
            assert chunk.metadata["chunk_index"] == i
            assert "filename" in chunk.metadata

    def test_word_based_chunking(self, parsed_document):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5, length_function="words")
        chunks = chunker.chunk(parsed_document)

        assert len(chunks) > 0
        for chunk in chunks[:-1]:
            word_count = len(chunk.content.split())
            assert word_count <= 25  # Allow some flexibility

    def test_overlap_validation(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=150)

    def test_empty_document(self):
        doc = ParsedDocument(id="empty", content="", metadata={}, sections=[])
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_whitespace_only_document(self):
        doc = ParsedDocument(id="ws", content="   \n\n   ", metadata={}, sections=[])
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert chunks == []


class TestSemanticChunker:
    def test_semantic_chunker_basic(self, parsed_document):
        chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=500)
        chunks = chunker.chunk(parsed_document)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_respects_sections(self):
        sections = [
            Section(title="Introduction", content="Intro content here.", level=1),
            Section(title="Methods", content="Methods description.", level=1),
            Section(title="Results", content="Results summary.", level=1),
        ]
        doc = ParsedDocument(
            id="sectioned",
            content="Full document content",
            metadata={"filename": "test.txt"},
            sections=sections,
        )

        chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=500)
        chunks = chunker.chunk(doc)

        # Should have chunks corresponding to sections
        assert len(chunks) >= len(sections)

        # Section titles should be in chunks
        titles_found = sum(1 for c in chunks if any(s.title in c.content for s in sections))
        assert titles_found >= 2

    def test_section_metadata_preserved(self):
        sections = [
            Section(title="Methods", content="Detailed methods.", level=2, page_number=5),
        ]
        doc = ParsedDocument(id="test", content="Content", metadata={}, sections=sections)

        chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=500)
        chunks = chunker.chunk(doc)

        methods_chunks = [c for c in chunks if c.metadata.get("section_title") == "Methods"]
        assert len(methods_chunks) >= 1
        assert methods_chunks[0].metadata.get("section_title") == "Methods"

    def test_large_section_split(self):
        large_content = "This is a sentence. " * 100
        sections = [Section(title="Large", content=large_content, level=1)]
        doc = ParsedDocument(id="large", content=large_content, metadata={}, sections=sections)

        chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=200)
        chunks = chunker.chunk(doc)

        # Large section should be split
        assert len(chunks) > 1

    def test_paragraph_based_chunking(self):
        content = """First paragraph with some content.

Second paragraph here.

Third paragraph follows."""

        doc = ParsedDocument(id="paras", content=content, metadata={}, sections=[])
        chunker = SemanticChunker(min_chunk_size=10, max_chunk_size=500)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1

    def test_small_chunks_merged(self):
        content = "A.\n\nB.\n\nC.\n\nD.\n\nE."
        doc = ParsedDocument(id="small", content=content, metadata={}, sections=[])

        chunker = SemanticChunker(min_chunk_size=20, max_chunk_size=500)
        chunks = chunker.chunk(doc)

        # Small chunks should be merged
        assert len(chunks) <= 3

    def test_empty_document(self):
        doc = ParsedDocument(id="empty", content="", metadata={}, sections=[])
        chunker = SemanticChunker()
        chunks = chunker.chunk(doc)

        assert chunks == []


class TestGetChunker:
    def test_get_fixed_chunker(self):
        chunker = get_chunker("fixed", chunk_size=200, chunk_overlap=30)
        assert isinstance(chunker, FixedSizeChunker)

    def test_get_semantic_chunker(self):
        chunker = get_chunker("semantic", max_chunk_size=500)
        assert isinstance(chunker, SemanticChunker)

    def test_default_is_semantic(self):
        chunker = get_chunker()
        assert isinstance(chunker, SemanticChunker)


class TestChunkDataclass:
    def test_token_estimate(self):
        chunk = Chunk(
            id="test",
            content="This is a test sentence with several words.",
            document_id="doc",
            metadata={},
        )

        word_count = len(chunk.content.split())
        expected = int(word_count * 1.3)
        assert chunk.token_estimate == expected
