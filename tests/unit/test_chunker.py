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
from nexusrag.ingestion.chunker import HierarchicalChunker


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
        content = " ".join(f"w{i}" for i in range(60))
        doc = ParsedDocument(id="test", content=content, metadata={}, sections=[])

        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5, length_function="words")
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2
        # The last 5 words of one chunk reappear as the first 5 of the next.
        assert chunks[0].content.split()[-5:] == chunks[1].content.split()[:5]

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


class TestPackParagraphs:
    """Branch-level tests for the shared paragraph packer."""

    def _chunker(self, **overrides) -> HierarchicalChunker:
        params = {
            "min_chunk_size": 20,
            "target_chunk_size": 80,
            "max_chunk_size": 120,
            "overlap_size": 60,
            "include_context": False,
        }
        params.update(overrides)
        return HierarchicalChunker(**params)

    def test_small_paragraphs_pack_into_one_chunk(self):
        packer = self._chunker(min_chunk_size=5)
        assert packer._pack_paragraphs(["one two", "three four"], "") == ["one two\n\nthree four"]

    def test_empty_paragraphs_skipped(self):
        packer = self._chunker(min_chunk_size=5)
        assert packer._pack_paragraphs(["", "  ", "real content here"], "") == ["real content here"]

    def test_header_prefix_on_every_emitted_chunk(self):
        paras = ["a" * 50, "b" * 50, "c" * 50]
        texts = self._chunker(overlap_size=10)._pack_paragraphs(paras, "[Methods]\n\n")
        assert len(texts) > 1
        assert all(t.startswith("[Methods]\n\n") for t in texts)

    def test_target_overflow_starts_new_chunk_with_overlap(self):
        paras = ["a" * 50, "b" * 50, "c" * 50]
        texts = self._chunker()._pack_paragraphs(paras, "")
        # overlap_size=60 fits one 50-char paragraph of tail context
        assert texts[0] == "a" * 50
        assert texts[1].startswith("a" * 50)  # carried overlap
        assert "b" * 50 in texts[1]

    def test_oversized_paragraph_split_by_sentences(self):
        big = " ".join(f"Sentence number {i} is here." for i in range(20))
        assert len(big) > 120
        texts = self._chunker()._pack_paragraphs([big], "")
        assert len(texts) > 1
        assert all(len(t) <= 120 for t in texts)
        assert all(len(t) >= 20 for t in texts)

    def test_boundaryless_oversized_paragraph_bounded_by_max(self):
        # No sentence boundaries at all: must still be hard-wrapped under max.
        giant = "word " * 200
        texts = self._chunker()._pack_paragraphs([giant.strip()], "")
        assert texts and all(len(t) <= 120 for t in texts)

    def test_short_tail_merges_into_same_batch_chunk(self):
        # A trailing under-min paragraph merges into the previous chunk of THIS
        # batch — never returned to a caller to attach across sections.
        texts = self._chunker()._pack_paragraphs(["a" * 100, "zz"], "")
        assert len(texts) == 1
        assert texts[0].startswith("a" * 100)
        assert texts[0].endswith("zz")

    def test_sole_short_content_still_emitted(self):
        # A single under-min paragraph with nothing to merge into is kept, not
        # dropped, so a short document is never lost.
        assert self._chunker()._pack_paragraphs(["tiny note"], "") == ["tiny note"]

    def test_oversized_paragraph_flushes_pending_content_first(self):
        big = " ".join(f"Sentence number {i} is here." for i in range(20))
        texts = self._chunker()._pack_paragraphs(["preamble text first", big], "")
        assert texts[0] == "preamble text first"


class TestChunkerNoDataLoss:
    def test_short_document_yields_a_chunk(self):
        doc = ParsedDocument(id="d", content="A short valid note.", metadata={})
        doc.sections = [Section(title="", content="A short valid note.", level=0)]
        chunks = SemanticChunker(include_context=False).chunk(doc)
        assert len(chunks) == 1
        assert "short valid note" in chunks[0].content

    def test_under_min_section_keeps_its_own_metadata(self):
        doc = ParsedDocument(id="d", content="x", metadata={})
        doc.sections = [
            Section(title="Alpha", content="alpha " * 300, level=0, page_number=1),
            Section(title="Beta", content="short beta note.", level=0, page_number=2),
        ]
        chunks = SemanticChunker(include_context=False).chunk(doc)
        beta = [c for c in chunks if c.section_title == "Beta"]
        assert beta and beta[0].page_number == 2
        # Beta's text is never attributed to Alpha.
        assert not any(c.section_title == "Alpha" and "beta" in c.content.lower() for c in chunks)
