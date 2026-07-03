"""Tests for DocumentParser."""

import pytest

from nexusrag.ingestion import DocumentParseError, DocumentParser, ParsedDocument


class TestDocumentParser:
    @pytest.fixture
    def parser(self):
        return DocumentParser()

    def test_parse_txt_file(self, parser, sample_text_file):
        doc = parser.parse(sample_text_file)

        assert isinstance(doc, ParsedDocument)
        assert doc.id is not None
        assert len(doc.id) == 16
        assert "introduction" in doc.content.lower()
        assert "methodology" in doc.content.lower()
        assert "results" in doc.content.lower()
        assert doc.metadata["filename"] == "sample.txt"
        assert doc.metadata["extension"] == ".txt"

    def test_parse_md_file(self, parser, sample_markdown_file):
        doc = parser.parse(sample_markdown_file)

        assert isinstance(doc, ParsedDocument)
        assert "research paper title" in doc.content.lower()
        assert doc.metadata["extension"] == ".md"

        # Should extract markdown headers as sections
        assert len(doc.sections) > 0
        section_titles = [s.title.lower() for s in doc.sections]
        assert any("abstract" in t for t in section_titles)
        assert any("introduction" in t for t in section_titles)

    def test_parse_nonexistent_file(self, parser, temp_dir):
        with pytest.raises(FileNotFoundError):
            parser.parse(temp_dir / "nonexistent.txt")

    def test_parse_unsupported_format(self, parser, temp_dir):
        unsupported = temp_dir / "file.xyz"
        unsupported.write_text("content")

        with pytest.raises(ValueError, match="Unsupported format"):
            parser.parse(unsupported)

    def test_generate_document_id(self, parser, sample_text_file):
        doc1 = parser.parse(sample_text_file)
        doc2 = parser.parse(sample_text_file)

        assert doc1.id == doc2.id
        assert len(doc1.id) == 16
        assert doc1.id.isalnum()

    def test_different_files_different_ids(self, parser, sample_text_file, sample_markdown_file):
        doc1 = parser.parse(sample_text_file)
        doc2 = parser.parse(sample_markdown_file)

        assert doc1.id != doc2.id

    def test_section_extraction_txt(self, parser, sample_text_file):
        doc = parser.parse(sample_text_file)

        # Text files use heuristic section detection (uppercase headers)
        assert len(doc.sections) >= 1

    def test_section_extraction_md(self, parser, sample_markdown_file):
        doc = parser.parse(sample_markdown_file)

        # Markdown uses header-based section extraction
        assert len(doc.sections) >= 4

        # Check section levels
        levels = [s.level for s in doc.sections]
        assert 1 in levels  # h1
        assert 2 in levels  # h2

    def test_section_content_preserved(self, parser, sample_markdown_file):
        doc = parser.parse(sample_markdown_file)

        # Find abstract section
        abstract_sections = [s for s in doc.sections if "abstract" in s.title.lower()]
        assert len(abstract_sections) == 1
        assert "novel findings" in abstract_sections[0].content.lower()

    def test_metadata_extraction(self, parser, sample_text_file):
        doc = parser.parse(sample_text_file)

        assert "filename" in doc.metadata
        assert "extension" in doc.metadata
        assert "size_bytes" in doc.metadata
        assert "created_at" in doc.metadata
        assert "modified_at" in doc.metadata
        assert doc.metadata["size_bytes"] > 0

    def test_word_count_property(self, parser, sample_text_file):
        doc = parser.parse(sample_text_file)

        assert doc.word_count > 0
        assert doc.word_count == len(doc.content.split())

    def test_empty_file(self, parser, empty_file):
        doc = parser.parse(empty_file)

        assert doc.content == ""
        assert doc.word_count == 0

    def test_parse_bytes(self, parser, sample_text_file):
        content = sample_text_file.read_bytes()
        doc = parser.parse_bytes(content, "uploaded.txt", ".txt")

        assert doc.metadata["original_filename"] == "uploaded.txt"
        assert "introduction" in doc.content.lower()

    def test_text_cleaning(self, parser, temp_dir):
        messy_file = temp_dir / "messy.txt"
        messy_file.write_text("Multiple   spaces\n\n\n\nToo many newlines")

        doc = parser.parse(messy_file)

        # Should not have excessive whitespace
        assert "   " not in doc.content
        assert "\n\n\n" not in doc.content


class TestPdfEdgeCases:
    def test_encrypted_pdf_actionable_error(self, temp_dir):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret")
        pdf_path = temp_dir / "locked.pdf"
        with open(pdf_path, "wb") as f:
            writer.write(f)

        with pytest.raises(DocumentParseError, match="password-protected"):
            DocumentParser().parse(pdf_path)

    def test_scanned_pdf_actionable_error(self, temp_dir, monkeypatch):
        class FakePage:
            images = [object()]

            def extract_text(self):
                return ""

        class FakeReader:
            def __init__(self, path):
                self.pages = [FakePage()]
                self.is_encrypted = False

        monkeypatch.setattr("nexusrag.ingestion.parser.PdfReader", FakeReader)
        pdf_path = temp_dir / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with pytest.raises(DocumentParseError, match="scanned"):
            DocumentParser().parse(pdf_path)


class TestDuplicateNormalization:
    def test_whitespace_only_edits_keep_same_id(self, temp_dir):
        parser = DocumentParser()
        doc = temp_dir / "doc.txt"
        doc.write_text("Alpha beta. Gamma delta.")
        original_id = parser.parse(doc).id

        doc.write_text("Alpha beta.\n\nGamma  delta. ")

        assert parser.parse(doc).id == original_id

    def test_content_edits_change_id(self, temp_dir):
        parser = DocumentParser()
        doc = temp_dir / "doc.txt"
        doc.write_text("Alpha beta. Gamma delta.")
        original_id = parser.parse(doc).id

        doc.write_text("Alpha beta. Gamma delta. Epsilon.")

        assert parser.parse(doc).id != original_id
