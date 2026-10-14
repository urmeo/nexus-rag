"""Document parsing for PDF, DOCX, TXT, and Markdown files."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


class DocumentParseError(ValueError):
    "Document parsing failed."


@dataclass
class Section:
    """A section body with optional source offsets; end_offset is exclusive."""

    title: str
    content: str
    level: int = 0
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass
class ParsedDocument:
    """Parsed document with extracted content and metadata."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


class DocumentParser:
    """Parse PDF, DOCX, Markdown, and UTF-8 text."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {extension}")

        parser_map = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".txt": self._parse_text,
            ".md": self._parse_markdown,
        }

        content, sections = parser_map[extension](path)
        doc_id = self._generate_id(path, content)

        return ParsedDocument(
            id=doc_id,
            content=content,
            metadata=self._extract_metadata(path),
            sections=sections,
        )

    def parse_bytes(self, data: bytes, filename: str, extension: str) -> ParsedDocument:
        """Parse document from bytes (for file uploads)."""
        import tempfile

        extension = extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"

        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            doc = self.parse(tmp_path)

            doc.id = self._generate_id(Path(filename), doc.content)
            doc.metadata["filename"] = filename
            doc.metadata["original_filename"] = filename
            doc.metadata["file_type"] = extension.lstrip(".")
            doc.metadata["display_name"] = filename
            return doc
        finally:
            tmp_path.unlink()

    def _parse_pdf(self, path: Path) -> tuple[str, list[Section]]:
        """Extract text and sections from PDF."""
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise DocumentParseError(
                "PDF is password-protected; remove the password and upload again"
            )

        sections: list[Section] = []
        full_text_parts: list[str] = []
        page_start = 0

        for page_num, page in enumerate(reader.pages, start=1):
            text = self._clean_text(page.extract_text() or "")
            if text:
                if full_text_parts:
                    page_start += 2
                full_text_parts.append(text)
                page_sections = self._extract_sections_from_text(text, page_num)
                for section in page_sections:
                    if section.start_offset is not None and section.end_offset is not None:
                        section.start_offset += page_start
                        section.end_offset += page_start
                sections.extend(page_sections)
                page_start += len(text)

        full_text = "\n\n".join(full_text_parts)
        if not full_text.strip() and self._has_images(reader):
            raise DocumentParseError(
                "PDF contains no extractable text but has images — it is likely "
                "a scanned document; OCR is not supported"
            )

        if not sections and full_text.strip():
            sections = [
                Section(
                    title="Content", content=full_text, start_offset=0, end_offset=len(full_text)
                )
            ]

        return full_text, sections

    def _parse_docx(self, path: Path) -> tuple[str, list[Section]]:
        """Extract text and sections from DOCX."""
        doc = DocxDocument(str(path))
        full_text_parts: list[str] = []
        headings: list[tuple[int, int, str, int]] = []
        position = 0

        for block in doc.iter_inner_content():
            text = (
                self._clean_text(block.text)
                if isinstance(block, Paragraph)
                else self._docx_table_text(block)
            )
            if not text:
                continue

            if full_text_parts:
                position += 2
            start = position
            full_text_parts.append(text)
            position += len(text)

            if (
                isinstance(block, Paragraph)
                and block.style
                and block.style.name.startswith("Heading")
            ):
                level = self._get_heading_level(block.style.name)
                headings.append((start, position, text, level))

        full_text = "\n\n".join(full_text_parts)
        return full_text, self._sections_from_headings(full_text, headings)

    def _docx_table_text(self, table: Table) -> str:
        """Keep row order and empty columns without repeating merged-cell text."""
        rows: list[str] = []
        seen_cells: set[object] = set()
        for row in table.rows:
            cells = [""] * row.grid_cols_before
            for cell in row.cells:
                if cell._tc in seen_cells:
                    cells.append("")
                    continue
                seen_cells.add(cell._tc)
                parts: list[str] = []
                for block in cell.iter_inner_content():
                    if isinstance(block, Paragraph):
                        text = self._clean_text(block.text)
                    else:
                        nested = self._docx_table_text(block)
                        text = "[" + nested.replace("\n", "; ") + "]" if nested else ""
                    if text:
                        parts.append(text)
                cells.append("; ".join(parts))
            cells.extend([""] * row.grid_cols_after)
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    @staticmethod
    def _read_text(path: Path) -> str:
        "Read UTF-8 text; reject invalid encoding."
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(
                "File is not valid UTF-8 text; re-save it as UTF-8 and upload again"
            ) from exc

    def _parse_text(self, path: Path) -> tuple[str, list[Section]]:
        content = self._clean_text(self._read_text(path))
        sections = self._extract_sections_from_text(content)
        return content, sections

    def _parse_markdown(self, path: Path) -> tuple[str, list[Section]]:
        """Parse Markdown file, extracting headers as sections."""
        content = self._clean_text(self._read_text(path))
        headings = [
            (match.start(), match.end(), match.group(2).strip(), len(match.group(1)))
            for match in re.finditer(r"^(#{1,6})[ \t]+(.+)$", content, re.MULTILINE)
        ]
        return content, self._sections_from_headings(content, headings)

    def _extract_sections_from_text(
        self, text: str, page_number: int | None = None
    ) -> list[Section]:
        """Heuristic section detection from plain text."""
        headings: list[tuple[int, int, str, int]] = []
        for line in re.finditer(r"^[^\n]+", text, re.MULTILINE):
            stripped = line.group().strip()

            is_header = (
                (stripped.isupper() and len(stripped) < 100)
                or re.match(r"^\d+\.\s+[A-Z]", stripped)
                or re.match(r"^[IVX]+\.\s+", stripped)
            )
            if is_header:
                headings.append((line.start(), line.end(), stripped, 1))
        return self._sections_from_headings(text, headings, page_number)

    @staticmethod
    def _sections_from_headings(
        text: str,
        headings: list[tuple[int, int, str, int]],
        page_number: int | None = None,
    ) -> list[Section]:
        """Slice section bodies from normalized text without relocating repeated passages."""
        starts = [(0, "", 0)] + [(end, title, level) for _, end, title, level in headings]
        ends = [start for start, _, _, _ in headings] + [len(text)]
        sections = []
        for (start, title, level), end in zip(starts, ends, strict=True):
            body = text[start:end]
            content = body.strip()
            if not content and not title:
                continue
            start += len(body) - len(body.lstrip())
            sections.append(
                Section(
                    title=title,
                    content=content,
                    level=level,
                    page_number=page_number,
                    start_offset=start,
                    end_offset=start + len(content),
                )
            )
        return sections

    @staticmethod
    def _has_images(reader: PdfReader) -> bool:
        """True when any page carries an image (best effort)."""
        try:
            return any(page.images for page in reader.pages)
        except Exception:
            return False

    def _get_heading_level(self, style_name: str) -> int:
        """Extract heading level from DOCX style name."""
        match = re.search(r"(\d+)", style_name)
        return int(match.group(1)) if match else 1

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _generate_id(self, path: Path, content: str) -> str:
        "Hash the name and normalized content."
        normalized = re.sub(r"\s+", " ", content).strip()
        hash_input = f"{path.name}:{normalized}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _extract_metadata(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        }
