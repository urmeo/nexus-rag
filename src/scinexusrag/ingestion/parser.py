"""Document parsing."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import BadZipFile

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.exceptions import InvalidXmlError
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentParseError(ValueError):
    """Parsing failed."""


@dataclass
class Section:
    """Source section."""

    title: str
    content: str
    level: int = 0
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    heading_only: bool = False


@dataclass
class ParsedDocument:
    """Parsed document."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


class DocumentParser:
    """Document parser."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
    MAX_PDF_PAGES = 1000
    MAX_EXTRACTED_CHARS = 2_000_000

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

        try:
            content, sections = parser_map[extension](path)
        except (
            PdfReadError,
            PackageNotFoundError,
            BadZipFile,
            InvalidXmlError,
            XMLSyntaxError,
        ) as exc:
            raise DocumentParseError(
                f"Cannot read {extension[1:].upper()} file; it may be damaged or have the wrong format"
            ) from exc
        if len(content) > self.MAX_EXTRACTED_CHARS:
            raise DocumentParseError(
                f"Document exceeds the extracted text limit of {self.MAX_EXTRACTED_CHARS:,} characters"
            )
        return ParsedDocument(
            id=self._generate_id(path, content),
            content=content,
            metadata=self._extract_metadata(path),
            sections=sections,
        )

    def parse_bytes(self, data: bytes, filename: str, extension: str) -> ParsedDocument:
        """Parse upload."""
        extension = extension.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {extension}")

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory) / f"upload{extension}"
            tmp_path.write_bytes(data)
            doc = self.parse(tmp_path)
            doc.id = self._generate_id(Path(filename), doc.content)
            doc.metadata["filename"] = filename
            doc.metadata["original_filename"] = filename
            doc.metadata["file_type"] = extension.lstrip(".")
            doc.metadata["display_name"] = filename
            return doc

    def _parse_pdf(self, path: Path) -> tuple[str, list[Section]]:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise DocumentParseError(
                "PDF is password-protected; remove the password and upload again"
            )

        pages = reader.pages
        if len(pages) > self.MAX_PDF_PAGES:
            raise DocumentParseError(f"PDF exceeds the page limit of {self.MAX_PDF_PAGES:,}")
        sections: list[Section] = []
        full_text_parts: list[str] = []
        page_start = 0
        extracted_chars = 0

        for page_num, page in enumerate(pages, start=1):
            raw_text = page.extract_text() or ""
            extracted_chars += len(raw_text)
            if extracted_chars > self.MAX_EXTRACTED_CHARS:
                raise DocumentParseError(
                    f"PDF exceeds the extracted text limit of {self.MAX_EXTRACTED_CHARS:,} characters"
                )
            text = self._clean_text(raw_text)
            if text:
                if full_text_parts:
                    page_start += 2
                if page_start + len(text) > self.MAX_EXTRACTED_CHARS:
                    raise DocumentParseError(
                        f"PDF exceeds the extracted text limit of {self.MAX_EXTRACTED_CHARS:,} characters"
                    )
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

        return full_text, sections

    def _parse_docx(self, path: Path) -> tuple[str, list[Section]]:
        try:
            doc = DocxDocument(str(path))
        except (KeyError, ValueError) as exc:
            raise DocumentParseError(
                "Cannot read DOCX file; required Word document content is missing or invalid"
            ) from exc
        if doc.element.tag != qn("w:document") or doc.element.body is None:
            raise DocumentParseError("Cannot read DOCX file; document body is missing or invalid")
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
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(
                "File is not valid UTF-8 text; re-save it as UTF-8 and upload again"
            ) from exc

    def _parse_text(self, path: Path) -> tuple[str, list[Section]]:
        content = self._clean_text(self._read_text(path))
        return content, self._extract_sections_from_text(content)

    def _parse_markdown(self, path: Path) -> tuple[str, list[Section]]:
        source = self._read_text(path)
        content = self._clean_text(source)
        source_lines = (line for line in source.split("\n") if line.strip())
        content_lines = (
            line for line in re.finditer(r"^[^\n]+", content, re.MULTILINE) if line.group().strip()
        )
        headings: list[tuple[int, int, str, int]] = []
        fence = ""
        for source_line, line in zip(source_lines, content_lines, strict=True):
            marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", source_line)
            if fence:
                if (
                    marker
                    and marker.group(1)[0] == fence[0]
                    and len(marker.group(1)) >= len(fence)
                    and not marker.group(2).strip()
                ):
                    fence = ""
                continue
            if marker and (marker.group(1)[0] == "~" or "`" not in marker.group(2)):
                fence = marker.group(1)
                continue
            heading = re.match(r" {0,3}(#{1,6})[ \t]+(.+)$", source_line)
            if heading:
                headings.append(
                    (
                        line.start(),
                        line.end(),
                        self._clean_text(heading.group(2)),
                        len(heading.group(1)),
                    )
                )
        return content, self._sections_from_headings(content, headings)

    def _extract_sections_from_text(
        self, text: str, page_number: int | None = None
    ) -> list[Section]:
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
        starts: list[tuple[int, int | None, str, int]] = [(0, None, "", 0)] + [
            (end, start, title, level) for start, end, title, level in headings
        ]
        ends = [start for start, _, _, _ in headings] + [len(text)]
        sections = []
        for (start, heading_start, title, level), end in zip(starts, ends, strict=True):
            body = text[start:end]
            heading_only = False
            if not body.strip() and heading_start is not None:
                heading_only = True
                body = text[heading_start:start]
                start = heading_start
            content = body.strip()
            if not content:
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
                    heading_only=heading_only,
                )
            )
        return sections

    @staticmethod
    def _has_images(reader: PdfReader) -> bool:
        try:
            return any(page.images for page in reader.pages)
        except Exception:
            return False

    @staticmethod
    def _get_heading_level(style_name: str) -> int:
        match = re.search(r"(\d+)", style_name)
        return int(match.group(1)) if match else 1

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _generate_id(path: Path, content: str) -> str:
        normalized = re.sub(r"\s+", " ", content).strip()
        hash_input = f"{path.name}:{normalized}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @staticmethod
    def _extract_metadata(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        }
