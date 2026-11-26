"""Document parsing."""

import hashlib
import io
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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

from scinexusrag.models import ParsedDocument, Section


class DocumentParseError(ValueError):
    """Parsing failed."""


class DocumentParser:
    """Document parser."""

    MAX_PDF_PAGES = 1000
    MAX_EXTRACTED_CHARS = 2_000_000

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if not path.is_file():
            raise DocumentParseError(f"File not found: {path.name}")
        document_format = _format(path.suffix)
        document = self._read(document_format, path.read_bytes(), path.name)
        stat = path.stat()
        document.metadata["created_at"] = datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat()
        document.metadata["modified_at"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        return document

    def parse_bytes(self, data: bytes, filename: str, extension: str) -> ParsedDocument:
        return self._read(_format(extension), data, filename)

    def _read(
        self, document_format: "DocumentFormat", data: bytes, filename: str
    ) -> ParsedDocument:
        try:
            content, sections = document_format.read(self, data)
        except (
            PdfReadError,
            PackageNotFoundError,
            BadZipFile,
            InvalidXmlError,
            XMLSyntaxError,
        ) as exc:
            raise DocumentParseError(
                f"Cannot read {document_format.label} file; it may be damaged or have the wrong format"
            ) from exc
        if len(content) > self.MAX_EXTRACTED_CHARS:
            raise DocumentParseError(
                f"Document exceeds the extracted text limit of {self.MAX_EXTRACTED_CHARS:,} characters"
            )
        return ParsedDocument(
            id=self._generate_id(Path(filename).name, content),
            content=content,
            metadata={
                "filename": filename,
                "extension": document_format.extension,
                "file_type": document_format.extension[1:],
                "size_bytes": len(data),
            },
            sections=sections,
        )

    def _parse_pdf(self, data: bytes) -> tuple[str, list[Section]]:
        reader = PdfReader(io.BytesIO(data))
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

    def _parse_docx(self, data: bytes) -> tuple[str, list[Section]]:
        try:
            doc = DocxDocument(io.BytesIO(data))
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
    def _read_text(data: bytes) -> str:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(
                "File is not valid UTF-8 text; re-save it as UTF-8 and upload again"
            ) from exc
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _parse_text(self, data: bytes) -> tuple[str, list[Section]]:
        content = self._clean_text(self._read_text(data))
        return content, self._extract_sections_from_text(content)

    def _parse_markdown(self, data: bytes) -> tuple[str, list[Section]]:
        source = self._read_text(data)
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
    def _generate_id(filename: str, content: str) -> str:
        normalized = re.sub(r"\s+", " ", content).strip()
        hash_input = f"{filename}:{normalized}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class DocumentFormat:
    """How a supported file type is recognized and read."""

    extension: str
    label: str
    content_types: tuple[str, ...]
    signatures: tuple[bytes, ...]
    read: Callable[[DocumentParser, bytes], tuple[str, list[Section]]]
    text: bool = False


_TEXT_TYPES = ("text/plain", "text/markdown")
FORMATS: dict[str, DocumentFormat] = {
    document_format.extension: document_format
    for document_format in (
        DocumentFormat(".pdf", "PDF", ("application/pdf",), (b"%PDF",), DocumentParser._parse_pdf),
        DocumentFormat(
            ".docx",
            "DOCX",
            (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/zip",
            ),
            (b"PK\x03\x04", b"PK\x05\x06"),
            DocumentParser._parse_docx,
        ),
        DocumentFormat(".txt", "TXT", _TEXT_TYPES, (), DocumentParser._parse_text, text=True),
        DocumentFormat(".md", "MD", _TEXT_TYPES, (), DocumentParser._parse_markdown, text=True),
    )
}


def _format(extension: str) -> DocumentFormat:
    extension = extension.lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    document_format = FORMATS.get(extension)
    if document_format is None:
        raise DocumentParseError(f"Unsupported format: {extension}")
    return document_format
