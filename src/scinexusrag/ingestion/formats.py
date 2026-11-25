from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentFormat:
    label: str
    content_types: tuple[str, ...]
    signatures: tuple[bytes, ...] = ()
    is_text: bool = False


_TEXT_CONTENT_TYPES = ("text/plain", "text/markdown")

DOCUMENT_FORMATS: Mapping[str, DocumentFormat] = {
    ".pdf": DocumentFormat("PDF", ("application/pdf",), (b"%PDF",)),
    ".docx": DocumentFormat(
        "DOCX",
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        ),
        (b"PK\x03\x04", b"PK\x05\x06"),
    ),
    ".txt": DocumentFormat("TXT", _TEXT_CONTENT_TYPES, is_text=True),
    ".md": DocumentFormat("MD", _TEXT_CONTENT_TYPES, is_text=True),
}
