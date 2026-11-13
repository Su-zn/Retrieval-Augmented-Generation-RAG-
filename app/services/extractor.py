"""Text extraction for uploaded PDF and plain-text documents."""

from __future__ import annotations

import asyncio
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})

_ALLOWED_CONTENT_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf", "application/x-pdf", "application/octet-stream"}),
    ".txt": frozenset({"text/plain", "application/octet-stream"}),
}


class ExtractionError(Exception):
    """Raised when an upload cannot be validated or converted to text."""


class UnsupportedFileTypeError(ExtractionError):
    """Raised when the extension or MIME type is not supported."""


class EmptyDocumentError(ExtractionError):
    """Raised when a file or the extracted text contains no usable content."""


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Normalised text plus the validated upload metadata."""

    text: str
    file_extension: str
    content_type: str


def _normalise_content_type(content_type: str | None) -> str:
    if not content_type:
        return "application/octet-stream"
    return content_type.split(";", maxsplit=1)[0].strip().lower() or "application/octet-stream"


def validate_upload(filename: str | None, content_type: str | None) -> tuple[str, str]:
    """Validate extension and MIME type, returning the normalised pair.

    Raises:
        UnsupportedFileTypeError: if the filename has no supported suffix or the
            declared MIME type does not match that suffix.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or filename or 'unknown'}'. Supported: {supported}."
        )

    normalised = _normalise_content_type(content_type)
    if normalised not in _ALLOWED_CONTENT_TYPES[suffix]:
        allowed = ", ".join(sorted(_ALLOWED_CONTENT_TYPES[suffix]))
        raise UnsupportedFileTypeError(
            f"Content type '{normalised}' does not match '{suffix}'. Allowed: {allowed}."
        )

    return suffix, normalised


def _normalise_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _extract_pdf_text(data: bytes) -> str:
    """Parse a PDF synchronously; executed in a worker thread."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("Password-protected PDFs are not supported.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 - any parser failure is a client error
        logger.warning("PDF parsing failed: %s", exc)
        raise ExtractionError("The PDF could not be parsed. The file may be corrupt.") from exc

    return _normalise_text("\n\n".join(page.strip() for page in pages if page.strip()))


def _extract_txt_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionError("The TXT file is not valid UTF-8.") from exc


async def extract_document(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
) -> ExtractedDocument:
    """Validate and extract text from a PDF or TXT upload.

    PDF parsing is CPU-bound and therefore offloaded with ``asyncio.to_thread``.
    """
    suffix, normalised_type = validate_upload(filename, content_type)

    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")

    if suffix == ".pdf":
        text = await asyncio.to_thread(_extract_pdf_text, data)
    else:
        text = _extract_txt_text(data)

    text = _normalise_text(text)
    if not text:
        raise EmptyDocumentError("No extractable text was found in the uploaded file.")

    return ExtractedDocument(text=text, file_extension=suffix, content_type=normalised_type)


def collapse_internal_whitespace(text: str) -> str:
    """Collapse runs of whitespace so PDF line breaks do not fragment sentences."""
    return re.sub(r"\s+", " ", text).strip()
