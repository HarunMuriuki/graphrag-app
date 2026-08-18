"""
Text extraction for uploaded documents.

Explicitly supported formats: PDF, DOCX, TXT, Markdown, plus common plain-text
formats (CSV, JSON, HTML, code files, ...) which are read directly as text.
For anything else, we fall back to a best-effort UTF-8 decode; if that fails
we raise a clear error rather than silently ingesting binary garbage. This
covers "any file that actually contains text" without pretending to support
things like images or proprietary binary formats without OCR/parsing support.
"""
from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument

# Extensions we know are plain text and can just be read directly.
PLAIN_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".html", ".htm",
    ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".java", ".c",
    ".cpp", ".rst",
}


class UnsupportedFileError(Exception):
    pass


def extract_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    if suffix in PLAIN_TEXT_EXTENSIONS:
        return _extract_plain_text(file_path)

    # Fallback: try to decode as text anyway. Many "unknown extension" files
    # (e.g. .cfg, .env, weird export formats) are plain text in practice.
    try:
        return _extract_plain_text(file_path)
    except UnicodeDecodeError as exc:
        raise UnsupportedFileError(
            f"Could not extract text from '{file_path}': binary or unsupported "
            f"format ({suffix or 'no extension'}). Supported: PDF, DOCX, TXT, "
            f"Markdown, and other plain-text formats."
        ) from exc


def _extract_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(pages)


def _extract_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    parts = [p.text for p in doc.paragraphs if p.text]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _extract_plain_text(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", raw, 0, 1, "no supported text encoding found")
