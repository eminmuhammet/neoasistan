from __future__ import annotations

import logging
from pathlib import Path

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)

# Claude summarizes what comes back rather than this tool doing any NLP, so
# there is no point extracting more text than fits comfortably in one
# request; this is generous enough for most reports/spreadsheets while
# keeping token cost predictable.
MAX_CHARS = 20_000

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt", ".md", ".csv"}


class UnsupportedDocumentError(Exception):
    """Raised when the file extension isn't one this tool knows how to read."""


class DocumentReadError(Exception):
    """Raised when the file exists and is a supported type, but reading it failed."""


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _read_xlsx(path: Path) -> str:
    import openpyxl

    workbook = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    lines: list[str] = []
    for sheet in workbook.worksheets:
        lines.append(f"--- {sheet.title} ---")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".xlsx": _read_xlsx,
    ".txt": _read_plain_text,
    ".md": _read_plain_text,
    ".csv": _read_plain_text,
}


def extract_text(path: Path) -> str:
    extension = path.suffix.lower()
    reader = _READERS.get(extension)
    if reader is None:
        raise UnsupportedDocumentError(
            f"'{extension}' uzantılı dosyaları okuyamıyorum. "
            f"Desteklenenler: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    try:
        text = reader(path)
    except Exception as exc:
        raise DocumentReadError(f"'{path.name}' okunurken bir hata oluştu.") from exc

    truncated = len(text) > MAX_CHARS
    return text[:MAX_CHARS] + ("\n\n[...metin kısaltıldı...]" if truncated else "")


class ReadDocumentTool(Tool):
    name = "read_document"
    description = (
        "Bir belgenin (PDF, Word .docx, Excel .xlsx, .txt/.md/.csv) metnini "
        "çıkarır ve sana gösterir, böylece içeriğini özetleyebilir ya da "
        "içinde arama yapabilirsin. Yol tam olarak bilinmiyorsa önce "
        "find_file ile dosyayı bul."
    )
    # Same privacy bar as capture_screen: reading a file the user points to
    # and sending its contents to the API is a one-time judgment call the
    # human should confirm, not something that should happen silently.
    risk = RiskLevel.MEDIUM
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Okunacak dosyanın tam yolu."},
        },
        "required": ["path"],
    }

    async def run(self, path: str, **kwargs: object) -> ToolResult:
        import asyncio

        file_path = Path(path).expanduser()
        if not file_path.is_file():
            return ToolResult(success=False, error=f"'{path}' diye bir dosya bulamadım.")

        try:
            text = await asyncio.to_thread(extract_text, file_path)
        except (UnsupportedDocumentError, DocumentReadError) as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(
            success=True,
            data={"path": str(file_path), "name": file_path.name, "text": text},
        )
