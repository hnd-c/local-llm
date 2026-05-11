"""Plain-text format loaders: TXT, Markdown, CSV, HTML."""

from __future__ import annotations

import csv as _csv
import io
import logging
from pathlib import Path

from docstack.models import DocumentRecord, make_doc_id

logger = logging.getLogger(__name__)


def _make(path: Path, mime: str, text: str, page: int = 1, block_type: str = "text") -> DocumentRecord:
    return DocumentRecord(
        doc_id=make_doc_id(path),
        source_path=str(path.resolve()),
        filename=path.name,
        mime=mime,
        page=page,
        block_type=block_type,
        text=text,
    )


def extract_txt_records(path: Path) -> list[DocumentRecord]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [_make(path, "txt", text)]


def extract_markdown_records(path: Path) -> list[DocumentRecord]:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []
    try:
        import markdown
        from bs4 import BeautifulSoup
        html = markdown.markdown(raw)
        text = BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
    except Exception:
        text = raw
    return [_make(path, "md", text or raw)]


def extract_html_records(path: Path) -> list[DocumentRecord]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(raw, "html.parser").get_text(separator="\n").strip()
    except Exception:
        text = raw.strip()
    if not text:
        return []
    return [_make(path, "html", text)]


def extract_csv_records(path: Path) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        reader = _csv.reader(io.StringIO(raw))
        rows = [row for row in reader if any(c.strip() for c in row)]
        if not rows:
            return []
        block_size = 50
        for i in range(0, len(rows), block_size):
            block = rows[i : i + block_size]
            text = "\n".join("\t".join(cell.strip() for cell in row) for row in block)
            records.append(_make(path, "csv", text, page=i // block_size + 1, block_type="table"))
    except Exception as e:
        logger.warning("CSV parse error %s: %s", path.name, e)
    return records
