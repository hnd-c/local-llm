"""Extract text and tables from PDFs via PyMuPDF and optional pdfplumber."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF

from docstack.models import DocumentRecord

logger = logging.getLogger(__name__)


def _doc_id(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha256(f"{path.resolve()}:{st.st_mtime_ns}".encode()).hexdigest()
    return h[:16]


def _avg_chars_per_page(doc: fitz.Document) -> float:
    if doc.page_count == 0:
        return 0.0
    total = 0
    for i in range(doc.page_count):
        total += len(doc.load_page(i).get_text("text").strip())
    return total / doc.page_count


def extract_pdf_records(
    pdf_path: Path,
    min_chars_per_page: int,
    *,
    ocr_pdf_path: Path | None = None,
) -> list[DocumentRecord]:
    """Extract DocumentRecords from a PDF (optionally already OCR'd file)."""
    path = ocr_pdf_path or pdf_path
    doc_id = _doc_id(pdf_path)
    filename = pdf_path.name
    records: list[DocumentRecord] = []
    doc = fitz.open(path)
    try:
        plumber_doc = None
        try:
            import pdfplumber

            plumber_doc = pdfplumber.open(path)
        except Exception as e:  # noqa: BLE001
            logger.debug("pdfplumber open skipped: %s", e)

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = page.get_text("text").strip()
            if not text:
                continue
            records.append(
                DocumentRecord(
                    doc_id=doc_id,
                    source_path=str(pdf_path.resolve()),
                    filename=filename,
                    mime="pdf",
                    page=page_idx + 1,
                    block_type="text",
                    text=text,
                    bbox=None,
                    ocr_confidence=None,
                    section_heading=None,
                )
            )
            if plumber_doc is not None and page_idx < len(plumber_doc.pages):
                try:
                    tables = plumber_doc.pages[page_idx].extract_tables()
                    for ti, table in enumerate(tables or []):
                        if not table:
                            continue
                        lines = []
                        for row in table:
                            lines.append("\t".join("" if c is None else str(c) for c in row))
                        ttext = "\n".join(lines)
                        if ttext.strip():
                            records.append(
                                DocumentRecord(
                                    doc_id=doc_id,
                                    source_path=str(pdf_path.resolve()),
                                    filename=filename,
                                    mime="pdf",
                                    page=page_idx + 1,
                                    block_type="table",
                                    text=ttext,
                                    bbox=None,
                                    ocr_confidence=None,
                                    section_heading=None,
                                    table_id=f"p{page_idx}_t{ti}",
                                )
                            )
                except Exception as e:  # noqa: BLE001
                    logger.debug("table extract page %s: %s", page_idx, e)
    finally:
        doc.close()
        if plumber_doc is not None:
            try:
                plumber_doc.close()
            except Exception:  # noqa: BLE001
                pass
    return records


def needs_ocr(pdf_path: Path, min_chars_per_page: int) -> bool:
    doc = fitz.open(pdf_path)
    try:
        avg = _avg_chars_per_page(doc)
        return avg < float(min_chars_per_page)
    finally:
        doc.close()
