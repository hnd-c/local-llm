"""Shared document models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


def make_doc_id(path: Path) -> str:
    """Stable 16-hex ID derived from file path + mtime (ns)."""
    st = path.stat()
    return hashlib.sha256(f"{path.resolve()}:{st.st_mtime_ns}".encode()).hexdigest()[:16]


@dataclass
class DocumentRecord:
    doc_id: str
    source_path: str
    filename: str
    mime: str
    page: int
    block_type: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    ocr_confidence: float | None = None
    section_heading: str | None = None
    table_id: str | None = None


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str
    page: int
    chunk_index: int
    source_filename: str
    section_heading: str | None = None
    table_id: str | None = None
