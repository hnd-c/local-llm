"""Shared document models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "filename": self.filename,
            "mime": self.mime,
            "page": self.page,
            "block_type": self.block_type,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox else None,
            "ocr_confidence": self.ocr_confidence,
            "section_heading": self.section_heading,
            "table_id": self.table_id,
        }


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
