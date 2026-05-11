"""Character-based chunking with overlap."""

from __future__ import annotations

import uuid

from docstack.config import get_settings
from docstack.models import DocumentRecord, TextChunk


def records_to_chunks(records: list[DocumentRecord]) -> list[TextChunk]:
    settings = get_settings()
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    if not records:
        return []

    doc_id = records[0].doc_id
    filename = records[0].filename
    current_heading: str | None = None
    parts: list[tuple[int, str, str | None, str | None]] = []

    for rec in records:
        if rec.block_type == "heading" and rec.text:
            current_heading = rec.text[:200]
        parts.append((rec.page, rec.text, current_heading, rec.table_id))

    big_parts = []
    for page, text, heading, tid in parts:
        for line in text.split("\n"):
            line = line.strip()
            if line:
                big_parts.append((page, line, heading, tid))

    flat: list[str] = []
    meta_pages: list[int] = []
    meta_heads: list[str | None] = []
    meta_tables: list[str | None] = []
    for page, line, heading, tid in big_parts:
        flat.append(line)
        meta_pages.append(page)
        meta_heads.append(heading)
        meta_tables.append(tid)

    full = "\n".join(flat)
    if not full.strip():
        return []

    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    n = len(full)
    while start < n:
        end = min(start + size, n)
        piece = full[start:end].strip()
        if piece:
            page_guess = meta_pages[0] if meta_pages else 1
            chunk_id = str(uuid.uuid4())[:12]
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=piece,
                    page=page_guess,
                    chunk_index=idx,
                    source_filename=filename,
                    section_heading=meta_heads[0] if meta_heads else None,
                    table_id=meta_tables[0] if meta_tables else None,
                )
            )
            idx += 1
        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks
