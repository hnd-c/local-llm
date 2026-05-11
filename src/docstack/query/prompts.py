"""RAG prompt templates."""

from __future__ import annotations

from typing import Any


def build_rag_system_prompt(chunks: list[dict[str, Any]], max_chars: int) -> str:
    lines = [
        "You are a precise assistant. Answer ONLY using the EXCERPTS below.",
        "If the answer is not in the excerpts, say you cannot find it in the uploaded document.",
        "At the end, add a short line: Sources: (list chunk_id and page for each excerpt you used).",
        "",
        "--- EXCERPTS ---",
    ]
    used = 0
    for i, ch in enumerate(chunks):
        meta = ch.get("metadata") or {}
        header = f"[chunk_id={ch.get('chunk_id')} page={meta.get('page', '?')} file={meta.get('source_filename', '')}]"
        block = f"{header}\n{ch.get('text', '')}\n"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)
