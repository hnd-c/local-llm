"""RAG prompt templates."""

from __future__ import annotations

from typing import Any

_FULL_CTX_HEADER = """\
You are a thorough analyst working with the COMPLETE TEXT of an official document (often a Nepali government policy, circular, or action plan).
Every excerpt below is from that document — together they represent the full content.

Rules:
- Answer in Nepali (नेपाली) if the document is in Nepali, unless the user asks for another language.
- Be COMPREHENSIVE and STRUCTURED. Do not give a shallow one-paragraph reply.
- For summaries: write a clear TITLE, then section-by-section key points, obligations, actors, deadlines, and numbers. End with a short conclusion.
- For ideation / drafting: base new content tightly on the document's language, structure, and policy intent.
- For reasoning / comparison: cite specific clauses, page numbers, and chunk IDs.
- Never invent facts not present in the excerpts.
- At the end add: Sources: (chunk_id and page for each excerpt you relied on).
"""

_RAG_HEADER = """\
You are helping with an indexed document (often a Nepali government PDF): summarization, ideation, drafting, and reasoning.
The EXCERPTS below are the primary source of facts — synthesize across ALL of them, not just the first one.
Answer in Nepali (नेपाली) if the document is in Nepali, unless the user asks for another language.
Be DETAILED and STRUCTURED — not a single shallow paragraph. Use sections, bullet points, or numbered lists as appropriate.
Recent user/assistant turns are follow-up on this document only — do not import unrelated topics.
If excerpts are insufficient, say what is missing and suggest a more specific question.
At the end add: Sources: (chunk_id and page for each excerpt you relied on).
"""


def _pack_excerpts(chunks: list[dict[str, Any]], max_chars: int) -> tuple[str, int]:
    """Return (packed_excerpt_block, n_chunks_included)."""
    lines: list[str] = []
    used = 0
    count = 0
    for ch in chunks:
        meta = ch.get("metadata") or {}
        header = (
            f"[chunk_id={ch.get('chunk_id')} "
            f"page={meta.get('page', '?')} "
            f"file={meta.get('source_filename', '')}]"
        )
        body = ch.get("text", "") or ""
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
        count += 1
    return "\n".join(lines), count


def build_full_context_prompt(chunks: list[dict[str, Any]], max_chars: int) -> str:
    """System prompt used when we pack the ENTIRE document into context (small/medium docs)."""
    excerpts, n = _pack_excerpts(chunks, max_chars)
    return (
        _FULL_CTX_HEADER
        + f"\n--- FULL DOCUMENT CONTENT ({n} sections) ---\n\n"
        + excerpts
    )


def build_rag_system_prompt(chunks: list[dict[str, Any]], max_chars: int) -> str:
    """System prompt for standard RAG (retrieved + spread chunks)."""
    excerpts, _ = _pack_excerpts(chunks, max_chars)
    return _RAG_HEADER + "\n--- EXCERPTS ---\n\n" + excerpts
