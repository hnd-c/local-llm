"""RAG prompt templates."""

from __future__ import annotations

import re
from typing import Any

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_DEVA_PUNCT = str.maketrans("", "", "।॥,;:!?.()")
_EN_STOPWORDS = {
    "a", "an", "the", "is", "in", "of", "to", "and", "or", "for", "this",
    "that", "it", "what", "how", "why", "me", "about", "can", "do", "i",
    "my", "with", "from", "at", "be", "are", "was", "were", "has", "have",
    "had", "not", "by", "on", "as", "its", "but", "if", "so", "we", "you",
}
_NE_STOPWORDS = {
    "को", "का", "की", "मा", "र", "छ", "छन्", "हो", "हुन्", "भयो",
    "भए", "गर्न", "गर्छ", "यो", "यस", "त्यो", "जो", "यी", "ती",
    "नै", "पनि", "गरी", "भनी", "तर", "किनकि", "जब", "जहाँ", "अनि",
    "वा", "एक", "दुई", "लागि", "सँग",
}


def _query_trigrams(query: str) -> set[str]:
    """Build a set of character trigrams from the query keywords."""
    is_nepali = bool(_DEVANAGARI_RE.search(query))
    stopwords = _NE_STOPWORDS if is_nepali else _EN_STOPWORDS
    tokens = [
        w.lower().translate(_DEVA_PUNCT)
        for w in query.split()
        if len(w) > 1 and w.lower() not in stopwords
    ]
    joined = " ".join(tokens)
    return {joined[i:i+3] for i in range(len(joined) - 2)} if len(joined) >= 3 else set()


def _score_chunk(chunk_text: str, q_trigrams: set[str]) -> float:
    """Trigram overlap score normalised by chunk length."""
    if not q_trigrams or not chunk_text:
        return 0.0
    t = chunk_text.lower()
    chunk_tg = {t[i:i+3] for i in range(len(t) - 2)}
    return len(q_trigrams & chunk_tg) / (len(q_trigrams) + 1) / (len(chunk_text) / 400 + 1)

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


def _chunk_block(ch: dict[str, Any]) -> str:
    meta = ch.get("metadata") or {}
    header = (
        f"[chunk_id={ch.get('chunk_id')} "
        f"page={meta.get('page', '?')} "
        f"file={meta.get('source_filename', '')}]"
    )
    return f"{header}\n{ch.get('text', '') or ''}\n"


def _pack_excerpts(chunks: list[dict[str, Any]], max_chars: int) -> tuple[str, int]:
    """Sequential pack — used for regular RAG where ChromaDB already ranks by relevance."""
    lines: list[str] = []
    used = 0
    for ch in chunks:
        block = _chunk_block(ch)
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines), len(lines)


def _pack_excerpts_query_aware(
    chunks: list[dict[str, Any]],
    max_chars: int,
    query: str,
) -> tuple[str, int]:
    """Query-aware pack for full-context path.

    When all chunks fit: return them all in document order (unchanged behaviour).
    When they overflow: apply trigram scoring to select the most query-relevant
    chunks while always anchoring the first 2 and last 2 chunks (intro + conclusion),
    then reassemble in original document order.

    This ensures the tail of large documents (final sections, conclusions, deadlines)
    is never silently dropped just because it appears late in the document.
    """
    blocks = [_chunk_block(ch) for ch in chunks]
    total = sum(len(b) for b in blocks)

    if total <= max_chars:
        # Everything fits — no selection needed
        return "\n".join(blocks), len(blocks)

    n = len(chunks)
    # Anchor indices: always include first 2 and last 2 chunks
    anchor_idxs: set[int] = {0, min(1, n - 1), max(n - 2, 0), n - 1}

    anchor_chars = sum(len(blocks[i]) for i in anchor_idxs)
    budget = max_chars - anchor_chars

    # Score non-anchor chunks by trigram overlap with the query
    q_tg = _query_trigrams(query)
    scored: list[tuple[float, int]] = []
    for i, (ch, block) in enumerate(zip(chunks, blocks)):
        if i in anchor_idxs:
            continue
        score = _score_chunk(ch.get("text", "") or "", q_tg)
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])

    selected: set[int] = set(anchor_idxs)
    remaining = budget
    for _score, i in scored:
        if remaining <= 0:
            break
        cost = len(blocks[i])
        if cost <= remaining:
            selected.add(i)
            remaining -= cost

    # Reassemble in original document order
    ordered = [blocks[i] for i in sorted(selected)]
    return "\n".join(ordered), len(ordered)


def build_full_context_prompt(
    chunks: list[dict[str, Any]],
    max_chars: int,
    query: str = "",
) -> str:
    """System prompt for full-context single-pass.

    Uses query-aware chunk selection so large documents don't lose their
    tail sections (conclusions, deadlines, final clauses) when they overflow
    max_chars. Falls back to sequential pack when query is empty.
    """
    if query:
        excerpts, n = _pack_excerpts_query_aware(chunks, max_chars, query)
    else:
        excerpts, n = _pack_excerpts(chunks, max_chars)
    return (
        _FULL_CTX_HEADER
        + f"\n--- FULL DOCUMENT CONTENT ({n} sections) ---\n\n"
        + excerpts
    )


def build_rag_system_prompt(chunks: list[dict[str, Any]], max_chars: int) -> str:
    """System prompt for standard RAG (retrieved + spread chunks).

    ChromaDB already ranks chunks by semantic similarity, so sequential
    packing is correct here — no additional selection needed.
    """
    excerpts, _ = _pack_excerpts(chunks, max_chars)
    return _RAG_HEADER + "\n--- EXCERPTS ---\n\n" + excerpts
