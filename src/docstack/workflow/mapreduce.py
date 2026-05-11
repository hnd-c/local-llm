"""Map–reduce over all indexed chunks for long-document summarization (e.g. Nepali government PDFs)."""

from __future__ import annotations

import asyncio
import logging

from docstack.config import get_settings
from docstack.models import TextChunk
from docstack.query.ollama_client import chat_once
from docstack.workflow.schemas import SummaryBullets

logger = logging.getLogger(__name__)

# Whole-corpus compression — stricter than generic “broad RAG” (no open-ended brainstorming).
_MAPREDUCE_DEVANAGARI = (
    "सारांश",
    "सङ्क्षेप",
    "संक्षेप",
    "निचोड",
    "संक्षिप्त",
    "यो कागजात",
    "यो दस्तावेज",
    "यो पत्र",
    "यो प्रतिवेदन",
    "मुख्य बुँदा",
    "मुख्य बुदा",
    "मुख्य बिन्दु",
    "व्याख्या",
    "नेपालीमा",
    "संक्षेपमा",
)

_MAPREDUCE_EN = (
    "summarize",
    "summarise",
    "summary",
    "overview",
    "entire document",
    "whole document",
    "full document",
    "all pages",
    "every section",
    "tell me what",
    "what is in",
    "what's in",
    "whats in",
    "summarize all",
    "describe this document",
    "outline this document",
)


def is_mapreduce_eligible_query(query: str) -> bool:
    if not query.strip():
        return False
    for s in _MAPREDUCE_DEVANAGARI:
        if s in query:
            return True
    t = query.lower()
    return any(s in t for s in _MAPREDUCE_EN)


def stratified_sample_chunks(chunks: list[TextChunk], max_n: int) -> list[TextChunk]:
    """Evenly subsample when the index is huge so map–reduce stays within time budget."""
    if max_n <= 0 or len(chunks) <= max_n:
        return sorted(chunks, key=lambda c: (c.doc_id, c.chunk_index))
    rows = sorted(chunks, key=lambda c: (c.doc_id, c.chunk_index))
    n = min(max_n, len(rows))
    indices: set[int] = set()
    if n == 1:
        indices.add(0)
    else:
        for j in range(n):
            idx = int(round(j * (len(rows) - 1) / (n - 1)))
            indices.add(min(max(idx, 0), len(rows) - 1))
    return [rows[i] for i in sorted(indices)]


def _map_system_prompt(user_query: str) -> str:
    q = user_query.strip() or "Summarize the document."
    return (
        "You are summarizing one excerpt from a long official document (often Nepali).\n"
        "Write 2–5 bullet points capturing obligations, actors, timelines, and numbers if present.\n"
        "If the excerpt is in Nepali (नेपाली), write bullets in Nepali unless the user explicitly asked for another language.\n"
        "Plain text bullets only (lines starting with - ). No preamble.\n"
        f"User request (for tone/focus): {q[:500]}"
    )


def _reduce_system_prompt(user_query: str, *, final: bool) -> str:
    q = user_query.strip() or "Merge partial summaries."
    tag = "final" if final else "intermediate"
    return (
        f"You are merging {tag} partial summaries of the same long document (often Nepali government text).\n"
        "Produce one coherent answer: sections or bullet list; remove duplicates; preserve important names, dates, and legal terms.\n"
        "If sources are in Nepali, write in Nepali unless the user asked otherwise.\n"
        f"User request: {q[:800]}"
    )


async def map_summarize_chunk(
    model: str,
    user_query: str,
    text: str,
    idx: int,
    *,
    num_gpu: int | None,
    max_chars: int,
) -> str:
    settings = get_settings()
    body = text[:max_chars]
    messages = [
        {"role": "system", "content": _map_system_prompt(user_query)},
        {"role": "user", "content": f"[Part {idx + 1}]\n{body}"},
    ]
    return await chat_once(model, messages, num_ctx=settings.num_ctx, num_gpu=num_gpu)


async def reduce_summaries_batch(
    model: str,
    user_query: str,
    partials: list[str],
    *,
    num_gpu: int | None,
    max_chars: int,
    final: bool,
) -> str:
    joined = "\n\n".join(f"--- Part {i + 1} ---\n{p}" for i, p in enumerate(partials))
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n\n[…truncated for length…]"
    settings = get_settings()
    messages = [
        {"role": "system", "content": _reduce_system_prompt(user_query, final=final)},
        {"role": "user", "content": joined},
    ]
    return await chat_once(model, messages, num_ctx=settings.num_ctx, num_gpu=num_gpu)


async def hierarchical_reduce(
    model: str,
    user_query: str,
    partials: list[str],
    *,
    num_gpu: int | None,
    batch: int,
    max_chars: int,
    deep_final: bool,
    deep_model: str,
    deep_num_gpu: int | None,
) -> str:
    cur = [p for p in partials if p.strip()]
    if not cur:
        return ""
    if len(cur) == 1:
        m = deep_model if deep_final else model
        g = deep_num_gpu if deep_final else num_gpu
        return await reduce_summaries_batch(
            m, user_query, cur, num_gpu=g, max_chars=max_chars, final=True
        )

    while len(cur) > batch:
        nxt: list[str] = []
        for i in range(0, len(cur), batch):
            group = cur[i : i + batch]
            merged = await reduce_summaries_batch(
                model,
                user_query,
                group,
                num_gpu=num_gpu,
                max_chars=max_chars,
                final=False,
            )
            nxt.append(merged)
        cur = nxt

    m = deep_model if deep_final else model
    g = deep_num_gpu if deep_final else num_gpu
    return await reduce_summaries_batch(
        m, user_query, cur, num_gpu=g, max_chars=max_chars, final=True
    )


async def map_reduce_long_document(
    user_query: str,
    chunks: list[TextChunk],
    *,
    model: str,
    num_gpu: int | None,
) -> str:
    """Run parallel map over chunks, then hierarchical reduce. Returns plain text."""
    settings = get_settings()
    if not chunks:
        return "No indexed chunks found. Upload and ingest a document first."

    max_map = settings.mapreduce_map_chunk_chars
    conc = max(1, settings.mapreduce_concurrency)
    batch = max(2, settings.mapreduce_reduce_batch)
    max_in = settings.mapreduce_reduce_input_chars
    deep_final = settings.mapreduce_deep_final_reduce

    sem = asyncio.Semaphore(conc)

    async def one(c: TextChunk, i: int) -> str:
        async with sem:
            return await map_summarize_chunk(
                model, user_query, c.text, i, num_gpu=num_gpu, max_chars=max_map
            )

    logger.info("Map–reduce: mapping %d chunks (concurrency=%d)", len(chunks), conc)
    partials = await asyncio.gather(*(one(c, i) for i, c in enumerate(chunks)))

    text = await hierarchical_reduce(
        model,
        user_query,
        list(partials),
        num_gpu=num_gpu,
        batch=batch,
        max_chars=max_in,
        deep_final=deep_final,
        deep_model=settings.deep_model,
        deep_num_gpu=settings.deep_model_num_gpu,
    )

    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.outputs_dir / "mapreduce_summary.txt"
    out_path.write_text(text, encoding="utf-8")
    return text


async def map_reduce_chunks(chunks: list[TextChunk], *, use_deep: bool = False) -> SummaryBullets:
    """Legacy helper: map–reduce with optional deep model for all stages."""
    settings = get_settings()
    model = settings.deep_model if use_deep else settings.fast_model
    num_gpu = settings.deep_model_num_gpu if use_deep else None
    text = await map_reduce_long_document("Summarize the document.", chunks, model=model, num_gpu=num_gpu)
    bullets = [
        b.strip("- ").strip() for b in text.split("\n") if b.strip() and b.strip().startswith("-")
    ]
    if not bullets:
        bullets = [ln.strip() for ln in text.split("\n") if ln.strip()][:50]
    out = SummaryBullets(
        bullets=bullets[:50],
        source_doc=chunks[0].source_filename if chunks else "",
        review_required=len(chunks) > 40,
    )
    out_path = settings.outputs_dir / "mapreduce_summary.json"
    out_path.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


def run_map_reduce_sync(chunks: list[TextChunk], *, use_deep: bool = False) -> SummaryBullets:
    return asyncio.run(map_reduce_chunks(chunks, use_deep=use_deep))
