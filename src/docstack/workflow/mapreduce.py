"""Map-reduce summarisation over many chunks (batch helper)."""

from __future__ import annotations

import asyncio
import logging

from docstack.config import get_settings
from docstack.models import TextChunk
from docstack.query.ollama_client import chat_once
from docstack.workflow.schemas import SummaryBullets

logger = logging.getLogger(__name__)


async def map_summarize_chunk(
    model: str, text: str, idx: int, *, num_gpu: int | None
) -> str:
    settings = get_settings()
    messages = [
        {
            "role": "system",
            "content": "Summarise the excerpt in 2-4 bullet points. Output plain text only.",
        },
        {"role": "user", "content": text[:8000]},
    ]
    return await chat_once(
        model, messages, num_ctx=settings.num_ctx, num_gpu=num_gpu
    )


async def reduce_summaries(model: str, partials: list[str], *, num_gpu: int | None) -> str:
    joined = "\n\n".join(f"--- Part {i+1} ---\n{p}" for i, p in enumerate(partials))
    messages = [
        {
            "role": "system",
            "content": "Merge these partial summaries into one coherent bullet list. Plain text.",
        },
        {"role": "user", "content": joined[:12000]},
    ]
    settings = get_settings()
    return await chat_once(
        model, messages, num_ctx=settings.num_ctx, num_gpu=num_gpu
    )


async def map_reduce_chunks(chunks: list[TextChunk], *, use_deep: bool = False) -> SummaryBullets:
    settings = get_settings()
    model = settings.deep_model if use_deep else settings.fast_model
    num_gpu = settings.deep_model_num_gpu if use_deep else None
    if not chunks:
        return SummaryBullets(bullets=[], source_doc="", review_required=False)

    partials: list[str] = []
    for i, c in enumerate(chunks):
        p = await map_summarize_chunk(model, c.text, i, num_gpu=num_gpu)
        partials.append(p)

    merged = await reduce_summaries(model, partials, num_gpu=num_gpu)
    bullets = [b.strip("- ").strip() for b in merged.split("\n") if b.strip()]
    out = SummaryBullets(
        bullets=bullets[:50],
        source_doc=chunks[0].source_filename,
        review_required=len(chunks) > 40,
    )
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.outputs_dir / "mapreduce_summary.json"
    out_path.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


def run_map_reduce_sync(chunks: list[TextChunk], *, use_deep: bool = False) -> SummaryBullets:
    return asyncio.run(map_reduce_chunks(chunks, use_deep=use_deep))
