"""Retrieve chunks for a query."""

from __future__ import annotations

from docstack.config import get_settings
from docstack.index.store import collection_chunk_count, list_chunks_spread_sample, query_similar
from docstack.workflow.mapreduce import is_mapreduce_eligible_query

# English / Latin — broad document tasks (summarize, ideation, drafting, reasoning).
_BROAD_DOC_SUBSTRINGS = (
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
    "what are in",
    "describe this",
    "outline this",
    "brainstorm",
    "new ideas",
    "new document",
    "draft a",
    "write a",
    "generate a",
    "reason about",
    "analyze this",
    "analyse this",
    "government",
    "policy",
    "ministry",
    "circular",
    "directive",
)

# Nepali (Devanagari) — do not lower(); matches government / summary phrasing.
_BROAD_DEVANAGARI = (
    "सारांश",
    "सङ्क्षेप",
    "संक्षेप",
    "संक्षिप्त",
    "निचोड",
    "व्याख्या",
    "मुख्य बुँदा",
    "मुख्य बुदा",
    "मुख्य बिन्दु",
    "यो कागजात",
    "यो दस्तावेज",
    "यो पत्र",
    "यो प्रतिवेदन",
    "नीति",
    "निर्देशिका",
    "परिपत्र",
    "सिफारिस",
    "निर्णय",
    "मन्त्रालय",
    "कार्यविधि",
    "कार्यालय",
    "आदेश",
    "सूचना",
    "नियमावली",
    "दस्तावेजमा",
    "कागजातमा",
    "नेपालीमा",
    "संक्षेपमा",
)


def is_broad_document_query(query: str) -> bool:
    if not query.strip():
        return False
    for s in _BROAD_DEVANAGARI:
        if s in query:
            return True
    t = query.lower()
    return any(s in t for s in _BROAD_DOC_SUBSTRINGS)


def _wide_document_intent(query: str) -> bool:
    """Broad RAG spread + floor: summarize / overview / Nepali equivalents / map-reduce-style."""
    return is_broad_document_query(query) or is_mapreduce_eligible_query(query)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    hits = query_similar(query, k)
    hits.sort(key=lambda x: -float(x.get("score") or 0.0))

    if settings.rag_breadth_chunks <= 0 or not _wide_document_intent(query):
        return hits

    seen = {str(h.get("chunk_id")) for h in hits if h.get("chunk_id")}
    spread = list_chunks_spread_sample(settings.rag_breadth_chunks, exclude_ids=seen)
    hits = hits + spread

    floor = settings.rag_min_hits_floor
    if floor <= 0:
        return hits

    total = collection_chunk_count()
    while len(hits) < floor and len(hits) < total:
        seen = {str(h.get("chunk_id")) for h in hits if h.get("chunk_id")}
        need = min(floor - len(hits), settings.rag_breadth_chunks, max(0, total - len(seen)))
        if need <= 0:
            break
        more = list_chunks_spread_sample(need, exclude_ids=seen)
        if not more:
            break
        hits = hits + more

    return hits
