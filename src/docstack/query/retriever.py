"""Retrieve chunks for a query."""

from __future__ import annotations

from docstack.config import get_settings
from docstack.index.store import query_similar


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    return query_similar(query, k)
