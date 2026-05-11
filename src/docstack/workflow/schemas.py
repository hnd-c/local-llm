"""Pydantic schema for validated map-reduce output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SummaryBullets(BaseModel):
    bullets: list[str] = Field(default_factory=list)
    source_doc: str = ""
    review_required: bool = False
