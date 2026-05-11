"""Pydantic schemas for validated LLM JSON outputs (optional post-processing)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str = ""
    page: int = 0
    source_filename: str = ""


class QAResult(BaseModel):
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    low_confidence: bool = False


class EntityItem(BaseModel):
    name: str = ""
    type: str = ""
    value: str = ""
    source_chunk: str = ""


class ExtractedEntities(BaseModel):
    entities: list[EntityItem] = Field(default_factory=list)


class SummaryBullets(BaseModel):
    bullets: list[str] = Field(default_factory=list)
    source_doc: str = ""
    review_required: bool = False


class ActionItem(BaseModel):
    text: str = ""
    owner: str = ""
    due_date: str = ""


class ActionItems(BaseModel):
    items: list[ActionItem] = Field(default_factory=list)
    source_doc: str = ""
