"""Route user text to fast (GPU) or deep (CPU) model."""

from __future__ import annotations

from docstack.config import get_settings

_DEEP_KEYWORDS = (
    "compare",
    "audit",
    "contradiction",
    "contradictions",
    "summarise all",
    "summarize all",
    "find differences",
    "analyse all",
    "analyze all",
    "deep analysis",
)


def select_model_for_prompt(user_text: str) -> tuple[str, int | None]:
    """Return (ollama_model_name, num_gpu or None). num_gpu=0 forces CPU for deep tier."""
    settings = get_settings()
    t = user_text.lower()
    for kw in _DEEP_KEYWORDS:
        if kw in t:
            return settings.deep_model, settings.deep_model_num_gpu
    return settings.fast_model, None
