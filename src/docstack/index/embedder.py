"""Sentence-transformers embeddings on CPU."""

from __future__ import annotations

import logging
from functools import lru_cache

from docstack.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _model(model_name: str):
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s on CPU…", model_name)
    return SentenceTransformer(model_name, device="cpu")


def clear_embedding_model_cache() -> None:
    """Drop cached SentenceTransformer instances (e.g. after changing embedding_model)."""
    _model.cache_clear()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    model = _model(settings.embedding_model)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()
