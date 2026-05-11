"""Sentence-transformers embeddings on CPU."""

from __future__ import annotations

import logging
from functools import lru_cache

from docstack.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _model():
    settings = get_settings()
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s on CPU…", settings.embedding_model)
    m = SentenceTransformer(settings.embedding_model, device="cpu")
    return m


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()
