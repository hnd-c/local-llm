"""Chroma vector store — replace mode (single active document collection)."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from docstack.config import get_settings
from docstack.index.embedder import embed_texts
from docstack.models import TextChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "docstack_chunks"


def _client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_collection() -> Collection:
    client = _client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def wipe_collection() -> None:
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[TextChunk]) -> int:
    if not chunks:
        return 0
    coll = get_collection()
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    embeddings = embed_texts(documents)
    metadatas: list[dict[str, Any]] = []
    for c in chunks:
        metadatas.append(
            {
                "doc_id": c.doc_id,
                "page": int(c.page),
                "chunk_index": int(c.chunk_index),
                "source_filename": c.source_filename,
                "section_heading": c.section_heading or "",
                "table_id": c.table_id or "",
            }
        )
    coll.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(chunks)


def query_similar(query: str, top_k: int) -> list[dict[str, Any]]:
    coll = get_collection()
    qemb = embed_texts([query])[0]
    res = coll.query(
        query_embeddings=[qemb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict[str, Any]] = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for i, cid in enumerate(res["ids"][0]):
        dist = res["distances"][0][i] if res.get("distances") else 0.0
        score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
        out.append(
            {
                "chunk_id": cid,
                "text": res["documents"][0][i] if res["documents"] else "",
                "metadata": res["metadatas"][0][i] if res["metadatas"] else {},
                "distance": float(dist) if dist is not None else 0.0,
                "score": score,
            }
        )
    return out
