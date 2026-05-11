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


def list_chunks_spread_sample(n: int, exclude_ids: set[str]) -> list[dict[str, Any]]:
    """Evenly spaced chunks (by chunk_index) for document-wide coverage; excludes exclude_ids."""
    if n <= 0:
        return []
    coll = get_collection()
    batch = coll.get(include=["documents", "metadatas"])
    ids = batch.get("ids") or []
    if not ids:
        return []
    documents = batch.get("documents") or []
    metadatas = batch.get("metadatas") or []
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for i, cid in enumerate(ids):
        if cid in exclude_ids:
            continue
        doc_text = documents[i] if i < len(documents) else ""
        meta_raw = metadatas[i] if i < len(metadatas) else {}
        meta = dict(meta_raw) if meta_raw else {}
        rows.append((cid, doc_text or "", meta))
    rows.sort(key=lambda x: (str(x[2].get("doc_id", "")), int(x[2].get("chunk_index", 0))))
    if not rows:
        return []
    m = min(n, len(rows))
    if m <= 0:
        return []
    indices: set[int] = set()
    if m == 1:
        indices.add(0)
    else:
        for j in range(m):
            idx = int(round(j * (len(rows) - 1) / (m - 1)))
            indices.add(min(max(idx, 0), len(rows) - 1))
    out: list[dict[str, Any]] = []
    for idx in sorted(indices):
        cid, text, meta = rows[idx]
        out.append(
            {
                "chunk_id": cid,
                "text": text,
                "metadata": meta,
                "distance": 0.0,
                "score": 0.2,
            }
        )
    return out


def collection_chunk_count() -> int:
    try:
        coll = get_collection()
        res = coll.get(include=[])
        return len(res.get("ids") or [])
    except Exception:  # noqa: BLE001
        return 0


def fetch_all_chunks_ordered() -> list[TextChunk]:
    """All chunks in the active collection, sorted by doc_id then chunk_index."""
    coll = get_collection()
    batch = coll.get(include=["documents", "metadatas"])
    ids = batch.get("ids") or []
    if not ids:
        return []
    documents = batch.get("documents") or []
    metadatas = batch.get("metadatas") or []
    out: list[TextChunk] = []
    for i, cid in enumerate(ids):
        meta_raw = metadatas[i] if i < len(metadatas) else {}
        meta = dict(meta_raw) if meta_raw else {}
        text = documents[i] if i < len(documents) else ""
        sh = meta.get("section_heading")
        tid = meta.get("table_id")
        out.append(
            TextChunk(
                chunk_id=str(cid),
                doc_id=str(meta.get("doc_id", "")),
                text=text or "",
                page=int(meta.get("page", 1) or 1),
                chunk_index=int(meta.get("chunk_index", i)),
                source_filename=str(meta.get("source_filename", "")),
                section_heading=str(sh) if sh else None,
                table_id=str(tid) if tid else None,
            )
        )
    out.sort(key=lambda x: (x.doc_id, x.chunk_index))
    return out


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
