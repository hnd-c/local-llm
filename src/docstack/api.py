"""FastAPI app: OpenAI-compatible /v1/chat/completions + ingest + minimal upload UI."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import fitz
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from docstack.chunk.chunker import records_to_chunks
from docstack.config import get_settings
from docstack.ingest.router import ingest_path
from docstack.index.store import index_chunks, wipe_collection
from docstack.query.ollama_client import chat_stream
from docstack.query.prompts import build_rag_system_prompt
from docstack.query.retriever import retrieve
from docstack.workflow.router import select_model_for_prompt

logger = logging.getLogger(__name__)

_ingest_lock = threading.Lock()
_ingest_state: dict[str, Any] = {
    "status": "idle",
    "job_id": None,
    "pages_done": 0,
    "pages_total": 0,
    "active_doc": None,
    "error": None,
    "started_at": None,
}


def _page_count(path: Path) -> int:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        try:
            return max(1, doc.page_count)
        finally:
            doc.close()
    return 1


def _run_ingest_job(job_id: str, saved_path: Path) -> None:
    settings = get_settings()
    global _ingest_state
    try:
        with _ingest_lock:
            _ingest_state.update(
                {
                    "status": "running",
                    "job_id": job_id,
                    "pages_done": 0,
                    "pages_total": _page_count(saved_path),
                    "active_doc": saved_path.name,
                    "error": None,
                    "started_at": time.time(),
                }
            )
        wipe_collection()
        records = ingest_path(saved_path, settings.min_chars_per_page)
        chunks = records_to_chunks(records)
        index_chunks(chunks)
        with _ingest_lock:
            _ingest_state.update(
                {
                    "status": "completed",
                    "pages_done": _ingest_state.get("pages_total", 1),
                    "error": None,
                }
            )
    except Exception as e:  # noqa: BLE001
        logger.exception("Ingest failed")
        with _ingest_lock:
            _ingest_state.update({"status": "failed", "error": str(e)})


app = FastAPI(title="DocStack RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DocStack</title></head>
<body>
<h1>DocStack</h1>
<p>POST a file to <code>/ingest</code> (multipart field <code>file</code>), then chat via Open WebUI pointing at <code>/v1</code>.</p>
<form action="/ingest" method="post" enctype="multipart/form-data">
<input type="file" name="file" required>
<button type="submit">Upload &amp; index</button>
</form>
<p><a href="/ingest/status">Ingest status</a> · <a href="/v1/models">Models</a> · <a href="/health">Health</a></p>
</body></html>"""


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ingest/status")
async def ingest_status() -> dict[str, Any]:
    with _ingest_lock:
        s = dict(_ingest_state)
    if s.get("started_at"):
        s["elapsed_s"] = round(time.time() - float(s["started_at"]), 2)
    else:
        s["elapsed_s"] = None
    return s


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> JSONResponse:
    settings = get_settings()
    dest = settings.uploads_dir / (file.filename or "upload.bin")
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    dest.write_bytes(data)
    job_id = secrets.token_hex(8)
    t = threading.Thread(target=_run_ingest_job, args=(job_id, dest), daemon=True)
    t.start()
    return JSONResponse({"job_id": job_id, "saved_as": str(dest), "message": "Indexing started"})


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = True
    temperature: float | None = None


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role != "user":
            continue
        c = m.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts: list[str] = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
            return "\n".join(parts)
    return ""


def _message_text(m: ChatMessage, fallback_user: str) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    if m.role == "user" and isinstance(c, list):
        return fallback_user
    return ""


def _build_ollama_messages(
    *,
    use_rag: bool,
    hits: list[dict[str, Any]],
    body: ChatCompletionRequest,
    user_text: str,
) -> list[dict[str, str]]:
    settings = get_settings()
    if use_rag:
        system = build_rag_system_prompt(hits, settings.max_ctx_chars)
        out: list[dict[str, str]] = [{"role": "system", "content": system}]
        for m in body.messages:
            if m.role == "system":
                continue
            text = _message_text(m, user_text)
            if text.strip():
                out.append({"role": m.role, "content": text})
        return out

    out = []
    for m in body.messages:
        if m.role == "system":
            continue
        text = _message_text(m, user_text)
        if text.strip():
            out.append({"role": m.role, "content": text})
    if not any(m["role"] == "user" for m in out) and user_text.strip():
        out.append({"role": "user", "content": user_text})
    return out


def _sse_chunk(
    *,
    content: str,
    model: str,
    cid: str,
    finish_reason: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": ({"content": content} if content else {}),
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    settings = get_settings()
    user_text = _last_user_text(body.messages)
    cid = f"chatcmpl-{secrets.token_hex(12)}"

    with _ingest_lock:
        idx_status = _ingest_state.get("status")

    async def stream_indexing_wait() -> Any:
        model = body.model or settings.fast_model
        msg = (
            f"⏳ Still indexing your document ({_ingest_state.get('active_doc', '…')}). "
            f"Progress: page {_ingest_state.get('pages_done', 0)}/{_ingest_state.get('pages_total', '?')}. "
            "Please wait and try again in a few seconds."
        )
        yield _sse_chunk(content=msg, model=model, cid=cid)
        yield _sse_chunk(content="", model=model, cid=cid, finish_reason="stop")
        yield "data: [DONE]\n\n"

    if idx_status == "running":
        return StreamingResponse(stream_indexing_wait(), media_type="text/event-stream")

    if idx_status == "failed":
        err = _ingest_state.get("error") or "unknown error"

        async def stream_err() -> Any:
            model = body.model or settings.fast_model
            yield _sse_chunk(
                content=f"Indexing failed: {err}",
                model=model,
                cid=cid,
            )
            yield _sse_chunk(content="", model=model, cid=cid, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_err(), media_type="text/event-stream")

    model_name, num_gpu = select_model_for_prompt(user_text)
    if body.model and body.model.strip():
        model_name = body.model.strip()

    hits = retrieve(user_text)
    max_score = max((h.get("score") or 0.0) for h in hits) if hits else 0.0
    use_rag = bool(hits) and max_score >= settings.retrieval_min_score
    ollama_messages = _build_ollama_messages(
        use_rag=use_rag, hits=hits, body=body, user_text=user_text
    )

    if not body.stream:

        async def collect() -> str:
            buf: list[str] = []
            async for p in chat_stream(
                model_name,
                ollama_messages,
                num_ctx=settings.num_ctx,
                num_gpu=num_gpu,
            ):
                buf.append(p)
            return "".join(buf)

        text = await collect()
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    async def stream_tokens() -> Any:
        async for piece in chat_stream(
            model_name,
            ollama_messages,
            num_ctx=settings.num_ctx,
            num_gpu=num_gpu,
        ):
            if piece:
                yield _sse_chunk(content=piece, model=model_name, cid=cid)
        yield _sse_chunk(content="", model=model_name, cid=cid, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    settings = get_settings()
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": settings.fast_model,
                "object": "model",
                "created": now,
                "owned_by": "docstack",
            },
            {
                "id": settings.deep_model,
                "object": "model",
                "created": now,
                "owned_by": "docstack",
            },
        ],
    }
