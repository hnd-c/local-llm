"""FastAPI app: OpenAI-compatible /v1/chat/completions + ingest + minimal upload UI."""

from __future__ import annotations

import hashlib
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
from docstack.index.store import (
    collection_chunk_count,
    fetch_all_chunks_ordered,
    index_chunks,
    wipe_collection,
)
from docstack.query.ollama_client import chat_stream
from docstack.query.prompts import build_full_context_prompt, build_rag_system_prompt
from docstack.query.retriever import retrieve
from docstack.workflow.mapreduce import (
    is_mapreduce_eligible_query,
    map_reduce_long_document,
    stratified_sample_chunks,
)
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


def _sha256(path: Path) -> str:
    """SHA-256 of file contents — true content fingerprint regardless of filename."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _registry_path() -> Path:
    """JSON file that maps content hashes → ingest metadata, stored next to ChromaDB."""
    return get_settings().chroma_dir / "indexed_hashes.json"


def _load_registry() -> dict[str, Any]:
    p = _registry_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save_registry(reg: dict[str, Any]) -> None:
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2))


def _clear_registry() -> None:
    p = _registry_path()
    if p.exists():
        p.unlink()


def _run_ingest_job(job_id: str, saved_path: Path, *, replace: bool = True) -> None:
    settings = get_settings()
    global _ingest_state

    file_hash = _sha256(saved_path)

    # Check if this exact file content is already indexed (skip-cache).
    # Only applies for additive ingests; a replace=True call always wipes and re-indexes.
    if not replace:
        reg = _load_registry()
        if file_hash in reg and collection_chunk_count() > 0:
            cached = reg[file_hash]
            logger.info(
                "Skipping ingest — file already indexed (hash=%s, chunks=%s, file=%s)",
                file_hash[:12],
                cached.get("chunk_count"),
                cached.get("filename"),
            )
            with _ingest_lock:
                _ingest_state.update(
                    {
                        "status": "completed",
                        "job_id": job_id,
                        "pages_done": cached.get("pages", 1),
                        "pages_total": cached.get("pages", 1),
                        "active_doc": saved_path.name,
                        "error": None,
                        "started_at": time.time(),
                        "cached": True,
                    }
                )
            return

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
                    "cached": False,
                }
            )
        if replace:
            wipe_collection()
            _clear_registry()
        records = ingest_path(saved_path, settings.min_chars_per_page)
        chunks = records_to_chunks(records)
        index_chunks(chunks)
        # Register the content hash so future uploads of the same file are skipped.
        reg = _load_registry()
        reg[file_hash] = {
            "filename": saved_path.name,
            "chunk_count": len(chunks),
            "pages": _ingest_state.get("pages_total", 1),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_registry(reg)
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


_UPLOAD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DocStack — Upload</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem}
  .card{background:#1e293b;border:1px solid #334155;border-radius:1rem;padding:2rem;width:100%;max-width:560px;box-shadow:0 25px 50px -12px rgba(0,0,0,.6)}
  h1{font-size:1.5rem;font-weight:700;color:#f8fafc;margin-bottom:.25rem}
  .sub{color:#94a3b8;font-size:.875rem;margin-bottom:1.5rem}
  .drop{border:2px dashed #475569;border-radius:.75rem;padding:2.5rem 1rem;text-align:center;cursor:pointer;transition:all .2s;background:#0f172a}
  .drop.over,.drop:hover{border-color:#6366f1;background:#1e1b4b}
  .drop svg{width:3rem;height:3rem;stroke:#475569;margin-bottom:.75rem;display:block;margin-left:auto;margin-right:auto}
  .drop.over svg,.drop:hover svg{stroke:#6366f1}
  .drop p{color:#94a3b8;font-size:.9rem}
  .drop strong{color:#e2e8f0}
  input[type=file]{display:none}
  .mode{display:flex;gap:.5rem;margin-top:1.25rem;background:#0f172a;border-radius:.5rem;padding:.25rem}
  .mode label{flex:1;text-align:center;padding:.4rem .5rem;border-radius:.375rem;cursor:pointer;font-size:.8rem;color:#94a3b8;transition:all .15s}
  .mode input[type=radio]{display:none}
  .mode input[type=radio]:checked+span{background:#6366f1;color:#fff;border-radius:.375rem;display:block;padding:.4rem .5rem;margin:-.4rem -.5rem}
  .btn{margin-top:1.25rem;width:100%;padding:.7rem;background:#6366f1;color:#fff;border:none;border-radius:.5rem;font-size:1rem;font-weight:600;cursor:pointer;transition:background .15s}
  .btn:hover{background:#4f46e5}
  .btn:disabled{background:#334155;color:#64748b;cursor:not-allowed}
  #status{margin-top:1.25rem;padding:.75rem 1rem;border-radius:.5rem;font-size:.875rem;display:none}
  #status.ok{background:#052e16;color:#86efac;border:1px solid #166534}
  #status.err{background:#2d0a0a;color:#fca5a5;border:1px solid #7f1d1d}
  #status.info{background:#0c1445;color:#93c5fd;border:1px solid #1d4ed8}
  .links{margin-top:1.5rem;display:flex;gap:1rem;justify-content:center;font-size:.8rem}
  .links a{color:#6366f1;text-decoration:none}
  .links a:hover{text-decoration:underline}
  #index-info{margin-top:1rem;padding:.6rem .875rem;background:#0f172a;border-radius:.5rem;font-size:.8rem;color:#64748b;text-align:center}
  #file-name{margin-top:.75rem;font-size:.85rem;color:#a5b4fc;text-align:center;min-height:1.2em}
</style>
</head>
<body>
<div class="card">
  <h1>📄 DocStack</h1>
  <p class="sub">Upload a document to index it for RAG queries in Open WebUI.</p>

  <div class="drop" id="drop-zone">
    <svg fill="none" viewBox="0 0 24 24" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/></svg>
    <p><strong>Drop a file here</strong><br>or click to browse</p>
    <p style="margin-top:.4rem;font-size:.75rem;color:#475569">PDF · DOCX · XLSX · PPTX · TXT · MD · CSV · HTML · JPG/PNG/TIFF · DOC/XLS/PPT/ODT (via LibreOffice)</p>
    <input type="file" id="file-input" accept=".pdf,.docx,.doc,.xlsx,.xls,.pptx,.ppt,.txt,.md,.csv,.html,.htm,.jpg,.jpeg,.png,.tiff,.tif,.bmp,.webp,.odt,.ods,.odp,.rtf">
  </div>
  <div id="file-name"></div>

  <div class="mode">
    <label><input type="radio" name="mode" value="add" checked><span>➕ Add to index</span></label>
    <label><input type="radio" name="mode" value="replace"><span>🔄 Replace all</span></label>
  </div>

  <button class="btn" id="upload-btn" disabled>Select a file first</button>
  <div id="status"></div>
  <div id="index-info">Loading index info…</div>

  <div class="links">
    <a href="/ingest/status">Ingest status (JSON)</a>
    <a href="/v1/models">Models</a>
    <a href="/health">Health</a>
  </div>
</div>

<script>
const drop = document.getElementById('drop-zone');
const input = document.getElementById('file-input');
const btn = document.getElementById('upload-btn');
const stat = document.getElementById('status');
const fname = document.getElementById('file-name');
const info = document.getElementById('index-info');
let chosen = null;

function setFile(f) {
  chosen = f;
  fname.textContent = f ? '📎 ' + f.name + ' (' + (f.size/1024/1024).toFixed(1) + ' MB)' : '';
  btn.disabled = !f;
  btn.textContent = f ? 'Upload & Index' : 'Select a file first';
}

drop.addEventListener('click', () => input.click());
input.addEventListener('change', () => setFile(input.files[0] || null));
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  setFile(e.dataTransfer.files[0] || null);
});

btn.addEventListener('click', async () => {
  if (!chosen) return;
  const mode = document.querySelector('input[name=mode]:checked').value;
  const fd = new FormData();
  fd.append('file', chosen);
  btn.disabled = true; btn.textContent = 'Uploading…';
  showStatus('info', '⏳ Uploading and starting ingest… (OCR may take a while for scanned PDFs)');
  try {
    const endpoint = mode === 'add' ? '/ingest/add' : '/ingest';
    const r = await fetch(endpoint, { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    showStatus('ok', '✅ ' + j.message + ' (job ' + j.job_id + '). Polling for completion…');
    poll();
  } catch(e) {
    showStatus('err', '❌ ' + e.message);
    btn.disabled = false; btn.textContent = 'Retry';
  }
});

async function poll() {
  let tries = 0;
  while (tries++ < 600) {
    await new Promise(r => setTimeout(r, 1500));
    try {
      const s = await (await fetch('/ingest/status')).json();
      if (s.status === 'completed') {
        showStatus('ok', '✅ Done! Document indexed. Go back to Open WebUI and start chatting.');
        btn.disabled = false; btn.textContent = 'Upload another';
        setFile(null); loadInfo(); return;
      }
      if (s.status === 'failed') {
        showStatus('err', '❌ Ingest failed: ' + (s.error || 'unknown'));
        btn.disabled = false; btn.textContent = 'Retry'; return;
      }
      if (s.status === 'running') {
        showStatus('info', '⏳ Indexing ' + (s.active_doc||'') + ' — page ' + s.pages_done + '/' + (s.pages_total||'?') + ' · ' + (s.elapsed_s||0).toFixed(0) + 's');
      }
    } catch(_){}
  }
}

function showStatus(cls, msg) {
  stat.className = cls; stat.textContent = msg; stat.style.display = 'block';
}

async function loadInfo() {
  try {
    const s = await (await fetch('/ingest/status')).json();
    const chunks = await (await fetch('/ingest/chunks')).json();
    if (chunks.count > 0) {
      info.textContent = '📚 Index: ' + chunks.count + ' chunks · last: ' + (s.active_doc || '—');
    } else {
      info.textContent = '📭 Index is empty — upload a document to get started.';
    }
  } catch(_) { info.textContent = ''; }
}
loadInfo();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return _UPLOAD_HTML


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


@app.get("/ingest/chunks")
async def ingest_chunks() -> dict[str, int]:
    return {"count": collection_chunk_count()}


@app.delete("/ingest/wipe")
async def ingest_wipe() -> dict[str, str]:
    """Wipe the entire vector index and hash registry."""
    wipe_collection()
    _clear_registry()
    return {"status": "wiped", "chunks": "0"}


async def _save_and_start(file: UploadFile, *, replace: bool) -> JSONResponse:
    settings = get_settings()
    dest = settings.uploads_dir / (file.filename or "upload.bin")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    job_id = secrets.token_hex(8)
    t = threading.Thread(
        target=_run_ingest_job, args=(job_id, dest), kwargs={"replace": replace}, daemon=True
    )
    t.start()
    action = "Replacing index and indexing" if replace else "Adding to index"
    return JSONResponse(
        {"job_id": job_id, "saved_as": str(dest), "message": f"{action} started"}
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a file and replace the entire index."""
    return await _save_and_start(file, replace=True)


@app.post("/ingest/add")
async def ingest_add(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a file and ADD it to the existing index (no wipe)."""
    return await _save_and_start(file, replace=False)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = True
    temperature: float | None = None


_OWUI_TASK_PREFIX = "### Task:\n"
_OWUI_RAG_MARKER = "### Task:\nRespond to the user query using the provided context"
_OWUI_CONTEXT_END = "</context>"
_DOCSTACK_NO_RAG = "[DocStack:no-rag]"


def _strip_owui_rag_template(text: str) -> str:
    """If Open WebUI has wrapped the user message in its RAG template
    (### Task: ... <context>...</context> REAL_QUERY), extract and return
    only the real query that follows the closing </context> tag.
    Falls back to the full text unchanged if the pattern is not found."""
    if not text.startswith(_OWUI_RAG_MARKER):
        return text
    idx = text.rfind(_OWUI_CONTEXT_END)
    if idx == -1:
        return text
    real_query = text[idx + len(_OWUI_CONTEXT_END):].strip()
    return real_query if real_query else text


def _is_owui_internal_task(messages: list[ChatMessage]) -> bool:
    """Return True when the request is an Open WebUI internal task sub-request
    (e.g. 'generate queries', 'title generation') that should be forwarded to
    Ollama as-is without any RAG context injection."""
    if len(messages) != 1:
        return False
    c = messages[0].content
    text = c if isinstance(c, str) else ""
    return text.startswith(_OWUI_TASK_PREFIX) and _OWUI_RAG_MARKER not in text


def _has_image_content(messages: list[ChatMessage]) -> bool:
    """Return True if any user message contains an inline image (image_url part)."""
    for m in messages:
        if m.role != "user":
            continue
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _to_ollama_vision_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert OpenAI-style messages (with image_url content parts) to Ollama's
    vision format, where images are a top-level list of raw base64 strings."""
    out: list[dict[str, Any]] = []
    for m in messages:
        # Strip DocStack system signals; keep real system prompts
        if m.role == "system":
            c = m.content if isinstance(m.content, str) else ""
            if _DOCSTACK_NO_RAG in c:
                continue
        content = m.content
        if isinstance(content, list):
            text_parts: list[str] = []
            images: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        images.append(url.split(",", 1)[1])
                    elif url:
                        images.append(url)
            msg: dict[str, Any] = {"role": m.role, "content": " ".join(text_parts).strip()}
            if images:
                msg["images"] = images
            out.append(msg)
        else:
            out.append({"role": m.role, "content": _message_text(m, "")})
    return out


def _is_no_rag_session(messages: list[ChatMessage]) -> bool:
    """Return True when the filter has signaled that no document is active
    in this chat session — forward straight to Ollama without RAG."""
    for m in messages:
        if m.role == "system":
            c = m.content
            text = c if isinstance(c, str) else ""
            if _DOCSTACK_NO_RAG in text:
                return True
    return False


def _strip_no_rag_signal(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Return Ollama-format messages with the no-rag system message removed."""
    out = []
    for m in messages:
        if m.role == "system":
            c = m.content if isinstance(m.content, str) else ""
            if _DOCSTACK_NO_RAG in c:
                continue  # drop the signal; don't send it to the LLM
        out.append({"role": m.role, "content": _message_text(m, "")})
    return out


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role != "user":
            continue
        c = m.content
        if isinstance(c, str):
            return _strip_owui_rag_template(c)
        if isinstance(c, list):
            parts: list[str] = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
            raw = "\n".join(parts)
            return _strip_owui_rag_template(raw)
    return ""


def _message_text(m: ChatMessage, fallback_user: str) -> str:
    c = m.content
    if isinstance(c, str):
        # Strip OWUI RAG template so Ollama sees only the real query,
        # not the empty <context> tags that override our system prompt.
        return _strip_owui_rag_template(c)
    if m.role == "user" and isinstance(c, list):
        return fallback_user
    return ""


def _rag_conversation_tail(messages: list[ChatMessage], max_messages: int) -> list[ChatMessage]:
    """Keep only user/assistant turns; cap length so unrelated earlier chat does not dilute RAG."""
    conv = [m for m in messages if m.role in ("user", "assistant")]
    if max_messages <= 0:
        return conv
    if len(conv) <= max_messages:
        return conv
    return conv[-max_messages:]


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
        tail = _rag_conversation_tail(body.messages, settings.rag_max_history_messages)
        for m in tail:
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
    cid = f"chatcmpl-{secrets.token_hex(12)}"

    # ── Open WebUI internal task sub-requests (generate queries, title gen…) ──
    # These are single-message prompts that start with "### Task:" but are NOT
    # the RAG-response template.  Forward them directly to Ollama with no RAG
    # context injection — injecting a 30 K-char document system prompt causes
    # Ollama to time out on these lightweight calls.
    if _is_owui_internal_task(body.messages):
        model_name = body.model or settings.fast_model
        ollama_msgs = [{"role": m.role, "content": m.content or ""} for m in body.messages]
        if not body.stream:
            async def _collect_task() -> str:
                buf: list[str] = []
                async for p in chat_stream(model_name, ollama_msgs, temperature=body.temperature):
                    buf.append(p)
                return "".join(buf)
            text = await _collect_task()
            return JSONResponse({
                "id": cid, "object": "chat.completion",
                "created": int(time.time()), "model": model_name,
                "choices": [{"index": 0,
                              "message": {"role": "assistant", "content": text},
                              "finish_reason": "stop"}],
            })
        async def _stream_task() -> Any:
            async for piece in chat_stream(model_name, ollama_msgs, temperature=body.temperature):
                if piece:
                    yield _sse_chunk(content=piece, model=model_name, cid=cid)
            yield _sse_chunk(content="", model=model_name, cid=cid, finish_reason="stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream_task(), media_type="text/event-stream")

    # ── Vision / image requests ─────────────────────────────────────────────
    # If the user's message contains an inline image (image_url), route to the
    # vision model and skip RAG entirely — document context is irrelevant for
    # visual Q&A. Images are converted from OpenAI data-URL format to Ollama's
    # images[] array format.
    if _has_image_content(body.messages):
        vision_model = body.model or settings.vision_model
        vision_msgs = _to_ollama_vision_messages(body.messages)
        if not body.stream:
            async def _collect_vision() -> str:
                buf: list[str] = []
                async for p in chat_stream(vision_model, vision_msgs, temperature=body.temperature):
                    buf.append(p)
                return "".join(buf)
            text = await _collect_vision()
            return JSONResponse({
                "id": cid, "object": "chat.completion",
                "created": int(time.time()), "model": vision_model,
                "choices": [{"index": 0,
                              "message": {"role": "assistant", "content": text},
                              "finish_reason": "stop"}],
            })
        async def _stream_vision() -> Any:
            async for piece in chat_stream(vision_model, vision_msgs, temperature=body.temperature):
                if piece:
                    yield _sse_chunk(content=piece, model=vision_model, cid=cid)
            yield _sse_chunk(content="", model=vision_model, cid=cid, finish_reason="stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream_vision(), media_type="text/event-stream")

    # ── No-document session bypass ──────────────────────────────────────────
    # The filter injects [DocStack:no-rag] when no file has been attached in
    # this chat tab, so a stale index from a previous session is never used.
    # Forward directly to Ollama as a plain assistant conversation.
    if _is_no_rag_session(body.messages):
        model_name = body.model or settings.fast_model
        ollama_msgs = _strip_no_rag_signal(body.messages)
        if not body.stream:
            async def _collect_task() -> str:
                buf: list[str] = []
                async for p in chat_stream(model_name, ollama_msgs, temperature=body.temperature):
                    buf.append(p)
                return "".join(buf)
            text = await _collect_task()
            return JSONResponse({
                "id": cid, "object": "chat.completion",
                "created": int(time.time()), "model": model_name,
                "choices": [{"index": 0,
                              "message": {"role": "assistant", "content": text},
                              "finish_reason": "stop"}],
            })
        async def _stream_task() -> Any:
            async for piece in chat_stream(model_name, ollama_msgs, temperature=body.temperature):
                if piece:
                    yield _sse_chunk(content=piece, model=model_name, cid=cid)
            yield _sse_chunk(content="", model=model_name, cid=cid, finish_reason="stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream_task(), media_type="text/event-stream")

    user_text = _last_user_text(body.messages)

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
        if model_name == settings.deep_model:
            num_gpu = settings.deep_model_num_gpu
        elif model_name == settings.fast_model:
            num_gpu = None
        else:
            num_gpu = None

    n_chunks = collection_chunk_count()
    quality_query = is_mapreduce_eligible_query(user_text)

    # ── Full-context single-pass (best quality for small/medium docs) ──────────
    # When the whole index fits in the context window, pack every chunk directly.
    use_full_context = (
        quality_query
        and settings.full_context_max_chunks > 0
        and 0 < n_chunks <= settings.full_context_max_chunks
    )
    if use_full_context:
        all_chunks = fetch_all_chunks_ordered()
        fc_dicts = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": {
                    "page": c.page,
                    "source_filename": c.source_filename,
                    "section_heading": c.section_heading or "",
                },
            }
            for c in all_chunks
        ]
        system_msg = build_full_context_prompt(fc_dicts, settings.max_ctx_chars)
        tail = _rag_conversation_tail(body.messages, settings.rag_max_history_messages)
        fc_messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
        for m in tail:
            if m.role == "system":
                continue
            txt = _message_text(m, user_text)
            if txt.strip():
                fc_messages.append({"role": m.role, "content": txt})

        if not body.stream:
            async def collect_fc() -> str:
                buf: list[str] = []
                async for p in chat_stream(
                    model_name, fc_messages, num_ctx=settings.num_ctx, num_gpu=num_gpu,
                    temperature=body.temperature,
                ):
                    buf.append(p)
                return "".join(buf)

            text = await collect_fc()
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

        async def stream_full_context() -> Any:
            async for piece in chat_stream(
                model_name, fc_messages, num_ctx=settings.num_ctx, num_gpu=num_gpu,
                temperature=body.temperature,
            ):
                if piece:
                    yield _sse_chunk(content=piece, model=model_name, cid=cid)
            yield _sse_chunk(content="", model=model_name, cid=cid, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_full_context(), media_type="text/event-stream")

    # ── Map–reduce (larger docs) ───────────────────────────────────────────────
    use_mapreduce = (
        quality_query
        and settings.mapreduce_enabled
        and n_chunks >= settings.mapreduce_min_chunks
    )
    if use_mapreduce:
        all_chunks = fetch_all_chunks_ordered()
        sub = stratified_sample_chunks(all_chunks, settings.mapreduce_max_chunks)
        if not body.stream:
            result_text = await map_reduce_long_document(
                user_text, sub, model=model_name, num_gpu=num_gpu, temperature=body.temperature
            )
            return JSONResponse(
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result_text},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        async def stream_mapreduce() -> Any:
            yield _sse_chunk(
                content=(
                    "_DocStack: starting **full-document map–reduce** over "
                    f"{len(sub)} chunk(s). This may take several minutes…_\n\n"
                ),
                model=model_name,
                cid=cid,
            )
            result_text = await map_reduce_long_document(
                user_text, sub, model=model_name, num_gpu=num_gpu, temperature=body.temperature
            )
            step = 200
            for i in range(0, len(result_text), step):
                yield _sse_chunk(
                    content=result_text[i : i + step], model=model_name, cid=cid
                )
            yield _sse_chunk(content="", model=model_name, cid=cid, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_mapreduce(), media_type="text/event-stream")

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
                num_gpu=num_gpu, temperature=body.temperature,
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
            num_gpu=num_gpu, temperature=body.temperature,
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
    # Deduplicate while preserving order: fast → deep → vision → extras
    seen: set[str] = set()
    model_ids: list[str] = []
    for mid in [settings.fast_model, settings.deep_model, settings.vision_model,
                *settings.required_models]:
        if mid and mid not in seen:
            seen.add(mid)
            model_ids.append(mid)
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "docstack"}
            for mid in model_ids
        ],
    }
