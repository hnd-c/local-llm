"""
DocStack Auto-Ingest Filter for Open WebUI
==========================================
How to install:
  1. Open WebUI → Admin Panel → Functions → ＋ New Function
  2. Paste this entire file, give it a name (e.g. "DocStack Ingest")
  3. Save, then toggle it ON in the Functions list

What it does:
  When you attach a PDF / DOCX / DOC to a chat message and send it,
  this filter intercepts the upload, sends the file bytes to DocStack's
  /ingest/add endpoint, waits for OCR + indexing to finish, then
  notifies the model that the document is ready. You see live status
  updates ("Indexing page 3/12…") in the chat while it runs.

Valves (configure in Open WebUI → Functions → gear icon):
  docstack_url   – base URL of the DocStack API (default: http://localhost:8000)
  replace_index  – wipe existing index on each upload? (default: False = accumulate)
  poll_timeout_s – max seconds to wait for indexing (default: 300)
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import requests
from pydantic import BaseModel, Field

# Tells Open WebUI: this filter owns file handling.
# Prevents Open WebUI from running its own (OCR-unaware) document extraction,
# which would throw "The content provided is empty" on scanned PDFs.
file_handler = True


class Filter:
    class Valves(BaseModel):
        docstack_url: str = Field(
            default="http://localhost:8000",
            description="DocStack API base URL (no trailing slash)",
        )
        replace_index: bool = Field(
            default=False,
            description="Wipe existing index before each upload (False = accumulate docs)",
        )
        poll_timeout_s: int = Field(
            default=300,
            description="Max seconds to wait for indexing before giving up (scanned PDFs can take 60–120s)",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _owui_upload_dir() -> str:
        """Return Open WebUI's uploads directory (works regardless of DATA_DIR env var)."""
        try:
            import open_webui.env as _e
            import os as _os
            # DATA_DIR is set by Open WebUI at runtime; uploads live inside it
            upload_dir = _os.path.join(str(_e.DATA_DIR), "uploads")
            if _os.path.isdir(upload_dir):
                return upload_dir
        except Exception:
            pass
        return ""

    def _download_file(
        self,
        file_info: dict,
        request: Any,
        user: Optional[dict],
    ) -> tuple[str, bytes]:
        """Read file bytes from Open WebUI's local uploads directory.
        Falls back to HTTP download if the local path cannot be found."""
        import glob as _glob
        import os as _os

        file_id = file_info.get("id", "")
        name = (
            file_info.get("name")
            or file_info.get("filename")
            or file_info.get("id", "upload.bin")
        )

        # ── Option 1: read directly from disk (fast, no auth needed) ────────
        upload_dir = self._owui_upload_dir()
        if upload_dir and file_id:
            # Open WebUI stores as: {uuid}_{original_filename}
            pattern = _os.path.join(upload_dir, f"{file_id}_*")
            matches = _glob.glob(pattern)
            if matches:
                local_path = matches[0]
                return name, open(local_path, "rb").read()

        # ── Option 2: HTTP download from Open WebUI's content endpoint ───────
        url_path = file_info.get("url", "")
        if not url_path and file_id:
            url_path = f"/api/v1/files/{file_id}/content"
        if url_path:
            try:
                base = str(request.base_url).rstrip("/")
            except Exception:
                base = "http://localhost:3000"
            full_url = base + url_path if url_path.startswith("/") else url_path
            token = (user or {}).get("token", "")
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            resp = requests.get(full_url, headers=headers, timeout=120)
            resp.raise_for_status()
            return name, resp.content

        raise ValueError(f"Cannot find file on disk or via HTTP for: {name} (id={file_id})")

    def _start_ingest(self, name: str, data: bytes) -> str:
        """POST file to DocStack; return job_id."""
        endpoint = "/ingest" if self.valves.replace_index else "/ingest/add"
        url = self.valves.docstack_url.rstrip("/") + endpoint
        resp = requests.post(url, files={"file": (name, data)}, timeout=30)
        resp.raise_for_status()
        return resp.json().get("job_id", "")

    def _poll_status(self) -> dict:
        url = self.valves.docstack_url.rstrip("/") + "/ingest/status"
        try:
            return requests.get(url, timeout=10).json()
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Open WebUI inlet hook (runs before message reaches the LLM)
    # ------------------------------------------------------------------

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__: Optional[Any] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        async def emit(description: str, done: bool = False) -> None:
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": description, "done": done}}
                )

        # Files come through body["files"] in Open WebUI
        raw_files = body.get("files", [])
        if not raw_files:
            return body

        # Filter to document types DocStack supports
        _SUPPORTED_EXT = {
            "pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt",
            "txt", "md", "markdown", "csv", "html", "htm",
            "jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp",
            "odt", "ods", "odp", "rtf",
        }
        doc_files = [
            f for f in raw_files
            if (f.get("name") or f.get("filename") or "").lower().rsplit(".", 1)[-1]
            in _SUPPORTED_EXT
        ]
        if not doc_files:
            return body

        ingested: list[str] = []
        errors: list[str] = []

        ingested: list[str] = []
        errors: list[str] = []

        for file_info in doc_files:
            name = file_info.get("name") or file_info.get("filename") or "document"

            # 1. Download
            await emit(f"⬇️  Downloading {name}…")
            try:
                fname, data = self._download_file(file_info, __request__, __user__)
            except Exception as e:
                errors.append(f"{name}: download failed — {e}")
                continue

            # 2. Start ingest (non-blocking on DocStack side — runs in bg thread)
            await emit(f"📤 Sending {fname} to DocStack for OCR + indexing…")
            try:
                self._start_ingest(fname, data)
            except Exception as e:
                errors.append(f"{fname}: upload failed — {e}")
                continue

            # 3. Poll until done — LLM response is held until index is complete
            deadline = asyncio.get_event_loop().time() + self.valves.poll_timeout_s
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(2)
                s = self._poll_status()
                status = s.get("status", "")
                if status == "completed":
                    ingested.append(fname)
                    await emit(f"✅ Indexed {fname} — generating answer…", done=True)
                    break
                if status == "failed":
                    errors.append(f"{fname}: {s.get('error', 'unknown error')}")
                    await emit(f"❌ Indexing failed for {fname}", done=True)
                    break
                if status == "running":
                    page = s.get("pages_done", 0)
                    total = s.get("pages_total", "?")
                    elapsed = int(s.get("elapsed_s") or 0)
                    await emit(
                        f"⏳ OCR + indexing {fname} — page {page}/{total} · {elapsed}s elapsed…"
                    )
            else:
                errors.append(f"{fname}: timed out after {self.valves.poll_timeout_s}s")
                await emit(f"⚠️ Timed out indexing {fname}", done=True)

        # ── Strip file parts from user message content ──────────────────────
        # Even with file_handler=True, the file reference stays inside the
        # user message content list and triggers Open WebUI's own RAG template
        # (which injects <source id="1"> metadata with no actual text).
        # Remove non-text parts so Open WebUI's RAG template never fires.
        for m in body.get("messages", []):
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                text_parts = [p for p in m["content"] if p.get("type") == "text"]
                if text_parts:
                    m["content"] = text_parts[0].get("text", "") if len(text_parts) == 1 else \
                        "\n".join(p.get("text", "") for p in text_parts)
                else:
                    m["content"] = ""

        # Inject context note so the LLM answers using the freshly indexed doc
        notes: list[str] = []
        if ingested:
            notes.append(
                "[DocStack] The following document(s) are fully indexed and ready:\n"
                + "\n".join(f"  • {d}" for d in ingested)
                + "\n\nAnswer the user's question using the indexed document content. "
                "If the document is in Nepali, reply in Nepali unless asked otherwise. "
                "Be comprehensive and structured."
            )
        if errors:
            notes.append("[DocStack] Errors:\n" + "\n".join(f"  • {e}" for e in errors))

        if notes:
            existing = [m for m in body.get("messages", []) if m.get("role") != "system"]
            body["messages"] = [{"role": "system", "content": "\n\n".join(notes)}] + existing

        return body
