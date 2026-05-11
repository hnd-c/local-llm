# Local Document RAG Stack with ChatGPT-like Interface

## Overview

A fully local, offline document processing and reasoning system that looks and feels like
ChatGPT/Claude. Users upload PDFs, scanned docs, or DOCX files, type a question, and get a
streamed answer with citations — all running on your own machine.

**Interface**: Open WebUI (free, local, ChatGPT-clone)
**LLM**: Qwen3 4B Q4 on GPU (fast) / Qwen3 8B on CPU (deep jobs) via Ollama
**RAG**: Custom Python pipeline — OCR, chunking, Chroma vector DB
**Glue**: FastAPI middleware (OpenAI-compatible) connecting Open WebUI → our RAG → Ollama

---

## Hardware Context

| Spec | Value |
|------|-------|
| CPU | Intel i7-11700 |
| RAM | 32 GB |
| GPU | GTX 1050 Ti — 4 GB VRAM |
| OS | Windows (WSL2 Ubuntu recommended) |

---

## How It Looks to the User

```
┌──────────────────────────────────────────────────────┐
│              Open WebUI  (localhost:3000)             │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  📎 Upload Document    [Choose file...]      │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  What are the key obligations in section 3? │    │
│  │                                    [Send ▶] │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ╔═════════════════════════════════════════════╗    │
│  ║  Based on section 3 (page 4 of contract.pdf)║    │
│  ║  the key obligations are...  ▌              ║    │
│  ╚═════════════════════════════════════════════╝    │
└──────────────────────────────────────────────────────┘
```

---

## Full System Architecture

```
User browser
     │
     │  http://localhost:3000
     ▼
┌─────────────┐
│  Open WebUI │  ← ChatGPT-like UI, conversation history,
│  (Docker)   │    streaming display, file upload button
└──────┬──────┘
       │
       │  OpenAI-compatible  POST /v1/chat/completions
       │  (Open WebUI thinks this is an OpenAI API)
       ▼
┌──────────────────────────┐
│  RAG Middleware           │  ← our FastAPI app
│  (localhost:8000)         │    intercepts every message,
│                           │    runs retrieval, builds prompt
│  1. parse user message    │
│  2. retrieve top-k chunks │
│  3. inject as context     │
│  4. forward to Ollama     │
│  5. stream response back  │
└──────┬───────────────────┘
       │
       ├──────────────────────────────────────┐
       │  retrieve                            │ generate
       ▼                                      ▼
┌─────────────┐                    ┌─────────────────────┐
│   Chroma DB │                    │  Ollama              │
│  (local)    │                    │  localhost:11434     │
└─────────────┘                    │                      │
                                   │  qwen3:4b  (GPU)     │
                                   │  qwen3:8b  (CPU)     │
                                   └─────────────────────┘

Separate ingest flow (triggered on file upload):

User uploads file
     │
     ▼
┌─────────────────────────────────┐
│  Ingest API  POST /ingest       │  ← our FastAPI endpoint
│  (same FastAPI app, port 8000)  │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────┐   ┌───────────────┐   ┌─────────────┐
│  OCR/Parse  │ → │    Chunker    │ → │  Embedder   │
│  router     │   │  800-1500 ch  │   │  CPU only   │
└─────────────┘   └───────────────┘   └──────┬──────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │   Chroma DB │
                                      └─────────────┘
```

---

## Components

### 1. Open WebUI (interface layer)

- Install via Docker: `docker run -d -p 3000:80 ghcr.io/open-webui/open-webui`
- Point its **OpenAI API base URL** to `http://localhost:8000/v1`
- Users get: conversation threads, streaming tokens, file uploads, markdown rendering, dark mode
- No code to write here — just configure it to talk to our middleware

---

### 2. RAG Middleware — FastAPI (`src/docstack/api.py`)

This is the core custom code. It is an **OpenAI-compatible API** that Open WebUI talks to.

**Endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Intercept chat message → RAG → Ollama → stream back |
| `GET /v1/models` | Return model list (required by Open WebUI to populate dropdown) |
| `POST /ingest` | Accept file → wipe DB → start background ingest job → return `job_id` immediately |
| `GET /ingest/status` | Poll ingest progress: `{status, pages_done, pages_total, active_doc, elapsed_s}` |

**Chat flow inside `/v1/chat/completions`:**

```
1. Check ingest status → if still indexing, stream back:
   "⏳ Still indexing your document (page 4/20). Please wait a moment."
   and return early.
2. Extract last user message
3. Retrieve top-k chunks from Chroma (default k=8)
4. If max retrieval score < 0.4 → answer from model knowledge (no RAG)
5. Build system prompt with retrieved chunks + citation instruction
6. Decide tier: deep task keywords → qwen3:8b; else → qwen3:4b
7. Forward to Ollama /api/chat with stream=True
8. Stream tokens back to Open WebUI in SSE format
9. Append citations block at end of response
```

**Ingest timing expectations (i7-11700):**

| Document type | Typical wait |
|---------------|-------------|
| Text-native PDF or DOCX | 5–12 seconds |
| Scanned PDF (10 pages) | 35–85 seconds |
| Scanned PDF (30 pages) | 2–4 minutes |

The upload returns immediately with a `job_id`. The background worker processes pages and updates
a shared status object. The chat endpoint checks this status on every request so the user always
gets a clear "still indexing" message rather than an empty or wrong answer.

---

### 3. Document Ingest Pipeline (`src/docstack/ingest/`)

| File type | Tool | Quality gate |
|-----------|------|-------------|
| PDF (text layer present) | PyMuPDF (`fitz`) | char count per page > 50 |
| PDF (scanned / image) | OCRmyPDF + Tesseract | triggered when char count ≤ 50 |
| Hard scan (skew/noise) | OpenCV deskew → Tesseract | OCR confidence < 70% → flag review |
| DOCX | python-docx | paragraphs + tables |
| Legacy .doc | LibreOffice headless → DOCX | subprocess; fail loudly if not installed |

**`DocumentRecord` dataclass** (internal schema):

```python
@dataclass
class DocumentRecord:
    doc_id: str        # hash(source_path + mtime)
    source_path: str
    mime: str          # "pdf" | "docx" | "doc"
    page: int
    block_type: str    # "text" | "table" | "heading"
    text: str
    bbox: tuple | None
    ocr_confidence: float | None  # None if not OCR'd
```

---

### 4. Chunking (`src/docstack/chunk/chunker.py`)

- Size: 800–1,500 characters with ~12% overlap
- Metadata per chunk: `doc_id`, `page`, `section_heading`, `source_filename`, `chunk_index`, `table_id`
- Tables serialised as Markdown inside chunk text
- Heading detection: font size (PDF) or docx styles

---

### 5. Embeddings + Vector Store (`src/docstack/index/`)

- Model: `BAAI/bge-small-en-v1.5` — CPU, ~130 MB, pinned version
- Store: Chroma (embedded, no server) persisted to `data/chroma/`
- **Document lifecycle — Replace mode**: every new upload **wipes the entire Chroma collection** and re-indexes only the newly uploaded document. There is always exactly one "active document" in the DB at any time.
  - On upload: `collection.delete()` → re-create → ingest new file
  - This keeps the system simple and queries always apply to the current document only
  - No per-document tracking or idempotency logic needed

---

### 6. LLM Routing (`src/docstack/workflow/router.py`)

```
user message contains: compare / audit / analyse / find differences / summarise all
    → qwen3:8b on CPU (slow, deeper)

everything else
    → qwen3:4b on GPU (fast, default)
```

---

### 7. Output Schemas (`src/docstack/workflow/schemas.py`)

All LLM JSON outputs validated by Pydantic v2 before any downstream action:

| Schema | Key fields |
|--------|-----------|
| `QAResult` | answer, citations (doc, page, chunk_id), low_confidence flag |
| `ExtractedEntities` | entities list (name, type, value, source_chunk) |
| `SummaryBullets` | bullets, source_doc, review_required |
| `ActionItems` | items (text, owner, due_date), source_doc |

---

### 8. Map-Reduce for Long Corpora (`src/docstack/workflow/mapreduce.py`)

For large document sets (e.g. summarise a folder of 20 contracts):

```
Step 1 — Map:    per-chunk summaries in parallel → qwen3:4b
Step 2 — Reduce: merge chunk summaries → qwen3:4b (or 8b for formal output)
Output:          SummaryBullets JSON, validated, written to data/outputs/
```

---

## Project Layout

```
local-llm/
├── PLAN.md
├── pyproject.toml
├── README.md
├── start.bat                   # one-click launcher for both services
├── configs/
│   └── settings.toml           # chunk_size, model names, ollama_url, paths
├── data/
│   ├── chroma/                 # vector DB (gitignored)
│   ├── uploads/                # raw uploaded files (gitignored)
│   └── outputs/                # JSON results, summaries (gitignored)
├── src/
│   └── docstack/
│       ├── __init__.py
│       ├── api.py              # FastAPI app (middleware + ingest endpoints)
│       ├── config.py           # pydantic-settings
│       ├── ingest/
│       │   ├── router.py       # detect mime → right loader
│       │   ├── pdf_loader.py   # PyMuPDF + OCRmyPDF fallback
│       │   ├── docx_loader.py  # python-docx
│       │   └── ocr.py          # OCRmyPDF wrapper + quality gate
│       ├── chunk/
│       │   └── chunker.py
│       ├── index/
│       │   ├── embedder.py     # sentence-transformers CPU
│       │   └── store.py        # Chroma wrapper + idempotency
│       ├── query/
│       │   ├── ollama_client.py  # streaming Ollama calls
│       │   ├── retriever.py
│       │   └── prompts.py
│       └── workflow/
│           ├── schemas.py
│           ├── router.py       # 4B vs 8B decision
│           └── mapreduce.py
└── tests/
```

---

## Windows Deployment (Native — no Docker, no WSL2 required)

### Why native Windows (not WSL2 or Docker)

- Ollama has a native Windows installer with CUDA support out of the box
- Open WebUI can be installed as a Python package (`pip install open-webui`) — no Docker needed
- Tesseract and Ghostscript have official Windows installers
- Everything runs as normal Windows processes; no virtualisation overhead

---

### Step-by-step setup (one-time)

**Step 1 — NVIDIA driver**
Ensure you have a recent NVIDIA driver (≥ 527.x) so Ollama can use the 1050 Ti.
Check: `nvidia-smi` in Command Prompt should show the GPU.

**Step 2 — Ollama**
Download and run the installer from https://ollama.com/download/windows
After install, Ollama runs as a system tray app and serves at `http://localhost:11434`.
```
ollama pull qwen3:4b
ollama pull qwen3:8b
```

**Step 3 — Tesseract (required for OCR)**
Download the UB-Mannheim build:
https://github.com/UB-Mannheim/tesseract/wiki
- Install to default path: `C:\Program Files\Tesseract-OCR\`
- During install, tick "Add to PATH" or add manually:
  System Properties → Environment Variables → PATH → add `C:\Program Files\Tesseract-OCR`
- Verify: open a new terminal and run `tesseract --version`

**Step 4 — Ghostscript (required for OCRmyPDF)**
Download from https://www.ghostscript.com/releases/gsdnld.html
- Install 64-bit version to default path: `C:\Program Files\gs\`
- Add `C:\Program Files\gs\gs<version>\bin` to PATH
- Verify: `gswin64c --version`

**Step 5 — LibreOffice (for legacy .doc files)**
Download from https://www.libreoffice.org/download/
- Install to default: `C:\Program Files\LibreOffice\`
- No PATH change needed; our code calls it at the full path:
  `C:\Program Files\LibreOffice\program\soffice.exe`

**Step 6 — Python environment**
Use Miniconda (recommended) from https://docs.conda.io/en/latest/miniconda.html
```
conda create -n docstack python=3.10 -y
conda activate docstack
cd C:\path\to\local-llm
pip install -e .
```

**Step 7 — Start the RAG middleware**
```
conda activate docstack
cd C:\path\to\local-llm
uvicorn docstack.api:app --host 0.0.0.0 --port 8000 --reload
```
Keep this terminal open. Middleware is now at `http://localhost:8000`.

**Step 8 — Start Open WebUI**
In a second terminal:
```
conda activate docstack
open-webui serve --port 3000
```
Open WebUI is now at `http://localhost:3000`.

**Step 9 — Connect Open WebUI to our middleware**
Open `http://localhost:3000` in your browser:
- Go to **Settings → Admin → Connections**
- Set **OpenAI API URL** to: `http://localhost:8000/v1`
- Set **API Key** to: `local` (any non-empty string)
- Save → the model dropdown will show `qwen3:4b` and `qwen3:8b`
- **Disable** the built-in RAG (Settings → Documents → toggle off) so all
  document handling goes through our pipeline

**Step 10 — Verify**
Upload any PDF in the chat and ask a question about it.

---

### Running day-to-day (after first-time setup)

Two terminals, two commands:
```
# Terminal 1
conda activate docstack && uvicorn docstack.api:app --host 0.0.0.0 --port 8000

# Terminal 2
conda activate docstack && open-webui serve --port 3000
```
Ollama starts automatically at Windows login (system tray).

Optionally both can be wrapped in a single `.bat` launcher:
```batch
@echo off
start "DocStack API" cmd /k "conda activate docstack && uvicorn docstack.api:app --port 8000"
start "Open WebUI"   cmd /k "conda activate docstack && open-webui serve --port 3000"
```

---

### Windows-specific notes in source code

| Issue | How we handle it |
|-------|-----------------|
| File paths | Use `pathlib.Path` everywhere — no hardcoded `/` separators |
| LibreOffice path | Read from `configs/settings.toml`: `libreoffice_path = "C:/Program Files/LibreOffice/program/soffice.exe"` |
| Tesseract path | OCRmyPDF finds it automatically if it is on PATH; fallback config key available |
| Background worker | Use `threading.Thread` (not `multiprocessing`) — avoids Windows `spawn` issues |
| Temp files | Use `tempfile.NamedTemporaryFile(delete=False)` pattern; clean up explicitly |

---

## Mac Development Setup (test here, deploy to Windows)

### Mac setup (one-time)

**Step 1 — Homebrew deps**
```bash
brew install tesseract ghostscript
brew install --cask libreoffice   # or download from libreoffice.org
```
Verify:
```bash
tesseract --version
gswin64c --version || gs --version   # Mac uses "gs", Windows uses "gswin64c"
```

**Step 2 — Ollama**
```bash
brew install ollama
ollama serve &          # starts API on localhost:11434
ollama pull qwen3:4b
ollama pull qwen3:8b
```
On Mac, Ollama uses Metal (Apple GPU) or CPU — same API, same speed characteristics for testing.

**Step 3 — Python environment**
```bash
conda create -n docstack python=3.10 -y
conda activate docstack
cd /path/to/local-llm
pip install -e .
```

**Step 4 — Start services**
```bash
# Terminal 1
conda activate docstack && uvicorn docstack.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
conda activate docstack && open-webui serve --port 3000
```

**Step 5 — Configure Open WebUI** (identical to Windows)
- Open `http://localhost:3000`
- Settings → Connections → OpenAI API URL: `http://localhost:8000/v1`
- Disable built-in RAG under Settings → Documents

Or use the Mac launcher script:
```bash
./start.sh
```

---

## Cross-Platform Code Design

This is the central principle: **all OS-specific values live in config, not in code.**
The Python source is identical on Mac and Windows. Only `settings.toml` changes.

### `configs/settings.toml`

```toml
[llm]
fast_model   = "qwen3:4b"
deep_model   = "qwen3:8b"
ollama_url   = "http://localhost:11434"
num_ctx      = 4096
max_ctx_chars = 12000

[ingest]
chunk_size   = 1200
chunk_overlap = 150
min_chars_per_page = 50      # below this triggers OCR
ocr_confidence_threshold = 70

[paths]
data_dir     = "data"
chroma_dir   = "data/chroma"
uploads_dir  = "data/uploads"
outputs_dir  = "data/outputs"

# Mac default (auto-detected if blank)
libreoffice_path = ""

# Windows: uncomment and set this when deploying
# libreoffice_path = "C:/Program Files/LibreOffice/program/soffice.exe"
```

### `src/docstack/config.py` — auto-detects OS defaults

```python
import platform
import sys
from pathlib import Path
from pydantic_settings import BaseSettings

def _default_libreoffice() -> str:
    if sys.platform == "win32":
        return r"C:\Program Files\LibreOffice\program\soffice.exe"
    elif sys.platform == "darwin":
        return "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    return "soffice"   # Linux / fallback — must be on PATH

class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    fast_model: str = "qwen3:4b"
    deep_model: str = "qwen3:8b"
    num_ctx: int = 4096
    max_ctx_chars: int = 12000
    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chars_per_page: int = 50
    ocr_confidence_threshold: int = 70
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/chroma")
    uploads_dir: Path = Path("data/uploads")
    outputs_dir: Path = Path("data/outputs")
    libreoffice_path: str = _default_libreoffice()

    model_config = {"toml_file": "configs/settings.toml"}
```

### Launcher scripts (two files, same commands)

**`start.sh`** (Mac / Linux):
```bash
#!/bin/bash
conda activate docstack
uvicorn docstack.api:app --host 0.0.0.0 --port 8000 &
open-webui serve --port 3000
```

**`start.bat`** (Windows):
```batch
@echo off
start "DocStack API" cmd /k "conda activate docstack && uvicorn docstack.api:app --port 8000"
start "Open WebUI"   cmd /k "conda activate docstack && open-webui serve --port 3000"
```

### What changes when switching Mac → Windows

| What | Mac value | Windows value |
|------|-----------|---------------|
| `libreoffice_path` in `settings.toml` | *(leave blank, auto-detected)* | `C:/Program Files/LibreOffice/program/soffice.exe` |
| Ghostscript binary name | `gs` | `gswin64c` (OCRmyPDF finds it automatically if on PATH) |
| Launcher | `./start.sh` | `start.bat` |
| Ollama GPU backend | Metal | CUDA (1050 Ti) |
| **Python source code** | **unchanged** | **unchanged** |

Switching is literally: edit one line in `settings.toml` (or leave it blank for auto-detection), run `start.bat` instead of `start.sh`.

---

## Python Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | RAG middleware API |
| `httpx` | Async Ollama streaming calls |
| `pymupdf` | PDF text extraction |
| `pdfplumber` | Table detection in PDFs |
| `ocrmypdf` | OCR pipeline wrapper (Tesseract) |
| `python-docx` | DOCX parsing |
| `opencv-python-headless` | Scan pre-processing |
| `sentence-transformers` | CPU embeddings |
| `chromadb` | Local vector DB |
| `pydantic>=2` | Output schemas + validation |
| `pydantic-settings` | Config from TOML/env |
| `typer` | Optional CLI |

---

## Implementation Milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| 1 | Ingest pipeline | pdf_loader, docx_loader, ocr wrapper, DocumentRecord |
| 2 | Chunk + index | chunker, embedder, Chroma store with idempotency |
| 3 | RAG middleware | FastAPI with `/v1/chat/completions` streaming + `/ingest` |
| 4 | Open WebUI wired up | docker-compose, settings configured, end-to-end chat working |
| 5 | Workflow JSON | Pydantic schemas, task router, citation block in responses |
| 6 | Map-reduce | Folder-level summarisation via deep tier |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Scanned tables / handwriting missed by OCR | Flag `review_required: true` in output; log OCR confidence |
| VRAM spike from large context on 4B | Cap `num_ctx`; byte-budget retrieved text to ~6k tokens |
| Legacy .doc conversion fails | Fail loudly with clear log; mark file `ingest_status: failed` |
| Open WebUI upload bypasses our OCR pipeline | Disable Open WebUI's built-in RAG; all ingestion goes through `/ingest` endpoint |
| User uploads new doc mid-conversation | DB wipe is immediate; old conversation context is gone — expected behaviour in Replace mode |
| Retrieval misses relevant chunks | Log low scores; tune chunk size and overlap iteratively |
