# DocStack

Local-only **document ingest (PDF / DOCX / .doc / scans)** → **Chroma + CPU embeddings** → **Ollama** with an **Open WebUI**–compatible API (`/v1/chat/completions`).

See [PLAN.md](PLAN.md) for architecture, Windows/Mac setup, and Open WebUI wiring.

## Python version (important)

Use **Python 3.11 or 3.12** for the project virtualenv. **Python 3.13+** is not supported yet because the **Open WebUI** PyPI package caps below 3.13. This repo's `requires-python` is set to `>=3.11,<3.13` so one venv can install **docstack + Open WebUI** together.

## Quick start (Mac or Linux)

```bash
# System: Tesseract, Ghostscript, LibreOffice (for .doc), Ollama
brew install tesseract ghostscript ollama
brew install --cask libreoffice

# One venv: Python 3.12 (or 3.11)
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[webui]"

# Pull all Ollama model weights declared in configs/settings.toml
# (qwen3:4b for text/RAG, qwen3:8b for deep reasoning, qwen2.5vl for vision/images)
docstack models pull

# Or pull each model individually:
ollama pull qwen3:4b
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b      # vision model — better OCR/text reading than llava

# After changing [ingest] embedding_model: docstack wipe-index → restart API → re-ingest (see section below).

# Terminal 1
uvicorn docstack.api:app --host 0.0.0.0 --port 8000

# Terminal 2
open-webui serve --port 3000
```

Or run both from the repo: **`./start.sh`** (uses `.venv`). You must include **`./`** — plain `start.sh` is not on `PATH` and will not run.

Open **Open WebUI** at **[http://localhost:3000](http://localhost:3000)**.

### Wiring Open WebUI to DocStack (required after first run or after `wipe-chats`)

> ⚠️ Without this step, Open WebUI talks directly to Ollama (`http://localhost:11434`) and **bypasses DocStack entirely** — no RAG, no URL fetching, no document context.

1. Go to **Admin Panel → Settings → Connections**
2. Under **OpenAI API**, click the **gear icon** next to the URL
3. Change the URL to: **`http://127.0.0.1:8000/v1`**
4. Set any non-empty API key (e.g. `docstack`)
5. Save

**Ollama API** (`http://localhost:11434`) can stay connected — it's fine for general chat. But only models selected from the DocStack endpoint (`http://127.0.0.1:8000/v1`) go through RAG, URL fetching, and document context. Models accessed via Ollama direct bypass all of that.

After saving, the model dropdown will show `qwen3:4b`, `qwen3:8b`, `qwen2.5vl:7b` from DocStack — select one of these for full functionality.

#### If summaries look thin or you see "Retrieved 1 source"

That UI line usually comes from **Open WebUI's own attachment / retrieval** when chat is wired to **native Ollama** (`http://127.0.0.1:11434`) instead of **DocStack** (`http://127.0.0.1:8000/v1`). DocStack does not drive that counter.

1. Set **OpenAI API URL** to **`http://127.0.0.1:8000/v1`** (any non-empty API key is fine).
2. Turn **off** Open WebUI's built-in **Documents / RAG** for attachments so all context comes from DocStack's index.
3. **Ingest the PDF through DocStack** (upload at `:8000` or via the chat attachment with the Filter Function installed). A 40-page PDF should produce 40–60 chunks.
4. **Restart the API** after pulling DocStack updates.

For summarize-style prompts DocStack uses **full-context single-pass**: it packs as many chunks as fit in `max_ctx_chars` (default 30 000 characters) into the system prompt in one shot. This is fast on 4 GB GPU hardware and typically answers in under 60 seconds.

## Uploading documents

There are three ways to get a file into DocStack's index:

### Option A — Drag & drop uploader (easiest)

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser. You get a full drag-and-drop UI with:
- **Add to index** (default) — accumulate multiple documents
- **Replace all** — wipe and re-index with just this file
- Live progress bar while OCR/ingest runs
- Index size shown at all times

### Option B — Open WebUI chat attachment (seamless)

Install the DocStack Filter Function into Open WebUI **once** and file attachments in the chat box will automatically route through DocStack:

1. Open **Open WebUI → Admin Panel → Functions → + New Function**
2. Set type to **Filter**
3. Copy-paste the contents of [`scripts/openwebui_ingest_filter.py`](scripts/openwebui_ingest_filter.py)
4. **Save** and **enable** the function (globally or per-model)

After that: drag a PDF / DOCX / spreadsheet into any Open WebUI chat message, hit send, and DocStack indexes it automatically. The model is notified that the document is ready and will answer using RAG.

**Re-uploading the same file is instant** — DocStack fingerprints each file with SHA-256 and skips re-indexing if the content hasn't changed. To force a full re-index, wipe first (see Option C).

> **Images (JPG / PNG / etc.)** are *not* routed to DocStack — they pass through to the LLM as native image attachments. To use them you need a vision-capable model in Ollama (e.g. `ollama pull qwen2.5vl:7b`). To OCR-index an image document, use the drag-and-drop uploader at `http://localhost:8000` instead.

> **Valve options** (edit in Open WebUI → Functions → gear icon):
> `docstack_url` (default `http://localhost:8000`) · `replace_index` (default `false` = accumulate) · `poll_timeout_s` (default `300`)

### Option C — CLI / curl

```bash
docstack ingest-file ./doc.pdf      # replace mode (wipes previous index)
docstack models pull                # Ollama tags from [llm] required_models
docstack wipe-index                 # delete data/chroma + clear embedding cache
docstack wipe-chats                 # clear Open WebUI chat history + uploads
docstack wipe-all                   # full clean slate (index + chats + uploads + logs)
```

Add `-y` to any wipe command to skip the confirmation prompt:
```bash
docstack wipe-chats -y
docstack wipe-all -y
```

```bash
curl -F "file=@/path/to/doc.pdf" http://127.0.0.1:8000/ingest/add   # add to index
curl -F "file=@/path/to/doc.pdf" http://127.0.0.1:8000/ingest        # replace index
curl -X DELETE http://127.0.0.1:8000/ingest/wipe                     # wipe index + hash registry
```

#### What each wipe command clears

| Command | What it removes | Effect on performance |
|---|---|---|
| `wipe-index` | `data/chroma/` vector store + embedding cache | Removes stale docs from RAG |
| `wipe-chats` | Open WebUI chat history (`webui.db`) + uploaded files cache | Faster Open WebUI startup; no inference speedup |
| `wipe-all` | Everything above + logs | Full clean slate |

None of these affect Ollama model weights — those live in `~/.ollama/` and are managed with `ollama rm <model>`.

## Embeddings & vector DB (change model, reset index)

Do these **in order** whenever you change **`[ingest] embedding_model`** in [configs/settings.toml](configs/settings.toml) (different models use different vector sizes; old Chroma data is invalid).

1. **Edit** `embedding_model` in `configs/settings.toml` (default is multilingual mpnet for Nepali + English).
2. **Wipe vectors** (pick one):
   - **`docstack wipe-index`** (recommended — also clears the in-process embedding cache), or
   - **`curl -X DELETE http://127.0.0.1:8000/ingest/wipe`** while the API is running, or
   - Manually: **`rm -rf data/chroma`** then **`mkdir -p data/chroma`** (Mac/Linux), or delete the `data\chroma` folder on Windows.
3. **Restart the DocStack API** (stop `uvicorn` / `./start.sh` / `start.bat` windows, then start again) so **`get_settings()`** reloads TOML and the new SentenceTransformer loads on first embed.
4. **Re-ingest** every document (`/` upload form, **`POST /ingest`**, or **`docstack ingest-file …`**). The first run may download the new embedding weights.

**Secrets / gitignore:** `.webui_secret_key`, `.env`, and `data/chroma` are ignored from git — see [.gitignore](.gitignore). Open WebUI creates `.webui_secret_key` locally on first run; you do not copy it from the repo.

**Runtime behavior (short):** normal chat uses **RAG** (retrieval + capped history + optional "broad" spread chunks). **Whole-document summarize**-style prompts (English + Nepali patterns) trigger **full-context single-pass**: all chunks are packed (up to `max_ctx_chars`) into one system prompt and sent to the LLM in a single call. Map–reduce is available but disabled by default (see Configuration).

## Configuration

Edit [configs/settings.toml](configs/settings.toml). Paths are resolved relative to the repo root.
`libreoffice_path` empty string uses the default per OS (see `docstack.config`).

Defaults target **long Nepali government PDFs** (on the order of **~100 pages**): multilingual `[ingest] embedding_model`, larger `chunk_size` / `chunk_overlap`, and higher `[llm]` limits (`retrieval_top_k`, `rag_breadth_chunks`, `max_ctx_chars`, `num_ctx`). Broad-query detection includes **Nepali (Devanagari)** phrases (see `docstack.query.retriever`).

RAG knobs: **`rag_max_history_messages`**, **`rag_breadth_chunks`**, **`rag_min_hits_floor`** (minimum chunks for summarize-style intent when the index is larger), **`retrieval_min_score`**. If **`num_ctx`** causes GPU OOM on **4B**, lower it (e.g. 8192) in `settings.toml`.

**Full-context single-pass** (default for summarize/overview queries): when a query matches whole-document patterns (English + Nepali; see `is_mapreduce_eligible_query` in `workflow/mapreduce.py`), DocStack packs all indexed chunks up to **`full_context_max_chunks`** / **`max_ctx_chars`** (30 000 chars ≈ 14–16 Nepali chunks) into the system prompt in one call. This is the recommended path on 4 GB GPU hardware.

**Map–reduce** (optional, disabled by default): set **`mapreduce_enabled = true`** to enable hierarchical map+reduce over large document sets. Tune **`mapreduce_concurrency`**, **`mapreduce_reduce_batch`**, **`mapreduce_max_chunks`**. Output is also written to **`data/outputs/mapreduce_summary.txt`**. On 4 GB GPU, a 96-chunk document takes 10+ minutes — only enable on more powerful hardware.

### Automatically activating `.venv` in new terminals

- **Cursor / VS Code**: [`.vscode/settings.json`](.vscode/settings.json) points the Python extension at `.venv` and turns on **`python.terminal.activateEnvironment`**, so new **integrated** terminals opened with this folder as the workspace should activate `.venv` (Python extension required; reload the window if an old terminal was already open).
- **Terminal.app / iTerm / etc.**: Install **[direnv](https://direnv.net)** (`brew install direnv`), add `eval "$(direnv hook zsh)"` to `~/.zshrc`, restart the shell, then run **`direnv allow`** once in this repo (see [.envrc](.envrc)).

## Windows

### 1. Get the code

```bat
cd %USERPROFILE%\Projects
git clone git@github.com:hnd-c/local-llm.git
cd local-llm
```

### 2. Install system tools (one-time)

| Tool | Why | Where |
|------|-----|--------|
| **Ollama** | LLM server (`localhost:11434`) | https://ollama.com/download/windows — leave the tray app running |
| **Tesseract** | OCR / scans | See detailed steps below |
| **Ghostscript** | OCRmyPDF | See detailed steps below |
| **LibreOffice** | Legacy `.doc` | https://www.libreoffice.org/download/ — default path is fine (`soffice.exe` is auto-detected) |

#### Tesseract (OCR)

1. Go to https://github.com/UB-Mannheim/tesseract/wiki and download the latest **64-bit installer** (e.g. `tesseract-ocr-w64-setup-5.x.x.exe`).
2. Run the installer. On the **"Choose Install Location"** screen, note the path — default is:
   ```
   C:\Program Files\Tesseract-OCR
   ```
3. Complete the install (defaults are fine; you can optionally add extra language packs during install).
4. **Add to PATH:**
   - Press `Win + S`, search **"Environment Variables"**, click **"Edit the system environment variables"**.
   - Click **"Environment Variables…"** → under **System variables**, select **Path** → click **Edit**.
   - Click **New** and add:
     ```
     C:\Program Files\Tesseract-OCR
     ```
   - Click **OK** on all dialogs.
5. **Verify** — open a new Command Prompt and run:
   ```bat
   tesseract --version
   ```
   You should see the version number printed.

#### Ghostscript (PDF rendering / OCRmyPDF)

1. Go to https://www.ghostscript.com/releases/gsdnld.html and download **Ghostscript AGPL Release** → **64-bit Windows** (e.g. `gs10xx w64.exe`).
2. Run the installer. Default install path is something like:
   ```
   C:\Program Files\gs\gs10.xx.x
   ```
3. **Add the `bin` folder to PATH:**
   - Press `Win + S`, search **"Environment Variables"** → **Edit the system environment variables** → **Environment Variables…**
   - Under **System variables**, select **Path** → **Edit** → **New**, then add (adjust version number):
     ```
     C:\Program Files\gs\gs10.05.1\bin
     ```
   - Click **OK** on all dialogs.
4. **Verify** — open a new Command Prompt and run:
   ```bat
   gswin64c --version
   ```
   You should see output like `10.05.1`. If it says "not recognized", double-check the path added matches the actual folder on disk.

### NVIDIA Driver Installation (GTX 1050 Ti — required for GPU acceleration)

Without the driver, Ollama runs everything on CPU and is very slow. Follow these steps once:

**Step 1 — Download the correct driver**

Go to https://www.nvidia.com/Download/index.aspx and fill in:

| Field | Value |
|-------|-------|
| Product Type | GeForce |
| Product Series | GeForce 10 Series |
| Product | GeForce GTX 1050 Ti |
| Operating System | Windows 10 64-bit / Windows 11 |
| Download Type | Game Ready Driver (GRD) |
| Language | English |

Click **Search → Download**. The installer will be a `.exe` file (~500 MB).

**Step 2 — Install**

1. Run the downloaded `.exe` as Administrator.
2. Choose **Express Installation** (installs driver + PhysX + HD Audio).
3. Restart when prompted.

**Step 3 — Restart the PC**

A full reboot is required after driver installation. Ollama will not detect the GPU until the system has restarted.

**Step 4 — Verify the driver**

Open a new **Command Prompt** and run:
```bat
nvidia-smi
```
You should see your GPU listed with driver version and memory usage.

**Step 5 — No code changes needed**

Ollama automatically detects and uses the NVIDIA GPU via CUDA — nothing needs to change in the codebase or config. Simply start Ollama (tray app or `ollama serve`) and run `start.bat` as normal.

**Step 6 — Confirm Ollama is using the GPU**

Run a model, then in a separate Command Prompt run `nvidia-smi`. You should see `ollama` or `ollama_llama_se` listed under **Processes** with non-zero GPU memory:
```bat
ollama run qwen3:4b "hello"
```
Or open **Task Manager → Performance → GPU** — GPU utilisation should spike during inference.

> **VRAM note (4 GB):** `qwen3:4b` (~3.5 GB) fits fully on GPU and runs fast. `qwen3:8b` and `qwen2.5vl` (~5–6 GB) exceed 4 GB and will partially offload layers to CPU — they will still be faster than pure CPU but slower than a full GPU fit.

More detail: [PLAN.md — Windows Deployment](PLAN.md#windows-deployment-native--no-docker-no-wsl2-required).

### 3. Python venv and Python packages

Install **Python 3.12** (python.org or Microsoft Store) so `py -3.12` works, then in **Command Prompt** or **PowerShell** inside the repo:

```bat
rmdir /s /q .venv 2>nul
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[webui]"

REM Pull all Ollama model weights (qwen3:4b, qwen3:8b, qwen2.5vl:7b)
docstack models pull

REM Or pull each model individually:
REM ollama pull qwen3:4b
REM ollama pull qwen3:8b
REM ollama pull qwen2.5vl:7b
```

After you change **`[ingest] embedding_model`**, run **`docstack wipe-index`**, restart the API, and re-ingest (same steps as the **Embeddings & vector DB** section above).

### 4. Run everything

From the repo folder, double-click **`start.bat`** or run:

```bat
start.bat
```

That checks **Ollama** (and can start `ollama serve` in a minimized window if needed), then opens two windows: **DocStack** on port **8000** and **Open WebUI** on **3000**. Windows **10+** includes **`curl`**, which `start.bat` uses.

If `py -3.12` is missing, install Python 3.12 or edit [start.bat](start.bat) / use `py -3.11` consistently with [pyproject.toml](pyproject.toml) (`>=3.11,<3.13`).

### 5. Same wiring as Mac

- **Open WebUI** at **[http://localhost:3000](http://localhost:3000)**: Settings → Connections → **OpenAI API URL** `http://127.0.0.1:8000/v1` (or `http://localhost:8000/v1`), API key any non-empty string.
- **Index files** via [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (Option A) or install the Filter Function for seamless chat-box uploads (Option B above).

## License

MIT (project scaffold — adjust as needed).
