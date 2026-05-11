# DocStack

Local-only **document ingest (PDF / DOCX / .doc / scans)** → **Chroma + CPU embeddings** → **Ollama** with an **Open WebUI**–compatible API (`/v1/chat/completions`).

See [PLAN.md](PLAN.md) for architecture, Windows/Mac setup, and Open WebUI wiring.

## Python version (important)

Use **Python 3.11 or 3.12** for the project virtualenv. **Python 3.13+** is not supported yet because the **Open WebUI** PyPI package caps below 3.13. This repo’s `requires-python` is set to `>=3.11,<3.13` so one venv can install **docstack + Open WebUI** together.

## Quick start (Mac or Linux)

```bash
# System: Tesseract, Ghostscript, LibreOffice (for .doc), Ollama
brew install tesseract ghostscript ollama
brew install --cask libreoffice

# After pip install (below), pull Ollama weights declared in configs/settings.toml:
#   docstack models pull

# One venv: Python 3.12 (or 3.11)
rm -rf .venv .venv-webui
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[webui]"

docstack models pull

# Terminal 1
uvicorn docstack.api:app --host 0.0.0.0 --port 8000

# Terminal 2
open-webui serve --port 3000
```

Or run both from the repo: **`./start.sh`** (uses `.venv`). You must include **`./`** — plain `start.sh` is not on `PATH` and will not run.

In **Open WebUI**: Settings → Connections → OpenAI API URL: `http://127.0.0.1:8000/v1` — disable built-in document RAG.

**Index a document** (replace mode — wipes previous index):

- Use the form at [http://127.0.0.1:8000/](http://127.0.0.1:8000/), or  
- `curl -F "file=@/path/to/doc.pdf" http://127.0.0.1:8000/ingest`

**Important:** Open WebUI’s in-chat file attachment does **not** call this repo’s `/ingest` or OCR pipeline. Index files through **`/ingest`** (form or `curl`) first, then ask questions in Open WebUI. Optionally disable Open WebUI’s built-in document/RAG features so all answers use your indexed Chroma store.

**CLI**

```bash
docstack ingest-file ./sample.pdf
docstack serve --port 8000
docstack models pull   # sync Ollama tags with [llm] required_models in settings.toml
```

## Configuration

Edit [configs/settings.toml](configs/settings.toml). Paths are resolved relative to the repo root.  
`libreoffice_path` empty string uses the default per OS (see `docstack.config`).

### Automatically activating `.venv` in new terminals

- **Cursor / VS Code**: [`.vscode/settings.json`](.vscode/settings.json) points the Python extension at `.venv` and turns on **`python.terminal.activateEnvironment`**, so new **integrated** terminals opened with this folder as the workspace should activate `.venv` (Python extension required; reload the window if an old terminal was already open).
- **Terminal.app / iTerm / etc.**: Install **[direnv](https://direnv.net)** (`brew install direnv`), add `eval "$(direnv hook zsh)"` to `~/.zshrc`, restart the shell, then run **`direnv allow`** once in this repo (see [.envrc](.envrc)).

## Windows

### 1. Get the code

```bat
cd %USERPROFILE%\Projects
git clone <YOUR_REPO_URL>
cd local-llm
```

(Replace `<YOUR_REPO_URL>` with your Git remote, or unzip the project into a folder and `cd` there.)

### 2. Install system tools (one-time)

| Tool | Why | Where |
|------|-----|--------|
| **Ollama** | LLM server (`localhost:11434`) | https://ollama.com/download/windows — leave the tray app running |
| **Tesseract** | OCR / scans | https://github.com/UB-Mannheim/tesseract/wiki — add install folder to **PATH** |
| **Ghostscript** | OCRmyPDF | https://www.ghostscript.com/releases/gsdnld.html — add `...\gs\bin` to **PATH**, verify with `gswin64c --version` |
| **LibreOffice** | Legacy `.doc` | https://www.libreoffice.org/download/ — default path is fine (`soffice.exe` is auto-detected) |

Optional: **NVIDIA driver** so Ollama can use the GPU (`nvidia-smi` in a new Command Prompt).

More detail: [PLAN.md — Windows Deployment](PLAN.md#windows-deployment-native--no-docker-no-wsl2-required).

### 3. Python venv and Python packages

Install **Python 3.12** (python.org or Microsoft Store) so `py -3.12` works, then in **Command Prompt** or **PowerShell** inside the repo:

```bat
rmdir /s /q .venv 2>nul
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[webui]"
docstack models pull
```

### 4. Run everything

From the repo folder, double-click **`start.bat`** or run:

```bat
start.bat
```

That checks **Ollama** (and can start `ollama serve` in a minimized window if needed), then opens two windows: **DocStack** on port **8000** and **Open WebUI** on **3000**. Windows **10+** includes **`curl`**, which `start.bat` uses.

If `py -3.12` is missing, install Python 3.12 or edit [start.bat](start.bat) / use `py -3.11` consistently with [pyproject.toml](pyproject.toml) (`>=3.11,<3.13`).

### 5. Same wiring as Mac

- **Open WebUI**: Settings → Connections → **OpenAI API URL** `http://127.0.0.1:8000/v1` (or `http://localhost:8000/v1`), API key any non-empty string — optionally turn off Open WebUI’s built-in document RAG so DocStack owns retrieval.
- **Index files** via [http://127.0.0.1:8000/](http://127.0.0.1:8000/) or `POST /ingest` — **not** via chat-only attachments in Open WebUI.

## License

MIT (project scaffold — adjust as needed).
