#!/usr/bin/env bash
# Start API + Open WebUI using repo .venv (Python 3.11/3.12).
# Ensures Ollama is reachable (starts `ollama serve` in the background if needed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"

# Base URL for the Ollama HTTP API (override if you bind elsewhere).
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

ollama_http_ok() {
  curl -sf --max-time 2 "${OLLAMA_URL}/api/tags" >/dev/null
}

ensure_ollama() {
  if ollama_http_ok; then
    echo "Ollama is running (${OLLAMA_URL})."
    return 0
  fi
  if command -v ollama >/dev/null 2>&1; then
    echo "Ollama not responding; starting \`ollama serve\` in the background…"
    mkdir -p "${ROOT}/data"
    ( ollama serve >>"${ROOT}/data/ollama-serve.log" 2>&1 ) &
    local i=0
    while (( i < 50 )); do
      if ollama_http_ok; then
        echo "Ollama is up (${OLLAMA_URL}). Log: ${ROOT}/data/ollama-serve.log"
        return 0
      fi
      sleep 0.2
      (( i++ )) || true
    done
    echo "Ollama did not become ready (tried ~10s). See ${ROOT}/data/ollama-serve.log"
    exit 1
  fi
  echo "Ollama is not running and the \`ollama\` command was not found on PATH."
  echo "Install from https://ollama.com or start the app/service, then re-run ./start.sh"
  exit 1
}

free_port() {
  local port="$1"
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "Stopping process(es) on port $port (PID: $pids)…"
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    local i=0
    while (( i < 20 )); do
      pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
      [[ -z "$pids" ]] && return 0
      sleep 0.2
      (( i++ )) || true
    done
    echo "$pids" | xargs kill -KILL 2>/dev/null || true
  fi
}

ensure_ollama

check_models() {
  # Read required_models from configs/settings.toml (handles quoted, comma-separated list)
  local config="$ROOT/configs/settings.toml"
  if [[ ! -f "$config" ]]; then return; fi
  local line
  line=$(grep '^required_models' "$config" | head -1)
  # Extract model names from: required_models = ["qwen3:4b", "qwen3:8b", "qwen2.5vl"]
  local models=()
  while IFS= read -r m; do
    models+=("$m")
  done < <(echo "$line" | grep -oE '"[^"]+"' | tr -d '"')

  local pulled
  pulled=$(curl -sf --max-time 5 "${OLLAMA_URL}/api/tags" | grep -oE '"name":"[^"]+"' | cut -d'"' -f4 || true)

  local missing=()
  for model in "${models[@]}"; do
    # Match model name with or without :latest tag
    local base="${model%%:*}"
    if ! echo "$pulled" | grep -q "^${base}"; then
      missing+=("$model")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo ""
    echo "⚠  Missing Ollama models: ${missing[*]}"
    echo "   Run:  ollama pull ${missing[0]}"
    for m in "${missing[@]:1}"; do
      echo "         ollama pull $m"
    done
    echo "   Or:   docstack models pull   (pulls all at once)"
    echo ""
  else
    echo "All required models are present: ${models[*]}"
  fi
}

check_models

free_port 8000
free_port 3000

if [[ ! -x "$VENV/bin/uvicorn" ]]; then
  echo "Missing $VENV. Create it with Python 3.12, then: pip install -e \".[webui]\""
  echo "  python3.12 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -e \".[webui]\""
  exit 1
fi
"$VENV/bin/uvicorn" docstack.api:app --host 0.0.0.0 --port 8000 &
UV_PID=$!
trap 'kill $UV_PID 2>/dev/null || true' EXIT
if [[ -x "$VENV/bin/open-webui" ]]; then
  exec "$VENV/bin/open-webui" serve --port 3000
fi
echo "open-webui not installed in .venv. Run: pip install -e \".[webui]\" inside Python 3.11/3.12 venv."
echo "API running on http://127.0.0.1:8000 — press Ctrl+C to stop."
wait "$UV_PID"
