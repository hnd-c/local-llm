@echo off
REM Requires: .venv with Python 3.12 and: pip install -e ".[webui]"
REM Ensures Ollama is reachable (starts ollama serve in a minimized window if needed).
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if not defined OLLAMA_URL set "OLLAMA_URL=http://127.0.0.1:11434"

curl -sf --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1
if not errorlevel 1 (
  echo Ollama is running at %OLLAMA_URL%
  goto :after_ollama
)

where ollama >nul 2>&1
if errorlevel 1 (
  echo Ollama is not running and ollama.exe was not found on PATH.
  echo Install from https://ollama.com or start the Ollama app, then re-run start.bat
  exit /b 1
)

echo Ollama not responding; starting ollama serve in the background...
if not exist "data" mkdir data
start "Ollama" /MIN /D "%~dp0" cmd /c "ollama serve >> data\ollama-serve.log 2>&1"

set i=0
:wait_ollama
curl -sf --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1
if not errorlevel 1 (
  echo Ollama is up at %OLLAMA_URL%  Log: %~dp0data\ollama-serve.log
  goto :after_ollama
)
set /a i+=1
if !i! geq 11 (
  echo Ollama did not become ready (tried ~10s). See %~dp0data\ollama-serve.log
  exit /b 1
)
timeout /t 1 /nobreak >nul
goto :wait_ollama

:after_ollama

REM ── Check required models are pulled ─────────────────────────────────────────
echo.
echo Checking Ollama models...
ollama list 2>nul
if errorlevel 1 (
  echo WARNING: Could not query Ollama model list.
) else (
  echo Required: qwen3:4b  qwen3:8b  qwen2.5vl
  echo If any are missing, run: docstack models pull
)
echo.

REM ── Free ports 8000 and 3000 before starting (kills any leftover processes) ──
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R ":8000 .*LISTENING" 2^>nul') do (
  echo Stopping process on port 8000 (PID: %%p)...
  taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R ":3000 .*LISTENING" 2^>nul') do (
  echo Stopping process on port 3000 (PID: %%p)...
  taskkill /PID %%p /F >nul 2>&1
)

if not exist ".venv\Scripts\uvicorn.exe" (
  echo Create .venv first: py -3.12 -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -U pip ^&^& pip install -e ".[webui]"
  exit /b 1
)
start "DocStack API" /D "%~dp0" cmd /k ".venv\Scripts\uvicorn.exe docstack.api:app --host 0.0.0.0 --port 8000"
if exist ".venv\Scripts\open-webui.exe" (
  start "Open WebUI" /D "%~dp0" cmd /k ".venv\Scripts\open-webui.exe serve --port 3000"
) else (
  echo open-webui not in .venv. Run: pip install -e ".[webui]" with Python 3.12
)

endlocal
