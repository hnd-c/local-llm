@echo off
REM Requires: .venv with Python 3.12 and: pip install -e ".[webui]"
REM Ensures Ollama is reachable (starts ollama serve in a minimized window if needed).
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if not defined OLLAMA_URL set "OLLAMA_URL=http://127.0.0.1:11434"

REM ── Check Ollama ──────────────────────────────────────────────────────────────
curl -sf --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1
if not errorlevel 1 (
  echo Ollama is running at %OLLAMA_URL%
  goto :after_ollama
)

where ollama >nul 2>&1
if errorlevel 1 (
  echo.
  echo ERROR: Ollama is not running and ollama.exe was not found on PATH.
  echo Install from https://ollama.com or start the Ollama tray app, then re-run start.bat
  echo.
  pause
  exit /b 1
)

echo Ollama not responding; starting ollama serve in the background...
if not exist "data" mkdir data
start "Ollama" /MIN /D "%~dp0" cmd /c "ollama serve >> data\ollama-serve.log 2>&1"

set i=0
:wait_ollama
curl -sf --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1
if not errorlevel 1 (
  echo Ollama is up at %OLLAMA_URL%
  goto :after_ollama
)
set /a i+=1
if !i! geq 11 (
  echo ERROR: Ollama did not become ready after 10s. See %~dp0data\ollama-serve.log
  pause
  exit /b 1
)
timeout /t 1 /nobreak >nul
goto :wait_ollama

:after_ollama

REM ── Check required models ─────────────────────────────────────────────────────
echo.
echo Checking Ollama models...
REM Note: "failed to get console mode for stderr" is a harmless Go warning — not an error.
ollama list
echo.
echo Required: qwen3:4b  qwen3:8b  qwen2.5vl:7b
echo If any are missing, run:  docstack models pull
echo.

REM ── Free ports 8000 and 3000 before starting ─────────────────────────────────
netstat -ano >"%TEMP%\ds_netstat.txt" 2>nul
for /f "tokens=5" %%p in ('findstr ":8000 " "%TEMP%\ds_netstat.txt" 2^>nul') do (
  if not "%%p"=="" taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('findstr ":3000 " "%TEMP%\ds_netstat.txt" 2^>nul') do (
  if not "%%p"=="" taskkill /PID %%p /F >nul 2>&1
)
del "%TEMP%\ds_netstat.txt" >nul 2>&1

REM ── Ensure .venv exists ───────────────────────────────────────────────────────
if not exist ".venv\Scripts\uvicorn.exe" (
  echo.
  echo .venv not found or incomplete. Setting up Python environment now...
  echo This may take 5-15 minutes on first run.
  echo.

  where py >nul 2>&1
  if errorlevel 1 (
    echo ERROR: Python launcher "py" not found.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
  )

  py -3.12 --version >nul 2>&1
  if errorlevel 1 (
    echo ERROR: Python 3.12 not found. Install it from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
  )

  echo Creating .venv with Python 3.12...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo ERROR: Failed to create .venv
    pause
    exit /b 1
  )

  echo Installing packages (this takes a while)...
  .venv\Scripts\pip install -U pip
  .venv\Scripts\pip install -e ".[webui]"
  if errorlevel 1 (
    echo ERROR: pip install failed. Check the output above for details.
    pause
    exit /b 1
  )

  echo.
  echo Setup complete!
  echo.
)

REM ── Launch DocStack and Open WebUI ────────────────────────────────────────────
echo Starting DocStack API on port 8000...
start "DocStack API" /D "%~dp0" cmd /k ".venv\Scripts\uvicorn.exe docstack.api:app --host 0.0.0.0 --port 8000"

if exist ".venv\Scripts\open-webui.exe" (
  echo Starting Open WebUI on port 3000...
  start "Open WebUI" /D "%~dp0" cmd /k ".venv\Scripts\open-webui.exe serve --port 3000"
) else (
  echo.
  echo WARNING: open-webui not found in .venv.
  echo Run: .venv\Scripts\pip install -e ".[webui]"
  echo.
)

echo.
echo DocStack upload UI:  http://127.0.0.1:8000
echo Open WebUI:          http://localhost:3000
echo.

endlocal
