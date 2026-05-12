@echo off
REM Requires: .venv with Python 3.12 and: pip install -e ".[webui]"
setlocal EnableExtensions
cd /d "%~dp0"

if not defined OLLAMA_URL set "OLLAMA_URL=http://127.0.0.1:11434"

REM ── Check Ollama ──────────────────────────────────────────────────────────────
curl -sf --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1
if not errorlevel 1 goto :ollama_ready

where ollama >nul 2>&1
if errorlevel 1 (
  echo ERROR: Ollama not found on PATH. Install from https://ollama.com
  pause & exit /b 1
)

echo Ollama not responding - starting ollama serve...
if not exist "data" mkdir data
start "Ollama" /MIN /D "%~dp0" cmd /c "ollama serve >> data\ollama-serve.log 2>&1"
call :wait_ollama
if errorlevel 1 (
  echo ERROR: Ollama did not start after 10s. Check data\ollama-serve.log
  pause & exit /b 1
)

:ollama_ready
echo Ollama is running at %OLLAMA_URL%

REM ── Show models ───────────────────────────────────────────────────────────────
echo.
echo Checking Ollama models...
ollama list
echo.
echo Required: qwen3:4b  qwen3:8b  qwen2.5vl:7b
echo If any are missing, run:  .venv\Scripts\docstack models pull
echo.

REM ── Free ports 8000 and 3000 (using PowerShell - avoids batch pipe issues) ────
powershell -NoProfile -Command "try { $p=Get-NetTCPConnection -LocalPort 8000 -State Listen -EA Stop; Stop-Process -Id $p.OwningProcess -Force -EA SilentlyContinue; Write-Host 'Freed port 8000' } catch {}" 2>nul
powershell -NoProfile -Command "try { $p=Get-NetTCPConnection -LocalPort 3000 -State Listen -EA Stop; Stop-Process -Id $p.OwningProcess -Force -EA SilentlyContinue; Write-Host 'Freed port 3000' } catch {}" 2>nul

REM ── Ensure .venv exists ───────────────────────────────────────────────────────
if exist ".venv\Scripts\uvicorn.exe" goto :launch

echo .venv not found or incomplete. Setting up Python environment...
echo This may take 5-15 minutes on first run.
echo.

where py >nul 2>&1
if errorlevel 1 goto :no_py

py -3.12 --version >nul 2>&1
if errorlevel 1 goto :no_py312

echo Creating .venv with Python 3.12...
py -3.12 -m venv .venv
if errorlevel 1 ( echo ERROR: Failed to create .venv & pause & exit /b 1 )

echo Installing packages (pip install -e .[webui])...
.venv\Scripts\pip install -U pip
.venv\Scripts\pip install -e ".[webui]"
if errorlevel 1 ( echo ERROR: pip install failed - check output above & pause & exit /b 1 )

echo.
echo Setup complete!
echo.
goto :launch

:no_py
echo ERROR: "py" launcher not found.
echo Install Python 3.12 from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during install.
pause & exit /b 1

:no_py312
echo ERROR: Python 3.12 not found.
echo Install Python 3.12 from https://www.python.org/downloads/
pause & exit /b 1

REM ── Launch ────────────────────────────────────────────────────────────────────
:launch
echo Starting DocStack API on port 8000...
start "DocStack API" /D "%~dp0" cmd /k ".venv\Scripts\uvicorn.exe docstack.api:app --host 0.0.0.0 --port 8000"

if not exist ".venv\Scripts\open-webui.exe" goto :no_webui
echo Starting Open WebUI on port 3000...
start "Open WebUI" /D "%~dp0" cmd /k ".venv\Scripts\open-webui.exe serve --port 3000"
goto :done

:no_webui
echo WARNING: open-webui not found in .venv
echo Run: .venv\Scripts\pip install -e ".[webui]"

:done
echo.
echo DocStack upload UI:  http://127.0.0.1:8000
echo Open WebUI:          http://localhost:3000
echo.
endlocal
goto :eof

REM ── Subroutine: wait up to 10s for Ollama ─────────────────────────────────────
:wait_ollama
set _w=0
:_wloop
ping -n 2 127.0.0.1 >nul
curl -sf --max-time 2 "%OLLAMA_URL%/api/tags" >nul 2>&1
if not errorlevel 1 exit /b 0
set /a _w+=1
if %_w% geq 10 exit /b 1
goto :_wloop
