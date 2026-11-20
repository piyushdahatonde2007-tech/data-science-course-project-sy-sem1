@echo off
setlocal
title Battery RUL Console
cd /d "%~dp0backend"
if "%PORT%"=="" set PORT=8000

set PY=
where python >nul 2>nul && set PY=python
if not defined PY where py >nul 2>nul && set PY=py -3
if not defined PY goto :nopython

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv || goto :fail
)
if not exist ".venv\.deps_ok" (
  echo Installing dependencies - first run only, about 1-2 minutes...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip -q
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt -q || goto :fail
  echo ok> ".venv\.deps_ok"
)

echo.
echo Starting Battery RUL Console on http://localhost:%PORT%
echo The browser opens by itself when the server is ready. Close this window to stop.
start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 400;$i++){try{Invoke-WebRequest 'http://localhost:%PORT%/api/health' -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process 'http://localhost:%PORT%'; break}catch{Start-Sleep 2}}"
".venv\Scripts\python.exe" -m uvicorn app.main:app --port %PORT%
goto :eof

:nopython
echo Python 3.10 or newer was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then run this file again.
pause
exit /b 1

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
