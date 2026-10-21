@echo off
rem Windows başlatıcı.
rem   run.bat          -> normal
rem   run.bat --demo   -> sentetik görüntü + demo segmentasyon
rem   run.bat --ml     -> PyTorch bağımlılıklarını da kur
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")

if not exist .venv\Scripts\python.exe (
  echo Sanal ortam olusturuluyor (.venv^)...
  %PY% -m venv .venv || (echo Python bulunamadi. https://www.python.org adresinden Python 3.10+ kurun. & pause & exit /b 1)
)
set "VPY=.venv\Scripts\python.exe"

for /f "skip=1 delims=" %%h in ('certutil -hashfile requirements.txt MD5') do if not defined WANT set "WANT=%%h"
set "HAVE="
if exist .venv\req-stamp.txt set /p HAVE=<.venv\req-stamp.txt
if not "!HAVE!"=="!WANT!" (
  %VPY% -m pip install --upgrade pip
  %VPY% -m pip install -r requirements.txt || (pause & exit /b 1)
  >.venv\req-stamp.txt echo !WANT!
)

set "ARGS="
for %%a in (%*) do (
  if /i "%%~a"=="--ml" (
    %VPY% -m pip install -r requirements-ml.txt
  ) else (
    set "ARGS=!ARGS! %%a"
  )
)

%VPY% -m backend !ARGS!
if errorlevel 1 pause
