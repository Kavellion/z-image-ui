@echo off
REM Z-Image Turbo UI one-click installer (Windows).
REM
REM What it does:
REM   1. Checks for Python 3.12 via the py launcher (py -3.12).
REM   2. Creates a Python 3.12 virtual environment (.venv) in this folder.
REM   3. Installs PyTorch with CUDA support from the cu130 index, then
REM      the rest of the requirements (Diffusers 0.42, Gradio, ...).
REM   4. Creates images_out\ for finished pictures.
REM
REM What it does NOT do:
REM   - Touch your Python 3.10 install, or any other Python you have.
REM     This app lives entirely in its own .venv, built from 3.12.
REM     Your other apps keep using whatever Python they already use.
REM   - Download the Z-Image model weights. Diffusers downloads the
REM     model (about 15 GB on the default quantized path) from
REM     Hugging Face on the first Generate, into the Hugging Face
REM     cache on your user profile -- not into this folder.
REM   - Ask for any account, token, or license acceptance. Z-Image
REM     Turbo is Apache 2.0 and ungated: the first download just
REM     works (see README.md).
REM
REM Requirements: Windows, Python 3.12 (see below), an NVIDIA GPU
REM (12 GB is plenty -- the app defaults to a quantized transformer
REM plus CPU offload, which is what makes it fit), about 20 GB free
REM disk for the first download plus room for pictures.

setlocal
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python 3.12 was not found.
    echo.
    echo Z-Image UI needs Python 3.12, side by side with any other
    echo Python you already have. Install it from python.org --
    echo tick "Add python.exe to PATH" during setup -- then run
    echo this file again. Nothing about your existing Python
    echo installs is changed by this.
    echo.
    pause
    exit /b 1
)

echo Python 3.12 found:
py -3.12 --version

if not exist .venv (
    echo Creating virtual environment from Python 3.12...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA support (this is the big download)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 (
    echo PyTorch install failed. Check your internet connection and
    echo that your GPU driver is up to date, then run this again.
    pause
    exit /b 1
)

echo Installing the remaining packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Package install failed. Run this again -- pip resumes
    echo downloads it already finished.
    pause
    exit /b 1
)

if not exist images_out mkdir images_out

echo.
echo Done. Double-click run.bat to start Z-Image UI.
echo.
echo First Generate downloads about 15 GB of model weights, so it
echo takes a while -- that is the one-time download, not a stuck
echo render. No token or license step is needed.
echo.
pause
