@echo off
REM Start Z-Image Turbo UI. Run install.bat once first.
REM
REM Pictures land in images_out\ next to this file, one subfolder per
REM batch name, with a settings.txt alongside the frames.

setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo The virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python z-image-ui\app.py
echo.
echo The app stopped. Double-click run.bat to start it again.
pause
