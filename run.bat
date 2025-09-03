@echo off
echo ========================================
echo    Video Processing Pipeline Starter
echo ========================================
echo.

:: Check if venv folder exists
if exist "venv" (
    echo Virtual environment found. Activating...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating new one...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Please make sure Python is installed and in PATH.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully. Activating...
    call venv\Scripts\activate.bat
)

echo.
echo Installing/updating requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Starting Video Processing Pipeline
echo ========================================
echo.

:: Run the main script
python main.py
if errorlevel 1 (
    echo.
    echo ERROR: Script execution failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    PROCESS COMPLETE
echo ========================================
echo.
echo Video processing pipeline has finished successfully!
echo Check the output directory for processed clips.
echo.
pause
