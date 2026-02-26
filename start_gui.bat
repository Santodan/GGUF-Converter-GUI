@echo off
setlocal enabledelayedexpansion
set PYTHONUTF8=1
chcp 65001

:: 1. SET AMD/ROCm PATHS (Ensure these are at the VERY START)
set HIP_PATH=E:\AI_Generated\hip65
set PATH=%HIP_PATH%\bin;%HIP_PATH%\rocm;%HIP_PATH%\cmake;%HIP_PATH%\include;%HIP_PATH%\lib;%PATH%

:: 2. AMD GPU CONFIG
set HSA_OVERRIDE_GFX_VERSION=10.3.0
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set MIOPEN_FIND_MODE=2

:: --- PATH CONFIGURATION ---
:: This specific line will be updated automatically if the path is wrong.
set VENV_PYTHON=E:\AI_Generated\ComfyUI7-13\venv\Scripts\python.exe

:: 1. CHECK IF PYTHON EXISTS
if not exist "%VENV_PYTHON%" (
    echo.
    echo Python not found at: "%VENV_PYTHON%"
    echo --------------------------------------------------------
    set /p NEW_PATH="Please drag-and-drop your python.exe here (or enter path): "
    
    :: Clean quotes from user input (in case of drag-and-drop)
    set NEW_PATH=!NEW_PATH:"=!

    if not exist "!NEW_PATH!" (
        echo.
        echo ERROR: The path "!NEW_PATH!" is invalid.
        pause
        exit /b
    )

    echo.
    echo New path verified. Updating script...
    
    :: Use PowerShell to find the specific line in this script and replace it
set VENV_PYTHON=E:\AI_Generated\ComfyUI7-13\venv\Scripts\python.exe
    
    :: Small delay so the user sees the success message
    timeout /t 1 >nul
    
    :: Restart the script and exit this process
    start "" "%~f0"
    exit /b
)

echo Starting Conversion GUI...
"%VENV_PYTHON%" gui_run_conversion.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Application exited with an error.
    pause
)
