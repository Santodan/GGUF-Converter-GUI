@echo off
setlocal enabledelayedexpansion
set PYTHONUTF8=1
chcp 65001

:: --- SET AMD/ROCm PATHS (Commented out as per your previous version) ---
::set HIP_PATH=E:\AI_Generated\hip65
::set PATH=%HIP_PATH%\bin;%HIP_PATH%\rocm;%HIP_PATH%\cmake;%HIP_PATH%\include;%HIP_PATH%\lib;%PATH%

:: --- AMD GPU CONFIG (Commented out as per your previous version) ---
::set HSA_OVERRIDE_GFX_VERSION=10.3.0
::set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
::set MIOPEN_FIND_MODE=2

echo.
echo ============================================================
echo           Select Python Environment to Launch GUI
echo ============================================================
echo [1] Custom Venv (Python 3.12 - ComfyUI)
echo [2] System Python (Python 3.11 - Default Path)
echo ============================================================
set /p CHOICE="Enter choice (1 or 2) [Default is 1]: "

if "%CHOICE%"=="2" (
    set RUN_PYTHON=python
    echo Status: Using System Python (3.11)
) else (
    set RUN_PYTHON=E:\AI_Generated\ComfyUI7-13\venv\Scripts\python.exe
    echo Status: Using Custom Venv (3.12)
)

:: 1. VERIFY SELECTION
if "%RUN_PYTHON%"=="python" (
    :: Check if 'python' exists in the system path
    where python >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo ERROR: 'python' was not found in your system PATH.
        pause
        exit /b
    )
) else (
    :: Check if the specific Venv path exists
    if not exist "%RUN_PYTHON%" (
        echo.
        echo Python not found at: "%RUN_PYTHON%"
        echo --------------------------------------------------------
        set /p NEW_PATH="Please drag-and-drop your custom python.exe here: "
        
        :: Clean quotes from user input
        set NEW_PATH=!NEW_PATH:"=!

        if not exist "!NEW_PATH!" (
            echo ERROR: The path "!NEW_PATH!" is invalid.
            pause
            exit /b
        )
        set RUN_PYTHON=!NEW_PATH!
    )
)

echo.
echo Launching GUI with:
"%RUN_PYTHON%" --version
echo.

:: 2. RUN THE GUI
"%RUN_PYTHON%" gui_run_conversion.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Application exited with an error code: %ERRORLEVEL%
    pause
)