@echo off
set PYTHONUTF8=1
chcp 65001

:: 1. SET AMD/ROCm PATHS (Ensure these are at the VERY START)
set HIP_PATH=E:\AI_Generated\hip65
set PATH=%HIP_PATH%\bin;%HIP_PATH%\rocm;%HIP_PATH%\cmake;%HIP_PATH%\include;%HIP_PATH%\lib;%PATH%

:: 2. AMD GPU CONFIG
set HSA_OVERRIDE_GFX_VERSION=10.3.0
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set MIOPEN_FIND_MODE=2

:: 3. DIRECT PATH TO VENV PYTHON (No "activate" needed)
set VENV_PYTHON=E:\AI_Generated\ComfyUI\venv\Scripts\python.exe

echo Starting Conversion GUI...
"%VENV_PYTHON%" gui_run_conversion.py

pause