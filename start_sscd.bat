@echo off
rem Start the SSCD web app - double-click this file.
rem
rem The first start installs the Python environment (a few minutes). It is kept in
rem %LOCALAPPDATA%\sscd\envs\ - outside this folder - so SSCD can live in any folder,
rem including long paths and Dropbox/OneDrive, without hitting Windows' 260-character
rem path limit (TensorFlow's files have long paths of their own).
rem Extra arguments are passed on to Streamlit, e.g. start_sscd.bat --server.port 8502
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv is not installed. Install it from
    echo   https://docs.astral.sh/uv/getting-started/installation/
    echo e.g. in PowerShell:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo then open a new window and start this file again.
    pause
    exit /b 1
)

rem one environment per SSCD folder: named after a short hash of this folder's path
for /f %%h in ('powershell -NoProfile -Command "[BitConverter]::ToString((New-Object Security.Cryptography.SHA1Managed).ComputeHash([Text.Encoding]::UTF8.GetBytes('%~dp0'))).Replace('-','').Substring(0,8)"') do set "SSCD_ENV_ID=%%h"
if not defined SSCD_ENV_ID set "SSCD_ENV_ID=default"
set "UV_PROJECT_ENVIRONMENT=%LOCALAPPDATA%\sscd\envs\sscd-%SSCD_ENV_ID%"

echo Preparing the SSCD environment in %UV_PROJECT_ENVIRONMENT%
echo (the first time this downloads and installs the packages - a few minutes)...
uv sync --extra gui
if errorlevel 1 (
    echo.
    echo Installing the environment failed - see the messages above.
    pause
    exit /b 1
)

echo.
echo Starting SSCD - it opens in your browser. Close this window to stop the app
echo (a run that is in progress keeps going).
rem the SSCD icon instead of Streamlit's in the browser tab, from the first moment
uv run --no-sync python -m sscd_libs.app_icon
uv run --no-sync python -m streamlit run sscd_app.py %*
if errorlevel 1 pause
