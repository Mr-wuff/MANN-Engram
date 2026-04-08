@echo off
:: Set console to UTF-8 encoding to prevent text rendering issues
chcp 65001 >nul
:: Set black background with green text for a geeky terminal look
color 0A

echo ===================================================
echo   🧠 MANN-Engram One-Click Launcher (Windows)
echo ===================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [Error] Python is not detected on your system!
    echo Please visit https://www.python.org/ to download and install Python (v3.10+ recommended).
    echo.
    echo ⚠️ IMPORTANT: Please ensure you check the "Add Python.exe to PATH" box at the bottom during installation!
    echo.
    pause
    exit /b
)

:: 2. Check and create an isolated virtual environment
if not exist ".venv" (
    echo [System] First run detected. Creating an isolated sandbox environment...
    echo          (This ensures your other system programs remain unaffected)
    python -m venv .venv
)

:: 3. Activate the virtual environment
echo [System] Activating sandbox environment...
call .venv\Scripts\activate

:: 4. Configure mirrors and silently install dependencies
echo [System] Checking and updating core components...
echo          (This may take a few minutes on the first run, please wait)
python -m pip install --upgrade pip -q

:: Force use of USTC mirror to solve slow download speeds in China 
:: (Remove the line below if your users are primarily outside of China)
python -m pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple >nul 2>&1

:: Install core SDK (-q for quiet installation to prevent screen clutter)
python -m pip install -e . -q
:: Install microservice dependencies
python -m pip install -r api_service\requirements.txt -q

:: 5. Pre-flight check: Are model weights in place?
if not exist "weights\skew_model_v4full_en.pt" (
    color 0E
    echo.
    echo ⚠️ [CRITICAL WARNING] Core routing engine tensor weights not found!
    echo.
    echo Please ensure you have downloaded 'skew_model_v4full_en.pt' from our release page
    echo and placed it inside the 'weights' folder in the project root directory.
    echo.
    pause
    exit /b
)

:: 6. Start the backend API service
color 0A
echo.
echo ✅ [Success] System check passed, engine modules are ready!
echo 🚀 Starting backend microservice...

:: Start FastAPI service and keep it running in a new, separate command window
:: (Remove 'set HF_ENDPOINT=https://hf-mirror.com &&' if targeting global users)
start "MANN-Engram API Server (DO NOT CLOSE)" cmd /k "call .venv\Scripts\activate && set HF_ENDPOINT=https://hf-mirror.com && uvicorn api_service.main:app --host 127.0.0.1 --port 8000"

:: Wait for 3 seconds to ensure the local service port is listening properly
timeout /t 3 >nul

:: 7. Automatically open the browser interface
echo 🌐 Opening the interactive interface in your default web browser...
start http://127.0.0.1:8000/docs

echo.
echo ===================================================
echo   🎉 MANN-Engram has successfully started!
echo.
echo   You can now test the system functionality directly 
echo   in the browser window that just opened.
echo.
echo   (To stop the service, simply close the popped-up 
echo   API Server console window)
echo ===================================================
echo.
pause