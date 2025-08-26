@echo off
echo 🚀 Starting Federated Learning Energy-Efficient Cloud Resource Allocation System
echo ==================================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
pip install --upgrade pip

REM Install dependencies
echo 📋 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "config\secrets" mkdir config\secrets

REM Set environment variables
set PYTHONPATH=%cd%\src
set FLASK_ENV=development

echo.
echo 🎯 Setup complete! Here's what you can do next:
echo.
echo 1. 🧪 Run simulation:
echo    python src\simulation\run_simulation.py
echo.
echo 2. 🌐 Start API server:
echo    python src\api\app.py
echo.
echo 3. 🐳 Run with Docker:
echo    docker-compose up
echo.
echo 4. ☁️  Deploy to Azure:
echo    azd up
echo.
echo 📖 For more information, see README.md
echo.

set /p choice="🤔 Would you like to run a quick simulation now? (y/n): "
if /i "%choice%"=="y" (
    echo 🏃‍♂️ Running simulation...
    python src\simulation\run_simulation.py
) else (
    echo 👋 Setup complete! Run any of the commands above to get started.
)

pause
