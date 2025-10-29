@echo off
REM =============================================================================
REM Fake News Detector - Startup Script (Windows)
REM =============================================================================
REM This script sets up the environment and starts the Fake News Detector application

setlocal enabledelayedexpansion

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "VENV_DIR=%PROJECT_ROOT%\venv"
set "REQUIREMENTS_FILE=%PROJECT_ROOT%\requirements.txt"
set "ENV_FILE=%PROJECT_ROOT%\.env"
set "ENV_EXAMPLE=%PROJECT_ROOT%\.env.example"

REM Colors (limited support in Windows CMD)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

echo %BLUE%==================================%NC%
echo %BLUE%  Fake News Detector Startup%NC%
echo %BLUE%==================================%NC%
echo.

REM Check if help is requested
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help

REM Change to project root directory
cd /d "%PROJECT_ROOT%"

REM Step 1: Check Python installation
echo %YELLOW%[STEP]%NC% Checking Python installation...

python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Python is not installed or not in PATH
    echo Please install Python 3.8 or higher and try again
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%[SUCCESS]%NC% Python %PYTHON_VERSION% found

REM Step 2: Setup virtual environment
echo %YELLOW%[STEP]%NC% Setting up virtual environment...

if not exist "%VENV_DIR%" (
    echo %BLUE%[INFO]%NC% Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo %RED%[ERROR]%NC% Failed to create virtual environment
        pause
        exit /b 1
    )
    echo %GREEN%[SUCCESS]%NC% Virtual environment created
) else (
    echo %BLUE%[INFO]%NC% Virtual environment already exists
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to activate virtual environment
    pause
    exit /b 1
)
echo %GREEN%[SUCCESS]%NC% Virtual environment activated

REM Step 3: Install dependencies
echo %YELLOW%[STEP]%NC% Installing dependencies...

if not exist "%REQUIREMENTS_FILE%" (
    echo %RED%[ERROR]%NC% Requirements file not found: %REQUIREMENTS_FILE%
    pause
    exit /b 1
)

REM Upgrade pip first
python -m pip install --upgrade pip
if errorlevel 1 (
    echo %YELLOW%[WARNING]%NC% Failed to upgrade pip, continuing...
)

REM Install requirements
python -m pip install -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to install dependencies
    pause
    exit /b 1
)
echo %GREEN%[SUCCESS]%NC% Dependencies installed

REM Step 4: Setup environment file
echo %YELLOW%[STEP]%NC% Setting up environment configuration...

if not exist "%ENV_FILE%" (
    if exist "%ENV_EXAMPLE%" (
        copy "%ENV_EXAMPLE%" "%ENV_FILE%" >nul
        echo %GREEN%[SUCCESS]%NC% Environment file created from template
        echo %BLUE%[INFO]%NC% Please edit .env file with your configuration before running the application
    ) else (
        echo %RED%[ERROR]%NC% Environment example file not found: %ENV_EXAMPLE%
        pause
        exit /b 1
    )
) else (
    echo %BLUE%[INFO]%NC% Environment file already exists
)

REM Step 5: Create directories
echo %YELLOW%[STEP]%NC% Creating necessary directories...

if not exist "%PROJECT_ROOT%\data\raw" mkdir "%PROJECT_ROOT%\data\raw"
if not exist "%PROJECT_ROOT%\data\processed" mkdir "%PROJECT_ROOT%\data\processed"
if not exist "%PROJECT_ROOT%\data\models" mkdir "%PROJECT_ROOT%\data\models"
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"
if not exist "%PROJECT_ROOT%\logs\training" mkdir "%PROJECT_ROOT%\logs\training"

echo %GREEN%[SUCCESS]%NC% Directories created

REM Step 6: Check for trained models
echo %YELLOW%[STEP]%NC% Checking for trained models...

dir "%PROJECT_ROOT%\data\models\*" >nul 2>&1
if errorlevel 1 (
    echo %BLUE%[INFO]%NC% No trained models found
    echo %BLUE%[INFO]%NC% The application will run in demo mode with mock predictions
    echo %BLUE%[INFO]%NC% To train models, run: python train_model.py
) else (
    echo %GREEN%[SUCCESS]%NC% Trained models found
)

REM Check if setup-only mode
if "%1"=="--setup-only" (
    echo.
    echo %GREEN%[SUCCESS]%NC% Setup completed! Run 'scripts\start.bat' to start the application
    pause
    exit /b 0
)

REM Step 7: Start application
echo.
echo %GREEN%[SUCCESS]%NC% Setup completed successfully!
echo.
echo %YELLOW%[STEP]%NC% Starting Fake News Detector application...

REM Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Streamlit is not installed
    echo %BLUE%[INFO]%NC% Installing Streamlit...
    python -m pip install streamlit
    if errorlevel 1 (
        echo %RED%[ERROR]%NC% Failed to install Streamlit
        pause
        exit /b 1
    )
)

echo %GREEN%[SUCCESS]%NC% Starting Streamlit application...
echo %BLUE%[INFO]%NC% The application will open in your default web browser
echo %BLUE%[INFO]%NC% If it doesn't open automatically, go to: http://localhost:8501
echo.
echo %BLUE%[INFO]%NC% Press Ctrl+C to stop the application
echo.

REM Start the application
streamlit run app.py

goto :end

:show_help
echo Fake News Detector Startup Script
echo.
echo Usage: %0 [OPTIONS]
echo.
echo Options:
echo   --setup-only    Only run setup steps, don't start the application
echo   --help, -h      Show this help message
echo.
echo Environment Variables:
echo   SKIP_VENV       Skip virtual environment setup (default: false)
echo.
goto :end

:end
pause