@echo off
REM TrafficLab Development Environment Startup Script for Windows
REM This script brings up the complete development environment for the TrafficLab project

setlocal enabledelayedexpansion

REM Colors (limited support in Windows batch)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Function to print colored output
:print_status
echo %BLUE%[INFO]%NC% %~1
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM Function to check if command exists
:command_exists
where %1 >nul 2>&1
if %errorlevel% equ 0 (
    exit /b 0
) else (
    exit /b 1
)

REM Function to check if port is in use
:port_in_use
netstat -an | findstr ":%1 " >nul 2>&1
if %errorlevel% equ 0 (
    exit /b 0
) else (
    exit /b 1
)

REM Function to start with Docker Compose
:start_docker
call :print_status "Starting TrafficLab with Docker Compose..."

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is not running. Please start Docker Desktop and try again."
    exit /b 1
)

REM Check if docker-compose is available
call :command_exists docker-compose
if %errorlevel% neq 0 (
    docker compose version >nul 2>&1
    if %errorlevel% neq 0 (
        call :print_error "Docker Compose is not available. Please install Docker Compose."
        exit /b 1
    )
    set "DOCKER_COMPOSE=docker compose"
) else (
    set "DOCKER_COMPOSE=docker-compose"
)

REM Stop any existing containers
call :print_status "Stopping any existing containers..."
%DOCKER_COMPOSE% down >nul 2>&1

REM Build and start services
call :print_status "Building and starting services..."
%DOCKER_COMPOSE% up --build -d

if %errorlevel% neq 0 (
    call :print_error "Failed to start services with Docker Compose."
    exit /b 1
)

call :print_success "All services are running!"
call :print_status "Services available at:"
echo   🌐 Frontend:    http://localhost:3000
echo   🔧 Backend API: http://localhost:8000
echo   🗄️  Database:    localhost:5432
echo.
call :print_status "To view logs: %DOCKER_COMPOSE% logs -f"
call :print_status "To stop: %DOCKER_COMPOSE% down"
goto :eof

REM Function to start locally (without Docker)
:start_local
call :print_status "Starting TrafficLab locally (without Docker)..."

REM Check prerequisites
call :command_exists python
if %errorlevel% neq 0 (
    call :print_error "Python is required but not installed."
    exit /b 1
)

call :command_exists node
if %errorlevel% neq 0 (
    call :print_error "Node.js is required but not installed."
    exit /b 1
)

call :command_exists npm
if %errorlevel% neq 0 (
    call :print_error "npm is required but not installed."
    exit /b 1
)

REM Check if PostgreSQL is running
call :port_in_use 5432
if %errorlevel% neq 0 (
    call :print_warning "PostgreSQL is not running on port 5432."
    call :print_status "Please start PostgreSQL and ensure it's accessible at localhost:5432"
    call :print_status "Database credentials: user/password, database: trafficlab"
)

REM Start backend
call :print_status "Starting backend server..."
cd backend

REM Check if virtual environment exists, create if not
if not exist "venv" (
    call :print_status "Creating Python virtual environment..."
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
call :print_status "Installing Python dependencies..."
pip install -r requirements.txt

REM Initialize database if needed
if exist "init_db.py" (
    call :print_status "Initializing database..."
    python init_db.py
)

REM Start backend in background
call :print_status "Starting FastAPI backend..."
start /b uvicorn main:app --host 0.0.0.0 --port 8000 --reload

REM Wait a moment for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend
call :print_status "Starting frontend development server..."
cd ..\frontend

REM Install frontend dependencies
call :print_status "Installing Node.js dependencies..."
npm install

REM Start frontend in background
call :print_status "Starting Vite development server..."
start /b npm run dev

REM Wait a moment for frontend to start
timeout /t 5 /nobreak >nul

call :print_success "All services are running locally!"
call :print_status "Services available at:"
echo   🌐 Frontend:    http://localhost:3000
echo   🔧 Backend API: http://localhost:8000
echo   🗄️  Database:    localhost:5432
echo.
call :print_status "Press any key to stop all services..."
pause >nul
goto :eof

REM Function to show help
:show_help
echo TrafficLab Development Environment Startup Script
echo.
echo Usage: %~nx0 [OPTION]
echo.
echo Options:
echo   docker, -d    Start with Docker Compose (recommended)
echo   local, -l     Start locally without Docker
echo   help, -h      Show this help message
echo.
echo Examples:
echo   %~nx0 docker     # Start with Docker Compose
echo   %~nx0 local      # Start locally
echo   %~nx0            # Default to Docker Compose
echo.
echo Prerequisites:
echo   Docker mode:  Docker Desktop, Docker Compose
echo   Local mode:   Python 3.8+, Node.js 16+, PostgreSQL
echo.
goto :eof

REM Main script logic
:main
REM Change to script directory
cd /d "%~dp0"

call :print_status "TrafficLab Development Environment Startup"
call :print_status "============================================="

REM Parse command line arguments
if "%1"=="docker" goto start_docker
if "%1"=="-d" goto start_docker
if "%1"=="local" goto start_local
if "%1"=="-l" goto start_local
if "%1"=="help" goto show_help
if "%1"=="-h" goto show_help
if "%1"=="--help" goto show_help
if "%1"=="" goto start_docker

call :print_error "Unknown option: %1"
call :show_help
exit /b 1

REM Run main function
goto main
