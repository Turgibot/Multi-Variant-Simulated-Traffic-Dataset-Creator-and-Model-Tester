#!/bin/bash

# TrafficLab Development Environment Startup Script
# This script brings up the complete development environment for the TrafficLab project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    print_status "Waiting for $service_name to be ready on $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z $host $port 2>/dev/null; then
            print_success "$service_name is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Function to start with Docker Compose
start_docker() {
    print_status "Starting TrafficLab with Docker Compose..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Stop any existing containers
    print_status "Stopping any existing containers..."
    if command_exists docker-compose; then
        docker-compose down 2>/dev/null || true
    else
        docker compose down 2>/dev/null || true
    fi
    
    # Build and start services
    print_status "Building and starting services..."
    if command_exists docker-compose; then
        docker-compose up --build -d
    else
        docker compose up --build -d
    fi
    
    # Wait for services to be ready
    wait_for_service localhost 5432 "PostgreSQL Database"
    wait_for_service localhost 8000 "Backend API"
    wait_for_service localhost 3000 "Frontend Development Server"
    
    print_success "All services are running!"
    print_status "Services available at:"
    echo "  🌐 Frontend:    http://localhost:3000"
    echo "  🔧 Backend API: http://localhost:8000"
    echo "  🗄️  Database:    localhost:5432"
    echo ""
    print_status "To view logs: docker-compose logs -f"
    print_status "To stop: docker-compose down"
}

# Function to start locally (without Docker)
start_local() {
    print_status "Starting TrafficLab locally (without Docker)..."
    
    # Check prerequisites
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    
    if ! command_exists node; then
        print_error "Node.js is required but not installed."
        exit 1
    fi
    
    if ! command_exists npm; then
        print_error "npm is required but not installed."
        exit 1
    fi
    
    # Check if PostgreSQL is running
    if ! port_in_use 5432; then
        print_warning "PostgreSQL is not running on port 5432."
        print_status "Please start PostgreSQL and ensure it's accessible at localhost:5432"
        print_status "Database credentials: user/password, database: trafficlab"
    fi
    
    # Start backend
    print_status "Starting backend server..."
    cd backend
    
    # Check if virtual environment exists, create if not
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies (torch first: CPU wheels; PyG index matches torch 2.1.0+cpu)
    print_status "Installing Python dependencies..."
    pip install --retries 10 torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    pip install --retries 10 -r requirements.txt -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    
    # Initialize database if needed
    if [ -f "init_db.py" ]; then
        print_status "Initializing database..."
        python init_db.py
    fi
    
    # Start backend in background
    print_status "Starting FastAPI backend..."
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    
    # Wait for backend to be ready
    wait_for_service localhost 8000 "Backend API"
    
    # Start frontend
    print_status "Starting frontend development server..."
    cd ../frontend
    
    # Install frontend dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    # Start frontend in background
    print_status "Starting Vite development server..."
    npm run dev &
    FRONTEND_PID=$!
    
    # Wait for frontend to be ready
    wait_for_service localhost 3000 "Frontend Development Server"
    
    print_success "All services are running locally!"
    print_status "Services available at:"
    echo "  🌐 Frontend:    http://localhost:3000"
    echo "  🔧 Backend API: http://localhost:8000"
    echo "  🗄️  Database:    localhost:5432"
    echo ""
    print_status "Process IDs:"
    echo "  Backend PID: $BACKEND_PID"
    echo "  Frontend PID: $FRONTEND_PID"
    echo ""
    print_status "To stop services: kill $BACKEND_PID $FRONTEND_PID"
    
    # Keep script running and show logs
    print_status "Press Ctrl+C to stop all services..."
    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
    wait
}

# Function to show help
show_help() {
    echo "TrafficLab Development Environment Startup Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  docker, -d    Start with Docker Compose (recommended)"
    echo "  local, -l     Start locally without Docker"
    echo "  help, -h      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker     # Start with Docker Compose"
    echo "  $0 local      # Start locally"
    echo "  $0            # Default to Docker Compose"
    echo ""
    echo "Prerequisites:"
    echo "  Docker mode:  Docker, Docker Compose"
    echo "  Local mode:   Python 3.8+, Node.js 16+, PostgreSQL"
    echo ""
}

# Main script logic
main() {
    # Change to script directory
    cd "$(dirname "$0")"
    
    print_status "TrafficLab Development Environment Startup"
    print_status "============================================="
    
    # Parse command line arguments
    case "${1:-docker}" in
        "docker"|"-d"|"")
            start_docker
            ;;
        "local"|"-l")
            start_local
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
