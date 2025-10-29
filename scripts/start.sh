#!/bin/bash
# =============================================================================
# Fake News Detector - Startup Script (Linux/macOS)
# =============================================================================
# This script sets up the environment and starts the Fake News Detector application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

# Functions
print_header() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}  Fake News Detector Startup${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_python() {
    print_step "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        echo "Please install Python 3.8 or higher and try again"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION found"
}

setup_virtual_environment() {
    print_step "Setting up virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
}

install_dependencies() {
    print_step "Installing dependencies..."
    
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install requirements
    pip install -r "$REQUIREMENTS_FILE"
    print_success "Dependencies installed"
}

setup_environment_file() {
    print_step "Setting up environment configuration..."
    
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$ENV_EXAMPLE" ]; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            print_success "Environment file created from template"
            print_info "Please edit .env file with your configuration before running the application"
        else
            print_error "Environment example file not found: $ENV_EXAMPLE"
            exit 1
        fi
    else
        print_info "Environment file already exists"
    fi
}

create_directories() {
    print_step "Creating necessary directories..."
    
    # Create data directories
    mkdir -p "$PROJECT_ROOT/data/raw"
    mkdir -p "$PROJECT_ROOT/data/processed"
    mkdir -p "$PROJECT_ROOT/data/models"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/logs/training"
    
    print_success "Directories created"
}

check_models() {
    print_step "Checking for trained models..."
    
    MODELS_DIR="$PROJECT_ROOT/data/models"
    if [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
        print_info "No trained models found"
        print_info "The application will run in demo mode with mock predictions"
        print_info "To train models, run: python train_model.py"
    else
        print_success "Trained models found"
    fi
}

start_application() {
    print_step "Starting Fake News Detector application..."
    
    cd "$PROJECT_ROOT"
    
    # Check if Streamlit is installed
    if ! command -v streamlit &> /dev/null; then
        print_error "Streamlit is not installed"
        print_info "Installing Streamlit..."
        pip install streamlit
    fi
    
    print_success "Starting Streamlit application..."
    print_info "The application will open in your default web browser"
    print_info "If it doesn't open automatically, go to: http://localhost:8501"
    print_info ""
    print_info "Press Ctrl+C to stop the application"
    print_info ""
    
    # Start the application
    streamlit run app.py
}

# Main execution
main() {
    print_header
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    # Setup steps
    check_python
    setup_virtual_environment
    install_dependencies
    setup_environment_file
    create_directories
    check_models
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    
    # Start the application
    start_application
}

# Handle script arguments
case "${1:-}" in
    --setup-only)
        print_header
        cd "$PROJECT_ROOT"
        check_python
        setup_virtual_environment
        install_dependencies
        setup_environment_file
        create_directories
        check_models
        print_success "Setup completed! Run './scripts/start.sh' to start the application"
        ;;
    --help|-h)
        echo "Fake News Detector Startup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --setup-only    Only run setup steps, don't start the application"
        echo "  --help, -h      Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  PYTHON_CMD      Python command to use (default: auto-detect)"
        echo "  SKIP_VENV       Skip virtual environment setup (default: false)"
        echo ""
        ;;
    *)
        main
        ;;
esac