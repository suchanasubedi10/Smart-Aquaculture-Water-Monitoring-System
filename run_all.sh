#!/usr/bin/env bash
# =============================================================================
# Water Intelligence Project - Complete Setup & Run Script
# =============================================================================
# Usage:
#   ./run_all.sh              # Full setup + training + start server
#   ./run_all.sh --skip-train # Skip training, just start server
#   ./run_all.sh --test       # Run tests only
#   ./run_all.sh --help       # Show help
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="${VENV_DIR:-.venv}"
DATA_DIR="${DATA_DIR:-data}"
MODELS_DIR="${MODELS_DIR:-models}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
SEED="${SEED:-42}"
N_SAMPLES="${N_SAMPLES:-10000}"
EPOCHS="${EPOCHS:-50}"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Help message
show_help() {
    echo "Water Intelligence Project - Setup & Run Script"
    echo ""
    echo "Usage: ./run_all.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-train    Skip data generation and model training"
    echo "  --skip-lstm     Skip LSTM training (faster, RF only)"
    echo "  --test          Run tests only, don't start server"
    echo "  --lint          Run linting and type checks"
    echo "  --no-server     Setup and train, but don't start server"
    echo "  --clean         Remove generated files and start fresh"
    echo "  --help          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  VENV_DIR        Virtual environment directory (default: venv)"
    echo "  DATA_DIR        Data directory (default: data)"
    echo "  MODELS_DIR      Models directory (default: models)"
    echo "  PORT            Server port (default: 8000)"
    echo "  HOST            Server host (default: 0.0.0.0)"
    echo "  SEED            Random seed (default: 42)"
    echo "  N_SAMPLES       Number of synthetic samples (default: 10000)"
    echo "  EPOCHS          LSTM training epochs (default: 50)"
    echo ""
    echo "Examples:"
    echo "  ./run_all.sh                      # Full setup and run"
    echo "  ./run_all.sh --skip-train         # Quick start with existing models"
    echo "  PORT=9000 ./run_all.sh            # Run on port 9000"
    echo "  N_SAMPLES=50000 ./run_all.sh      # Generate more data"
}

# Parse command line arguments
SKIP_TRAIN=false
SKIP_LSTM=false
TEST_ONLY=false
LINT_ONLY=false
NO_SERVER=false
CLEAN=false

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --skip-lstm) SKIP_LSTM=true ;;
        --test) TEST_ONLY=true ;;
        --lint) LINT_ONLY=true ;;
        --no-server) NO_SERVER=true ;;
        --clean) CLEAN=true ;;
        --help) show_help; exit 0 ;;
        *) log_error "Unknown option: $arg"; show_help; exit 1 ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
    log_warn "Cleaning generated files..."
    rm -rf "$VENV_DIR"
    rm -rf "$DATA_DIR"/*.csv "$DATA_DIR"/*.json
    rm -rf "$MODELS_DIR"/*.pkl "$MODELS_DIR"/*.pth "$MODELS_DIR"/*.json
    rm -rf logs/*.log
    rm -rf __pycache__ .pytest_cache .mypy_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    log_success "Cleaned!"
fi

# Check Python version
log_info "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.10"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    log_error "Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
    exit 1
fi
log_success "Python $PYTHON_VERSION detected"

# Activate existing virtual environment (assumes venv already exists)
if [ -d "$VENV_DIR" ]; then
    log_info "Activating virtual environment: $VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment activated"
else
    log_error "Virtual environment not found at $VENV_DIR"
    log_error "Please create it first: python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Create directories if they don't exist
mkdir -p "$DATA_DIR" "$MODELS_DIR" logs

# Copy .env.example to .env if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    log_info "Creating .env from .env.example..."
    cp .env.example .env
    log_success ".env created (edit as needed)"
fi

# Run linting if requested
if [ "$LINT_ONLY" = true ]; then
    log_info "Running linting and type checks..."
    echo ""
    log_info "Running flake8..."
    flake8 scripts/ src/ --max-line-length=100 --ignore=E501,W503 || true
    echo ""
    log_info "Running mypy..."
    mypy scripts/ src/ --ignore-missing-imports || true
    echo ""
    log_info "Checking formatting with black..."
    black scripts/ src/ --check --diff || true
    log_success "Linting complete!"
    exit 0
fi

# Run tests if requested
if [ "$TEST_ONLY" = true ]; then
    log_info "Running tests..."
    echo ""
    pytest tests/ -v --tb=short
    log_success "Tests complete!"
    exit 0
fi

# Data generation and training
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    log_info "=============================================="
    log_info "STEP 1: Generating synthetic data"
    log_info "=============================================="
    python scripts/gen_synthetic_full.py \
        --out-dir "$DATA_DIR" \
        --n-samples "$N_SAMPLES" \
        --seed "$SEED" \
        --add-anomalies 50
    log_success "Data generation complete!"

    echo ""
    log_info "=============================================="
    log_info "STEP 2: Training classical ML models"
    log_info "=============================================="
    python scripts/train_all_models.py \
        --data-csv "$DATA_DIR/raw_combined.csv" \
        --models-dir "$MODELS_DIR" \
        --seed "$SEED"
    log_success "Classical models trained!"

    if [ "$SKIP_LSTM" = false ]; then
        echo ""
        log_info "=============================================="
        log_info "STEP 3: Training LSTM forecasters"
        log_info "=============================================="
        python scripts/train_lstm_pytorch.py \
            --data-csv "$DATA_DIR/cleaned.csv" \
            --models-dir "$MODELS_DIR" \
            --epochs "$EPOCHS" \
            --seed "$SEED"
        log_success "LSTM models trained!"
    else
        log_warn "Skipping LSTM training (--skip-lstm)"
    fi

    echo ""
    log_info "=============================================="
    log_info "STEP 4: Evaluating classifier"
    log_info "=============================================="
    python scripts/evaluate_classifier.py evaluate \
        --data-csv "$DATA_DIR/cleaned.csv" \
        --models-dir "$MODELS_DIR" \
        --save-labeled
    log_success "Classifier evaluation complete!"

else
    log_warn "Skipping training (--skip-train)"
    
    # Verify models exist
    if [ ! -f "$MODELS_DIR/classifier_rf.pkl" ]; then
        log_error "No trained models found in $MODELS_DIR"
        log_error "Run without --skip-train first to train models"
        exit 1
    fi
fi

# Start server
if [ "$NO_SERVER" = false ]; then
    echo ""
    log_info "=============================================="
    log_info "Starting FastAPI server"
    log_info "=============================================="
    log_info "Server: http://$HOST:$PORT"
    log_info "API Docs: http://localhost:$PORT/docs"
    log_info "Dashboard: http://localhost:$PORT/static/index.html"
    log_info "Press Ctrl+C to stop"
    echo ""
    
    python -m uvicorn src.backend.main:app --reload

else
    log_success "Setup complete! (--no-server specified)"
    echo ""
    log_info "To start the server manually:"
    echo "  cd src/backend && uvicorn main:app --host $HOST --port $PORT --reload"
fi

