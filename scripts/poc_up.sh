#!/usr/bin/env bash
# Insurance MVP PoC — One-command startup
# Usage:
#   ./scripts/poc_up.sh          # Mock/Replay mode (no GPU)
#   ./scripts/poc_up.sh --gpu    # Real GPU inference mode
#   ./scripts/poc_up.sh --down   # Stop all services
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.poc.yml"
ENV_FILE="$PROJECT_ROOT/.env.poc"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Parse args
GPU_MODE=false
DOWN_MODE=false
for arg in "$@"; do
    case "$arg" in
        --gpu)  GPU_MODE=true ;;
        --down) DOWN_MODE=true ;;
        -h|--help)
            echo "Usage: $0 [--gpu] [--down]"
            echo "  --gpu   Enable real GPU inference (requires NVIDIA GPU with >=14GB VRAM)"
            echo "  --down  Stop all PoC services"
            exit 0
            ;;
    esac
done

# Stop mode
if [ "$DOWN_MODE" = true ]; then
    info "Stopping PoC services..."
    docker compose -f "$COMPOSE_FILE" --profile gpu down 2>/dev/null || true
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    info "All PoC services stopped."
    exit 0
fi

# Check docker
command -v docker >/dev/null 2>&1 || error "Docker not found. Install: https://docs.docker.com/get-docker/"

# Create .env.poc if missing
if [ ! -f "$ENV_FILE" ]; then
    warn ".env.poc not found — creating from template"
    cp "$PROJECT_ROOT/.env.poc.example" "$ENV_FILE"
fi

# Build and start
info "=== Insurance MVP PoC ==="
if [ "$GPU_MODE" = true ]; then
    info "Mode: Real GPU Inference (Qwen2.5-VL-7B-Instruct)"

    # Check NVIDIA GPU
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        error "nvidia-smi not found. Install NVIDIA drivers for GPU mode."
    fi
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" -lt 14000 ]; then
        warn "GPU VRAM: ${VRAM_MB}MB — Qwen2.5-VL-7B requires >=14GB"
    else
        info "GPU VRAM: ${VRAM_MB}MB — OK"
    fi

    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile gpu up --build -d
else
    info "Mode: Mock + Replay (no GPU required)"
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up --build -d
fi

# Wait for health
info "Waiting for dashboard to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        echo ""
        info "Dashboard ready!"
        echo ""
        echo "  Open: http://localhost:8501"
        echo ""
        if [ "$GPU_MODE" = true ]; then
            echo "  Backend: Real (Qwen2.5-VL-7B on GPU)"
            echo "  Select 'Real (Qwen2.5-VL-7B)' in the sidebar"
        else
            echo "  Backend: Mock + Replay"
            echo "  Select 'Replay' in the sidebar to view RTX 5090 results"
        fi
        echo ""
        echo "  Stop:  ./scripts/poc_up.sh --down"
        echo ""
        exit 0
    fi
    printf "."
    sleep 2
done

warn "Dashboard did not become healthy within 60s"
warn "Check logs: docker compose -f $COMPOSE_FILE logs"
exit 1
