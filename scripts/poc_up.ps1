# Insurance MVP PoC — One-command startup (PowerShell)
# Usage:
#   .\scripts\poc_up.ps1          # Mock/Replay mode (no GPU)
#   .\scripts\poc_up.ps1 -Gpu     # Real GPU inference mode
#   .\scripts\poc_up.ps1 -Down    # Stop all services
param(
    [switch]$Gpu,
    [switch]$Down,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ComposeFile = Join-Path $ProjectRoot "docker-compose.poc.yml"
$EnvFile = Join-Path $ProjectRoot ".env.poc"

function Write-Info  { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

if ($Help) {
    Write-Host "Usage: .\scripts\poc_up.ps1 [-Gpu] [-Down]"
    Write-Host "  -Gpu   Enable real GPU inference (requires NVIDIA GPU with >=14GB VRAM)"
    Write-Host "  -Down  Stop all PoC services"
    exit 0
}

# Stop mode
if ($Down) {
    Write-Info "Stopping PoC services..."
    docker compose -f $ComposeFile --profile gpu down 2>$null
    docker compose -f $ComposeFile down 2>$null
    Write-Info "All PoC services stopped."
    exit 0
}

# Check docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Err "Docker not found. Install: https://docs.docker.com/get-docker/"
}

# Create .env.poc if missing
if (-not (Test-Path $EnvFile)) {
    Write-Warn ".env.poc not found - creating from template"
    Copy-Item "$ProjectRoot\.env.poc.example" $EnvFile
}

# Build and start
Write-Host ""
Write-Info "=== Insurance MVP PoC ==="

if ($Gpu) {
    Write-Info "Mode: Real GPU Inference (Qwen2.5-VL-7B-Instruct)"

    # Check NVIDIA GPU
    try {
        $vramInfo = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($vramInfo -and [int]$vramInfo -lt 14000) {
            Write-Warn "GPU VRAM: ${vramInfo}MB - Qwen2.5-VL-7B requires >=14GB"
        } else {
            Write-Info "GPU VRAM: ${vramInfo}MB - OK"
        }
    } catch {
        Write-Err "nvidia-smi not found. Install NVIDIA drivers for GPU mode."
    }

    docker compose -f $ComposeFile --env-file $EnvFile --profile gpu up --build -d
} else {
    Write-Info "Mode: Mock + Replay (no GPU required)"
    docker compose -f $ComposeFile --env-file $EnvFile up --build -d
}

# Wait for health
Write-Info "Waiting for dashboard to be ready..."
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch { }
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 2
}

Write-Host ""
if ($ready) {
    Write-Info "Dashboard ready!"
    Write-Host ""
    Write-Host "  Open: http://localhost:8501"
    Write-Host ""
    if ($Gpu) {
        Write-Host "  Backend: Real (Qwen2.5-VL-7B on GPU)"
        Write-Host "  Select 'Real (Qwen2.5-VL-7B)' in the sidebar"
    } else {
        Write-Host "  Backend: Mock + Replay"
        Write-Host "  Select 'Replay' in the sidebar to view RTX 5090 results"
    }
    Write-Host ""
    Write-Host "  Stop:  .\scripts\poc_up.ps1 -Down"
    Write-Host ""
} else {
    Write-Warn "Dashboard did not become healthy within 60s"
    Write-Warn "Check logs: docker compose -f $ComposeFile logs"
    exit 1
}
