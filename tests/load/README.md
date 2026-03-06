# SOPilot Load Tests

## Prerequisites

```bash
pip install locust>=2.29.0
# or
pip install sopilot[dev]
```

## Quick Start

```bash
# Interactive UI mode (http://localhost:8089)
locust -f tests/load/locustfile.py --host http://localhost:8000

# Headless mode — 50 users, 5/s spawn rate, 60 seconds
locust -f tests/load/locustfile.py \
  --host http://localhost:8000 \
  --users 50 --spawn-rate 5 \
  --run-time 60s --headless

# Generate HTML report
locust -f tests/load/locustfile.py \
  --host http://localhost:8000 \
  --users 50 --spawn-rate 5 \
  --run-time 60s --headless \
  --html load-report.html
```

## Scenarios

| Scenario | Command | Purpose |
|----------|---------|---------|
| Spike | `--shape-class SpikeScenario` | 200 users in 10s → memory/GC pressure |
| Soak | `--shape-class SoakScenario` | 20 users × 10 min → memory leak detection |
| Stress | `--shape-class StressScenario` | Step +50 users/30s → find breaking point |

```bash
locust -f tests/load/locustfile.py \
  -f tests/load/scenarios.py \
  --host http://localhost:8000 \
  --shape-class SpikeScenario --headless
```

## Automated Run

```bash
bash scripts/run_load_test.sh
```

This script starts the server, waits for `/health`, runs 60-second load test,
saves `load-report.html`, and shuts down the server.

## Traffic Distribution

| Endpoint | Weight |
|----------|--------|
| GET /health | 10% |
| GET /readiness | 5% |
| GET /videos | 20% |
| GET /analytics | 15% |
| GET /vigil/sessions | 15% |
| GET /vigil/perception/early-warning | 10% |
| GET /vigil/perception/health-score | 10% |
| GET /vigil/perception/daily-report | 5% |
| GET /metrics | 10% |
