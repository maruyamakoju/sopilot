# Multi-stage Dockerfile for SOPilot (CPU-only, production)
# Targets: sopilot-api (uvicorn), sopilot-worker (rq), sopilot-watch
# For GPU support, use docker/Dockerfile.gpu

# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install deps first (layer caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 sopilot \
    && mkdir -p /app /data/raw /data/embeddings /data/models /data/reports /data/index \
    && chown -R sopilot:sopilot /app /data

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
WORKDIR /app
COPY --chown=sopilot:sopilot src/ ./src/
COPY --chown=sopilot:sopilot pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps -e .

USER sopilot

# Defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SOPILOT_DATA_DIR=/data \
    SOPILOT_QUEUE_BACKEND=rq \
    SOPILOT_REDIS_URL=redis://redis:6379/0

EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Default: API server
CMD ["uvicorn", "sopilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
