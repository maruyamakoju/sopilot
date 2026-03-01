# =============================================================================
# SOPilot -- Multi-stage Docker build
# =============================================================================
# Build:  docker build -t sopilot .
# Run:    docker run -p 8000:8000 -v sopilot-data:/app/data sopilot
# =============================================================================

# =============================================================================
# Stage 1: builder -- install Python deps into an isolated prefix
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time tools required to compile C-extension wheels
# (numpy, scipy, numba, etc.) but keep the layer as small as possible.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the package manifest first so this expensive layer is cached
# independently from application source changes.
COPY pyproject.toml ./

# Install core deps + numba JIT (speed extra) into /install so we can copy
# the whole prefix into the runtime stage without a venv activation wrapper.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefix=/install ".[speed]"

# =============================================================================
# Stage 2: runtime -- minimal production image
# =============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="SOPilot Team"
LABEL description="SOPilot -- on-prem SOP video evaluation service (v0.9)"

# Runtime system libraries:
#   libglib2.0-0  -- required by opencv-python-headless
#   curl          -- used by the HEALTHCHECK instruction
# NOTE: libgl1 is NOT needed because we use the headless OpenCV variant.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Suppress .pyc generation and enable unbuffered stdout/stderr so log lines
# appear immediately in `docker logs` output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Bring in the pre-built packages from the builder stage.
# Placing them under /usr/local means they are on sys.path automatically.
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy only the application package (includes ui/*.html and core/templates).
# Tests, docs, scripts, and data directories are excluded via .dockerignore.
COPY sopilot/ ./sopilot/

# Create a non-root system user/group for container security.
# --no-create-home keeps the image smaller; the app never needs a home dir.
RUN groupadd --gid 1000 sopilot \
 && useradd --uid 1000 --gid sopilot --no-create-home --shell /sbin/nologin sopilot \
 && mkdir -p /app/data/raw \
 && chown -R sopilot:sopilot /app/data

# Persistent data volume: SQLite database + raw video files live here.
# Mount a named volume or host bind-mount at runtime; never bake data into
# the image.
VOLUME ["/app/data"]

# Sensible production defaults -- all can be overridden via environment or
# docker-compose.yml.
ENV SOPILOT_DATA_DIR=/app/data \
    SOPILOT_EMBEDDER_BACKEND=color-motion \
    SOPILOT_ALLOW_EMBEDDER_FALLBACK=true \
    SOPILOT_LOG_JSON=true \
    SOPILOT_LOG_LEVEL=INFO

EXPOSE 8000

# Lightweight liveness probe.  /health is on PUBLIC_PATHS so it bypasses the
# API-key middleware -- no credentials needed here.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

USER sopilot

# --factory instructs uvicorn to call create_app() and use its return value
# as the ASGI application, which is required by SOPilot's app-factory pattern.
# --log-config /dev/null suppresses uvicorn's own logging so that only the
# application's structured JSON logs (via logging_config.py) appear.
CMD ["uvicorn", "sopilot.main:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
