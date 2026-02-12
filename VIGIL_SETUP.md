# VIGIL-RAG Setup Guide

This guide walks you through setting up the VIGIL-RAG infrastructure (Postgres + Qdrant) and running your first end-to-end video retrieval.

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker + docker-compose (Linux)
- Python 3.10+
- ~5GB disk space for containers + data

## Quick Start (5 minutes)

### 1. Start Infrastructure Services

```bash
# Copy environment template
cp .env.example .env

# Start Postgres + Qdrant (detached mode)
docker-compose up -d postgres qdrant

# Verify services are healthy
docker-compose ps
# Both should show "Up" and "healthy"
```

**Expected output:**
```
NAME             STATUS                   PORTS
vigil-postgres   Up 10 seconds (healthy)  0.0.0.0:5432->5432/tcp
vigil-qdrant     Up 10 seconds (healthy)  0.0.0.0:6333->6333/tcp, 0.0.0.0:6334->6334/tcp
```

### 2. Run Database Migrations

```bash
# Install vigil dependencies (includes alembic, psycopg2, qdrant-client)
pip install -e ".[vigil]"

# Apply migrations (creates all 8 tables)
alembic upgrade head
```

**Expected output:**
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> 4033bc64052e, Initial VIGIL-RAG schema
```

### 3. Verify Database Setup

```bash
# Connect to Postgres
docker exec -it vigil-postgres psql -U vigil_user -d vigil

# Check tables exist
\dt
# Should list: videos, clips, embeddings, events, queries, ingest_jobs, score_jobs, training_jobs

# Exit psql
\q
```

### 4. Verify Qdrant Setup

```bash
# Check Qdrant health
curl http://localhost:6333/health
# Should return: {"title":"qdrant - vector search engine","version":"1.7.4"}

# List collections (should be empty initially)
curl http://localhost:6333/collections
# Should return: {"result":{"collections":[]}}
```

## Done Conditions ✅

- [ ] `docker-compose ps` shows postgres and qdrant as "healthy"
- [ ] `alembic upgrade head` completes without errors
- [ ] `psql` shows 8 tables (videos, clips, embeddings, events, queries, ingest_jobs, score_jobs, training_jobs)
- [ ] Qdrant responds to health checks on port 6333

## Next Steps

### Option A: Run E2E Smoke Test (Recommended)
```bash
# Coming soon: scripts/vigil_smoke_e2e.py
python scripts/vigil_smoke_e2e.py --video test_video.mp4
```

### Option B: Start SOPilot Services (Full Stack)
```bash
# Start all services (API + workers + Postgres + Qdrant)
docker-compose up -d

# Check logs
docker-compose logs -f sopilot-api
```

## Troubleshooting

### Postgres connection refused
```bash
# Check if Postgres is running
docker-compose ps postgres

# Restart if needed
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

### Qdrant not responding
```bash
# Check Qdrant logs
docker-compose logs qdrant

# Verify port not in use
netstat -an | findstr "6333"  # Windows
# or
lsof -i :6333  # Mac/Linux
```

### Alembic migration fails
```bash
# Check VIGIL_POSTGRES_URL in .env
cat .env | grep VIGIL_POSTGRES_URL

# Should be: postgresql://vigil_user:vigil_dev_password@localhost:5432/vigil

# Verify Postgres is accessible
psql -U vigil_user -h localhost -d vigil -c "SELECT 1"
# Enter password: vigil_dev_password
```

## Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────┐
│  VIGIL-RAG Infrastructure                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐               │
│  │  Postgres    │      │   Qdrant     │               │
│  │  Port: 5432  │      │  Port: 6333  │               │
│  │              │      │              │               │
│  │  8 tables:   │      │  4 levels:   │               │
│  │  - videos    │      │  - shot      │               │
│  │  - clips     │      │  - micro     │               │
│  │  - embeddings│      │  - meso      │               │
│  │  - events    │      │  - macro     │               │
│  │  - queries   │      │              │               │
│  │  - *_jobs    │      │              │               │
│  └──────────────┘      └──────────────┘               │
│         │                      │                       │
│         └──────────┬───────────┘                       │
│                    │                                   │
│              ┌─────▼──────┐                            │
│              │   Python   │                            │
│              │  Services  │                            │
│              └────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

## Data Persistence

All data is persisted in Docker volumes:
- `postgres_data`: Database files (~100MB baseline, grows with video metadata)
- `qdrant_data`: Vector index files (~1GB per 10K clips @ 768-dim)

To **reset all data** (WARNING: destructive):
```bash
docker-compose down -v  # Removes containers AND volumes
docker-compose up -d postgres qdrant
alembic upgrade head
```

## Production Notes

For production deployment, update:
1. `.env`: Change passwords and tokens
2. `docker-compose.yml`: Add resource limits (CPU/memory)
3. Qdrant: Configure persistence settings
4. Postgres: Configure connection pooling (pgbouncer)

See `ARCHITECTURE.md` for full production deployment guide.
