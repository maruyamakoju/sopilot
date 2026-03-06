-- SOPilot v4.0.0 — PostgreSQL VigilPilot schema
-- Tables: vigil_sessions, vigil_events, webhooks, camera_groups

CREATE TABLE IF NOT EXISTS vigil_sessions (
    id          TEXT PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    frame_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_vigil_sessions_camera ON vigil_sessions(camera_id);
CREATE INDEX IF NOT EXISTS idx_vigil_sessions_status ON vigil_sessions(status);

CREATE TABLE IF NOT EXISTS vigil_events (
    id            SERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES vigil_sessions(id),
    camera_id     TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    severity      TEXT NOT NULL DEFAULT 'low',
    description   TEXT,
    frame_path    TEXT,
    confidence    DOUBLE PRECISION,
    rule_id       TEXT,
    zone_id       TEXT,
    metadata_json TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vigil_events_session   ON vigil_events(session_id);
CREATE INDEX IF NOT EXISTS idx_vigil_events_camera    ON vigil_events(camera_id);
CREATE INDEX IF NOT EXISTS idx_vigil_events_type      ON vigil_events(event_type);
CREATE INDEX IF NOT EXISTS idx_vigil_events_severity  ON vigil_events(severity);
CREATE INDEX IF NOT EXISTS idx_vigil_events_created   ON vigil_events(created_at);

CREATE TABLE IF NOT EXISTS webhooks (
    id         SERIAL PRIMARY KEY,
    url        TEXT NOT NULL UNIQUE,
    secret     TEXT,
    enabled    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS camera_groups (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    camera_ids  TEXT NOT NULL DEFAULT '[]',
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
