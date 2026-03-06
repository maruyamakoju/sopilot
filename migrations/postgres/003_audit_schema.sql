-- SOPilot v4.0.0 — PostgreSQL audit log schema
-- Used by Item 3 (OAuth2 + Audit Logging)

CREATE TABLE IF NOT EXISTS audit_log (
    id          SERIAL PRIMARY KEY,
    ts          TEXT NOT NULL,
    method      TEXT NOT NULL,
    path        TEXT NOT NULL,
    status_code INTEGER,
    duration_ms DOUBLE PRECISION,
    client_id   TEXT,
    role        TEXT,
    ip_address  TEXT,
    request_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_ts        ON audit_log(ts);
CREATE INDEX IF NOT EXISTS idx_audit_client    ON audit_log(client_id);
CREATE INDEX IF NOT EXISTS idx_audit_path      ON audit_log(path);
CREATE INDEX IF NOT EXISTS idx_audit_method    ON audit_log(method);
