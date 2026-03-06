-- SOPilot v4.0.0 — PostgreSQL initial schema
-- Core tables: videos, clips, score_jobs, task_profiles, sop_steps, score_reviews

CREATE TABLE IF NOT EXISTS videos (
    id                   SERIAL PRIMARY KEY,
    task_id              TEXT NOT NULL,
    site_id              TEXT,
    camera_id            TEXT,
    operator_id_hash     TEXT,
    recorded_at          TEXT,
    is_gold              BOOLEAN NOT NULL DEFAULT FALSE,
    file_path            TEXT,
    status               TEXT NOT NULL DEFAULT 'processing',
    clip_count           INTEGER NOT NULL DEFAULT 0,
    step_boundaries_json TEXT NOT NULL DEFAULT '[]',
    embedding_model      TEXT NOT NULL DEFAULT 'color-motion-v1',
    original_filename    TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    error                TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_task       ON videos(task_id);
CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);
CREATE INDEX IF NOT EXISTS idx_videos_operator   ON videos(operator_id_hash);
CREATE INDEX IF NOT EXISTS idx_videos_site       ON videos(site_id);

CREATE TABLE IF NOT EXISTS clips (
    id             SERIAL PRIMARY KEY,
    video_id       INTEGER NOT NULL REFERENCES videos(id),
    clip_index     INTEGER NOT NULL,
    start_sec      DOUBLE PRECISION NOT NULL,
    end_sec        DOUBLE PRECISION NOT NULL,
    embedding_json TEXT NOT NULL,
    quality_flag   TEXT,
    created_at     TEXT NOT NULL,
    UNIQUE(video_id, clip_index)
);

CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id);

CREATE TABLE IF NOT EXISTS score_jobs (
    id               SERIAL PRIMARY KEY,
    gold_video_id    INTEGER NOT NULL,
    trainee_video_id INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'queued',
    score_json       TEXT,
    weights_json     TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    error            TEXT
);

CREATE INDEX IF NOT EXISTS idx_score_jobs_status         ON score_jobs(status);
CREATE INDEX IF NOT EXISTS idx_score_jobs_gold           ON score_jobs(gold_video_id);
CREATE INDEX IF NOT EXISTS idx_score_jobs_trainee        ON score_jobs(trainee_video_id);
CREATE INDEX IF NOT EXISTS idx_score_jobs_created_at     ON score_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_score_jobs_status_created ON score_jobs(status, created_at);

CREATE TABLE IF NOT EXISTS task_profiles (
    task_id              TEXT PRIMARY KEY,
    task_name            TEXT NOT NULL,
    pass_score           DOUBLE PRECISION NOT NULL,
    retrain_score        DOUBLE PRECISION NOT NULL,
    default_weights_json TEXT NOT NULL,
    deviation_policy_json TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sop_steps (
    id                    SERIAL PRIMARY KEY,
    task_id               TEXT NOT NULL,
    step_index            INTEGER NOT NULL,
    name_ja               TEXT NOT NULL DEFAULT '',
    name_en               TEXT NOT NULL DEFAULT '',
    expected_duration_sec DOUBLE PRECISION,
    min_duration_sec      DOUBLE PRECISION,
    max_duration_sec      DOUBLE PRECISION,
    is_critical           BOOLEAN NOT NULL DEFAULT FALSE,
    description           TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    UNIQUE(task_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_sop_steps_task ON sop_steps(task_id);

CREATE TABLE IF NOT EXISTS score_reviews (
    job_id     INTEGER PRIMARY KEY REFERENCES score_jobs(id),
    verdict    TEXT NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
