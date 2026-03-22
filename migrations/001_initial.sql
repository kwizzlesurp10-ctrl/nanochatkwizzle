-- PostgreSQL 14+. Apply with: psql $DATABASE_URL -f migrations/001_initial.sql
-- Or rely on SQLAlchemy create_all (scripts/chat_web.py / scripts/seed_db.py) for dev.

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    ab_group        VARCHAR(8) NOT NULL,
    model           TEXT NOT NULL,
    backend         VARCHAR(32) NOT NULL CHECK (backend IN ('ollama', 'nanochat')),
    created_at      TIMESTAMPTZ NOT NULL,
    last_seen_at    TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions (created_at DESC);

CREATE TABLE IF NOT EXISTS analytics_events (
    id              BIGSERIAL PRIMARY KEY,
    event_type      VARCHAR(32) NOT NULL CHECK (event_type IN ('feedback', 'metric')),
    session_id      TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    payload         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analytics_session_time ON analytics_events (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_type_time ON analytics_events (event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_metric_name ON analytics_events ((payload ->> 'metric_name'))
    WHERE event_type = 'metric';

CREATE TABLE IF NOT EXISTS user_training_text (
    id                  BIGSERIAL PRIMARY KEY,
    body                TEXT NOT NULL CHECK (char_length(body) > 0),
    source              TEXT NOT NULL DEFAULT 'save_to_training',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingested_into_run   TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_training_created ON user_training_text (created_at DESC);
