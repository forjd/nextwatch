-- Web app support. See docs/plans/08-app.md.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX title_name_trgm_idx ON title USING gin (name gin_trgm_ops);

-- Owner settings: streaming services, type mix, re-rank knobs. One JSON value per key.
CREATE TABLE app_setting (
    key         text PRIMARY KEY,
    value       jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
