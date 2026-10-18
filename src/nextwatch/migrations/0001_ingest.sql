-- Raw TMDB data and the ingest work queue. See docs/plans/01-ingest.md.

CREATE TABLE tmdb_title (
    media_type        text        NOT NULL CHECK (media_type IN ('movie', 'tv')),
    tmdb_id           integer     NOT NULL,
    name              text,
    vote_count        integer,
    vote_average      real,
    popularity        real,
    release_date      date,
    eligible          boolean     NOT NULL DEFAULT false,
    deleted_at        timestamptz,
    light_fetched_at  timestamptz,
    full_fetched_at   timestamptz,
    payload           jsonb,
    PRIMARY KEY (media_type, tmdb_id)
);

CREATE INDEX tmdb_title_refresh_idx ON tmdb_title (full_fetched_at NULLS FIRST)
    WHERE eligible AND deleted_at IS NULL;

CREATE TABLE ingest_queue (
    media_type       text        NOT NULL CHECK (media_type IN ('movie', 'tv')),
    tmdb_id          integer     NOT NULL,
    kind             text        NOT NULL CHECK (kind IN ('light', 'full')),
    reason           text        NOT NULL
        CHECK (reason IN ('backfill', 'new', 'changed', 'refresh', 'promote', 'verify')),
    -- Lower runs first. Within a priority, more popular titles run first.
    priority         smallint    NOT NULL DEFAULT 10,
    popularity       real        NOT NULL DEFAULT 0,
    attempts         smallint    NOT NULL DEFAULT 0,
    next_attempt_at  timestamptz NOT NULL DEFAULT now(),
    claimed_until    timestamptz,
    last_error       text,
    PRIMARY KEY (media_type, tmdb_id, kind)
);

CREATE INDEX ingest_queue_ready_idx ON ingest_queue (priority, popularity DESC)
    WHERE claimed_until IS NULL;

-- MovieLens movies are always eligible, whatever their TMDB vote count.
CREATE TABLE movielens_link (
    movielens_id  integer PRIMARY KEY,
    imdb_id       text,
    tmdb_id       integer
);

CREATE INDEX movielens_link_tmdb_idx ON movielens_link (tmdb_id);

-- Small key-value store for job state, such as the last successful sync date.
CREATE TABLE ingest_state (
    key    text PRIMARY KEY,
    value  text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
