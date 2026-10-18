-- Normalised catalogue, owner data and model storage. See docs/plans/02-storage.md.
-- Catalogue tables are rebuilt from tmdb_title by `nextwatch normalise`.

CREATE TABLE collection (
    id    integer PRIMARY KEY,
    name  text NOT NULL
);

CREATE TABLE title (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    media_type         text    NOT NULL CHECK (media_type IN ('movie', 'tv')),
    tmdb_id            integer NOT NULL,
    imdb_id            text,
    name               text    NOT NULL,
    original_name      text,
    original_language  text,
    overview           text,
    tagline            text,
    release_date       date,
    year               smallint,
    runtime            smallint,
    status             text,
    number_of_seasons  smallint,
    vote_count         integer NOT NULL DEFAULT 0,
    vote_average       real,
    popularity         real,
    poster_path        text,
    certification_gb   text,
    collection_id      integer REFERENCES collection (id),
    -- tmdb_title.full_fetched_at of the payload this row was built from.
    source_fetched_at  timestamptz NOT NULL,
    normalised_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (media_type, tmdb_id)
);

CREATE INDEX title_imdb_idx ON title (imdb_id);

CREATE TABLE genre (
    id    integer PRIMARY KEY,
    name  text NOT NULL
);

CREATE TABLE title_genre (
    title_id  bigint  NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    genre_id  integer NOT NULL REFERENCES genre (id),
    PRIMARY KEY (title_id, genre_id)
);

CREATE TABLE keyword (
    id    integer PRIMARY KEY,
    name  text NOT NULL
);

CREATE TABLE title_keyword (
    title_id    bigint  NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    keyword_id  integer NOT NULL REFERENCES keyword (id),
    PRIMARY KEY (title_id, keyword_id)
);

CREATE INDEX title_keyword_keyword_idx ON title_keyword (keyword_id);

CREATE TABLE person (
    id                    integer PRIMARY KEY,
    name                  text NOT NULL,
    gender                smallint,
    known_for_department  text,
    popularity            real
);

CREATE TABLE credit (
    title_id       bigint  NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    person_id      integer NOT NULL REFERENCES person (id),
    role           text    NOT NULL CHECK (role IN ('cast', 'crew')),
    department     text,
    job            text,
    character      text,
    cast_order     smallint,
    episode_count  smallint
);

CREATE INDEX credit_title_idx ON credit (title_id);
CREATE INDEX credit_person_idx ON credit (person_id);

CREATE TABLE company (
    id              integer PRIMARY KEY,
    name            text NOT NULL,
    origin_country  text
);

CREATE TABLE title_company (
    title_id    bigint  NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    company_id  integer NOT NULL REFERENCES company (id),
    kind        text    NOT NULL CHECK (kind IN ('production', 'network')),
    PRIMARY KEY (title_id, company_id, kind)
);

CREATE TABLE title_country (
    title_id  bigint NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    country   text   NOT NULL,
    PRIMARY KEY (title_id, country)
);

CREATE TABLE provider (
    id    integer PRIMARY KEY,
    name  text NOT NULL
);

CREATE TABLE title_provider (
    title_id          bigint   NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    provider_id       integer  NOT NULL REFERENCES provider (id),
    kind              text     NOT NULL CHECK (kind IN ('flatrate', 'free', 'ads', 'rent', 'buy')),
    display_priority  smallint,
    PRIMARY KEY (title_id, provider_id, kind)
);

-- TMDB's own recommendations, by TMDB ID because the target may not be eligible.
CREATE TABLE title_tmdb_rec (
    title_id        bigint  NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    rank            smallint NOT NULL,
    rec_media_type  text    NOT NULL,
    rec_tmdb_id     integer NOT NULL,
    PRIMARY KEY (title_id, rank)
);

-- Owner data: the only data we cannot rebuild from TMDB. RESTRICT so a TMDB delete
-- never removes a rating.
CREATE TABLE rating (
    title_id  bigint       PRIMARY KEY REFERENCES title (id) ON DELETE RESTRICT,
    score     numeric(2, 1) NOT NULL CHECK (score BETWEEN 0.5 AND 5 AND score * 2 = floor(score * 2)),
    source    text         NOT NULL CHECK (source IN ('app', 'onboarding', 'import')),
    rated_at  timestamptz  NOT NULL DEFAULT now()
);

CREATE TABLE event (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title_id  bigint REFERENCES title (id) ON DELETE RESTRICT,
    kind      text   NOT NULL
        CHECK (kind IN ('seen', 'dismiss', 'watchlist', 'unwatchlist', 'impression', 'click')),
    context   jsonb,
    at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX event_title_idx ON event (title_id, kind);

CREATE TABLE model_artifact (
    name        text        NOT NULL,
    version     integer     NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    blob        bytea       NOT NULL,
    meta        jsonb       NOT NULL DEFAULT '{}',
    PRIMARY KEY (name, version)
);

CREATE TABLE eval_run (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    git_sha     text,
    config      jsonb NOT NULL,
    metrics     jsonb NOT NULL
);
