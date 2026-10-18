import datetime as dt

import pytest
from psycopg.types.json import Jsonb

from nextwatch.normalise import normalise, parse
from tests.payloads import movie_payload, tv_payload


def test_parse_movie():
    p = parse("movie", 603, movie_payload())
    t = p.title
    assert (t["name"], t["year"], t["runtime"], t["imdb_id"]) == (
        "The Matrix",
        1999,
        136,
        "tt0133093",
    )
    assert t["certification_gb"] == "18"  # theatrical beats digital
    assert t["collection_id"] == 2344
    # Cast in billing order; crew limited to the jobs we keep.
    assert [(c.person_id, c.cast_order) for c in p.credits if c.role == "cast"] == [
        (6384, 0),
        (2975, 1),
    ]
    assert sorted(c.job for c in p.credits if c.role == "crew") == ["Director", "Writer"]
    assert p.providers == [(8, "Netflix", "flatrate", 1), (2, "Apple TV", "rent", 4)]
    assert p.recs == [("movie", 604), ("tv", 1399)]


def test_parse_tv():
    p = parse("tv", 1396, tv_payload())
    t = p.title
    assert (t["name"], t["year"], t["runtime"], t["number_of_seasons"]) == (
        "Breaking Bad", 2008, 47, 5,
    )  # fmt: skip
    assert t["tagline"] is None and t["imdb_id"] == "tt0903747"
    assert t["certification_gb"] == "18"
    cast = [(c.person_id, c.cast_order, c.episode_count) for c in p.credits if c.role == "cast"]
    assert cast[:2] == [(17419, 0, 62), (84497, 1, 62)]
    crew = sorted((c.person_id, c.job) for c in p.credits if c.role == "crew")
    assert crew == [(66633, "Creator"), (66633, "Writer")]
    assert (174, "AMC", "US", "network") in p.companies


def test_parse_handles_sparse_payload():
    p = parse("movie", 1, {"id": 1, "title": "X", "credits": {"cast": [{"id": 3, "name": "A"}]}})
    assert p.title["name"] == "X" and p.title["year"] is None
    assert len(p.credits) == 1 and p.title["certification_gb"] is None


def store(conn, media_type, tmdb_id, payload, fetched=None, deleted=False):
    conn.execute(
        """
        INSERT INTO tmdb_title (media_type, tmdb_id, eligible, full_fetched_at, payload, deleted_at)
        VALUES (%s, %s, true, %s, %s, %s)
        ON CONFLICT (media_type, tmdb_id) DO UPDATE SET full_fetched_at = EXCLUDED.full_fetched_at,
            payload = EXCLUDED.payload, deleted_at = EXCLUDED.deleted_at
        """,
        (media_type, tmdb_id, fetched or dt.datetime.now(dt.UTC), Jsonb(payload),
         dt.datetime.now(dt.UTC) if deleted else None),
    )  # fmt: skip


def counts(conn):
    tables = ["title", "title_genre", "title_keyword", "credit", "person", "title_company",
              "title_country", "title_provider", "title_tmdb_rec", "genre", "keyword"]  # fmt: skip
    return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}


@pytest.mark.db
def test_normalise_is_idempotent(conn):
    store(conn, "movie", 603, movie_payload())
    store(conn, "tv", 1396, tv_payload())
    assert normalise(conn) == 2
    first = counts(conn)
    assert first["title"] == 2 and first["credit"] == 2 + 2 + 3 + 2
    assert normalise(conn) == 0
    assert counts(conn) == first


@pytest.mark.db
def test_normalise_rebuilds_changed_titles(conn):
    store(conn, "movie", 603, movie_payload())
    normalise(conn)
    title_id = conn.execute("SELECT id FROM title WHERE tmdb_id = 603").fetchone()[0]

    changed = movie_payload(genres=[{"id": 28, "name": "Action"}], vote_count=27000)
    store(conn, "movie", 603, changed, fetched=dt.datetime.now(dt.UTC) + dt.timedelta(seconds=1))
    assert normalise(conn) == 1
    row = conn.execute("SELECT id, vote_count FROM title WHERE tmdb_id = 603").fetchone()
    assert row == (title_id, 27000)  # same id, so owner data stays linked
    assert conn.execute("SELECT count(*) FROM title_genre").fetchone()[0] == 1


@pytest.mark.db
def test_rebuild_from_scratch_gives_same_counts(conn):
    store(conn, "movie", 603, movie_payload())
    store(conn, "tv", 1396, tv_payload())
    normalise(conn)
    before = counts(conn)
    conn.execute("TRUNCATE title, person, genre, keyword, company, provider, collection CASCADE")
    normalise(conn)
    assert counts(conn) == before


@pytest.mark.db
def test_deleted_titles_removed_unless_rated(conn):
    store(conn, "movie", 603, movie_payload())
    store(conn, "movie", 604, movie_payload(604, title="Reloaded"))
    normalise(conn)
    rated = conn.execute("SELECT id FROM title WHERE tmdb_id = 604").fetchone()[0]
    conn.execute("INSERT INTO rating (title_id, score, source) VALUES (%s, 4.5, 'app')", (rated,))

    conn.execute("UPDATE tmdb_title SET deleted_at = now()")
    normalise(conn)
    assert [r[0] for r in conn.execute("SELECT tmdb_id FROM title")] == [604]


@pytest.mark.db
def test_rating_score_must_be_half_steps(conn):
    store(conn, "movie", 603, movie_payload())
    normalise(conn)
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO rating (title_id, score, source) SELECT id, 4.3, 'app' FROM title"
        )
