import datetime as dt

import pytest

from nextwatch.ingest import jobs, queue
from tests.conftest import write_export

pytestmark = pytest.mark.db

TODAY = dt.date(2026, 10, 18)


def movie(i, popularity=1.0, **kw):
    return {"id": i, "adult": False, "video": False, "popularity": popularity, **kw}


def tv(i, popularity=1.0):
    return {"id": i, "popularity": popularity}


def titles(conn):
    return {
        (r[0], r[1]): r[2:]
        for r in conn.execute(
            "SELECT media_type, tmdb_id, eligible, payload IS NOT NULL, deleted_at IS NOT NULL"
            " FROM tmdb_title"
        )
    }


@pytest.fixture
def world(conn, fake_tmdb, tmp_path):
    """Exports and a fake API with: an eligible movie, an obscure one, an obscure MovieLens
    movie, a 404, an adult movie that the export filters out, and an eligible TV series."""
    write_export(
        tmp_path,
        "movie",
        TODAY,
        [
            movie(1, 50),
            movie(2),
            movie(3),
            movie(4),
            movie(5, adult=True),
        ],
    )
    write_export(tmp_path, "tv", TODAY, [tv(10, 20)])
    fake_tmdb.add("movie", 1, votes=100)
    fake_tmdb.add("movie", 2, votes=3)
    fake_tmdb.add("movie", 3, votes=1)  # in MovieLens
    fake_tmdb.add("tv", 10, votes=40)
    conn.execute("INSERT INTO movielens_link VALUES (1, 'tt1', 3), (2, 'tt2', NULL)")
    return tmp_path


def test_backfill(conn, settings, fake_tmdb, world):
    stats = jobs.backfill(settings, world, TODAY, transport=fake_tmdb.transport)

    assert titles(conn) == {
        ("movie", 1): (True, True, False),
        ("movie", 2): (False, False, False),
        ("movie", 3): (True, True, False),
        ("tv", 10): (True, True, False),
    }
    assert stats.light == 4 and stats.full == 3 and stats.deleted == 1
    assert queue.counts(conn) == {}
    assert fake_tmdb.count(r"/movie/5") == 0  # adult, filtered from the export

    payload = conn.execute(
        "SELECT payload FROM tmdb_title WHERE media_type = 'tv' AND tmdb_id = 10"
    ).fetchone()[0]
    assert "aggregate_credits" in payload and "content_ratings" in payload
    assert payload["watch/providers"] == {"results": {"GB": {"flatrate": [{"provider_id": 8}]}}}
    assert payload["recommendations"] == [{"id": 99, "media_type": "movie"}]


def test_backfill_again_fetches_nothing(conn, settings, fake_tmdb, world):
    jobs.backfill(settings, world, TODAY, transport=fake_tmdb.transport)
    before = len(fake_tmdb.requests)
    stats = jobs.backfill(settings, world, TODAY, transport=fake_tmdb.transport)
    # Only the 404 (never stored) is tried again.
    assert len(fake_tmdb.requests) - before == 1
    assert stats.light == 0 and stats.full == 0


def test_backfill_resumes_after_crash(conn, settings, fake_tmdb, world):
    # A crash after a light fetch but before the full one: eligible, no payload, empty queue.
    conn.execute(
        "INSERT INTO tmdb_title (media_type, tmdb_id, eligible, light_fetched_at)"
        " VALUES ('movie', 1, true, now())"
    )
    conn.execute(
        "INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, claimed_until)"
        " VALUES ('movie', 2, 'light', 'backfill', now() + interval '1 hour')"
    )
    jobs.backfill(settings, world, TODAY, transport=fake_tmdb.transport)
    assert titles(conn)[("movie", 1)] == (True, True, False)
    assert ("movie", 2) in titles(conn)  # stale claim was released
    assert fake_tmdb.count(r"/movie/1\?") == 1  # only the full fetch


def test_429_pauses_and_retries(conn, settings, fake_tmdb, world):
    fake_tmdb.fail_once[("movie", 1)] = 2
    stats = jobs.backfill(settings, world, TODAY, transport=fake_tmdb.transport)
    assert stats.failed == 0
    assert titles(conn)[("movie", 1)] == (True, True, False)


def test_server_errors_back_off_then_park(conn, settings, fake_tmdb, world):
    fake_tmdb.always_fail.add(("tv", 10))
    jobs.backfill(settings, world, TODAY, transport=fake_tmdb.transport)
    row = conn.execute(
        "SELECT attempts, next_attempt_at > now(), last_error FROM ingest_queue"
        " WHERE media_type = 'tv' AND tmdb_id = 10"
    ).fetchone()
    assert row[0] == 1 and row[1] and "503" in row[2]
    assert queue.counts(conn) == {"waiting": 1}

    conn.execute("UPDATE ingest_queue SET attempts = 4, next_attempt_at = now()")
    jobs.work(settings, transport=fake_tmdb.transport)
    assert queue.counts(conn) == {"parked": 1}


def test_sync(conn, settings, fake_tmdb, world):
    jobs.backfill(settings, world, TODAY, transport=fake_tmdb.transport)
    tomorrow = TODAY + dt.timedelta(days=1)
    # Tomorrow: movie 6 is new, movie 2 vanished from the export and from TMDB,
    # movie 1 changed, movie 3 changed and gained votes.
    write_export(world, "movie", tomorrow, [movie(1), movie(3), movie(6)])
    write_export(world, "tv", tomorrow, [tv(10)])
    del fake_tmdb.titles[("movie", 2)]
    fake_tmdb.add("movie", 6, votes=500)
    fake_tmdb.changes = {"movie": [1, 3], "tv": []}
    before = conn.execute(
        "SELECT full_fetched_at FROM tmdb_title WHERE tmdb_id = 1 AND media_type = 'movie'"
    ).fetchone()[0]

    jobs.sync(settings, world, tomorrow, transport=fake_tmdb.transport)

    t = titles(conn)
    assert t[("movie", 6)] == (True, True, False)
    assert t[("movie", 2)][2] is True  # deleted
    after = conn.execute(
        "SELECT full_fetched_at FROM tmdb_title WHERE tmdb_id = 1 AND media_type = 'movie'"
    ).fetchone()[0]
    assert after > before
    assert (
        conn.execute("SELECT value FROM ingest_state WHERE key = 'last_sync_date'").fetchone()[0]
        == tomorrow.isoformat()
    )
    assert fake_tmdb.count(r"start_date=2026-10-18") == 2  # first sync looks back one day

    # Second run the same day: nothing new, only the changes list again.
    before = len(fake_tmdb.requests)
    fake_tmdb.changes = {}
    jobs.sync(settings, world, tomorrow, transport=fake_tmdb.transport)
    assert len(fake_tmdb.requests) - before == 2


def test_refresh_queues_a_thirtieth(conn, settings, fake_tmdb, world):
    jobs.backfill(settings, world, TODAY, transport=fake_tmdb.transport)
    before = len(fake_tmdb.requests)
    stats = jobs.refresh(settings, transport=fake_tmdb.transport)
    # 3 eligible titles / 30 days rounds up to 1 a day.
    assert stats.full == 1 and len(fake_tmdb.requests) - before == 1


def test_only_one_job_at_a_time(conn, settings, world):
    conn.execute("SELECT pg_advisory_lock(%s)", (jobs.LOCK_KEY,))
    try:
        with pytest.raises(jobs.JobLocked):
            jobs.work(settings)
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (jobs.LOCK_KEY,))
