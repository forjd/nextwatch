"""The ingest jobs: backfill, sync, refresh and work. See docs/plans/01-ingest.md."""

import asyncio
import datetime as dt
import logging
import math
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path

import httpx
from psycopg import AsyncConnection, Connection
from psycopg_pool import AsyncConnectionPool

from nextwatch import db
from nextwatch.config import Settings
from nextwatch.ingest import queue
from nextwatch.ingest.exports import download_export, read_export
from nextwatch.ingest.worker import Ingester, Stats
from nextwatch.tmdb import TmdbClient

log = logging.getLogger(__name__)

MEDIA_TYPES = ("movie", "tv")
CHANGES_MAX_DAYS = 14
REFRESH_CYCLE_DAYS = 30
LOCK_KEY = 7_201_884_311  # any constant; one ingest job at a time

AsyncPrep = Callable[[AsyncConnection, TmdbClient], Awaitable[None]]


class JobLocked(RuntimeError):
    pass


@contextmanager
def ingest_lock(conn: Connection):
    if not conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,)).fetchone()[0]:
        raise JobLocked("another ingest job is running")
    try:
        yield
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))


def _run(
    settings: Settings,
    prepare: Callable[[Connection], None] | None = None,
    prepare_async: AsyncPrep | None = None,
    after: Callable[[Connection, Stats], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Stats:
    with db.connect(settings.database_url, autocommit=True) as conn, ingest_lock(conn):
        queue.release_claims(conn)
        if prepare:
            prepare(conn)
        stats = asyncio.run(_work(settings, prepare_async, transport))
        if after:
            after(conn, stats)
        log.info("queue left: %s", queue.counts(conn) or "empty")
        db.log_db_size(conn)
        return stats


async def _work(
    settings: Settings, prepare_async: AsyncPrep | None, transport: httpx.AsyncBaseTransport | None
) -> Stats:
    pool_size = min(settings.tmdb_workers, 10)
    async with (
        AsyncConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=pool_size,
            open=False,
            kwargs={"autocommit": True},
        ) as pool,
        TmdbClient(settings.tmdb_api_token, settings.tmdb_rps, transport=transport) as client,
    ):
        if prepare_async:
            async with pool.connection() as conn:
                await prepare_async(conn, client)
        return await Ingester(pool, client, settings).run()


def _load_exports(conn: Connection, data_dir: Path, today: dt.date | None = None) -> None:
    """Load today's export IDs into the temp table export_ids. Call inside a transaction."""
    conn.execute(
        "CREATE TEMP TABLE export_ids (media_type text, tmdb_id integer, popularity real)"
        " ON COMMIT DROP"
    )
    with conn.cursor() as cur:
        for media_type in MEDIA_TYPES:
            path = download_export(media_type, data_dir / "exports", today)
            n = 0
            with cur.copy("COPY export_ids FROM STDIN") as copy:
                for row in read_export(path):
                    copy.write_row((media_type, row.tmdb_id, row.popularity))
                    n += 1
            log.info("%s export: %d IDs", media_type, n)
    conn.execute("CREATE INDEX ON export_ids (media_type, tmdb_id)")
    conn.execute("ANALYZE export_ids")


def backfill(
    settings: Settings,
    data_dir: Path,
    today: dt.date | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Stats:
    """Queue every ID in the exports and every MovieLens movie, then work the queue."""

    def prepare(conn: Connection) -> None:
        with conn.transaction():
            _load_exports(conn, data_dir, today)
            cur = conn.execute(
                """
                INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, priority, popularity)
                SELECT e.media_type, e.tmdb_id, 'light', 'backfill', %s, e.popularity
                FROM export_ids e
                WHERE NOT EXISTS (
                    SELECT 1 FROM tmdb_title t
                    WHERE t.media_type = e.media_type AND t.tmdb_id = e.tmdb_id
                      AND t.light_fetched_at IS NOT NULL)
                ON CONFLICT DO NOTHING
                """,
                (queue.PRIORITY["backfill"],),
            )
            log.info("queued %d export IDs", cur.rowcount)
        _queue_movielens(conn)
        _queue_missing_payloads(conn)

    return _run(settings, prepare=prepare, transport=transport)


def _queue_movielens(conn: Connection) -> None:
    cur = conn.execute(
        """
        INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, priority)
        SELECT DISTINCT 'movie', l.tmdb_id, 'light', 'backfill', %s
        FROM movielens_link l
        WHERE l.tmdb_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM tmdb_title t
            WHERE t.media_type = 'movie' AND t.tmdb_id = l.tmdb_id
              AND t.light_fetched_at IS NOT NULL)
        ON CONFLICT (media_type, tmdb_id, kind) DO UPDATE
        SET priority = LEAST(ingest_queue.priority, EXCLUDED.priority)
        """,
        (queue.PRIORITY["movielens"],),
    )
    log.info("queued %d MovieLens IDs", cur.rowcount)


def _queue_missing_payloads(conn: Connection) -> None:
    """Eligible titles with no payload yet, for example after a crash mid-backfill."""
    cur = conn.execute(
        """
        INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, priority, popularity)
        SELECT media_type, tmdb_id, 'full', 'promote', %s, coalesce(popularity, 0)
        FROM tmdb_title
        WHERE eligible AND payload IS NULL AND deleted_at IS NULL
        ON CONFLICT DO NOTHING
        """,
        (queue.PRIORITY["promote"],),
    )
    if cur.rowcount:
        log.info("queued %d eligible titles with no payload", cur.rowcount)


def _get_state(conn: Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM ingest_state WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def _set_state(conn: Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO ingest_state (key, value) VALUES (%s, %s)"
        " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        (key, value),
    )


def changes_window(last_sync: dt.date | None, today: dt.date) -> tuple[dt.date, bool]:
    """Start date for /changes, and whether changes were lost because the gap is too long."""
    if last_sync is None:
        return today - dt.timedelta(days=1), False
    if (today - last_sync).days > CHANGES_MAX_DAYS:
        return today - dt.timedelta(days=CHANGES_MAX_DAYS), True
    return last_sync, False


def sync(
    settings: Settings,
    data_dir: Path,
    today: dt.date | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Stats:
    """Daily: new and vanished IDs from the export, changed IDs from /changes."""
    today = today or dt.datetime.now(dt.UTC).date()
    window: dict[str, dt.date] = {}

    def prepare(conn: Connection) -> None:
        with conn.transaction():
            _load_exports(conn, data_dir, today)
            new = conn.execute(
                """
                INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, priority, popularity)
                SELECT e.media_type, e.tmdb_id, 'light', 'new', %s, e.popularity
                FROM export_ids e
                WHERE NOT EXISTS (SELECT 1 FROM tmdb_title t
                    WHERE t.media_type = e.media_type AND t.tmdb_id = e.tmdb_id)
                ON CONFLICT (media_type, tmdb_id, kind) DO UPDATE
                SET priority = LEAST(ingest_queue.priority, EXCLUDED.priority)
                """,
                (queue.PRIORITY["new"],),
            ).rowcount
            # Gone from the export: fetch again rather than trust it. A 404 marks it deleted.
            gone = conn.execute(
                """
                INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, priority)
                SELECT t.media_type, t.tmdb_id, 'light', 'verify', %s
                FROM tmdb_title t
                WHERE t.deleted_at IS NULL AND NOT EXISTS (SELECT 1 FROM export_ids e
                    WHERE e.media_type = t.media_type AND e.tmdb_id = t.tmdb_id)
                ON CONFLICT DO NOTHING
                """,
                (queue.PRIORITY["verify"],),
            ).rowcount
            log.info("export diff: %d new, %d to verify", new, gone)

        last = _get_state(conn, "last_sync_date")
        start, lost = changes_window(dt.date.fromisoformat(last) if last else None, today)
        if lost:
            log.warning(
                "last sync was %s, more than %d days ago; changes are lost, refreshing all",
                last,
                CHANGES_MAX_DAYS,
            )
            _queue_refresh(conn, limit=None)
        window["start"] = start

    async def prepare_async(conn: AsyncConnection, client: TmdbClient) -> None:
        for media_type in MEDIA_TYPES:
            changed = await client.changes(media_type, window["start"], today)
            ids = sorted({int(c["id"]) for c in changed if not c.get("adult")})
            await _queue_changed(conn, media_type, ids)
            log.info("%s changes since %s: %d IDs", media_type, window["start"], len(ids))

    def after(conn: Connection, stats: Stats) -> None:
        left = queue.counts(conn)
        if left.get("light", 0) or left.get("full", 0):
            log.warning("queue not drained; not advancing last_sync_date")
            return
        _set_state(conn, "last_sync_date", today.isoformat())

    return _run(settings, prepare, prepare_async, after, transport)


async def _queue_changed(conn: AsyncConnection, media_type: str, ids: list[int]) -> None:
    if not ids:
        return
    # Eligible titles get a full fetch. Everything else a light one, which may promote it.
    await conn.execute(
        """
        INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, priority, popularity)
        SELECT %(mt)s, i.id,
               CASE WHEN t.eligible THEN 'full' ELSE 'light' END,
               CASE WHEN t.tmdb_id IS NULL THEN 'new' ELSE 'changed' END,
               CASE WHEN t.eligible THEN %(p_full)s ELSE %(p_light)s END,
               coalesce(t.popularity, 0)
        FROM unnest(%(ids)s::integer[]) AS i(id)
        LEFT JOIN tmdb_title t ON t.media_type = %(mt)s AND t.tmdb_id = i.id
        WHERE t.deleted_at IS NULL
        ON CONFLICT (media_type, tmdb_id, kind) DO UPDATE
        SET priority = LEAST(ingest_queue.priority, EXCLUDED.priority)
        """,
        {
            "mt": media_type,
            "ids": ids,
            "p_full": queue.PRIORITY["changed_full"],
            "p_light": queue.PRIORITY["changed_light"],
        },
    )


def _queue_refresh(conn: Connection, limit: int | None) -> int:
    cur = conn.execute(
        """
        INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, priority, popularity)
        SELECT media_type, tmdb_id, 'full', 'refresh', %s, coalesce(popularity, 0)
        FROM tmdb_title
        WHERE eligible AND deleted_at IS NULL
        ORDER BY full_fetched_at NULLS FIRST
        LIMIT %s
        ON CONFLICT DO NOTHING
        """,
        (queue.PRIORITY["refresh"], limit),
    )
    return cur.rowcount


def refresh(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> Stats:
    """Re-fetch the stalest eligible titles, enough to cycle all of them every 30 days."""

    def prepare(conn: Connection) -> None:
        total = conn.execute(
            "SELECT count(*) FROM tmdb_title WHERE eligible AND deleted_at IS NULL"
        ).fetchone()[0]
        n = _queue_refresh(conn, math.ceil(total / REFRESH_CYCLE_DAYS))
        log.info("queued %d of %d eligible titles for refresh", n, total)

    return _run(settings, prepare=prepare, transport=transport)


def work(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> Stats:
    """Work whatever is in the queue. Use it to resume after a crash."""
    return _run(settings, transport=transport)
