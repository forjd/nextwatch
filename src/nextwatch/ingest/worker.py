"""Workers that take items off the queue, fetch them from TMDB and store them."""

import asyncio
import datetime as dt
import logging
import time
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from nextwatch.config import Settings
from nextwatch.ingest import queue
from nextwatch.ingest.queue import QueueItem
from nextwatch.ingest.trim import trim_payload
from nextwatch.tmdb import NotFound, TmdbClient, TransientError

log = logging.getLogger(__name__)

CLAIM_BATCH = 10


@dataclass(frozen=True)
class Stub:
    name: str | None
    vote_count: int
    vote_average: float | None
    popularity: float | None
    release_date: dt.date | None


def extract_stub(media_type: str, data: dict[str, Any]) -> Stub:
    date_key = "release_date" if media_type == "movie" else "first_air_date"
    name_key = "title" if media_type == "movie" else "name"
    raw_date = data.get(date_key) or None
    try:
        release_date = dt.date.fromisoformat(raw_date) if raw_date else None
    except ValueError:
        release_date = None
    return Stub(
        name=data.get(name_key),
        vote_count=int(data.get("vote_count") or 0),
        vote_average=data.get("vote_average"),
        popularity=data.get("popularity"),
        release_date=release_date,
    )


def is_excluded(media_type: str, data: dict[str, Any]) -> bool:
    """Adult titles and movie 'video' entries are treated as if they do not exist."""
    return bool(data.get("adult")) or (media_type == "movie" and bool(data.get("video")))


@dataclass
class Stats:
    light: int = 0
    full: int = 0
    deleted: int = 0
    failed: int = 0
    promoted: int = 0


class Ingester:
    def __init__(self, pool: AsyncConnectionPool, client: TmdbClient, settings: Settings):
        self.pool = pool
        self.client = client
        self.settings = settings
        self.stats = Stats()
        self.busy = 0

    async def run(self) -> Stats:
        """Work the queue with the configured number of workers until nothing is ready."""
        started = time.monotonic()
        reporter = asyncio.create_task(self._report(started))
        try:
            await asyncio.gather(*(self._worker() for _ in range(self.settings.tmdb_workers)))
        finally:
            reporter.cancel()
        self._log_progress(started)
        return self.stats

    async def _report(self, started: float) -> None:
        while True:
            await asyncio.sleep(60)
            self._log_progress(started)

    def _log_progress(self, started: float) -> None:
        s = self.stats
        elapsed = max(time.monotonic() - started, 1e-9)
        log.info(
            "light %d, full %d, promoted %d, deleted %d, failed %d (%.1f req/s)",
            s.light,
            s.full,
            s.promoted,
            s.deleted,
            s.failed,
            (s.light + s.full) / elapsed,
        )

    async def _worker(self) -> None:
        while True:
            async with self.pool.connection() as conn:
                items = await queue.claim(conn, CLAIM_BATCH)
            if not items:
                # Busy workers may still queue follow-up work (a light fetch promotes to full),
                # so only stop once every worker is idle.
                if self.busy == 0:
                    return
                await asyncio.sleep(0.5)
                continue
            self.busy += 1
            try:
                for item in items:
                    await self.process(item)
            finally:
                self.busy -= 1

    async def process(self, item: QueueItem) -> None:
        try:
            data = await self.client.details(
                item.media_type, item.tmdb_id, full=item.kind == "full"
            )
        except NotFound:
            await self._mark_deleted(item)
            return
        except TransientError as e:
            await self._fail(item, str(e))
            return
        except Exception as e:  # noqa: BLE001 - one bad title must not stop the job
            log.exception("unexpected error on %s/%s", item.media_type, item.tmdb_id)
            await self._fail(item, repr(e))
            return

        if is_excluded(item.media_type, data):
            await self._mark_deleted(item)
            return

        async with self.pool.connection() as conn, conn.transaction():
            if item.kind == "light":
                await self._store_light(conn, item, data)
            else:
                await self._store_full(conn, item, data)
            await queue.complete(conn, item)

    async def _eligible(self, conn: AsyncConnection, item: QueueItem, stub: Stub) -> bool:
        if stub.vote_count >= self.settings.min_votes_for(item.media_type):
            return True
        if item.media_type != "movie":
            return False
        cur = await conn.execute(
            "SELECT EXISTS (SELECT 1 FROM movielens_link WHERE tmdb_id = %s)", (item.tmdb_id,)
        )
        return (await cur.fetchone())[0]

    async def _store_light(self, conn: AsyncConnection, item: QueueItem, data: dict) -> None:
        stub = extract_stub(item.media_type, data)
        eligible = await self._eligible(conn, item, stub)
        cur = await conn.execute(
            """
            INSERT INTO tmdb_title (media_type, tmdb_id, name, vote_count, vote_average,
                popularity, release_date, eligible, light_fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (media_type, tmdb_id) DO UPDATE SET
                name = EXCLUDED.name, vote_count = EXCLUDED.vote_count,
                vote_average = EXCLUDED.vote_average, popularity = EXCLUDED.popularity,
                release_date = EXCLUDED.release_date,
                eligible = tmdb_title.eligible OR EXCLUDED.eligible,
                light_fetched_at = now(), deleted_at = NULL
            RETURNING eligible, full_fetched_at
            """,
            (
                item.media_type,
                item.tmdb_id,
                stub.name,
                stub.vote_count,
                stub.vote_average,
                stub.popularity,
                stub.release_date,
                eligible,
            ),
        )
        now_eligible, full_fetched_at = await cur.fetchone()
        self.stats.light += 1
        if now_eligible and full_fetched_at is None:
            await queue.enqueue(
                conn,
                item.media_type,
                item.tmdb_id,
                "full",
                "promote",
                queue.PRIORITY["promote"],
                stub.popularity or 0,
            )
            self.stats.promoted += 1

    async def _store_full(self, conn: AsyncConnection, item: QueueItem, data: dict) -> None:
        stub = extract_stub(item.media_type, data)
        eligible = await self._eligible(conn, item, stub)
        await conn.execute(
            """
            INSERT INTO tmdb_title (media_type, tmdb_id, name, vote_count, vote_average,
                popularity, release_date, eligible, light_fetched_at, full_fetched_at, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now(), now(), %s)
            ON CONFLICT (media_type, tmdb_id) DO UPDATE SET
                name = EXCLUDED.name, vote_count = EXCLUDED.vote_count,
                vote_average = EXCLUDED.vote_average, popularity = EXCLUDED.popularity,
                release_date = EXCLUDED.release_date,
                eligible = tmdb_title.eligible OR EXCLUDED.eligible,
                light_fetched_at = now(), full_fetched_at = now(),
                payload = EXCLUDED.payload, deleted_at = NULL
            """,
            (
                item.media_type,
                item.tmdb_id,
                stub.name,
                stub.vote_count,
                stub.vote_average,
                stub.popularity,
                stub.release_date,
                eligible,
                Jsonb(trim_payload(data)),
            ),
        )
        self.stats.full += 1

    async def _mark_deleted(self, item: QueueItem) -> None:
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                "UPDATE tmdb_title SET deleted_at = now()"
                " WHERE media_type = %s AND tmdb_id = %s AND deleted_at IS NULL",
                (item.media_type, item.tmdb_id),
            )
            await queue.complete(conn, item)
        self.stats.deleted += 1

    async def _fail(self, item: QueueItem, error: str) -> None:
        async with self.pool.connection() as conn:
            await queue.fail(conn, item, error)
        self.stats.failed += 1
