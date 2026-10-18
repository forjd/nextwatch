"""The ingest work queue in Postgres. Workers claim rows with FOR UPDATE SKIP LOCKED."""

import datetime as dt
from dataclasses import dataclass

from psycopg import AsyncConnection, Connection

MAX_ATTEMPTS = 5

# Lower runs first. See the reason column in migrations/0001_ingest.sql.
PRIORITY = {
    "movielens": 0,
    "changed_full": 3,
    "promote": 4,
    "new": 5,
    "changed_light": 6,
    "verify": 8,
    "refresh": 9,
    "backfill": 10,
}


@dataclass(frozen=True)
class QueueItem:
    media_type: str
    tmdb_id: int
    kind: str
    reason: str
    attempts: int


ENQUEUE_SQL = """
INSERT INTO ingest_queue (media_type, tmdb_id, kind, reason, priority, popularity)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (media_type, tmdb_id, kind) DO UPDATE
SET priority = LEAST(ingest_queue.priority, EXCLUDED.priority)
"""


async def enqueue(
    conn: AsyncConnection,
    media_type: str,
    tmdb_id: int,
    kind: str,
    reason: str,
    priority: int,
    popularity: float = 0,
) -> None:
    await conn.execute(ENQUEUE_SQL, (media_type, tmdb_id, kind, reason, priority, popularity))


def release_claims(conn: Connection) -> None:
    """Clear leases left by a job that died. Only call while holding the ingest lock."""
    conn.execute("UPDATE ingest_queue SET claimed_until = NULL WHERE claimed_until IS NOT NULL")


async def claim(conn: AsyncConnection, n: int) -> list[QueueItem]:
    cur = await conn.execute(
        """
        UPDATE ingest_queue q SET claimed_until = now() + interval '15 minutes'
        FROM (
            SELECT media_type, tmdb_id, kind FROM ingest_queue
            WHERE claimed_until IS NULL AND next_attempt_at <= now()
            ORDER BY priority, popularity DESC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        ) c
        WHERE q.media_type = c.media_type AND q.tmdb_id = c.tmdb_id AND q.kind = c.kind
        RETURNING q.media_type, q.tmdb_id, q.kind, q.reason, q.attempts
        """,
        (n,),
    )
    return [QueueItem(*row) for row in await cur.fetchall()]


async def complete(conn: AsyncConnection, item: QueueItem) -> None:
    await conn.execute(
        "DELETE FROM ingest_queue WHERE media_type = %s AND tmdb_id = %s AND kind = %s",
        (item.media_type, item.tmdb_id, item.kind),
    )


def backoff(attempts: int) -> dt.timedelta:
    """Delay before retry number `attempts` (1-based): 1, 2, 4, 8 minutes, capped at 6 hours."""
    return min(dt.timedelta(minutes=2 ** (attempts - 1)), dt.timedelta(hours=6))


async def fail(conn: AsyncConnection, item: QueueItem, error: str) -> None:
    """Record a failure. After MAX_ATTEMPTS the row stays, parked, for someone to inspect."""
    attempts = item.attempts + 1
    if attempts >= MAX_ATTEMPTS:
        next_at = "infinity"
    else:
        next_at = (dt.datetime.now(dt.UTC) + backoff(attempts)).isoformat()
    await conn.execute(
        """
        UPDATE ingest_queue
        SET attempts = %s, next_attempt_at = %s::timestamptz, claimed_until = NULL,
            last_error = %s
        WHERE media_type = %s AND tmdb_id = %s AND kind = %s
        """,
        (attempts, next_at, error[:1000], item.media_type, item.tmdb_id, item.kind),
    )


def counts(conn: Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT CASE WHEN next_attempt_at = 'infinity' THEN 'parked'
                    WHEN next_attempt_at > now() THEN 'waiting'
                    ELSE kind END,
               count(*)
        FROM ingest_queue GROUP BY 1
        """
    ).fetchall()
    return dict(rows)
