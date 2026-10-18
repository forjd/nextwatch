"""Export and import the owner's data: ratings and events.

Rows are keyed by (media_type, tmdb_id), not title.id, so a backup still applies after the
catalogue is rebuilt from scratch.
"""

import base64
import datetime as dt
import gzip
import json
import logging
from typing import Any

import httpx
from psycopg import Connection

from nextwatch.http import SSL_CONTEXT

log = logging.getLogger(__name__)

FORMAT_VERSION = 1


def export_data(conn: Connection) -> dict[str, Any]:
    ratings = conn.execute(
        """
        SELECT t.media_type, t.tmdb_id, r.score, r.source, r.rated_at
        FROM rating r JOIN title t ON t.id = r.title_id
        ORDER BY t.media_type, t.tmdb_id
        """
    ).fetchall()
    events = conn.execute(
        """
        SELECT t.media_type, t.tmdb_id, e.kind, e.context, e.at
        FROM event e LEFT JOIN title t ON t.id = e.title_id
        ORDER BY e.at, e.id
        """
    ).fetchall()
    return {
        "version": FORMAT_VERSION,
        "ratings": [
            {
                "media_type": m,
                "tmdb_id": i,
                "score": float(s),
                "source": src,
                "rated_at": at.isoformat(),
            }
            for m, i, s, src, at in ratings
        ],
        "events": [
            {"media_type": m, "tmdb_id": i, "kind": k, "context": c, "at": at.isoformat()}
            for m, i, k, c, at in events
        ],
    }


def dumps(data: dict[str, Any]) -> bytes:
    """Deterministic gzip, so an unchanged export gives identical bytes."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return gzip.compress(raw, mtime=0)


def loads(blob: bytes) -> dict[str, Any]:
    data = json.loads(gzip.decompress(blob))
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"unknown backup format version {data.get('version')}")
    return data


def import_data(conn: Connection, data: dict[str, Any]) -> tuple[int, int, int]:
    """Restore ratings (newest wins) and events (skipping exact duplicates).

    Returns (ratings, events, skipped). Titles not in the catalogue are skipped and logged.
    """
    ids = {(m, i): tid for tid, m, i in conn.execute("SELECT id, media_type, tmdb_id FROM title")}
    skipped = 0
    n_ratings = n_events = 0
    with conn.transaction(), conn.cursor() as cur:
        for r in data["ratings"]:
            title_id = ids.get((r["media_type"], r["tmdb_id"]))
            if title_id is None:
                skipped += 1
                continue
            cur.execute(
                """
                INSERT INTO rating (title_id, score, source, rated_at) VALUES (%s, %s, %s, %s)
                ON CONFLICT (title_id) DO UPDATE SET score = EXCLUDED.score,
                    source = EXCLUDED.source, rated_at = EXCLUDED.rated_at
                WHERE rating.rated_at < EXCLUDED.rated_at
                """,
                (title_id, r["score"], r["source"], dt.datetime.fromisoformat(r["rated_at"])),
            )
            n_ratings += 1
        for e in data["events"]:
            title_id = ids.get((e["media_type"], e["tmdb_id"]))
            if e["tmdb_id"] is not None and title_id is None:
                skipped += 1
                continue
            at = dt.datetime.fromisoformat(e["at"])
            cur.execute(
                """
                INSERT INTO event (title_id, kind, context, at)
                SELECT %(t)s, %(k)s, %(c)s, %(at)s
                WHERE NOT EXISTS (SELECT 1 FROM event WHERE title_id IS NOT DISTINCT FROM %(t)s
                                  AND kind = %(k)s AND at = %(at)s)
                """,
                {"t": title_id, "k": e["kind"], "c": json.dumps(e["context"]), "at": at},
            )
            n_events += cur.rowcount
    if skipped:
        log.warning("skipped %d rows for titles not in the catalogue", skipped)
    return n_ratings, n_events, skipped


GITHUB_API = "https://api.github.com"
BACKUP_PATH = "owner-data.json.gz"


def upload_github(
    blob: bytes,
    repo: str,
    token: str,
    path: str = BACKUP_PATH,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """Write the backup to a GitHub repo through the contents API. Git history keeps old ones.

    Returns False without writing when the content is unchanged.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    with httpx.Client(headers=headers, timeout=30, transport=transport, verify=SSL_CONTEXT) as http:
        current = http.get(url)
        sha = None
        if current.status_code == 200:
            body = current.json()
            sha = body["sha"]
            if base64.b64decode(body.get("content", "")) == blob:
                log.info("backup unchanged; not writing")
                return False
        elif current.status_code != 404:
            current.raise_for_status()
        payload = {
            "message": f"backup: owner data {dt.date.today().isoformat()}",
            "content": base64.b64encode(blob).decode(),
            **({"sha": sha} if sha else {}),
        }
        http.put(url, json=payload).raise_for_status()
    log.info("backup written to %s/%s", repo, path)
    return True
