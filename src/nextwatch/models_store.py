"""Trained models as rows in model_artifact. See docs/plans/02-storage.md."""

from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb


def save(conn: Connection, name: str, blob: bytes, meta: dict[str, Any]) -> int:
    """Store a new version of a model. Returns the version number."""
    with conn.transaction():
        version = conn.execute(
            "SELECT coalesce(max(version), 0) + 1 FROM model_artifact WHERE name = %s", (name,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO model_artifact (name, version, blob, meta) VALUES (%s, %s, %s, %s)",
            (name, version, blob, Jsonb(meta)),
        )
    return version


def load_latest(conn: Connection, name: str) -> tuple[int, bytes, dict[str, Any]] | None:
    row = conn.execute(
        "SELECT version, blob, meta FROM model_artifact WHERE name = %s"
        " ORDER BY version DESC LIMIT 1",
        (name,),
    ).fetchone()
    return (row[0], bytes(row[1]), row[2]) if row else None


def prune(conn: Connection, name: str, keep: int = 3) -> int:
    """Delete all but the newest `keep` versions, to hold the database to its size budget."""
    return conn.execute(
        "DELETE FROM model_artifact WHERE name = %s AND version NOT IN ("
        " SELECT version FROM model_artifact WHERE name = %s ORDER BY version DESC LIMIT %s)",
        (name, name, keep),
    ).rowcount
