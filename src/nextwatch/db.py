"""Database connections and the migration runner."""

import logging
from importlib import resources

import psycopg

log = logging.getLogger(__name__)


def connect(url: str, **kwargs) -> psycopg.Connection:
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(url, **kwargs)


def migration_files() -> list[tuple[str, str]]:
    """Return (version, sql) for every migration, in order."""
    files = resources.files("nextwatch.migrations")
    out = []
    for f in sorted(files.iterdir(), key=lambda p: p.name):
        if f.name.endswith(".sql"):
            out.append((f.name.removesuffix(".sql"), f.read_text()))
    return out


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply pending migrations, each in its own transaction. Returns the versions applied."""
    with conn.transaction():
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migration ("
            " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migration")}
    applied = []
    for version, sql in migration_files():
        if version in done:
            continue
        with conn.transaction():
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migration (version) VALUES (%s)", (version,))
        log.info("applied migration %s", version)
        applied.append(version)
    return applied


def db_size(conn: psycopg.Connection) -> list[tuple[str, int]]:
    """Total size first, then each table's size including indexes and TOAST, largest first."""
    total = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
    rows = conn.execute(
        "SELECT relname, pg_total_relation_size(c.oid) FROM pg_class c"
        " JOIN pg_namespace n ON n.oid = c.relnamespace"
        " WHERE c.relkind = 'r' AND n.nspname = 'public'"
        " ORDER BY 2 DESC"
    ).fetchall()
    return [("(database)", total), *rows]


def log_db_size(conn: psycopg.Connection) -> None:
    total = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
    log.info("database size %.1f MB", total / 1e6)
