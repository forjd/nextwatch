"""The `nextwatch` command."""

import logging
import sys
from pathlib import Path

import click

from nextwatch import db
from nextwatch.config import load_settings

DATA_DIR = Path("data")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def main(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # httpx logs every request at INFO, which drowns everything at 20 req/s.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@main.command()
def migrate() -> None:
    """Apply pending database migrations."""
    with db.connect(load_settings().database_url, autocommit=True) as conn:
        applied = db.migrate(conn)
    click.echo(f"applied {len(applied)} migration(s): {', '.join(applied) or 'none'}")


@main.command("db-size")
def db_size() -> None:
    """Print the database size and the size of each table."""
    with db.connect(load_settings().database_url) as conn:
        for name, size in db.db_size(conn):
            click.echo(f"{size / 1e6:10.1f} MB  {name}")


@main.command()
def backfill() -> None:
    """Queue every TMDB ID and MovieLens movie, then fetch them. Safe to stop and rerun."""
    from nextwatch.ingest import jobs

    jobs.backfill(load_settings(), DATA_DIR)


@main.command()
def sync() -> None:
    """Daily: fetch new, vanished and changed titles."""
    from nextwatch.ingest import jobs

    jobs.sync(load_settings(), DATA_DIR)


@main.command()
def refresh() -> None:
    """Daily: re-fetch the stalest eligible titles (all of them every 30 days)."""
    from nextwatch.ingest import jobs

    jobs.refresh(load_settings())


@main.command()
def work() -> None:
    """Work whatever is left in the ingest queue."""
    from nextwatch.ingest import jobs

    jobs.work(load_settings())


@main.command("queue")
def queue_status() -> None:
    """Show how many items are waiting in the ingest queue."""
    from nextwatch.ingest import queue

    with db.connect(load_settings().database_url) as conn:
        counts = queue.counts(conn)
    for kind, n in sorted(counts.items()):
        click.echo(f"{n:10d}  {kind}")
    if not counts:
        click.echo("empty")
