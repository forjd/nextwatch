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


@main.command()
def normalise() -> None:
    """Rebuild the catalogue tables from new or changed TMDB payloads."""
    from nextwatch import normalise as n

    with db.connect(load_settings().database_url, autocommit=True) as conn:
        n.normalise(conn)
        db.log_db_size(conn)


@main.command("export")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=DATA_DIR / "backups" / "owner-data.json.gz",
)
def export_cmd(output: Path) -> None:
    """Write ratings and events to a gzipped JSON file."""
    from nextwatch import backup

    with db.connect(load_settings().database_url) as conn:
        data = backup.export_data(conn)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(backup.dumps(data))
    click.echo(f"wrote {len(data['ratings'])} ratings, {len(data['events'])} events to {output}")


@main.command("import-backup")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def import_backup(path: Path) -> None:
    """Restore ratings and events from an export file."""
    from nextwatch import backup

    with db.connect(load_settings().database_url) as conn:
        r, e, skipped = backup.import_data(conn, backup.loads(path.read_bytes()))
    click.echo(f"restored {r} ratings, {e} events ({skipped} skipped)")
