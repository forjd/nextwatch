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


@main.group("movielens")
def movielens_group() -> None:
    """MovieLens 32M data (research licence, personal use only)."""


@movielens_group.command("fetch")
def movielens_fetch() -> None:
    """Download MovieLens 32M, convert it to Parquet and load the TMDB links."""
    from nextwatch import movielens

    movielens.fetch(DATA_DIR)
    with db.connect(load_settings().database_url) as conn:
        n = movielens.load_links(conn, DATA_DIR)
    click.echo(f"loaded {n} links; run `nextwatch backfill` to ingest MovieLens movies")


@main.group("eval")
def eval_group() -> None:
    """Offline evaluation (docs/plans/03-evaluation.md)."""


@eval_group.command("run")
@click.option("--model", "model_name", required=True, help="Model name, see `eval models`.")
@click.option("--split", type=click.Choice(["validation", "test"]), default="validation")
@click.option("--users", "n_users", default=5000, show_default=True)
@click.option("--golden", type=click.Path(path_type=Path), default=Path("eval/golden.yaml"))
def eval_run(model_name: str, split: str, n_users: int, golden: Path) -> None:
    """Score a model on T1, T2 and the golden set, and save the run."""
    from nextwatch.evaluation import data, harness
    from nextwatch.evaluation.models import Catalogue
    from nextwatch.registry import MODELS

    if model_name not in MODELS:
        raise click.BadParameter(f"unknown model; choose from {', '.join(MODELS)}")
    factory = MODELS[model_name]
    with db.connect(load_settings().database_url, autocommit=True) as conn:
        movie_cat = Catalogue.load(conn, "movie")
        full_cat = Catalogue.load(conn)
        interactions = data.load_interactions(DATA_DIR, data.movielens_to_title(conn))
        users = data.eval_users(interactions, split, n_users)
        train = data.train_interactions(interactions)
        result = harness.evaluate(
            factory(movie_cat, train, conn),
            users,
            movie_cat,
            model_all=factory(full_cat, train, conn),
            full_cat=full_cat,
            golden=harness.load_golden(golden),
        )
        config = {
            "model": model_name,
            "split": split,
            "users": n_users,
            "seed": data.SEED,
            "k": harness.K,
            "git": harness.git_sha(),
        }
        run_id = harness.save_run(conn, config, result, Path("eval/reports"))
    t1 = result["t1"]
    click.echo(
        f"run {run_id}: nDCG@20 {t1['ndcg@20']['mean']:.4f}, "
        f"recall@20 {t1['recall@20']['mean']:.4f}, T2 {result['t2']['recall@20']['mean']:.4f}"
    )


@eval_group.command("compare")
@click.argument("a", type=int)
@click.argument("b", type=int)
def eval_compare(a: int, b: int) -> None:
    """Compare two eval runs, with paired bootstrap intervals."""
    from nextwatch.evaluation import harness

    with db.connect(load_settings().database_url) as conn:
        click.echo(harness.compare(conn, a, b))


@eval_group.command("models")
def eval_models() -> None:
    """List the models `eval run` can score."""
    from nextwatch.registry import MODELS

    click.echo("\n".join(MODELS))


@main.command()
def embed() -> None:
    """Embed titles whose text changed (plan 04). Downloads the model on first run."""
    from nextwatch.features import embed as e

    embedder = e.fastembed_embedder(DATA_DIR / "models")
    with db.connect(load_settings().database_url, autocommit=True) as conn:
        n = e.embed_titles(conn, embedder)
        n_feel = e.embed_titles(conn, embedder, e.FEEL_MODEL)
        db.log_db_size(conn)
    click.echo(f"embedded {n} titles and {n_feel} feel sentences")


@main.command()
@click.argument("ref")
@click.option("--model", "model_name", default="content", show_default=True)
@click.option("-k", default=10, show_default=True)
def similar(ref: str, model_name: str, k: int) -> None:
    """Print titles similar to REF, written as movie/603 or tv/1396."""
    from nextwatch.evaluation.models import Catalogue
    from nextwatch.registry import MODELS

    media_type, tmdb_id = ref.split("/")
    with db.connect(load_settings().database_url) as conn:
        cat = Catalogue.load(conn)
        seed = cat.by_key.get((media_type, int(tmdb_id)))
        if seed is None:
            raise click.ClickException(f"{ref} is not in the catalogue")
        model = MODELS[model_name](cat, None, conn)
        names = dict(conn.execute("SELECT id, name FROM title"))
        for t in model.similar(seed, k):
            mt, tid = cat.tmdb_key[t]
            click.echo(f"{mt}/{tid}\t{names[t]}")


@main.command()
@click.option("--limit", type=int, help="Tag at most this many titles.")
@click.option("--spike", is_flag=True, help="Tag a mixed sample and write data/vibes-spike.md.")
def vibes(limit: int | None, spike: bool) -> None:
    """Tag untagged titles with vibe tags through OpenRouter (plan 05)."""
    from nextwatch.vibes import job

    settings = load_settings()
    with db.connect(settings.database_url, autocommit=True) as conn:
        stats = job.run(conn, settings, limit=limit, spike=spike)
        if spike:
            job.spike_report(conn, stats, DATA_DIR / "vibes-spike.md")
            click.echo(f"wrote {DATA_DIR / 'vibes-spike.md'}")


@main.group("train")
def train_group() -> None:
    """Train models on MovieLens train users and store them in model_artifact."""


@train_group.command("als")
@click.option("--factors", default=128, show_default=True)
@click.option("--regularization", default=0.05, show_default=True)
@click.option("--alpha", default=10.0, show_default=True)
@click.option("--iterations", default=15, show_default=True)
def train_als(factors: int, regularization: float, alpha: float, iterations: int) -> None:
    """Implicit ALS (plan 06). Evaluate the result with `eval run --model als`."""
    import time

    from nextwatch import models_store
    from nextwatch.cf import als
    from nextwatch.evaluation import data, harness

    params = als.AlsParams(factors, regularization, alpha, iterations)
    with db.connect(load_settings().database_url, autocommit=True) as conn:
        interactions = data.load_interactions(DATA_DIR, data.movielens_to_title(conn))
        started = time.monotonic()
        model, _ = als.train(data.train_interactions(interactions), params)
        meta = {
            "params": params.__dict__,
            "items": len(model.item_ids),
            "seconds": round(time.monotonic() - started, 1),
            "git": harness.git_sha(),
        }
        version = models_store.save(conn, als.ARTIFACT, model.to_bytes(), meta)
        models_store.prune(conn, als.ARTIFACT)
    click.echo(f"saved als version {version}: {meta}")
