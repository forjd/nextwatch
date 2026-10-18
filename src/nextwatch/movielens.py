"""MovieLens 32M: download, convert to Parquet, load the TMDB links.

Research licence: personal use only, never redistributed or committed. Files live in
data/movielens/, which is gitignored.
"""

import io
import logging
import zipfile
from pathlib import Path

import httpx
import polars as pl
from psycopg import Connection

from nextwatch.http import SSL_CONTEXT

log = logging.getLogger(__name__)

URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
RATINGS = "ratings.parquet"
LINKS = "links.parquet"


def fetch(data_dir: Path, url: str = URL, transport: httpx.BaseTransport | None = None) -> Path:
    """Download the zip, write ratings and links as Parquet, then delete the zip."""
    out = data_dir / "movielens"
    out.mkdir(parents=True, exist_ok=True)
    if (out / RATINGS).exists() and (out / LINKS).exists():
        log.info("MovieLens already converted in %s", out)
        return out

    zip_path = out / "ml-32m.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".part")
        with httpx.Client(
            timeout=300, transport=transport, follow_redirects=True, verify=SSL_CONTEXT
        ) as http:
            with http.stream("GET", url) as resp, tmp.open("wb") as f:
                resp.raise_for_status()
                for chunk in resp.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.rename(zip_path)
        log.info("downloaded %s", url)

    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).name: n for n in z.namelist()}
        with z.open(names["ratings.csv"]) as f:
            ratings = pl.read_csv(
                io.BytesIO(f.read()),
                schema={
                    "userId": pl.Int32,
                    "movieId": pl.Int32,
                    "rating": pl.Float32,
                    "timestamp": pl.Int64,
                },
            )
        ratings.write_parquet(out / RATINGS)
        log.info("wrote %d ratings", ratings.height)
        with z.open(names["links.csv"]) as f:
            links = pl.read_csv(
                io.BytesIO(f.read()),
                schema={"movieId": pl.Int32, "imdbId": pl.Utf8, "tmdbId": pl.Int32},
            )
        links.write_parquet(out / LINKS)
    zip_path.unlink()
    return out


def load_links(conn: Connection, data_dir: Path) -> int:
    """Replace movielens_link from links.parquet. Returns the number of rows."""
    links = pl.read_parquet(data_dir / "movielens" / LINKS)
    rows = [
        (m, f"tt{imdb}" if imdb else None, tmdb)
        for m, imdb, tmdb in links.select("movieId", "imdbId", "tmdbId").iter_rows()
    ]
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("TRUNCATE movielens_link")
        with cur.copy("COPY movielens_link (movielens_id, imdb_id, tmdb_id) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(row)
    log.info("loaded %d MovieLens links", len(rows))
    return len(rows)
