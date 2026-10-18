"""TMDB daily ID exports: one gzipped JSON object per line, published by 08:00 UTC."""

import datetime as dt
import gzip
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

EXPORT_URL = "https://files.tmdb.org/p/exports/{name}_{date:%m_%d_%Y}.json.gz"
EXPORT_NAMES = {"movie": "movie_ids", "tv": "tv_series_ids"}


@dataclass(frozen=True)
class ExportRow:
    tmdb_id: int
    popularity: float


def export_url(media_type: str, date: dt.date) -> str:
    return EXPORT_URL.format(name=EXPORT_NAMES[media_type], date=date)


def download_export(
    media_type: str,
    dest_dir: Path,
    today: dt.date | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Download the newest export: today's if it is out, else yesterday's. Returns the path."""
    today = today or dt.datetime.now(dt.UTC).date()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, transport=transport, follow_redirects=True) as http:
        for date in (today, today - dt.timedelta(days=1)):
            url = export_url(media_type, date)
            path = dest_dir / url.rsplit("/", 1)[1]
            if path.exists():
                return path
            with http.stream("GET", url) as resp:
                if resp.status_code in (403, 404):
                    log.info("export not available yet: %s", url)
                    continue
                resp.raise_for_status()
                tmp = path.with_suffix(".part")
                with tmp.open("wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                tmp.rename(path)
                log.info("downloaded %s", url)
                return path
    raise RuntimeError(f"no {media_type} export found for {today} or the day before")


def read_export(path: Path) -> Iterator[ExportRow]:
    """Yield every non-adult, non-video row in an export file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("adult") or row.get("video"):
                continue
            yield ExportRow(int(row["id"]), float(row.get("popularity") or 0))
