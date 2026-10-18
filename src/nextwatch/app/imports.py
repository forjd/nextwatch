"""Import ratings from a Letterboxd or IMDb CSV export. See plan 08, "Getting started"."""

import csv
import datetime as dt
import io
from dataclasses import dataclass, field

from psycopg import Connection


@dataclass
class ImportResult:
    kind: str
    matched: int = 0
    unmatched: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.matched + len(self.unmatched)


def detect(header: list[str]) -> str:
    if "Letterboxd URI" in header and "Rating" in header:
        return "letterboxd"
    if "Const" in header and "Your Rating" in header:
        return "imdb"
    raise ValueError("not a Letterboxd ratings.csv or an IMDb ratings export")


def _half_steps(x: float) -> float:
    return min(5.0, max(0.5, round(x * 2) / 2))


def _date(value: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value.strip()).replace(tzinfo=dt.UTC)
    except ValueError:
        return dt.datetime.now(dt.UTC)


def _by_name(conn: Connection, name: str, year: int | None) -> int | None:
    """Best movie match by title and year: exact year first, then a year either side."""
    rows = conn.execute(
        """
        SELECT id, year FROM title
        WHERE media_type = 'movie'
          AND (lower(name) = lower(%(n)s) OR lower(original_name) = lower(%(n)s))
          AND (%(y)s::int IS NULL OR year BETWEEN %(y)s - 1 AND %(y)s + 1)
        ORDER BY (year IS NOT DISTINCT FROM %(y)s) DESC, vote_count DESC
        LIMIT 1
        """,
        {"n": name, "y": year},
    ).fetchone()
    return rows[0] if rows else None


def import_csv(conn: Connection, text: str) -> ImportResult:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    kind = detect(reader.fieldnames or [])
    result = ImportResult(kind)
    with conn.transaction():
        for row in reader:
            if kind == "letterboxd":
                if not row.get("Rating"):
                    continue
                year = int(row["Year"]) if row.get("Year", "").isdigit() else None
                title_id = _by_name(conn, row["Name"], year)
                score = _half_steps(float(row["Rating"]))
                when, label = _date(row.get("Date", "")), f"{row['Name']} ({row.get('Year')})"
            else:
                found = conn.execute(
                    "SELECT id FROM title WHERE imdb_id = %s ORDER BY vote_count DESC LIMIT 1",
                    (row["Const"],),
                ).fetchone()
                title_id = found[0] if found else None
                score = _half_steps(float(row["Your Rating"]) / 2)
                when, label = (
                    _date(row.get("Date Rated", "")),
                    f"{row.get('Title')} ({row['Const']})",
                )
            if title_id is None:
                result.unmatched.append(label)
                continue
            conn.execute(
                """
                INSERT INTO rating (title_id, score, source, rated_at)
                VALUES (%s, %s, 'import', %s)
                ON CONFLICT (title_id) DO UPDATE SET score = EXCLUDED.score,
                    source = EXCLUDED.source, rated_at = EXCLUDED.rated_at
                WHERE rating.source = 'import'
                """,
                (title_id, score, when),
            )
            result.matched += 1
    return result
