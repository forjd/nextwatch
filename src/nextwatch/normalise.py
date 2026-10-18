"""Build the normalised catalogue tables from raw TMDB payloads. See docs/plans/02-storage.md.

Idempotent: a title is rebuilt only when its payload is newer than its last normalise, and
each rebuild deletes and reinserts that title's link rows.
"""

import datetime as dt
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection

log = logging.getLogger(__name__)

BATCH = 500
MAX_CAST = 20
CREW_JOBS = {
    "Director",
    "Screenplay",
    "Writer",
    "Story",
    "Novel",
    "Characters",
    "Creator",
    "Original Music Composer",
    "Director of Photography",
    "Editor",
    "Producer",
}
PROVIDER_KINDS = ("flatrate", "free", "ads", "rent", "buy")
# TMDB release types, most useful certification first: theatrical, digital, then the rest.
RELEASE_TYPE_ORDER = {3: 0, 4: 1, 2: 2, 5: 3, 6: 4, 1: 5}


@dataclass
class Credit:
    person_id: int
    role: str
    department: str | None = None
    job: str | None = None
    character: str | None = None
    cast_order: int | None = None
    episode_count: int | None = None


@dataclass
class Parsed:
    """Everything normalise writes for one title."""

    title: dict[str, Any]
    collection: tuple[int, str] | None = None
    genres: list[tuple[int, str]] = field(default_factory=list)
    keywords: list[tuple[int, str]] = field(default_factory=list)
    people: dict[int, tuple] = field(default_factory=dict)
    credits: list[Credit] = field(default_factory=list)
    companies: list[tuple[int, str, str | None, str]] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    providers: list[tuple[int, str, str, int | None]] = field(default_factory=list)
    recs: list[tuple[str, int]] = field(default_factory=list)


def _date(value: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _order(c: dict) -> int:
    order = c.get("order")
    return order if order is not None else 999


def _person(p: dict) -> tuple:
    return (p["id"], p.get("name") or "", p.get("gender"), p.get("known_for_department"),
            p.get("popularity"))  # fmt: skip


def _certification_gb(media_type: str, payload: dict) -> str | None:
    if media_type == "tv":
        for r in (payload.get("content_ratings") or {}).get("results", []):
            if r.get("iso_3166_1") == "GB" and r.get("rating"):
                return r["rating"]
        return None
    for r in (payload.get("release_dates") or {}).get("results", []):
        if r.get("iso_3166_1") != "GB":
            continue
        dates = [d for d in r.get("release_dates", []) if d.get("certification")]
        dates.sort(key=lambda d: RELEASE_TYPE_ORDER.get(d.get("type"), 9))
        if dates:
            return dates[0]["certification"]
    return None


def _movie_credits(payload: dict, out: Parsed) -> None:
    credits = payload.get("credits") or {}
    cast = sorted(credits.get("cast", []), key=_order)[:MAX_CAST]
    for c in cast:
        out.people[c["id"]] = _person(c)
        out.credits.append(
            Credit(c["id"], "cast", "Acting", None, c.get("character"), c.get("order"))
        )
    for c in credits.get("crew", []):
        if c.get("job") in CREW_JOBS:
            out.people[c["id"]] = _person(c)
            out.credits.append(Credit(c["id"], "crew", c.get("department"), c["job"]))


def _tv_credits(payload: dict, out: Parsed) -> None:
    credits = payload.get("aggregate_credits") or {}
    cast = sorted(
        credits.get("cast", []),
        key=lambda c: (-(c.get("total_episode_count") or 0), _order(c)),
    )[:MAX_CAST]
    for order, c in enumerate(cast):
        out.people[c["id"]] = _person(c)
        roles = c.get("roles") or [{}]
        out.credits.append(
            Credit(
                c["id"],
                "cast",
                "Acting",
                None,
                roles[0].get("character"),
                order,
                c.get("total_episode_count"),
            )  # fmt: skip
        )
    for c in credits.get("crew", []):
        for j in c.get("jobs") or []:
            if j.get("job") in CREW_JOBS:
                out.people[c["id"]] = _person(c)
                out.credits.append(
                    Credit(
                        c["id"],
                        "crew",
                        c.get("department"),
                        j["job"],
                        episode_count=j.get("episode_count"),
                    )  # fmt: skip
                )
    for c in payload.get("created_by") or []:
        out.people.setdefault(c["id"], _person(c))
        out.credits.append(Credit(c["id"], "crew", "Writing", "Creator"))


def parse(
    media_type: str, tmdb_id: int, payload: dict, fetched_at: dt.datetime | None = None
) -> Parsed:
    if media_type == "movie":
        name = payload.get("title")
        original_name = payload.get("original_title")
        release_date = _date(payload.get("release_date"))
        runtime = payload.get("runtime") or None
        keywords = (payload.get("keywords") or {}).get("keywords", [])
    else:
        name = payload.get("name")
        original_name = payload.get("original_name")
        release_date = _date(payload.get("first_air_date"))
        runtimes = [r for r in payload.get("episode_run_time") or [] if r]
        runtime = int(statistics.median(runtimes)) if runtimes else None
        keywords = (payload.get("keywords") or {}).get("results", [])

    imdb_id = payload.get("imdb_id") or (payload.get("external_ids") or {}).get("imdb_id")
    collection = payload.get("belongs_to_collection")
    out = Parsed(
        title={
            "media_type": media_type,
            "tmdb_id": tmdb_id,
            "imdb_id": imdb_id or None,
            "name": name or original_name or "",
            "original_name": original_name,
            "original_language": payload.get("original_language"),
            "overview": payload.get("overview") or None,
            "tagline": payload.get("tagline") or None,
            "release_date": release_date,
            "year": release_date.year if release_date else None,
            "runtime": runtime,
            "status": payload.get("status"),
            "number_of_seasons": payload.get("number_of_seasons"),
            "vote_count": payload.get("vote_count") or 0,
            "vote_average": payload.get("vote_average"),
            "popularity": payload.get("popularity"),
            "poster_path": payload.get("poster_path"),
            "certification_gb": _certification_gb(media_type, payload),
            "collection_id": collection["id"] if collection else None,
            "source_fetched_at": fetched_at,
        },
        collection=(collection["id"], collection.get("name") or "") if collection else None,
        genres=[(g["id"], g["name"]) for g in payload.get("genres") or []],
        keywords=[(k["id"], k["name"]) for k in keywords],
        countries=sorted({c["iso_3166_1"] for c in payload.get("production_countries") or []}),
    )

    if media_type == "movie":
        _movie_credits(payload, out)
    else:
        _tv_credits(payload, out)

    seen_companies = set()
    for kind, key in (("production", "production_companies"), ("network", "networks")):
        for c in payload.get(key) or []:
            if (c["id"], kind) not in seen_companies:
                seen_companies.add((c["id"], kind))
                out.companies.append((c["id"], c.get("name") or "", c.get("origin_country"), kind))

    gb = ((payload.get("watch/providers") or {}).get("results") or {}).get("GB") or {}
    for kind in PROVIDER_KINDS:
        for p in gb.get(kind) or []:
            out.providers.append(
                (p["provider_id"], p.get("provider_name") or "", kind, p.get("display_priority"))
            )

    for r in payload.get("recommendations") or []:
        out.recs.append((r.get("media_type") or media_type, r["id"]))
    return out


TITLE_COLUMNS = (
    "media_type", "tmdb_id", "imdb_id", "name", "original_name", "original_language",
    "overview", "tagline", "release_date", "year", "runtime", "status", "number_of_seasons",
    "vote_count", "vote_average", "popularity", "poster_path", "certification_gb",
    "collection_id", "source_fetched_at",
)  # fmt: skip

UPSERT_TITLE = f"""
INSERT INTO title ({", ".join(TITLE_COLUMNS)}, normalised_at)
VALUES ({", ".join(f"%({c})s" for c in TITLE_COLUMNS)}, now())
ON CONFLICT (media_type, tmdb_id) DO UPDATE SET
{", ".join(f"{c} = EXCLUDED.{c}" for c in TITLE_COLUMNS[2:])}, normalised_at = now()
RETURNING id
"""

LINK_TABLES = (
    "title_genre", "title_keyword", "credit", "title_company", "title_country",
    "title_provider", "title_tmdb_rec",
)  # fmt: skip


def _write_batch(conn: Connection, batch: list[Parsed]) -> None:
    def dedupe(rows):
        return list({r[0]: r for r in rows}.values())

    with conn.cursor() as cur:
        # Shared rows first, so link rows can reference them.
        cur.executemany(
            "INSERT INTO collection VALUES (%s, %s)"
            " ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            dedupe(p.collection for p in batch if p.collection),
        )
        cur.executemany(
            "INSERT INTO genre VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            dedupe(g for p in batch for g in p.genres),
        )
        cur.executemany(
            "INSERT INTO keyword VALUES (%s, %s)"
            " ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            dedupe(k for p in batch for k in p.keywords),
        )
        cur.executemany(
            "INSERT INTO person VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET"
            " name = EXCLUDED.name, gender = EXCLUDED.gender,"
            " known_for_department = EXCLUDED.known_for_department,"
            " popularity = EXCLUDED.popularity",
            dedupe(person for p in batch for person in p.people.values()),
        )
        cur.executemany(
            "INSERT INTO company VALUES (%s, %s, %s) ON CONFLICT (id) DO UPDATE SET"
            " name = EXCLUDED.name, origin_country = EXCLUDED.origin_country",
            dedupe(c[:3] for p in batch for c in p.companies),
        )
        cur.executemany(
            "INSERT INTO provider VALUES (%s, %s)"
            " ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            dedupe(pr[:2] for p in batch for pr in p.providers),
        )

        ids = []
        for p in batch:
            cur.execute(UPSERT_TITLE, p.title)
            ids.append(cur.fetchone()[0])

        for table in LINK_TABLES:
            cur.execute(f"DELETE FROM {table} WHERE title_id = ANY(%s)", (ids,))

        pairs = list(zip(ids, batch, strict=True))
        cur.executemany(
            "INSERT INTO title_genre VALUES (%s, %s)",
            [(i, g[0]) for i, p in pairs for g in dict(p.genres).items()],
        )
        cur.executemany(
            "INSERT INTO title_keyword VALUES (%s, %s)",
            [(i, k) for i, p in pairs for k in dict(p.keywords)],
        )
        cur.executemany(
            "INSERT INTO credit (title_id, person_id, role, department, job, character,"
            " cast_order, episode_count) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    i,
                    c.person_id,
                    c.role,
                    c.department,
                    c.job,
                    c.character,
                    c.cast_order,
                    c.episode_count,
                )  # fmt: skip
                for i, p in pairs
                for c in p.credits
            ],
        )
        cur.executemany(
            "INSERT INTO title_company VALUES (%s, %s, %s)",
            [(i, c[0], c[3]) for i, p in pairs for c in p.companies],
        )
        cur.executemany(
            "INSERT INTO title_country VALUES (%s, %s)",
            [(i, c) for i, p in pairs for c in p.countries],
        )
        cur.executemany(
            "INSERT INTO title_provider VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [(i, pr[0], pr[2], pr[3]) for i, p in pairs for pr in p.providers],
        )
        cur.executemany(
            "INSERT INTO title_tmdb_rec VALUES (%s, %s, %s, %s)",
            [(i, rank, mt, tid) for i, p in pairs for rank, (mt, tid) in enumerate(p.recs)],
        )


def _remove_deleted(conn: Connection) -> None:
    """Drop titles TMDB deleted, unless the owner rated or interacted with them."""
    removed = conn.execute(
        """
        DELETE FROM title t USING tmdb_title r
        WHERE r.media_type = t.media_type AND r.tmdb_id = t.tmdb_id AND r.deleted_at IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM rating WHERE title_id = t.id)
          AND NOT EXISTS (SELECT 1 FROM event WHERE title_id = t.id)
        """
    ).rowcount
    kept = conn.execute(
        "SELECT t.media_type, t.tmdb_id FROM title t JOIN tmdb_title r"
        " USING (media_type, tmdb_id) WHERE r.deleted_at IS NOT NULL"
    ).fetchall()
    if removed:
        log.info("removed %d deleted titles", removed)
    for media_type, tmdb_id in kept:
        log.warning("%s/%s was deleted on TMDB but has owner data; kept", media_type, tmdb_id)


def normalise(conn: Connection) -> int:
    """Rebuild every title whose payload changed. Returns the number of titles written."""
    total = 0
    while True:
        rows = conn.execute(
            """
            SELECT r.media_type, r.tmdb_id, r.payload, r.full_fetched_at
            FROM tmdb_title r
            LEFT JOIN title t USING (media_type, tmdb_id)
            WHERE r.payload IS NOT NULL AND r.deleted_at IS NULL
              AND t.source_fetched_at IS DISTINCT FROM r.full_fetched_at
            ORDER BY r.media_type, r.tmdb_id
            LIMIT %s
            """,
            (BATCH,),
        ).fetchall()
        if not rows:
            break
        batch = [parse(mt, tid, payload, fetched) for mt, tid, payload, fetched in rows]
        with conn.transaction():
            _write_batch(conn, batch)
        total += len(batch)
        if total % 10_000 < BATCH:
            log.info("normalised %d titles", total)
    with conn.transaction():
        _remove_deleted(conn)
    log.info("normalised %d titles in total", total)
    return total
