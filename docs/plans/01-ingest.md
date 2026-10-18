# 01 — Ingest

## Goal

Keep a local, current copy of the TMDB data the engine needs: every movie and TV series worth recommending, with its credits, keywords, genres, ratings, external IDs and UK watch providers.

Ingest only fetches and stores raw API responses. Turning them into tables is [02 — Storage](02-storage.md). Turning them into features is later plans.

## Sources

| Source | What we use it for |
| --- | --- |
| Daily ID exports, `https://files.tmdb.org/p/exports/{movie_ids,tv_series_ids}_MM_DD_YYYY.json.gz` | The full list of valid IDs. One JSON object per line with `id`, `adult`, `video`, `popularity`. Published by 08:00 UTC, kept for 3 months, no auth. |
| `GET /3/movie/{id}` and `GET /3/tv/{id}` with `append_to_response` | Title details plus related data in one request. |
| `GET /3/movie/changes` and `GET /3/tv/changes` | IDs changed in a date range (max 14 days, 100 per page). |
| MovieLens 32M `links.csv` | `tmdbId` for every MovieLens movie. These are always ingested, see below. |

We do not fetch people, seasons or episodes. Credits give us person ID, name, department, job or character and popularity, which is all the engine uses. TV `aggregate_credits` covers the whole series, so per-season data adds nothing.

### What we append

| Movie | TV |
| --- | --- |
| `credits` | `aggregate_credits` |
| `keywords` | `keywords` |
| `external_ids` | `external_ids` |
| `release_dates` | `content_ratings` |
| `watch/providers` | `watch/providers` |
| `recommendations` | `recommendations` |

`recommendations` is TMDB's own list (first page, 20 items). It is a cheap extra signal and a sanity check for our results. We keep only the IDs.

Requests use `language=en-US`. `original_language` is in the payload anyway.

## Two passes

Most of TMDB's ~1.1M movies have almost no votes, overview or credits. Fetching and storing all of them in full would blow the 5 GB Railway Hobby volume and add nothing. The export file has `popularity` but not vote counts, and popularity is too volatile to filter on.

1. **Light pass.** Fetch plain details (no appends) for every non-adult, non-`video` ID in the exports. Store only a stub: `media_type`, `tmdb_id`, `title`, `vote_count`, `vote_average`, `popularity`, `release_date` or `first_air_date`, `fetched_at`.
2. **Full pass.** Fetch with appends for titles that are **eligible**:
   - `vote_count >= MIN_VOTES` (config, start at 10), or
   - in MovieLens `links.csv`, whatever its vote count (we need them to train and evaluate).

Pick `MIN_VOTES` after the light pass from a histogram of vote counts, so the eligible set and its estimated size fit the budget below.

A title that gains votes later is promoted by the daily sync. A title that drops below the threshold is kept; we never demote.

## Payload trimming

Before storing a full payload:

- Keep only the `GB` entry from `watch/providers`.
- Keep only `GB` and `US` from `release_dates` and `content_ratings`.
- Reduce `recommendations` to a list of IDs.
- Drop `backdrop_path`, `belongs_to_collection.backdrop_path`, `production_companies[].logo_path` and similar image fields we will not show. Keep `poster_path`.

## Storage

Two tables, both written only by ingest. [02 — Storage](02-storage.md) builds normalised tables from them.

- `tmdb_title` — one row per `(media_type, tmdb_id)`: stub columns, `eligible`, `deleted_at`, `light_fetched_at`, `full_fetched_at`, and `payload jsonb` (null until the full pass).
- `ingest_queue` — `(media_type, tmdb_id, kind, reason, attempts, next_attempt_at, last_error)`, where `kind` is `light` or `full` and `reason` is `backfill`, `new`, `changed`, `refresh` or `promote`. Workers claim rows with `FOR UPDATE SKIP LOCKED`.

Upserts are idempotent: fetching a title twice gives the same row.

## Jobs

All jobs read config from environment variables (`DATABASE_URL`, `TMDB_API_TOKEN`, `MIN_VOTES`, `TMDB_RPS`) and exit when done, so each one can be a Railway cron service later.

| Job | When | What it does |
| --- | --- | --- |
| `backfill` | Once, locally | Load the latest exports and MovieLens links, queue every ID for `light`, work the queue, then queue eligible titles for `full` and work that. Safe to kill and restart. |
| `sync` | Daily, after 08:00 UTC | Diff today's export against `tmdb_title` and queue new IDs for `light` and missing IDs as deleted. Call `/changes` from the last successful sync (max 14 days) and queue changed IDs for `full` if they are eligible, or `light` if not. Queue promotions. Work the queue. |
| `refresh` | Daily, after `sync` | Queue the eligible titles with the oldest `full_fetched_at`, enough to cycle the whole set every 30 days. Vote counts and watch providers drift without always showing in `/changes`. |

If `sync` has not run for more than 14 days, the change window is lost. It then logs a warning and queues every eligible title for `refresh`.

## Rate limits and errors

- TMDB has no published limit, only "somewhere in the 40 requests per second range". We run at `TMDB_RPS=20` with a token bucket and a small worker pool.
- On `429`, honour `Retry-After` if present, else back off exponentially, and slow the whole pool, not just that worker.
- On `404`, set `deleted_at` and drop the queue row.
- On `5xx` or a network error, retry with backoff up to 5 attempts, then leave the row with `last_error` for inspection.
- Ignore `adult` titles everywhere. Handle `adult: null` in `/changes` results as not adult and let the details call decide.

## Time and size estimates

To check after the light pass. These are guesses.

- Light pass: ~1.3M requests at 20/s is about 18 hours. Run it once in a `tmux` session on the VM.
- Full pass: if ~150k titles are eligible, about 2 hours.
- Daily sync: a few thousand requests, a few minutes.
- Full payloads after trimming: maybe 5–20 KB each, about 1–2 GB in Postgres after TOAST compression. Stubs for ~1.3M titles add about 200 MB. That leaves room in 5 GB for normalised tables and embeddings, but not much more. If the measured size is too big, raise `MIN_VOTES`.

## Stack

Python, like the rest of the project: `httpx` (async) for TMDB and `psycopg` 3 for Postgres.

## Open questions

- **TV threshold.** TV has far fewer votes per title than movies. It may need its own, lower `MIN_VOTES`.
- **Spike first.** Confirm with a few real requests that `aggregate_credits` works in `append_to_response`, and measure real payload sizes before the full pass.

## How we know it works

- After `backfill`, every non-adult, non-video ID in the export has a `tmdb_title` row, and every eligible one has a `payload`.
- Every MovieLens `tmdbId` that still exists on TMDB has a `payload`.
- Kill `backfill` halfway and restart it. It finishes without duplicate rows or refetching completed titles.
- Edit nothing, run `sync` twice in a row. The second run fetches almost nothing.
- Pick a title from yesterday's `/changes`. After `sync`, its `full_fetched_at` is today.
- The database size is logged at the end of every job and stays under budget.
