# 02 — Storage

## Goal

One Postgres database holds the catalogue, the features the engine needs at request time, the owner's ratings and the trained models. It fits in the 5 GB Railway Hobby volume with room to spare.

## Decisions

- **Postgres 17 with `pgvector`.** Locally the `pgvector/pgvector:pg17` image. On Railway, the pgvector template (the standard Railway image has no pgvector).
- **No ORM.** `psycopg` 3 with plain SQL. The queries are few and specific, and the bulk loads want `COPY`.
- **Migrations are numbered SQL files** in `migrations/` (`0001_ingest.sql`, ...), applied in order by a small `nextwatch migrate` command that records them in `schema_migration`. Forward only.
- **Raw and normalised data are separate.** Ingest owns `tmdb_title` ([01](01-ingest.md)). A `normalise` step rebuilds the tables below from it. Normalised tables can always be dropped and rebuilt from raw.
- **MovieLens stays out of Postgres** except the ID mapping. The 32M ratings live as Parquet in `data/movielens/` and only the training job reads them ([06](06-collaborative-filtering.md)).
- **Trained models are rows**, not files, so there is no bucket or shared volume to manage: `model_artifact` holds a `bytea` blob per model version.

## Schema

Keys: every normalised table uses `title.id` (a `bigint` identity), not `(media_type, tmdb_id)`. `title` has a unique index on `(media_type, tmdb_id)`.

### Catalogue

| Table | Columns (main ones) |
| --- | --- |
| `title` | `id`, `media_type` (`movie`/`tv`), `tmdb_id`, `imdb_id`, `name`, `original_name`, `original_language`, `overview`, `tagline`, `release_date`, `year`, `runtime` (TV: median episode runtime), `status`, `number_of_seasons`, `vote_count`, `vote_average`, `popularity`, `poster_path`, `certification_gb`, `normalised_at` |
| `genre`, `title_genre` | TMDB genre ID and name. Movie and TV genres share one table. |
| `keyword`, `title_keyword` | TMDB keyword ID and name. |
| `person` | `id` (TMDB person ID), `name`, `gender`, `known_for_department`, `popularity` |
| `credit` | `title_id`, `person_id`, `role` (`cast`/`crew`), `department`, `job`, `character`, `cast_order`, `episode_count` |
| `company`, `title_company` | Production companies and, for TV, networks (`kind`). |
| `title_country` | Production countries. |
| `collection`, `title.collection_id` | Movie collections (franchises). |
| `provider`, `title_provider` | GB watch providers: `provider_id`, `kind` (`flatrate`/`free`/`ads`/`rent`/`buy`), `display_priority`. |
| `title_tmdb_rec` | TMDB's own recommendations: `title_id`, `rec_tmdb_id`, `rank`. Kept by TMDB ID, since the target may not be eligible. |
| `movielens_link` | `movielens_id`, `tmdb_id`. From `links.csv`. |

What `normalise` keeps from credits, to hold the table to a sensible size:

- Cast: the first 20 by `order` (TV: by `total_episode_count`, then order).
- Crew: only Director, Screenplay, Writer, Story, Novel, Characters, Creator, Original Music Composer, Director of Photography, Editor, Producer. Not executive producers, which are noisy.

### Features

| Table | Columns | Written by |
| --- | --- | --- |
| `title_embedding` | `title_id`, `model`, `embedding halfvec(N)`, `source_hash` | [04](04-content-similarity.md) |
| `title_vibe` | `title_id`, `model`, `prompt_version`, `tags jsonb`, `feel`, `input_tokens`, `output_tokens`, `cost_usd`, `created_at` | [05](05-vibe-tags.md) |

`title_embedding` gets an HNSW index (`halfvec_cosine_ops`). `source_hash` is a hash of the text that was embedded, so we only re-embed when the text changes.

### Owner data

The only data we cannot rebuild from TMDB. Single user, so no `user` table.

| Table | Columns |
| --- | --- |
| `rating` | `title_id`, `score` (1–5, half steps), `source` (`app`, `onboarding`, `import`), `rated_at` |
| `event` | `title_id`, `kind` (`seen`, `dismiss`, `watchlist`, `unwatchlist`, `impression`, `click`), `context jsonb`, `at` |

### Models and evaluation

| Table | Columns |
| --- | --- |
| `model_artifact` | `name` (`als`, `ranker`, ...), `version`, `created_at`, `blob bytea`, `meta jsonb` (params, metrics, git SHA). The app loads the newest version of each name at start. |
| `eval_run` | `id`, `created_at`, `git_sha`, `config jsonb`, `metrics jsonb`. See [03](03-evaluation.md). |

## Normalise

`nextwatch normalise` finds rows in `tmdb_title` where `full_fetched_at > normalised_at` (or `normalised_at` is null) and, per title, in one transaction: upserts `title`, `person`, `genre`, `keyword`, `company`, `provider`, then deletes and reinserts that title's link rows. It is idempotent and runs after every `sync`.

Deleted titles (`deleted_at` set) are removed from `title`, and link tables follow with `ON DELETE CASCADE`. Owner tables reference `title` with `ON DELETE RESTRICT` instead: TMDB deletes are rare and losing a rating is worse, so `normalise` skips the delete and logs it if the title has owner data.

## Size budget

Guesses for ~150k eligible titles, to check after the first full load. Postgres also needs room for WAL, vacuum and indexes, so we aim to stay under **3.5 GB** total.

| Data | Estimate |
| --- | --- |
| `tmdb_title` payloads (TOAST-compressed) and stubs | 1.2–2.2 GB |
| `credit` (~5M rows) with indexes | 400–600 MB |
| Other catalogue tables | 150 MB |
| `title_embedding`, 384-dim `halfvec`, plus HNSW | 250–350 MB |
| `title_vibe` | 100 MB |
| `model_artifact` (a few versions kept) | 100–200 MB |
| Total | 2.2–3.6 GB |

If it is too tight, the first lever is to null out `tmdb_title.payload` once it has been normalised (we can refetch from TMDB at any time). The second is raising `MIN_VOTES`.

`nextwatch db-size` prints the size of every table, and each job logs the total at the end.

## Backups

- Catalogue and features can be rebuilt from TMDB and the jobs, so they need no backup beyond convenience.
- `rating` and `event` cannot. `nextwatch export` writes them to a gzipped JSON file in the private repo `forjd/nextwatch-backups` ([09](09-deploy.md)).
- `model_artifact` can be retrained locally, so it is not exported.

## How we know it works

- `migrate` on an empty database, then on an up-to-date one, both succeed and the second does nothing.
- `normalise` twice in a row gives identical row counts and the second run touches no titles.
- Drop every normalised table, rerun `migrate` and `normalise`, and get the same counts.
- Spot-check 10 titles against the TMDB website: cast order, director, genres, GB providers.
- `db-size` total is under 3.5 GB after the full load.
