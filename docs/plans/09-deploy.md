# 09 — Deploy

## Goal

Run the app and the daily jobs on Railway Hobby for roughly $10–20 a month, with the same containers we run locally.

## Local first

`compose.yaml` runs the same parts as Railway:

| Service | Image | Notes |
| --- | --- | --- |
| `db` | `pgvector/pgvector:pg17` | Data in a named volume. Starts with plain `docker compose up -d`. |
| `web` | Our image, `nextwatch web` | Profile `app`. Port 8000, exposed with Tailscale Serve ([08](08-app.md)) |
| `daily` | Our image, `nextwatch daily` | Profile `jobs`: `docker compose --profile jobs run --rm daily` |

One `Dockerfile` (Python 3.12 slim, `uv sync --frozen --no-editable`, multi-stage) builds one image. The command decides what it does. The embedding model is baked in at `/opt/models`, so no job or request downloads it. Config comes from `.env` locally and from Railway variables in production.

Measured: the image is 1.16 GB and builds in under a minute. The VM's disk is the limit for local image builds; remove the image after testing if space is short.

Heavy one-off work stays local: the TMDB `backfill` ([01](01-ingest.md)), training ALS and the ranker ([06](06-collaborative-filtering.md), [07](07-ranking.md)), and evaluation ([03](03-evaluation.md)). Their results reach Railway as database rows, so Railway only runs light work.

## Railway project

| Service | Source | Schedule | Resources |
| --- | --- | --- | --- |
| `postgres` | `pgvector/pgvector:pg17` image, `PGDATA` in a subdirectory of the volume | Always on | 5 GB volume |
| `web` | This repo, `nextwatch web` | Always on, or Serverless (below) | ~0.5–1 GB RAM |
| `daily` | This repo, `nextwatch daily` | Cron `30 8 * * *` (UTC, after TMDB's 08:00 exports) | Runs for minutes, then exits |

`nextwatch daily` runs the steps in order and exits: `sync`, `refresh`, `normalise`, `embed`, `vibes`, `export`. A failed step is logged and the rest still run; the job exits non-zero so Railway shows the failure. Each step is idempotent, so the next run catches up. If a run is still going when the next is due, Railway skips the new run.

The Railway project is described in `.railway/railway.ts` (Railway Infrastructure as Code; `railway.json` is deprecated and stops being read on 2026-12-01). Railway's managed Postgres has no pgvector and the IaC `postgres()` helper cannot pick the pgvector template, so the database runs the same `pgvector/pgvector:pg17` image as `compose.yaml`, on a volume. Secrets are `preserve()`d: set in the dashboard, never in the file.

Services reach Postgres over the private network: `DATABASE_URL` is built from `${{postgres.POSTGRES_PASSWORD}}` and `${{postgres.RAILWAY_PRIVATE_DOMAIN}}`. `web` runs `nextwatch migrate` as its pre-deploy command and listens on `::` so the private network (IPv6) reaches it.

## Getting the data there

The first load is a local backfill and a `pg_dump`, restored into Railway with `pg_restore`. Rough size 2–3 GB, so this is minutes, not hours. After that, `daily` keeps it current in place.

Models trained locally go to Railway with `nextwatch push-model <name>`, which copies the newest `model_artifact` row across. The app picks it up within a minute.

## Backups

- Turn on Railway's daily volume backups for `postgres` (kept 6 days) and weekly (kept 27 days).
- `export` in `daily` also writes the owner's data ([02](02-storage.md)) as gzipped JSON to the private repo `forjd/nextwatch-backups` (created), so it survives losing the whole Railway project. Tested end to end: the first export wrote a commit, the second saw no change and wrote nothing.
  - It writes one file, `owner-data.json.gz`, through the GitHub contents API (`PUT /repos/forjd/nextwatch-backups/contents/owner-data.json.gz`). No `git` in the image. Git history keeps every earlier version.
  - It authenticates with `BACKUP_GITHUB_TOKEN`, a fine-grained token with Contents read and write on that one repo only.
  - It skips the write if the content has not changed since the last run, so the history only grows when the owner rates something.
  - `nextwatch import-backup` restores from the newest file, or from a commit SHA.

## Cost

Railway bills RAM at $10 per GB per month, CPU at $20 per vCPU per month and volumes at $0.15 per GB per month. Hobby is $5 a month including $5 of usage.

| Item | Estimate per month |
| --- | --- |
| `postgres`, ~0.5–1 GB RAM, light CPU | $5–10 |
| `web`, ~0.5–1 GB RAM (models in memory), light CPU | $5–10 |
| `daily`, a few minutes a day | < $1 |
| Volume, 5 GB | < $1 |
| **Total**, including the $5 Hobby fee | **~$11–21** |

Plus LLM spend for vibe tags ([05](05-vibe-tags.md)) through OpenRouter: about $1 a month after the backfill.

**Serverless** on `web` puts it to sleep after 5–10 minutes with no outbound traffic and wakes it on the next request. For a personal app used in the evenings, that could halve its cost. The trade-offs: a cold start that reloads every model (seconds), a possible 502 on the first request, and the app must close its database pool when idle or it never sleeps. Try it after launch and measure.

## Secrets

`TMDB_API_TOKEN`, `OPENROUTER_API_KEY`, `BACKUP_GITHUB_TOKEN`, `APP_PASSWORD`, `SESSION_SECRET`, `DATABASE_URL`. Railway variables in production, `.env` locally. Never in git.

## Checks before going live

- `migrate` runs on deploy (as a pre-deploy command) before the new `web` starts.
- `/healthz` checks the database and that every model is loaded. Railway uses it as the health check.
- After the first `daily` run, `db-size` is under 3.5 GB and the logs show no errors.
- The app works on the owner's phone over the public URL, and the password is required.

## First deploy (runbook)

Do this once the local backfill, embeddings and models exist ([01](01-ingest.md)–[07](07-ranking.md)).

1. Create a fine-grained GitHub token with Contents read and write on `forjd/nextwatch-backups` only.
2. `cd .railway && npm install`, then `railway link` to a new empty project, then `railway config plan`. Check the plan creates `postgres`, `pgdata`, `web` and `daily`, then `railway config apply`.
3. In the dashboard, set the preserved secrets: `POSTGRES_PASSWORD` (generate one), `APP_PASSWORD`, `SESSION_SECRET`, `TMDB_API_TOKEN`, `OPENROUTER_API_KEY`, `BACKUP_GITHUB_TOKEN`. Turn on daily and weekly volume backups for `postgres`.
4. Copy the data: `pg_dump -Fc` locally, `pg_restore --no-owner` into Railway through `railway connect postgres` or a temporary TCP proxy, then remove the proxy.
5. Generate a domain for `web`, open it on the phone, log in, and check `/healthz` reports every model loaded.
6. Watch the first `daily` run's logs. `db-size` should stay under 3.5 GB.
