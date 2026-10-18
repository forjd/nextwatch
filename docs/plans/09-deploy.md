# 09 — Deploy

## Goal

Run the app and the daily jobs on Railway Hobby for roughly $10–20 a month, with the same containers we run locally.

## Local first

`compose.yaml` runs the same parts as Railway:

| Service | Image | Notes |
| --- | --- | --- |
| `db` | `pgvector/pgvector:pg17` | Data in a named volume |
| `web` | Our image, `nextwatch web` | Port 8000, exposed with Tailscale Serve ([08](08-app.md)) |
| `jobs` | Our image, `nextwatch daily` | Run by hand or by a host cron |

One `Dockerfile` (Python 3.12 slim, `uv sync --frozen`) builds one image. The command decides what it does. Config comes from `.env` locally and from Railway variables in production.

Heavy one-off work stays local: the TMDB `backfill` ([01](01-ingest.md)), training ALS and the ranker ([06](06-collaborative-filtering.md), [07](07-ranking.md)), and evaluation ([03](03-evaluation.md)). Their results reach Railway as database rows, so Railway only runs light work.

## Railway project

| Service | Source | Schedule | Resources |
| --- | --- | --- | --- |
| `postgres` | Railway pgvector template | Always on | 5 GB volume |
| `web` | This repo, `nextwatch web` | Always on, or Serverless (below) | ~0.5–1 GB RAM |
| `daily` | This repo, `nextwatch daily` | Cron `30 8 * * *` (UTC, after TMDB's 08:00 exports) | Runs for minutes, then exits |

`nextwatch daily` runs the steps in order and exits: `sync`, `refresh`, `normalise`, `embed`, `vibes`, `export`. Each step is idempotent, so a failed run is fixed by the next one. If a run is still going when the next is due, Railway skips the new run.

Services reach Postgres over the private network with `DATABASE_URL` from the template's reference variable.

## Getting the data there

The first load is a local backfill and a `pg_dump`, restored into Railway with `pg_restore`. Rough size 2–3 GB, so this is minutes, not hours. After that, `daily` keeps it current in place.

Models trained locally go to Railway with `nextwatch push-model <name>`, which copies the newest `model_artifact` row across. The app picks it up within a minute.

## Backups

- Turn on Railway's daily volume backups for `postgres` (kept 6 days) and weekly (kept 27 days).
- `export` in `daily` also writes the owner's data ([02](02-storage.md)) as gzipped JSON to the private repo `forjd/nextwatch-backups`, so it survives losing the whole Railway project.
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
