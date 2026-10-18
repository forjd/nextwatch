# AGENTS.md

`nextwatch` is a personal movie and TV recommendation engine. It keeps a local copy of TMDB's catalogue and ranks titles with a hybrid of content similarity, collaborative filtering and a learned ranker.

Plans live in `docs/plans/`. Code is being built plan by plan; the status column in the Plans table says how far each one is. Where code and a plan disagree, fix one of them in the same change.

## Scope and constraints

- Personal use only. No public service, no other users. This keeps us inside the TMDB API terms for non-commercial use and the MovieLens research licence.
- There is no existing watch history. Collaborative signal comes from MovieLens until the owner's own ratings exist.
- Movies and TV are one catalogue. A show can recommend a movie and the reverse.
- TMDB requires attribution in any UI: "This product uses the TMDB API but is not endorsed or certified by TMDB."
- Never commit API keys or downloaded datasets. Keys go in `.env` (gitignored). Data goes in `data/` (gitignored).

## Approach

The plans expand on each part. In short:

1. **Ingest.** Seed IDs from TMDB's daily export files, fetch each title with `append_to_response` (credits, keywords, external IDs, watch providers, release dates or content ratings), then sync daily with the `/changes` endpoints.
2. **Store.** Postgres with `pgvector`. Raw API responses in `jsonb`, plus normalised tables for titles, people, credits, keywords and genres.
3. **Features.** TF-IDF weighted sparse features (cast, crew, keywords, genres), text embeddings of overviews, LLM-extracted "vibe" tags (tone, pacing, themes), and implicit-feedback collaborative filtering trained on MovieLens.
4. **Rank.** Candidate generation from each signal, then a learning-to-rank model (LightGBM LambdaRank) combines them. Weights are learned, not hand-tuned.
5. **Re-rank.** Diversity (MMR), popularity debiasing, and filtering by UK watch providers.
6. **Evaluate.** Offline recall@k and nDCG on held-out MovieLens histories. Every change to ranking must be measured against this before it is kept.

## Stack

Python 3.12 for everything (ingest, features, training, API), managed with `uv`. The ranking and training libraries (`implicit`, LightGBM) are Python, so one language keeps it simple.

## Hosting

- Develop locally with Docker Compose (Postgres with pgvector plus the app services) on the `ai-agent` VM.
- Deploy to Railway later, on the Hobby plan. Design for it from the start:
  - Each part is its own container: Postgres, the API/UI, and jobs (TMDB sync, model training) that run, finish and exit so they can be Railway cron services.
  - Use Railway's pgvector Postgres template, not the standard Postgres image.
  - Hobby volumes are capped at 5 GB per service. Keep the database under that: index only titles above a vote threshold, keep raw JSON lean, and train on MovieLens outside Postgres.
  - All config comes from environment variables (`DATABASE_URL`, `TMDB_API_TOKEN`, ...). No hard-coded hosts or paths.

## Layout

| Path | What it is |
| --- | --- |
| `src/nextwatch/cli.py` | The `nextwatch` command (click) |
| `src/nextwatch/config.py` | Settings from environment variables and `.env` |
| `src/nextwatch/db.py` | Connections, migration runner, `db-size` |
| `src/nextwatch/migrations/` | Numbered SQL migrations, forward only |
| `src/nextwatch/tmdb.py` | Async TMDB client with a shared rate limit |
| `src/nextwatch/ingest/` | Exports, payload trimming, queue, workers and the ingest jobs |
| `src/nextwatch/normalise.py` | Builds the catalogue tables from `tmdb_title` payloads |
| `src/nextwatch/backup.py` | Export and import of the owner's ratings and events |
| `src/nextwatch/movielens.py` | MovieLens 32M download, Parquet conversion, TMDB links |
| `src/nextwatch/evaluation/` | Splits, metrics, baselines (`MODELS`) and the eval harness |
| `src/nextwatch/features/` | Sparse TF-IDF families (`sparse.py`), embeddings (`embed.py`), content models (`content.py`) |
| `src/nextwatch/vibes/` | Vocabulary, prompt, OpenRouter client and the `vibes` job |
| `src/nextwatch/cf/als.py` | Implicit ALS training, numpy fold-in, `als` recommender |
| `src/nextwatch/models_store.py` | Save, load and prune `model_artifact` rows |
| `src/nextwatch/ranking/` | `signals.py` (aligned inputs), `features.py` (candidates, features), `ranker.py` (LambdaRank, re-rank, explain) |
| `src/nextwatch/app/` | Web app: `main.py` (routes), `engine.py` (models, profile, filters), `imports.py`, `templates/`, `static/` |
| `src/nextwatch/registry.py` | Imports every model module; use its `MODELS` |
| `src/nextwatch/http.py` | Shared HTTP settings (system CA store) |
| `eval/golden.yaml` | Golden set for T3, written by the owner |
| `tests/` | pytest. `conftest.py` has a fake TMDB API and a throwaway test database |
| `compose.yaml` | Local Postgres with pgvector on `127.0.0.1:5433`; `web` and `daily` behind the `app` and `jobs` profiles |
| `Dockerfile` | One image for every service, embedding model baked in |
| `.railway/railway.ts` | Railway project as code (TypeScript IaC; `npm install` in `.railway/` to plan) |
| `scripts/init-test-db.sql` | Creates `nextwatch_test` on first start of the database |
| `.github/workflows/ci.yml` | Lint and tests on the self-hosted `forjd` runner |
| `docs/plans/` | Numbered plan files, one per area (`NN-topic.md`) |
| `AGENTS.md` | This file |
| `CLAUDE.md` | Just `@AGENTS.md` |
| `LICENSE` | MIT |

Update this table when you add top-level paths.

## Commands

```sh
cp .env.example .env          # then set TMDB_API_TOKEN
docker compose up -d --wait   # Postgres on 127.0.0.1:5433
uv sync
uv run nextwatch migrate
uv run nextwatch backfill     # first load; safe to stop and rerun
uv run nextwatch sync         # daily
uv run nextwatch refresh      # daily, after sync
uv run nextwatch normalise    # after sync/refresh: rebuild changed titles
uv run nextwatch export       # owner data to data/backups/owner-data.json.gz
uv run nextwatch embed        # after normalise: embed new or changed titles
uv run nextwatch vibes --spike # 200-title sample to read; then `vibes` for the rest
uv run nextwatch similar movie/603             # quick look at "more like this"
uv run nextwatch movielens fetch               # once, before backfill
uv run nextwatch train als                     # ~70 s on the full train split
uv run nextwatch train ranker                  # after train als; needs embeddings
uv run nextwatch eval run --model popular      # see `eval models`
uv run nextwatch eval compare <run> <run>
uv run nextwatch queue        # what is left in the ingest queue
uv run nextwatch db-size
uv run nextwatch web          # app on 127.0.0.1:8000; needs APP_PASSWORD, SESSION_SECRET
uv run nextwatch daily        # every daily step in order; what Railway's cron runs
```

Before committing, run `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest -q`. Database tests need `TEST_DATABASE_URL` (in `.env.example`) and skip without it. Never point it at the real database: the tests drop the `public` schema.

## Conventions

- SQL lives next to the code that runs it, as plain strings with `%s` parameters. No ORM.
- Every job is idempotent and safe to kill: state is in Postgres, not in memory.
- Ingest jobs take a Postgres advisory lock, so only one runs at a time.
- Tests never call the real TMDB or OpenRouter APIs. Add behaviour to `FakeTmdb` or `FakeOpenRouter` instead. Payload fixtures shaped like real responses live in `tests/payloads.py`.
- Use `nextwatch.http.SSL_CONTEXT` for every HTTP client: certifi's bundle rejects `files.grouplens.org`.
- Recommenders get history as `(title_id, rating, unix_time)`, oldest first.
- A new recommender must register in `MODELS` (and its module be imported in `registry.py`) and be compared with `eval compare` against the current best before it is kept.
- Adding or removing a ranker feature means editing `FEATURES` in `ranking/features.py` and retraining; old ranker artifacts refuse to load.
- To show the app to the owner, expose it with Tailscale Serve on its own port (not 443), e.g. `tailscale serve --bg --https=8443 http://127.0.0.1:8000`, and turn it off afterwards.
- Backups go to the private repo `forjd/nextwatch-backups` when `BACKUP_GITHUB_TOKEN` is set.
- The VM disk is small. After a local `docker build`, remove the image when done. Never prune Docker caches broadly: other projects share them.
- Each plan owns its tables: add them in a new migration when that plan is built, not earlier.

## CI

- Runs on the self-hosted `forjd` runner. The repo is public, so keep the `if:` guard that stops fork PRs running on it.
- Pin every action to a full commit SHA with a `# vX.Y.Z` comment. Dependabot bumps them monthly.
- Keep `permissions: contents: read`, `persist-credentials: false` and caching off: the runner is shared and persistent.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, for example `docs(plans): add ingest plan` or `feat(ingest): fetch TMDB daily exports`.

## Plans

| Plan | Covers | Status |
| --- | --- | --- |
| [01 — Ingest](docs/plans/01-ingest.md) | TMDB exports, two-pass fetch, daily sync | Built, tested against a fake API. Not yet run against real TMDB. |
| [02 — Storage](docs/plans/02-storage.md) | Postgres schema, normalise, size budget, backups | Built and tested. GitHub upload of backups is part of 09. |
| [03 — Evaluation](docs/plans/03-evaluation.md) | MovieLens splits, metrics, golden set, harness | Built and tested on synthetic data. Real runs wait for the TMDB backfill. Golden set has 3 starter cases. |
| [04 — Content similarity](docs/plans/04-content-similarity.md) | TF-IDF feature families, embeddings, profiles | Built and tested; real model checked. Experiments wait for the backfill. |
| [05 — Vibe tags](docs/plans/05-vibe-tags.md) | LLM tags from a fixed vocabulary, through OpenRouter | Built and tested against a fake API. Needs `OPENROUTER_API_KEY` and the spike. |
| [06 — Collaborative filtering](docs/plans/06-collaborative-filtering.md) | Implicit ALS on MovieLens, owner fold-in, coverage gaps | Built and tested; training timed on real MovieLens. Eval waits for the backfill. |
| [07 — Ranking](docs/plans/07-ranking.md) | Candidates, features, LambdaRank, re-ranking | Built and tested on synthetic data. Real training and latency wait for the backfill. |
| [08 — App](docs/plans/08-app.md) | FastAPI and HTMX app, onboarding, feedback, access | Built and tested; runs against an empty catalogue. |
| [09 — Deploy](docs/plans/09-deploy.md) | Docker Compose locally, Railway services, cost | Built: image tested, IaC type-checked, backup upload tested. Not deployed. |

Build in this order, except that 03 must exist before any of 04–07 is tuned.

## Writing plans

- One file per area, numbered in the order it should be built: `01-ingest.md`, `02-storage.md`, and so on.
- Each plan states its goal, the decisions made and why, open questions, and how we know it works.
- When a decision changes, edit the plan rather than adding a contradicting note below it.
- Keep plans short enough to read in one sitting. Split a plan that grows past that.
