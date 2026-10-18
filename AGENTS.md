# AGENTS.md

`nextwatch` is a personal movie and TV recommendation engine. It keeps a local copy of TMDB's catalogue and ranks titles with a hybrid of content similarity, collaborative filtering and a learned ranker.

The project is in the planning stage. There is no code yet. Plans live in `docs/plans/` and are the source of truth until code exists.

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
| `docs/plans/` | Numbered plan files, one per area (`NN-topic.md`) |
| `AGENTS.md` | This file |
| `CLAUDE.md` | Just `@AGENTS.md` |
| `LICENSE` | MIT |

Update this table when you add top-level paths.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, for example `docs(plans): add ingest plan` or `feat(ingest): fetch TMDB daily exports`.

## Plans

| Plan | Covers |
| --- | --- |
| [01 — Ingest](docs/plans/01-ingest.md) | TMDB exports, two-pass fetch, daily sync |
| [02 — Storage](docs/plans/02-storage.md) | Postgres schema, normalise, size budget, backups |
| [03 — Evaluation](docs/plans/03-evaluation.md) | MovieLens splits, metrics, golden set, harness |
| [04 — Content similarity](docs/plans/04-content-similarity.md) | TF-IDF feature families, embeddings, profiles |
| [05 — Vibe tags](docs/plans/05-vibe-tags.md) | LLM tags from a fixed vocabulary, through OpenRouter |
| [06 — Collaborative filtering](docs/plans/06-collaborative-filtering.md) | Implicit ALS on MovieLens, owner fold-in, coverage gaps |
| [07 — Ranking](docs/plans/07-ranking.md) | Candidates, features, LambdaRank, re-ranking |
| [08 — App](docs/plans/08-app.md) | FastAPI and HTMX app, onboarding, feedback, access |
| [09 — Deploy](docs/plans/09-deploy.md) | Docker Compose locally, Railway services, cost |

Build in this order, except that 03 must exist before any of 04–07 is tuned.

## Writing plans

- One file per area, numbered in the order it should be built: `01-ingest.md`, `02-storage.md`, and so on.
- Each plan states its goal, the decisions made and why, open questions, and how we know it works.
- When a decision changes, edit the plan rather than adding a contradicting note below it.
- Keep plans short enough to read in one sitting. Split a plan that grows past that.
