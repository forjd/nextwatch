# 03 — Evaluation

## Goal

A repeatable offline benchmark that says whether a change makes recommendations better. It exists before any recommender, and nothing in [04](04-content-similarity.md)–[07](07-ranking.md) is kept unless it beats the current best here.

## The problem with a single user

There is one user and no history, so there is nothing of the owner's to evaluate against. We borrow 200k users from MovieLens 32M and measure how well each model predicts what they went on to like. Then we add a small hand-made set for the owner's taste and for TV, which MovieLens does not cover.

## Data

- MovieLens 32M: 32M ratings (0.5–5 stars) from ~200k users on ~87k movies, with timestamps up to late 2023. Research licence: fine for this personal project, never redistributed or committed. Downloaded to `data/movielens/` by `nextwatch movielens fetch` and converted to Parquet.
- Only movies whose `tmdbId` maps to an eligible `title` take part.
- A rating of **4.0 or more** is a positive. Everything else the user rated is "seen", so we never count it as a hit or recommend it back.

## Splits

- **Users** are split once, at random with a fixed seed: 80% train, 10% validation, 10% test. Training ([06](06-collaborative-filtering.md), [07](07-ranking.md)) only sees train users. Tuning uses validation. Test is for the final number of a change and is not used for tuning.
- **Within each evaluated user**, sort by time. The first 80% of their ratings are the history the model sees. The last 20% are the held-out ratings it must predict. This is a temporal split, so models cannot use the future.
- To keep runs to a few minutes, evaluate a fixed sample of 5,000 users from validation (or test). Users need at least 5 history ratings and 1 held-out positive.

## Tasks and metrics

**T1. "For you."** Given a user's history, rank all eligible titles they have not seen. Score the top 20 against held-out positives.

- Recall@20 and nDCG@20 (main metrics).
- Coverage: share of the catalogue that appears in anyone's top 20.
- Popularity bias: mean log `vote_count` of recommended titles versus the held-out positives.
- Intra-list diversity: mean pairwise embedding distance inside each top 20.

**T2. "More like this."** For a seed title, rank similar titles. MovieLens gives a proxy: titles that the same users rate highly within a short time of each other. For each test user, take a positive from history as the seed and their held-out positives as targets. Report Recall@20.

**T3. Golden set.** `eval/golden.yaml`, written by the owner: about 50 seeds, half TV, each with a few titles that must appear in the top 20 and a few that must not. Example: seed *Severance*, must include *Mr. Robot* and *Black Mirror*, must not include *The Office*. Report pass rate. This is the only check on TV and on taste, so it matters more than its size suggests.

The headline is nDCG@20 on T1. A change that raises it but drops T3 is not an improvement.

## Baselines

Every report shows these, so numbers have something to compare against:

- **Popular:** top by `vote_count` that the user has not seen.
- **Popular in genre:** the same, weighted by the user's genre mix.
- **TMDB recommendations:** union of `title_tmdb_rec` for the user's positives, ranked by how often each appears.

## Harness

- `nextwatch eval run --model <name> --split validation` builds each user's history, calls the model through one interface (`recommend(history, k) -> [title_id]` and `similar(title_id, k)`), computes metrics, and writes an `eval_run` row plus `eval/reports/<timestamp>-<model>.md`.
- `nextwatch eval compare <run> <run>` prints a side-by-side table with bootstrap 95% intervals, so we can see whether a difference is real or noise.
- Fixed seeds everywhere. The same model and data give the same numbers.

## Limits we accept

- MovieLens users are not the owner. Their taste skews toward popular Hollywood films, and their data stops in 2023, so recent titles cannot be evaluated by T1.
- MovieLens has no TV. TV quality rests on T3 and on content features working the same way for TV as for movies.
- Offline metrics reward predicting what people watched, not what they enjoyed discovering. Once the owner uses the app, their ratings of recommendations ([08](08-app.md)) are the real test. `nextwatch eval live` reports the share of recommendations rated 4+.

## How we know it works

- The popular baseline gets non-zero but low recall, and the TMDB baseline beats it. If not, the harness is broken.
- Running the same model twice gives identical numbers.
- A model that "cheats" by seeing the held-out set scores near 1.0. We run this once as a test of the harness.
