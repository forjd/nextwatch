# 06 — Collaborative filtering

## Goal

Learn from what people watch and like together, which content features cannot see. People who love *Arrival* also love *Annihilation*, though the two share no cast, crew or keywords. In most recommenders this is the strongest single signal.

## Data

- **MovieLens 32M** (train users only, see [03](03-evaluation.md)). Movies only, data up to 2023.
- **The owner's ratings** ([08](08-app.md)). Not for training. They are folded in at request time as one more user.
- **TMDB recommendations** (`title_tmdb_rec`). TMDB does not say how they are made, but they behave partly like co-watch data, and they are our only collaborative signal for TV and post-2023 titles. They are used as features and candidates in [07](07-ranking.md), not trained on here.

## Model

**Implicit ALS** with the `implicit` library.

- Matrix: train users × movies that map to an eligible `title`.
- Confidence: `1 + α · max(rating − 2.5, 0)`. Ratings of 2.5 and below count only as weak "seen" signals. Start with α = 10 and tune on validation.
- Factors: 128. Regularisation and iterations tuned on validation.
- Output: item factors (~87k × 128 floats, ~45 MB) and the parameters, saved as a `model_artifact` named `als`.

**Owner fold-in.** At request time, compute a user vector from the owner's ratings with the same confidence weighting, using `implicit`'s `recalculate_user`. This is a small least-squares solve, a few milliseconds. No retraining when the owner rates something.

**Item-item.** `sim_als` between two titles is the cosine of their item factors.

Later, if ALS stops improving: try EASE on the top ~20k movies (closed form and often stronger than ALS on MovieLens, but its item matrix is ~1.6 GB, which is too big to serve on Railway Hobby), or a two-tower model that can take content features as input and so covers TV.

## Coverage gaps

| Titles | CF signal |
| --- | --- |
| Movies in MovieLens with enough ratings | `cf_score`, `sim_als` |
| Movies released after 2023, or rarely rated | None from ALS. Content, vibes and TMDB recommendations only. |
| All TV | None from ALS. Content, vibes and TMDB recommendations only. |

The ranker ([07](07-ranking.md)) sees missing CF features as missing (LightGBM handles `NaN`), plus an `has_cf` flag, so it learns to lean on other features for these titles instead of treating them as bad.

## When it trains

MovieLens does not change, so ALS only retrains when the code, parameters or eligible set change. `nextwatch train als` runs locally, takes minutes and ~4–8 GB of RAM, and writes a new `model_artifact` version with its validation metrics in `meta`. Nothing about this needs to run on Railway.

## How we know it works

- On [03](03-evaluation.md) T1, ALS alone beats every baseline and content-only by a clear margin. This is the expected result on MovieLens, so if it does not happen, look for a bug.
- On T2, `sim_als` neighbours for well-known seeds look right to a human.
- Fold-in for a test user gives nearly the same recommendations as training with that user included.
