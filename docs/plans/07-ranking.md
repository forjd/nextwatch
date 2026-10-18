# 07 — Ranking

## Goal

Turn all the signals from [04](04-content-similarity.md)–[06](06-collaborative-filtering.md) into one ordered list that is accurate, varied and watchable tonight. The weights are learned from data, not set by hand.

## Pipeline

```
candidates (~500)  →  features  →  LightGBM ranker  →  re-rank  →  top N
```

The same pipeline serves "for you" (input: a user profile) and "more like this" (input: one seed title, treated as a profile with a single 5-star rating plus the seed's own features).

## 1. Candidates

Union of the top results from each source, deduplicated:

| Source | Count |
| --- | --- |
| ALS score for the user (movies only) | 200 |
| Dense content profile (`sim_embed`) | 150 |
| Sparse content profile | 150 |
| Feel embedding profile (`sim_feel`) | 100 |
| TMDB recommendations of the user's top-rated titles | 100 |
| Popular in the user's top genres, as a safety net | 50 |

Remove anything rated, seen or dismissed. Each candidate keeps which sources found it.

## 2. Features

Per user–candidate pair. All are cheap to compute from data already in memory or in Postgres.

| Group | Features |
| --- | --- |
| Collaborative | `cf_score`, `has_cf`, max and mean `sim_als` to the user's positives |
| Content | Per family: max and mean `sim_*` to the user's positives, and profile score |
| Dense | Profile score and max `sim_embed` to a positive |
| Vibe | Profile `sim_vibe`, `sim_feel`, `knowledge` |
| TMDB | How many of the user's positives list it as a TMDB recommendation |
| Title | log `vote_count`, `vote_average`, `popularity`, `year`, `runtime`, `media_type`, `original_language == en` |
| User fit | User's mean rating for the candidate's genres, years since the user's mean year |
| Retrieval | Which sources found it, best rank in each |

## 3. Ranker

- LightGBM with the `lambdarank` objective, optimising nDCG@20.
- Training data: MovieLens train users, built exactly as [03](03-evaluation.md) does. Each user's history goes through steps 1–2. Held-out ratings are labels: 2 for ≥ 4.5 stars, 1 for 4.0, 0 otherwise or not rated. One query group per user.
- The serving ALS model was trained on these users' full ratings, so its `cf_score` would leak their held-out labels. `train ranker` therefore trains a second ALS on each train user's history part only and uses it for the training features. Validation (early stopping) uses validation users and the serving ALS, as evaluation does.
- Groups with no positive among their candidates are dropped: they carry no ranking signal.
- LightGBM's native `lgb.train` API, not `LGBMRanker`, which would need scikit-learn.
- MovieLens has no TV, so the model never sees a TV label. TV candidates rely on features that mean the same thing for both types (content, vibe, TMDB). Check TV with T3. If it suffers, train a second model with CF features masked on a random half of the users, so the model learns to rank without them.
- Tune on validation and report on test. Saved as a `model_artifact` named `ranker`, with feature importances in `meta`.
- Retrain whenever a feature changes (`nextwatch train ranker`). The artifact stores the feature list, and loading a ranker trained on a different list fails loudly. Locally, minutes.
- Evaluate as `ranker` (pure ranking) and `ranker-rerank` (with step 4).

## 4. Re-rank

The ranker only orders by predicted liking. Re-ranking makes a list a person wants to use:

- **Diversity:** MMR on `sim_embed`, λ = 0.7 to start. Stops the top 10 being five *Marvel* films.
- **Popularity:** a small, tunable penalty on log `vote_count`, so the list is not only famous titles.
- **Availability:** a filter, not a penalty. "On my services" keeps only `flatrate`, `free` or `ads` providers from the owner's chosen list ([08](08-app.md)).
- **Type mix:** honour the movie / TV / both setting. On "both", stop either type exceeding 70% of the top 20.
- **Explanation:** for each result, the user positive with the largest contribution, for "Because you liked *X*". Taken from the max `sim_*` or `sim_als` feature, not from the model.

Re-rank settings are measured on [03](03-evaluation.md) too: diversity and coverage should go up a lot while nDCG@20 drops only a little.

## Latency

Target under 300 ms for a request in the app: candidates ~100 ms (vector queries and sparse products), features ~100 ms, ranker and re-rank ~20 ms. Results are cached per profile version, so they only change after the owner rates something.

## How we know it works

- On [03](03-evaluation.md) T1, the ranker beats the best single source by a clear margin.
- Removing one feature group at a time (ablation) shows each group earns its place. Drop the ones that do not.
- T3 TV seeds pass as often as movie seeds.
- The owner's own live hit rate ([03](03-evaluation.md), `eval live`) goes up over the first month.
