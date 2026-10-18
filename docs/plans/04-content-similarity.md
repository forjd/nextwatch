# 04 — Content similarity

## Goal

Say how alike two titles are from what TMDB tells us about them: who made them, what they are about and how they are described. This works for every title, including TV and new releases that no one has rated yet.

It replaces the hand-tuned weighting from the previous attempt. We compute one similarity per kind of feature and let the ranker ([07](07-ranking.md)) learn how much each one matters.

## Two kinds of signal

### Sparse: shared people, keywords and facts

Each title becomes a bag of tokens, grouped into families:

| Family | Tokens |
| --- | --- |
| `cast` | `cast:<person_id>`, weighted by billing: order 0–2 = 1.0, 3–9 = 0.6, 10–19 = 0.3 |
| `director` | `director:<id>`, and for TV `creator:<id>` |
| `writer` | Screenplay, Writer, Story, Novel, Characters |
| `crew` | Composer, Director of Photography, Editor, Producer |
| `keyword` | `kw:<id>` |
| `genre` | `genre:<id>` |
| `meta` | `lang:<code>`, `country:<code>`, `decade:<1990s>`, `company:<id>`, `collection:<id>` |

Each family is a separate sparse matrix (titles × tokens) with TF-IDF weighting and L2 rows. IDF does the job the manual weights used to: a person in 3 films is a strong link, a genre shared by 40% of titles is a weak one.

For a pair of titles we get one cosine score per family: `sim_cast`, `sim_director`, `sim_writer`, `sim_crew`, `sim_keyword`, `sim_genre`, `sim_meta`.

### Dense: what it is about

One text per title, embedded:

```
{name} ({year}). {genres}. {tagline} {overview} Keywords: {keywords}.
```

- Model: a small local embedding model through `fastembed` (ONNX, no PyTorch), starting with `BAAI/bge-small-en-v1.5` (384 dims). It runs on CPU on the VM and in a Railway job. ~150k short texts should take well under an hour.
- Stored in `title_embedding` as `halfvec(384)` with an HNSW index ([02](02-storage.md)).
- `sim_embed` is cosine similarity.
- Only re-embed when `source_hash` changes.
- Measured on the VM: ~490 short texts a second on CPU, so the first full run is 5–15 minutes. The model files (~65 MB) are cached in `data/models/`.

If a bigger model clearly wins on [03](03-evaluation.md), we switch. The model name is a column, so two can live side by side while we compare.

## Using it

**Title to title ("more like this").** Candidates are the top 200 by `sim_embed` from pgvector, plus the top 200 by a combined sparse score (sum of family scores with equal weights, only for candidate generation). The ranker then scores the union with every `sim_*` as a feature.

**User profile ("for you").** A profile is a weighted mix of the titles the user rated: weight `score − 3` (so 5 stars = +2, 1 star = −2), with a time decay of half-life one year on `rated_at`. The sparse profile is the weighted sum of title rows per family. The dense profile is the weighted mean of embeddings. A profile scores every title with one sparse matrix-vector product per family and one vector query.

## Where it runs

- The sparse matrices are built from Postgres by `nextwatch features build` and kept in memory by the app (~150k titles × tens of tokens each, well under 200 MB). The app rebuilds them on start and after each daily `sync`.
- Embedding happens in a job (`nextwatch embed`), which runs after `normalise`.
- The app also needs the embedding model at request time, for mood search ([08](08-app.md)). That adds ~150 MB of RAM.

## Experiments

Run on [03](03-evaluation.md) T1, T2 and T3, each against the baselines:

1. Sparse only, equal family weights.
2. Dense only.
3. Both, equal weights.
4. Cast billing weights on versus flat.
5. Two embedding models.

The winning combination becomes the content candidate generator and the source of `sim_*` features for [07](07-ranking.md).

## How we know it works

- On T2, content similarity beats both popular baselines.
- On T3, the TV seeds pass at about the same rate as the movie seeds.
- Sanity list in the report: the top 10 for 10 well-known seeds, for a human to read. *The Godfather* should return *The Godfather Part II* and other crime sagas, not every film with Al Pacino in it.
