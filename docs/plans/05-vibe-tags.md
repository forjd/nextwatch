# 05 — Vibe tags

## Goal

Describe how a title *feels*: tone, pacing, themes, how much attention it needs. TMDB has none of this, and it is often why someone likes a title. Two films with no shared people or keywords can still be "bleak, slow-burn, about grief", and that is the link a good recommender finds.

An LLM reads what we know about each title, plus what it already knows about well-known titles, and returns tags from a fixed vocabulary.

## Vocabulary

A fixed list, versioned in `vibes/vocabulary.yaml`, so tags are comparable across titles and usable as features. Free-text tags would not be. First draft:

| Dimension | Values | Shape |
| --- | --- | --- |
| `tone` | dark, bleak, melancholic, tense, eerie, warm, uplifting, light, whimsical, absurd, satirical, cynical, romantic, earnest, campy | Up to 4, each with strength 1–3 |
| `pacing` | slow-burn, measured, brisk, relentless | One |
| `attention` | background-friendly, easy-watch, engaged, demanding | One |
| `themes` | ~60 values: grief, revenge, coming-of-age, identity, obsession, isolation, found family, class, power, corruption, addiction, faith, survival, justice, ambition, betrayal, memory, parenthood, ... | Up to 6 |
| `setting` | era (period, contemporary, near-future, far-future, timeless) and place type (small town, big city, workplace, school, wilderness, space, ...) | One era, up to 2 places |
| `form` | twist ending, nonlinear, unreliable narrator, ensemble, anthology, mockumentary, found footage, single location, procedural, serialised | Up to 4 |
| `intensity` | violence, gore, sexual content, language, each none/mild/strong | Fixed |
| `knowledge` | known, partial, overview-only | One. How much the model knows about this title beyond our input. |

Plus a free-text `feel`: one sentence of up to 30 words on what it is like to watch, for example "A cold, patient procedural about obsession that leaves you uneasy rather than satisfied."

Changing the vocabulary bumps `prompt_version`, and the affected titles are re-tagged.

## Input and output

Input per title: name, year, media type, genres, keywords, tagline, overview, director or creator, top 5 cast, runtime or seasons, original language.

The system prompt holds the vocabulary and instructions and is the same for every request, so the provider caches it. The response uses structured outputs (`response_format` of type `json_schema` with `strict: true`, the schema made from the vocabulary), so every result parses and only uses allowed values.

The instructions ask the model to use its own knowledge for well-known titles and to stay close to the overview for unknown ones, and to say which it did in `knowledge`.

## Model

[OpenRouter](https://openrouter.ai), model `meta/muse-spark-1.3-contributor`.

- OpenRouter's API is OpenAI-compatible: `POST https://openrouter.ai/api/v1/chat/completions` with `Authorization: Bearer $OPENROUTER_API_KEY`. We call it with `httpx`, which ingest already uses.
- Each request sets `provider: {"require_parameters": true}`, so OpenRouter only routes to endpoints that support structured outputs.
- It is a reasoning model. Set `reasoning: {"effort": "low"}`: tagging does not need deep thought, and reasoning tokens are billed as output.
- The model ID is config (`VIBE_MODEL`) and is stored in `title_vibe.model`, so we can switch models or compare two later without a schema change.
- "Contributor" is a cheaper tier, and we could not find what it means for data use. It may mean the provider keeps prompts for training. Our prompts hold only public TMDB data, so that is acceptable here.

## Cost

Prices from OpenRouter in October 2026: $0.10 per million input tokens, $0.20 per million output tokens, $0.002 per million cached input tokens. At ~500 uncached input tokens, ~1,500 cached system tokens and ~600 output tokens (reasoning included) per title:

| Run | Estimate |
| --- | --- |
| Backfill, ~150k titles | ~$25 |
| Daily, ~100–200 new titles | ~$0.03 |

These are guesses. The spike measures real token counts.

## Running it

OpenRouter has no batch API, so we make normal requests with a small async worker pool.

- `nextwatch vibes` picks eligible titles without a `title_vibe` for the current `VIBE_MODEL` and `prompt_version`, most popular first, and tags them with `VIBE_CONCURRENCY` (start at 8) requests in flight.
- On `429`, back off and slow the pool, the same as TMDB ingest ([01](01-ingest.md)). A `502` is a failed generation, which OpenRouter does not bill: retry it. A response that fails schema validation is retried once, then logged and skipped.
- Each result is written as it arrives, so the job can stop and restart at any time.
- A spend cap, `VIBE_BUDGET_USD`, stops the job once the month's spend reaches it. Spend comes from the `usage` in each response.
- The backfill runs locally. At ~2–5 requests a second it takes about 10–20 hours.
- Daily, after `normalise`, `vibes` tags the new eligible titles. That takes a minute or two.

**Spike first:** tag 200 titles (100 well-known, 100 obscure, half TV), read them, and check the real token counts and cost before the backfill.

## Features

- **Tag vector.** Each tag becomes a weighted dimension (strength for tone, 1 for the rest; themes are IDF-weighted like [04](04-content-similarity.md)). `sim_vibe` is cosine similarity between tag vectors.
- **Feel embedding.** The `feel` sentence is embedded with the same model as [04](04-content-similarity.md) and stored in `title_embedding` under its own `model` name. `sim_feel` is cosine similarity.
- **Filters.** `attention`, `pacing` and `intensity` become filters in the app: "nothing demanding tonight", "no gore".
- **Mood search.** A free-text query ("something cosy and funny, under 2 hours") is embedded and matched against feel embeddings, then re-ranked by the user's profile ([08](08-app.md)).
- `knowledge` is a ranker feature, so the model can learn to trust tags on obscure titles less.

## How we know it works

- Read the tags for 50 titles you know well. Most should feel right, and none should be nonsense.
- Tag the same 100 titles twice. Tone and themes should mostly agree. If they do not, the vocabulary is too vague.
- Adding `sim_vibe` and `sim_feel` to the ranker improves T1 and T3 on [03](03-evaluation.md). If it does not, tags are not worth what they cost, and we stop paying for new ones.
