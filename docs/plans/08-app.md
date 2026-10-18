# 08 — App

## Goal

A small web app, mostly used on a phone, where the owner gets recommendations, rates what they have seen and finds something to watch tonight. Every interaction goes back into their profile.

## Stack

- **FastAPI** with **Jinja2** templates and **HTMX**. Server-rendered, one language, no JavaScript build. HTMX handles the rate-without-reload interactions.
- Mobile first. Poster grid, big tap targets.
- Posters load directly from TMDB's image CDN (`https://image.tmdb.org/t/p/w342{poster_path}`). No images stored.
- The app loads the sparse matrices ([04](04-content-similarity.md)), ALS factors ([06](06-collaborative-filtering.md)), ranker ([07](07-ranking.md)) and embedding model at start, and reloads them when a newer `model_artifact` or `sync` appears.

## Getting started with no history

The engine knows nothing about the owner at first. Two ways in, both on the first run and always available later:

1. **Quick rate.** A grid of ~60 well-known titles, chosen to spread across genres, decades, movies and TV. Tap a rating or skip. After every 10 ratings the next 10 are picked to fill the gaps in what we know (active learning: titles the current profile is most unsure about). 30 ratings is enough to start.
2. **Import.** Upload a Letterboxd export (`ratings.csv`) or an IMDb ratings export. Match by IMDb ID first, then by title and year. Show the unmatched ones.

## Pages

| Page | What it does |
| --- | --- |
| **For you** | The ranked feed. Filters: movie / TV / both, "on my services", max runtime, attention level, no gore. Each card shows "Because you liked *X*". |
| **Title** | Poster, overview, year, runtime, GB providers, vibe tags and feel sentence, cast and crew. "More like this" list below. |
| **Mood** | A text box: "something cosy and funny, under 2 hours". Matched against feel embeddings and vibe filters, then re-ranked for the owner ([05](05-vibe-tags.md)). |
| **Search** | Title search over `title` (Postgres trigram index), to rate a specific thing. |
| **Watchlist** | Saved titles, sorted by predicted score and filterable by provider. |
| **Ratings** | Everything rated, editable. |
| **Settings** | Which streaming services the owner has, default type mix, diversity and popularity knobs. |

## Feedback on every card

- Rate: 1–5 stars, in half steps.
- **Seen**, without a rating.
- **Not interested**: hide it, and use it as a weak negative.
- **Watchlist**.

Every action writes to `rating` or `event` ([02](02-storage.md)) and bumps the profile version, so the next page load re-ranks. Impressions and clicks are logged too: they show what is recommended but ignored, which becomes a negative signal later.

## Access

- One owner, one password: `APP_PASSWORD` in the environment, a signed session cookie (`SESSION_SECRET`), and a 30-day login.
- Locally, through Tailscale Serve on its own port, for example `tailscale serve --bg --https=8443 http://127.0.0.1:8000`. Not port 443.
- On Railway, the public `*.up.railway.app` domain behind the same password ([09](09-deploy.md)).

## TMDB attribution

The footer and an About page show the TMDB logo and "This product uses the TMDB API but is not endorsed or certified by TMDB." Provider data shows "Powered by JustWatch", which TMDB requires for watch provider data.

## How we know it works

- From an empty database of owner data, quick rate gets to a usable "For you" feed in under 3 minutes.
- Rating a title changes the feed on the next load.
- A Letterboxd export imports with over 95% of rows matched.
- Pages load in under 1 second on a phone over 4G.
