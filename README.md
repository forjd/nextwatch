# nextwatch

A personal movie and TV recommendation engine. It keeps a local copy of [TMDB](https://www.themoviedb.org/)'s catalogue and ranks titles with content similarity, collaborative filtering and a learned ranker.

It is a work in progress. The design is in [`docs/plans/`](docs/plans/), and [`AGENTS.md`](AGENTS.md) has the commands and the status of each part.

## Quick start

```sh
cp .env.example .env          # set TMDB_API_TOKEN
docker compose up -d --wait
uv sync
uv run nextwatch migrate
uv run nextwatch backfill
```

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.

## Licence

[MIT](LICENSE)
