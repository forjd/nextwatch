"""Settings from environment variables. A .env file in the working directory is loaded first."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    tmdb_api_token: str
    min_votes: int
    min_votes_tv: int
    tmdb_rps: float
    tmdb_workers: int
    openrouter_api_key: str = ""
    vibe_model: str = "meta/muse-spark-1.3-contributor"
    vibe_concurrency: int = 8
    vibe_budget_usd: float = 30.0

    def min_votes_for(self, media_type: str) -> int:
        return self.min_votes_tv if media_type == "tv" else self.min_votes


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.environ.get("DATABASE_URL", ""),
        tmdb_api_token=os.environ.get("TMDB_API_TOKEN", ""),
        min_votes=int(os.environ.get("MIN_VOTES", "10")),
        min_votes_tv=int(os.environ.get("MIN_VOTES_TV", os.environ.get("MIN_VOTES", "10"))),
        tmdb_rps=float(os.environ.get("TMDB_RPS", "20")),
        tmdb_workers=int(os.environ.get("TMDB_WORKERS", "16")),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        vibe_model=os.environ.get("VIBE_MODEL", "meta/muse-spark-1.3-contributor"),
        vibe_concurrency=int(os.environ.get("VIBE_CONCURRENCY", "8")),
        vibe_budget_usd=float(os.environ.get("VIBE_BUDGET_USD", "30")),
    )
