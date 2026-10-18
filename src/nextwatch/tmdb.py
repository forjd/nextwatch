"""Async TMDB API client with a shared rate limit.

TMDB publishes no hard limit, only "somewhere in the 40 requests per second range". We run
below that with a token bucket, and on a 429 every request through this client waits, not
just the one that got it.
"""

import asyncio
import datetime as dt
import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

API_BASE = "https://api.themoviedb.org/3"

MOVIE_APPENDS = (
    "credits",
    "keywords",
    "external_ids",
    "release_dates",
    "watch/providers",
    "recommendations",
)
TV_APPENDS = (
    "aggregate_credits",
    "keywords",
    "external_ids",
    "content_ratings",
    "watch/providers",
    "recommendations",
)


class NotFound(Exception):
    """The title no longer exists on TMDB."""


class TransientError(Exception):
    """A 5xx, timeout or network error. Safe to retry later."""


class RateLimiter:
    """Token bucket shared by every request, with a global pause for 429s."""

    def __init__(self, rate: float, burst: float | None = None):
        self.rate = rate
        self.capacity = burst if burst is not None else max(1.0, rate)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.paused_until = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                if now < self.paused_until:
                    await asyncio.sleep(self.paused_until - now)
                    continue
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)

    def pause(self, seconds: float) -> None:
        self.paused_until = max(self.paused_until, time.monotonic() + seconds)


class TmdbClient:
    def __init__(
        self,
        token: str,
        rate: float = 20,
        transport: httpx.AsyncBaseTransport | None = None,
        max_429_retries: int = 8,
    ):
        if not token:
            raise RuntimeError("TMDB_API_TOKEN is not set")
        self.http = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=httpx.Timeout(30.0),
            transport=transport,
        )
        self.limiter = RateLimiter(rate)
        self.max_429_retries = max_429_retries

    async def __aenter__(self) -> "TmdbClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.http.aclose()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        backoff = 1.0
        for _ in range(self.max_429_retries + 1):
            await self.limiter.acquire()
            try:
                resp = await self.http.get(path, params=params)
            except httpx.TransportError as e:
                raise TransientError(f"{path}: {e!r}") from e
            if resp.status_code == 429:
                wait = _retry_after(resp) or backoff
                log.warning("TMDB 429 on %s, pausing all requests for %.1fs", path, wait)
                self.limiter.pause(wait)
                backoff = min(backoff * 2, 60)
                continue
            if resp.status_code == 404:
                raise NotFound(path)
            if resp.status_code >= 500:
                raise TransientError(f"{path}: HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        raise TransientError(f"{path}: still rate limited after {self.max_429_retries} retries")

    async def details(self, media_type: str, tmdb_id: int, full: bool) -> dict[str, Any]:
        params: dict[str, Any] = {"language": "en-US"}
        if full:
            appends = MOVIE_APPENDS if media_type == "movie" else TV_APPENDS
            params["append_to_response"] = ",".join(appends)
        return await self.get(f"/{media_type}/{tmdb_id}", params)

    async def changes(self, media_type: str, start: dt.date, end: dt.date) -> list[dict[str, Any]]:
        """Every changed ID between start and end (max 14 days apart), across all pages."""
        results: list[dict[str, Any]] = []
        page = 1
        while True:
            data = await self.get(
                f"/{media_type}/changes",
                {"start_date": start.isoformat(), "end_date": end.isoformat(), "page": page},
            )
            results.extend(data.get("results", []))
            if page >= data.get("total_pages", 1):
                return results
            page += 1


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
