"""OpenRouter chat completions with structured output. See docs/plans/05-vibe-tags.md."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from nextwatch.http import SSL_CONTEXT
from nextwatch.vibes.vocab import json_schema

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta/muse-spark-1.3-contributor"
# Fallback prices (USD per token) if a response has no usage.cost. October 2026.
PRICE_IN = 0.10 / 1e6
PRICE_OUT = 0.20 / 1e6


class RetryableError(Exception):
    """429, 502, 5xx or a network error. Worth trying again later."""


class BadOutput(Exception):
    """The response arrived but did not parse as the requested JSON. It was still billed."""

    def __init__(self, message: str, cost_usd: float = 0.0):
        super().__init__(message)
        self.cost_usd = cost_usd


@dataclass(frozen=True)
class Completion:
    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost_usd: float


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.http = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Title": "nextwatch",
                "HTTP-Referer": "https://github.com/forjd/nextwatch",
            },
            timeout=httpx.Timeout(120.0),
            transport=transport,
            verify=SSL_CONTEXT,
        )
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.paused_until = 0.0
        self.schema = json_schema()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.http.aclose()

    def body(self, system: str, user: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "vibe", "strict": True, "schema": self.schema},
            },
            "reasoning": {"effort": "low"},
            # Only route to endpoints that honour response_format.
            "provider": {"require_parameters": True},
            "usage": {"include": True},
        }

    async def _wait_if_paused(self) -> None:
        delay = self.paused_until - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    async def complete(self, system: str, user: str) -> Completion:
        """One structured completion, retrying rate limits and failed generations."""
        backoff = self.initial_backoff
        for attempt in range(self.max_retries + 1):
            await self._wait_if_paused()
            try:
                resp = await self.http.post(API_URL, json=self.body(system, user))
            except httpx.TransportError as e:
                err: Exception = RetryableError(repr(e))
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json())
                if resp.status_code in (408, 429, 502, 503) or resp.status_code >= 500:
                    err = RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    if resp.status_code == 429:
                        wait = float(resp.headers.get("Retry-After") or backoff)
                        log.warning("OpenRouter 429, pausing all requests for %.0fs", wait)
                        self.paused_until = max(self.paused_until, time.monotonic() + wait)
                else:
                    resp.raise_for_status()
            if attempt < self.max_retries:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
        raise err

    def _parse(self, body: dict[str, Any]) -> Completion:
        if "error" in body:
            raise RetryableError(f"provider error: {body['error']}")
        usage = body.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        cost = usage.get("cost")
        cost = float(cost) if cost is not None else tin * PRICE_IN + tout * PRICE_OUT
        try:
            content = body["choices"][0]["message"]["content"]
            data = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            raise BadOutput(f"unparseable response: {e!r}", cost) from e
        return Completion(data, tin, tout, cost)
