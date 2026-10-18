"""The `vibes` job: tag titles that have no tags for the current model and prompt version."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from psycopg import AsyncConnection, Connection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from nextwatch.config import Settings
from nextwatch.vibes import vocab
from nextwatch.vibes.client import BadOutput, Completion, OpenRouterClient, RetryableError
from nextwatch.vibes.prompt import PROMPT_VERSION, TitleInput, system_prompt, title_inputs

log = logging.getLogger(__name__)

INPUT_CHUNK = 100


@dataclass
class Stats:
    tagged: int = 0
    failed: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_for_budget: bool = False
    results: list[tuple[TitleInput, dict]] = field(default_factory=list)


def titles_to_tag(conn: Connection, model: str, limit: int | None) -> list[int]:
    """Untagged (or outdated) titles, most popular first."""
    rows = conn.execute(
        """
        SELECT t.id FROM title t
        LEFT JOIN title_vibe v ON v.title_id = t.id AND v.model = %s
        WHERE v.title_id IS NULL OR v.prompt_version < %s
        ORDER BY t.popularity DESC NULLS LAST, t.id
        LIMIT %s
        """,
        (model, PROMPT_VERSION, limit),
    ).fetchall()
    return [r[0] for r in rows]


def spike_titles(conn: Connection, per_group: int = 50) -> list[int]:
    """Well-known and obscure titles, half TV: the sample to read before a backfill."""
    ids: list[int] = []
    for media_type in ("movie", "tv"):
        for order in ("vote_count DESC", "vote_count ASC"):
            rows = conn.execute(
                f"SELECT id FROM title WHERE media_type = %s AND overview IS NOT NULL"
                f" ORDER BY {order}, id LIMIT %s",
                (media_type, per_group),
            ).fetchall()
            ids += [r[0] for r in rows if r[0] not in ids]
    return ids


def month_spend(conn: Connection, model: str) -> float:
    row = conn.execute(
        "SELECT coalesce(sum(cost_usd), 0) FROM title_vibe"
        " WHERE model = %s AND created_at >= date_trunc('month', now())",
        (model,),
    ).fetchone()
    return float(row[0])


async def _store(conn: AsyncConnection, model: str, ti: TitleInput, tags: dict, c: Completion):
    feel = tags.pop("feel")
    await conn.execute(
        """
        INSERT INTO title_vibe (title_id, model, prompt_version, tags, feel, input_tokens,
            output_tokens, cost_usd)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (title_id, model) DO UPDATE SET prompt_version = EXCLUDED.prompt_version,
            tags = EXCLUDED.tags, feel = EXCLUDED.feel, input_tokens = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens, cost_usd = EXCLUDED.cost_usd,
            created_at = now()
        """,
        (ti.title_id, model, PROMPT_VERSION, Jsonb(tags), feel, c.input_tokens,
         c.output_tokens, c.cost_usd),
    )  # fmt: skip
    tags["feel"] = feel


async def _run(
    settings: Settings,
    inputs: list[TitleInput],
    spent_before: float,
    transport: httpx.AsyncBaseTransport | None,
    keep_results: bool,
) -> Stats:
    stats = Stats()
    system = system_prompt()
    work: asyncio.Queue[TitleInput] = asyncio.Queue()
    for ti in inputs:
        work.put_nowait(ti)

    async with (
        AsyncConnectionPool(
            settings.database_url, min_size=1, max_size=4, open=False,
            kwargs={"autocommit": True},
        ) as pool,
        OpenRouterClient(settings.openrouter_api_key, settings.vibe_model, transport) as client,
    ):  # fmt: skip

        async def worker() -> None:
            while not work.empty():
                if spent_before + stats.cost_usd >= settings.vibe_budget_usd:
                    stats.stopped_for_budget = True
                    return
                ti = work.get_nowait()
                for attempt in (1, 2):
                    try:
                        c = await client.complete(system, ti.text)
                        # Billed whether or not the tags turn out valid.
                        stats.cost_usd += c.cost_usd
                        stats.input_tokens += c.input_tokens
                        stats.output_tokens += c.output_tokens
                        tags = vocab.validate(c.data)
                    except (BadOutput, vocab.InvalidTags) as e:
                        if isinstance(e, BadOutput):
                            stats.cost_usd += e.cost_usd
                        if attempt == 2:
                            log.warning("title %d: giving up: %s", ti.title_id, e)
                            stats.failed += 1
                        continue
                    except RetryableError as e:
                        log.warning("title %d: %s", ti.title_id, e)
                        stats.failed += 1
                        break
                    async with pool.connection() as conn:
                        await _store(conn, client.model, ti, tags, c)
                    stats.tagged += 1
                    if keep_results:
                        stats.results.append((ti, tags))
                    if stats.tagged % 500 == 0:
                        log.info("tagged %d, $%.2f", stats.tagged, stats.cost_usd)
                    break

        await asyncio.gather(*(worker() for _ in range(settings.vibe_concurrency)))
    return stats


def load_inputs(conn: Connection, title_ids: list[int]) -> list[TitleInput]:
    """Inputs in the same order as title_ids."""
    out = []
    for start in range(0, len(title_ids), INPUT_CHUNK):
        chunk = title_ids[start : start + INPUT_CHUNK]
        by_id = {ti.title_id: ti for ti in title_inputs(conn, chunk)}
        out += [by_id[t] for t in chunk if t in by_id]
    return out


def run(
    conn: Connection,
    settings: Settings,
    limit: int | None = None,
    spike: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Stats:
    ids = spike_titles(conn) if spike else titles_to_tag(conn, settings.vibe_model, limit)
    spent = month_spend(conn, settings.vibe_model)
    log.info(
        "%d titles to tag with %s; $%.2f of $%.2f budget spent this month",
        len(ids), settings.vibe_model, spent, settings.vibe_budget_usd,
    )  # fmt: skip
    if spent >= settings.vibe_budget_usd:
        log.warning("monthly vibe budget reached; nothing to do")
        return Stats(stopped_for_budget=True)
    inputs = load_inputs(conn, ids)
    stats = asyncio.run(_run(settings, inputs, spent, transport, keep_results=spike))
    log.info(
        "tagged %d, failed %d, $%.4f, %d in / %d out tokens%s",
        stats.tagged, stats.failed, stats.cost_usd, stats.input_tokens, stats.output_tokens,
        " (stopped: budget)" if stats.stopped_for_budget else "",
    )  # fmt: skip
    return stats


def spike_report(conn: Connection, stats: Stats, path: Path) -> None:
    """Write the spike's tags for a human to read, with a cost projection for the backfill."""
    total = conn.execute("SELECT count(*) FROM title").fetchone()[0]
    per_title = stats.cost_usd / stats.tagged if stats.tagged else 0
    lines = [
        "# Vibe spike",
        "",
        f"{stats.tagged} tagged, {stats.failed} failed, ${stats.cost_usd:.4f} in total.",
        f"Mean tokens per title: {stats.input_tokens / max(stats.tagged, 1):.0f} in, "
        f"{stats.output_tokens / max(stats.tagged, 1):.0f} out.",
        f"Projected backfill for {total} titles: ${per_title * total:.2f}.",
        "",
    ]
    for ti, tags in stats.results:
        tone = ", ".join(f"{t['value']}({t['strength']})" for t in tags["tone"])
        lines += [
            f"## {ti.text.splitlines()[0]}",
            "",
            f"> {tags['feel']}",
            "",
            f"- tone: {tone}; pacing: {tags['pacing']}; attention: {tags['attention']}",
            f"- themes: {', '.join(tags['themes'])}",
            f"- era: {tags['era']}; places: {', '.join(tags['places']) or '-'}; "
            f"form: {', '.join(tags['form']) or '-'}",
            f"- intensity: {', '.join(f'{k} {v}' for k, v in tags['intensity'].items())}",
            f"- knowledge: {tags['knowledge']}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
