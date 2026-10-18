import json

import httpx
import numpy as np
import pytest

from nextwatch.evaluation.models import Catalogue
from nextwatch.features import embed, sparse
from nextwatch.features.content import ContentFeatures, ContentModel
from nextwatch.normalise import normalise
from nextwatch.vibes import job, vocab
from nextwatch.vibes.client import OpenRouterClient
from nextwatch.vibes.prompt import PROMPT_VERSION, system_prompt, title_inputs
from tests.payloads import movie_payload, tv_payload
from tests.test_content import fake_embedder
from tests.test_normalise import store

GOOD = {
    "tone": [{"value": "dark", "strength": 3}, {"value": "tense", "strength": 2}],
    "pacing": "brisk",
    "attention": "engaged",
    "themes": ["identity", "rebellion", "technology"],
    "era": "near-future",
    "places": ["big city"],
    "form": [],
    "intensity": {"violence": "strong", "gore": "mild", "sexual_content": "none",
                  "language": "mild"},
    "knowledge": "known",
    "feel": "Sleek and propulsive, a slick action ride that keeps asking what is real.",
}  # fmt: skip


def test_vocabulary_and_schema():
    v = vocab.load()
    assert "slow-burn" in v.pacing and "grief" in v.themes
    schema = vocab.json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["tone"]["items"]["properties"]["value"]["enum"] == list(v.tone)
    # Every value in the prompt comes from the vocabulary.
    assert all(t in system_prompt() for t in v.themes)


def test_validate_trims_and_rejects():
    out = vocab.validate(
        {**GOOD, "themes": ["grief"] * 2 + list(vocab.load().themes[:8]), "feel": "  a  b "}
    )
    assert out["themes"][0] == "grief" and len(out["themes"]) == vocab.MAX_THEMES
    assert out["feel"] == "a b"
    for bad in (
        {**GOOD, "pacing": "glacial"},
        {**GOOD, "tone": [{"value": "dark", "strength": 5}]},
        {**GOOD, "intensity": {"violence": "lots"}},
        {k: v for k, v in GOOD.items() if k != "era"},
        {**GOOD, "feel": ""},
    ):
        with pytest.raises(vocab.InvalidTags):
            vocab.validate(bad)


class FakeOpenRouter:
    def __init__(self):
        self.calls = 0
        self.script: list = []  # per call: dict (tags), str (raw content), or int (status)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["provider"] == {"require_parameters": True}
        assert request.headers["Authorization"] == "Bearer test-key"
        step = self.script.pop(0) if self.script else GOOD
        if isinstance(step, int):
            return httpx.Response(step, headers={"Retry-After": "0"})
        content = step if isinstance(step, str) else json.dumps(step)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.001},
        })  # fmt: skip

    @property
    def transport(self):
        return httpx.MockTransport(self.handler)


async def test_client_retries_rate_limits_and_failed_generations():
    fake = FakeOpenRouter()
    fake.script = [429, 502]
    async with OpenRouterClient("test-key", transport=fake.transport, initial_backoff=0) as client:
        c = await client.complete("sys", "user")
    assert c.data == GOOD and c.cost_usd == 0.001 and fake.calls == 3


@pytest.fixture
def world(conn):
    store(conn, "movie", 603, movie_payload())
    store(conn, "movie", 604, movie_payload(604, title="The Matrix Reloaded", vote_count=5))
    store(conn, "tv", 1396, tv_payload())
    normalise(conn)
    return {k: v for v, k in conn.execute("SELECT id, tmdb_id FROM title")}


@pytest.mark.db
def test_title_inputs(conn, world):
    (ti,) = title_inputs(conn, [world[1396]])
    assert ti.text.splitlines()[0] == "TV series: Breaking Bad (2008)"
    assert "Created by: Vince Gilligan" in ti.text
    assert "Starring: Bryan Cranston, Aaron Paul" in ti.text
    assert "Runtime: 47 min per episode" in ti.text


@pytest.mark.db
def test_vibes_job(conn, world, settings):
    s = settings.__class__(**{**settings.__dict__, "openrouter_api_key": "test-key",
                              "vibe_concurrency": 2})  # fmt: skip
    fake = FakeOpenRouter()
    # One bad answer that is retried, one that fails twice and is skipped.
    fake.script = ["not json", GOOD, {**GOOD, "pacing": "glacial"}, {**GOOD, "pacing": "x"}]
    s1 = s.__class__(**{**s.__dict__, "vibe_concurrency": 1})
    stats = job.run(conn, s1, transport=fake.transport)
    assert (stats.tagged, stats.failed) == (2, 1)
    assert fake.calls == 5 and stats.cost_usd == pytest.approx(0.001 * 5)  # every call billed
    rows = conn.execute("SELECT title_id, prompt_version, feel FROM title_vibe").fetchall()
    assert len(rows) == 2 and all(r[1] == PROMPT_VERSION for r in rows)

    # Second run only tags what is missing.
    stats = job.run(conn, s, transport=fake.transport)
    assert stats.tagged == 1
    assert job.titles_to_tag(conn, s.vibe_model, None) == []


@pytest.mark.db
def test_vibes_budget_stops_the_job(conn, world, settings):
    s = settings.__class__(**{**settings.__dict__, "openrouter_api_key": "test-key",
                              "vibe_concurrency": 1, "vibe_budget_usd": 0.0015})  # fmt: skip
    stats = job.run(conn, s, transport=FakeOpenRouter().transport)
    assert stats.tagged == 2 and stats.stopped_for_budget
    # The month's spend now exceeds the budget, so the next run does nothing.
    assert job.run(conn, s, transport=FakeOpenRouter().transport).stopped_for_budget


@pytest.mark.db
def test_spike_report(conn, world, settings, tmp_path):
    s = settings.__class__(**{**settings.__dict__, "openrouter_api_key": "test-key"})
    stats = job.run(conn, s, spike=True, transport=FakeOpenRouter().transport)
    assert stats.tagged == 3
    path = tmp_path / "spike.md"
    job.spike_report(conn, stats, path)
    text = path.read_text()
    assert "Projected backfill for 3 titles" in text and "dark(3)" in text


@pytest.mark.db
def test_vibe_features(conn, world, settings):
    s = settings.__class__(**{**settings.__dict__, "openrouter_api_key": "test-key"})
    job.run(conn, s, transport=FakeOpenRouter().transport)
    assert embed.embed_titles(conn, fake_embedder, embed.FEEL_MODEL) == 3
    assert embed.embed_titles(conn, fake_embedder, embed.FEEL_MODEL) == 0

    idx = sparse.build(conn)
    a, b = idx.index[world[603]], idx.index[world[1396]]
    assert idx.pair(a, np.array([b]))["vibe"][0] == pytest.approx(1.0)  # identical fake tags

    embed.embed_titles(conn, fake_embedder)
    feats = ContentFeatures.load(conn)
    assert feats.feel is not None
    model = ContentModel("content-vibe", Catalogue.load(conn), feats, use_vibe=True)
    assert model.use_feel and "vibe" in model.families
    assert len(model.similar(world[603], 2)) == 2
