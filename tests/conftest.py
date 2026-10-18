import gzip
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psycopg
import pytest
from dotenv import load_dotenv

from nextwatch import db
from nextwatch.config import Settings

load_dotenv()


@pytest.fixture
def test_db_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    return url


@pytest.fixture
def conn(test_db_url):
    """A migrated, empty test database."""
    with psycopg.connect(test_db_url, autocommit=True) as c:
        c.execute("DROP SCHEMA public CASCADE")
        c.execute("CREATE SCHEMA public")
        db.migrate(c)
        yield c


@pytest.fixture
def settings(test_db_url) -> Settings:
    return Settings(
        database_url=test_db_url,
        tmdb_api_token="test-token",
        min_votes=10,
        min_votes_tv=10,
        tmdb_rps=1000,
        tmdb_workers=4,
    )


def write_export(data_dir: Path, media_type: str, date, rows: list[dict]) -> None:
    from nextwatch.ingest.exports import export_url

    path = data_dir / "exports" / export_url(media_type, date).rsplit("/", 1)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@dataclass
class FakeTmdb:
    """A fake TMDB API. Titles are keyed by (media_type, id)."""

    titles: dict[tuple[str, int], dict] = field(default_factory=dict)
    changes: dict[str, list[int]] = field(default_factory=dict)
    fail_once: dict[tuple[str, int], int] = field(default_factory=dict)
    always_fail: set[tuple[str, int]] = field(default_factory=set)
    requests: list[str] = field(default_factory=list)

    def add(self, media_type: str, tmdb_id: int, votes: int, **extra) -> None:
        name_key = "title" if media_type == "movie" else "name"
        self.titles[(media_type, tmdb_id)] = {
            "id": tmdb_id,
            name_key: f"{media_type} {tmdb_id}",
            "vote_count": votes,
            "vote_average": 7.0,
            "popularity": float(votes),
            **extra,
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        assert request.headers["Authorization"] == "Bearer test-token"
        if m := re.fullmatch(r"/3/(movie|tv)/changes", request.url.path):
            ids = self.changes.get(m[1], [])
            return httpx.Response(
                200,
                json={
                    "results": [{"id": i, "adult": None} for i in ids],
                    "page": 1,
                    "total_pages": 1,
                },
            )
        m = re.fullmatch(r"/3/(movie|tv)/(\d+)", request.url.path)
        assert m, request.url
        key = (m[1], int(m[2]))
        if self.fail_once.get(key):
            self.fail_once[key] -= 1
            return httpx.Response(429, headers={"Retry-After": "0"})
        if key in self.always_fail:
            return httpx.Response(503)
        if key not in self.titles:
            return httpx.Response(404, json={"status_code": 34})
        body = dict(self.titles[key])
        appends = request.url.params.get("append_to_response")
        if appends:
            for a in appends.split(","):
                body[a] = _append(a)
        return httpx.Response(200, json=body)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def count(self, pattern: str) -> int:
        return sum(1 for r in self.requests if re.search(pattern, r))


def _append(name: str) -> dict:
    if name == "watch/providers":
        return {"results": {"GB": {"flatrate": [{"provider_id": 8}]}, "US": {"buy": []}}}
    if name in ("release_dates", "content_ratings"):
        return {"results": [{"iso_3166_1": "GB"}, {"iso_3166_1": "US"}, {"iso_3166_1": "FR"}]}
    if name == "recommendations":
        return {"results": [{"id": 99, "media_type": "movie", "overview": "x"}]}
    return {"results": [], "cast": [], "crew": []}


@pytest.fixture
def fake_tmdb() -> FakeTmdb:
    return FakeTmdb()
