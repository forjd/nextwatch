import pytest
from fastapi.testclient import TestClient

from nextwatch.app import imports
from nextwatch.app.main import create_app
from nextwatch.features import embed
from nextwatch.normalise import normalise
from tests.payloads import movie_payload, tv_payload
from tests.test_content import fake_embedder
from tests.test_normalise import store

pytestmark = pytest.mark.db


@pytest.fixture
def world(conn):
    store(conn, "movie", 603, movie_payload())
    store(conn, "movie", 604, movie_payload(604, title="The Matrix Reloaded", vote_count=900,
                                            release_date="2003-05-15"))  # fmt: skip
    store(conn, "movie", 605, movie_payload(605, title="The Matrix Revolutions", vote_count=800))
    store(conn, "tv", 1396, tv_payload())
    store(conn, "movie", 862, {"id": 862, "title": "Toy Story", "overview": "Toys come alive.",
                               "release_date": "1995-10-30", "vote_count": 18000,
                               "imdb_id": "tt0114709",
                               "genres": [{"id": 16, "name": "Animation"}]})  # fmt: skip
    normalise(conn)
    embed.embed_titles(conn, fake_embedder)
    return {k: v for v, k in conn.execute("SELECT id, tmdb_id FROM title")}


@pytest.fixture
def client(world, test_db_url):
    app = create_app(test_db_url, "pw", "secret", embedder_factory=lambda: fake_embedder)
    with TestClient(app) as c:
        yield c


def login(c):
    r = c.post("/login", data={"password": "pw"}, follow_redirects=False)
    assert r.status_code == 303


def test_requires_login(client):
    assert client.get("/for-you", follow_redirects=False).headers["location"] == "/login"
    assert client.get("/for-you", headers={"HX-Request": "true"}).headers["HX-Redirect"] == "/login"
    assert "Wrong password" in client.post("/login", data={"password": "nope"}).text
    health = client.get("/healthz").json()
    assert health["ok"] and health["models"]["catalogue"] and not health["models"]["ranker"]


def test_onboarding_rate_and_for_you(client, conn, world):
    login(client)
    r = client.get("/", follow_redirects=False)
    assert r.headers["location"] == "/start"
    start = client.get("/start").text
    assert "Toy Story" in start and 'value="onboarding"' in start

    r = client.post(f"/rate/{world[603]}", data={"score": "5", "source": "onboarding"},
                    headers={"HX-Request": "true"})  # fmt: skip
    assert r.status_code == 200 and "selected>★★★★★ 5.0" in r.text
    assert conn.execute("SELECT score, source FROM rating").fetchone() == (5, "onboarding")
    assert client.post(f"/rate/{world[603]}", data={"score": "4.3"}).status_code == 400

    assert client.get("/", follow_redirects=False).headers["location"] == "/for-you"
    page = client.get("/for-you").text
    # Content fallback: the sequels rank first, and the rated title is not recommended back.
    assert page.index("The Matrix Reloaded") < page.index("Toy Story")
    assert f'id="card-{world[603]}"' not in page
    assert "Because you liked" in page
    n = conn.execute("SELECT count(*) FROM event WHERE kind = 'impression'").fetchone()[0]
    assert n > 0


def test_card_actions_and_watchlist(client, conn, world):
    login(client)
    client.post(f"/rate/{world[603]}", data={"score": "5"})
    r = client.post(f"/event/{world[862]}/watchlist")
    assert "Saved" in r.text
    assert "Toy Story" in client.get("/watchlist").text
    client.post(f"/event/{world[862]}/unwatchlist")
    assert "Toy Story" not in client.get("/watchlist").text

    assert client.post(f"/event/{world[604]}/dismiss").text == ""
    assert f'id="card-{world[604]}"' not in client.get("/for-you").text
    assert client.post(f"/event/{world[604]}/nope").status_code == 400


def test_title_search_settings_about(client, conn, world):
    login(client)
    page = client.get(f"/title/{world[603]}").text
    assert "Welcome to the Real World." in page and "Netflix (flatrate)" in page
    assert "More like this" in page and "Lilly Wachowski" in page
    assert client.get("/title/999999").status_code == 404

    found = client.get("/search", params={"q": "matrix"}, headers={"HX-Request": "true"}).text
    assert "The Matrix Reloaded" in found and "<html" not in found

    form = {"providers": ["8"], "media_type": "movie", "mmr_lambda": "0.5",
            "popularity_penalty": "0.2"}  # fmt: skip
    r = client.post("/settings", data=form)
    assert r.status_code == 200 and "Saved." in r.text
    assert "checked" in client.get("/settings").text
    client.post(f"/rate/{world[603]}", data={"score": "5"})
    page = client.get("/for-you", params={"services": "1"}).text
    assert "Toy Story" not in page  # not on Netflix in the fixtures
    assert "TMDB" in client.get("/about").text


def test_mood(client, world):
    login(client)
    page = client.get("/mood", params={"q": "toys come alive"}).text
    assert page.index("Toy Story") < page.index("The Matrix")


def test_import(client, conn, world):
    login(client)
    letterboxd = (
        "Date,Name,Year,Letterboxd URI,Rating\n"
        "2024-01-02,The Matrix,1999,https://boxd.it/x,4.5\n"
        "2024-01-03,Toy Story,1996,https://boxd.it/y,3\n"  # year off by one still matches
        "2024-01-04,Unknown Film,2001,https://boxd.it/z,2\n"
    )
    r = client.post("/import", files={"file": ("ratings.csv", letterboxd, "text/csv")})
    assert "matched 2 of 3" in r.text and "Unknown Film (2001)" in r.text
    imdb = (
        "Const,Your Rating,Date Rated,Title,Title Type,Year\n"
        "tt0903747,9,2024-02-01,Breaking Bad,tvSeries,2008\n"
    )
    r = client.post("/import", files={"file": ("ratings.csv", imdb, "text/csv")})
    assert "matched 1 of 1" in r.text
    scores = dict(conn.execute(
        "SELECT t.tmdb_id, r.score FROM rating r JOIN title t ON t.id = r.title_id"))  # fmt: skip
    assert {k: float(v) for k, v in scores.items()} == {603: 4.5, 862: 3.0, 1396: 4.5}
    r = client.post("/import", files={"file": ("x.csv", "a,b\n1,2\n", "text/csv")})
    assert "not a Letterboxd" in r.text


def test_import_never_overwrites_app_ratings(conn, world):
    conn.execute("INSERT INTO rating (title_id, score, source) VALUES (%s, 2, 'app')",
                 (world[603],))  # fmt: skip
    imports.import_csv(
        conn, "Date,Name,Year,Letterboxd URI,Rating\n2024-01-01,The Matrix,1999,u,5\n"
    )
    assert conn.execute("SELECT score FROM rating").fetchone()[0] == 2
