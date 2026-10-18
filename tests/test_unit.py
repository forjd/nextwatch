import datetime as dt
import gzip
import json

from nextwatch.ingest.exports import export_url, read_export
from nextwatch.ingest.jobs import changes_window
from nextwatch.ingest.queue import backoff
from nextwatch.ingest.trim import trim_payload
from nextwatch.ingest.worker import extract_stub, is_excluded


def test_export_url():
    assert export_url("movie", dt.date(2026, 10, 5)) == (
        "https://files.tmdb.org/p/exports/movie_ids_10_05_2026.json.gz"
    )
    assert "tv_series_ids_10_05_2026" in export_url("tv", dt.date(2026, 10, 5))


def test_read_export_skips_adult_and_video(tmp_path):
    path = tmp_path / "e.json.gz"
    with gzip.open(path, "wt") as f:
        for row in [
            {"id": 1, "adult": False, "video": False, "popularity": 2.5},
            {"id": 2, "adult": True, "video": False, "popularity": 1},
            {"id": 3, "adult": False, "video": True, "popularity": 1},
            {"id": 4, "popularity": None},
        ]:
            f.write(json.dumps(row) + "\n")
        f.write("\n")
    assert [(r.tmdb_id, r.popularity) for r in read_export(path)] == [(1, 2.5), (4, 0.0)]


def test_trim_payload():
    payload = {
        "id": 1,
        "poster_path": "/p.jpg",
        "backdrop_path": "/b.jpg",
        "production_companies": [{"id": 5, "logo_path": "/l.png", "name": "X"}],
        "credits": {"cast": [{"id": 9, "profile_path": "/f.jpg", "name": "A"}]},
        "watch/providers": {"results": {"GB": {"flatrate": [1]}, "US": {"buy": [2]}}},
        "release_dates": {"results": [{"iso_3166_1": c} for c in ("GB", "US", "FR")]},
        "recommendations": {"page": 1, "results": [{"id": 7, "media_type": "movie", "x": 1}]},
    }
    out = trim_payload(payload)
    assert out["poster_path"] == "/p.jpg"
    assert "backdrop_path" not in out
    assert out["production_companies"] == [{"id": 5, "name": "X"}]
    assert out["credits"]["cast"] == [{"id": 9, "name": "A"}]
    assert out["watch/providers"] == {"results": {"GB": {"flatrate": [1]}}}
    assert [r["iso_3166_1"] for r in out["release_dates"]["results"]] == ["GB", "US"]
    assert out["recommendations"] == [{"id": 7, "media_type": "movie"}]
    assert "backdrop_path" in payload  # input untouched


def test_trim_payload_without_gb_providers():
    out = trim_payload({"watch/providers": {"results": {"US": {}}}})
    assert out["watch/providers"] == {"results": {}}


def test_extract_stub():
    movie = extract_stub("movie", {"title": "A", "vote_count": 12, "release_date": "1999-03-31"})
    assert (movie.name, movie.vote_count, movie.release_date) == ("A", 12, dt.date(1999, 3, 31))
    tv = extract_stub("tv", {"name": "B", "first_air_date": "", "vote_count": None})
    assert (tv.name, tv.vote_count, tv.release_date) == ("B", 0, None)


def test_is_excluded():
    assert is_excluded("movie", {"adult": True})
    assert is_excluded("movie", {"video": True})
    assert not is_excluded("tv", {"video": True})
    assert not is_excluded("movie", {"adult": False, "video": False})


def test_changes_window():
    today = dt.date(2026, 10, 18)
    assert changes_window(None, today) == (dt.date(2026, 10, 17), False)
    assert changes_window(dt.date(2026, 10, 10), today) == (dt.date(2026, 10, 10), False)
    assert changes_window(dt.date(2026, 9, 1), today) == (dt.date(2026, 10, 4), True)


def test_backoff():
    assert backoff(1) == dt.timedelta(minutes=1)
    assert backoff(3) == dt.timedelta(minutes=4)
    assert backoff(20) == dt.timedelta(hours=6)
