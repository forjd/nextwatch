import numpy as np
import pytest

from nextwatch.evaluation import data, harness, metrics
from nextwatch.evaluation.data import Interactions
from nextwatch.evaluation.models import Catalogue, Popular, PopularInGenre, TmdbRecs, top_k


def test_recall_and_ndcg():
    assert metrics.recall_at_k([1, 2, 3], {2, 9}, 3) == 0.5
    assert metrics.recall_at_k([1, 2], set(), 2) == 0.0
    assert metrics.ndcg_at_k([2, 1], {2}, 2) == 1.0
    assert metrics.ndcg_at_k([1, 2], {2}, 2) == pytest.approx(1 / np.log2(3))
    # Recall is capped by k, so a perfect top 2 of 5 relevant scores 1.
    assert metrics.recall_at_k([1, 2], {1, 2, 3, 4, 5}, 2) == 1.0


def test_bootstrap_ci_contains_mean():
    lo, hi = metrics.bootstrap_ci([0.0, 1.0] * 50)
    assert lo < 0.5 < hi
    assert metrics.paired_bootstrap_ci([1.0] * 10, [1.0] * 10) == (0.0, 0.0)


def test_top_k_skips_excluded_and_breaks_ties_stably():
    ids = np.array([10, 11, 12, 13])
    scores = np.array([1.0, 3.0, 3.0, 2.0])
    assert top_k(scores, ids, {11}, 2) == [12, 13]
    assert top_k(scores, ids, set(), 10) == [11, 12, 13, 10]


def synthetic(n_users=200, n_titles=50, seed=1) -> Interactions:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        titles = rng.choice(n_titles, size=rng.integers(5, 20), replace=False)
        for i, t in enumerate(titles):
            rows.append((u, int(t) + 1, float(rng.choice([2.0, 3.5, 4.0, 5.0])), 1000 * u + i))
    rows.sort(key=lambda r: (r[0], r[3]))
    u, t, r, ts = zip(*rows, strict=True)
    return Interactions(
        np.array(u, dtype=np.int32), np.array(t), np.array(r, dtype=np.float32), np.array(ts)
    )


def test_split_users_is_fixed_and_disjoint():
    users = synthetic().user
    a, b = data.split_users(users), data.split_users(users)
    assert all(np.array_equal(a[s], b[s]) for s in data.SPLITS)
    assert sum(len(a[s]) for s in data.SPLITS) == len(np.unique(users))
    assert not set(a["train"]) & set(a["test"])


def test_eval_users_are_temporal():
    inter = synthetic()
    users = data.eval_users(inter, "validation", 10)
    assert 0 < len(users) <= 10
    for u in users:
        mask = inter.user == u.user_id
        ts, titles, ratings = inter.ts[mask], inter.title[mask], inter.rating[mask]
        cut = int(mask.sum() * data.HISTORY_FRACTION)
        assert np.array_equal(u.history, titles[:cut])
        assert u.history_ts.max() < ts[cut:].min()
        assert u.holdout_positives == {
            int(t) for t, r in zip(titles[cut:], ratings[cut:], strict=True) if r >= 4
        }
    assert [u.user_id for u in users] == [
        u.user_id for u in data.eval_users(inter, "validation", 10)
    ]


def catalogue(n=50) -> Catalogue:
    ids = np.arange(1, n + 1)
    return Catalogue(
        ids=ids,
        media_type={int(i): "movie" for i in ids},
        tmdb_key={int(i): ("movie", int(i) + 1000) for i in ids},
        votes={int(i): int(i) * 10 for i in ids},
        genres={int(i): frozenset({int(i) % 3}) for i in ids},
        tmdb_recs={1: [7, 8], 2: [8]},
    )


def test_baselines():
    cat = catalogue()
    assert Popular(cat).recommend([], {50}, 3) == [49, 48, 47]
    genre = PopularInGenre(cat).recommend([(3, 5.0), (6, 5.0)], {3, 6}, 3)
    assert all(t % 3 == 0 for t in genre)
    assert TmdbRecs(cat).recommend([(1, 5.0), (2, 4.0)], {1, 2}, 3) == [8, 7, 50]
    assert TmdbRecs(cat).similar(1, 2) == [7, 8]


class Cheat:
    """Knows the held-out answers. The harness should score it near 1."""

    name = "cheat"

    def __init__(self, users):
        self.answers = {tuple(u.history.tolist()): sorted(u.holdout_positives) for u in users}

    def recommend(self, history, exclude, k):
        return self.answers[tuple(t for t, _ in history)][:k]

    def similar(self, title_id, k):
        return []


def test_harness_sanity():
    inter = synthetic(n_users=400)
    cat = catalogue()
    users = data.eval_users(inter, "validation", 50)
    cheat = harness.run_t1(Cheat(users), users)
    assert np.mean(cheat["ndcg"]) == pytest.approx(1.0)
    pop = harness.evaluate(Popular(cat), users, cat)
    assert 0 < pop["t1"]["ndcg@20"]["mean"] < 1
    assert pop == harness.evaluate(Popular(cat), users, cat)  # deterministic


def test_golden(tmp_path):
    cat = catalogue()
    path = tmp_path / "golden.yaml"
    path.write_text(
        "- seed: movie/1001\n  must: [movie/1007]\n  must_not: [movie/1003]\n"
        "- seed: movie/1001\n  must: [movie/1002]\n"
        "- seed: tv/1\n  must: []\n"
    )
    result = harness.run_t3(TmdbRecs(cat), cat, harness.load_golden(path))
    assert [c["status"] for c in result["cases"]] == ["pass", "fail", "missing"]
    assert result["pass_rate"] == 0.5
    assert harness.load_golden(tmp_path / "none.yaml") == []


def test_render_report_and_compare_shape():
    cat = catalogue()
    users = data.eval_users(synthetic(n_users=400), "validation", 30)
    result = harness.evaluate(Popular(cat), users, cat)
    config = {"model": "popular", "split": "validation", "users": 30, "seed": 1, "k": 20}
    report = harness.render_report(1, config, result)
    assert "T1 nDCG@20" in report and "T2 Recall@20" in report


@pytest.mark.db
def test_catalogue_load_and_compare(conn, tmp_path):
    from nextwatch.normalise import normalise
    from tests.payloads import movie_payload, tv_payload
    from tests.test_normalise import store

    store(conn, "movie", 603, movie_payload())
    store(conn, "movie", 604, movie_payload(604, title="Reloaded", recommendations=[]))
    store(conn, "tv", 1396, tv_payload())
    normalise(conn)
    movies = Catalogue.load(conn, "movie")
    full = Catalogue.load(conn)
    assert len(movies) == 2 and len(full) == 3
    m603 = movies.by_key[("movie", 603)]
    assert movies.tmdb_recs[m603] == [movies.by_key[("movie", 604)]]  # TV rec dropped
    assert len(full.tmdb_recs[m603]) == 1  # 1399 is not in the catalogue

    users = data.eval_users(synthetic(n_users=400), "validation", 30)
    cfg = {"model": "popular", "split": "validation", "users": 30, "seed": 1, "k": 20}
    result = harness.evaluate(Popular(catalogue()), users, catalogue())
    a = harness.save_run(conn, cfg, result, tmp_path)
    b = harness.save_run(conn, {**cfg, "model": "again"}, result, tmp_path)
    assert len(list(tmp_path.glob("*.md"))) == 2
    table = harness.compare(conn, a, b)
    assert "+0.0000 (+0.0000, +0.0000)" in table
