import numpy as np
import pytest

from nextwatch.cf import als
from nextwatch.evaluation import data, harness
from nextwatch.evaluation.models import Catalogue, Popular
from nextwatch.features import sparse
from nextwatch.features.content import ContentFeatures
from nextwatch.ranking import ranker
from nextwatch.ranking.features import FEATURES, candidates, features, user_context
from nextwatch.ranking.signals import Meta, Signals
from tests.test_als import clustered

N = 60
TV = set(range(51, 61))


def group(t: int) -> int:
    return 0 if t <= 30 else 1


@pytest.fixture(scope="module")
def world():
    ids = np.arange(1, N + 1)
    rng = np.random.default_rng(0)
    cat = Catalogue(
        ids=ids,
        media_type={int(t): "tv" if t in TV else "movie" for t in ids},
        tmdb_key={int(t): ("tv" if t in TV else "movie", int(t)) for t in ids},
        votes={int(t): int(rng.integers(10, 1000)) for t in ids},
        genres={int(t): frozenset({group(t)}) for t in ids},
        tmdb_recs={int(t): [int(x) for x in ids if group(x) == group(t) and x != t][:5]
                   for t in ids},
    )  # fmt: skip
    matrices, vocab = {}, {}
    for f in sparse.FAMILIES:
        rows = [{f"g{group(t)}": 1.0, f"t{t}": 0.5} if f == "keyword" else {} for t in ids]
        matrices[f], vocab[f] = sparse.tfidf(rows, N)
    centres = rng.standard_normal((2, 16))
    dense = np.array([centres[group(t)] + 0.3 * rng.standard_normal(16) for t in ids])
    dense = (dense / np.linalg.norm(dense, axis=1, keepdims=True)).astype(np.float32)
    content = ContentFeatures(sparse.SparseIndex(ids, matrices, vocab), dense, None)
    meta = Meta(
        log_votes=np.log1p([cat.votes[int(t)] for t in ids]).astype(np.float32),
        vote_average=np.full(N, 7.0, np.float32),
        log_popularity=np.ones(N, np.float32),
        year=np.full(N, 2000.0, np.float32),
        runtime=np.full(N, 100.0, np.float32),
        is_tv=np.array([float(t in TV) for t in ids], np.float32),
        is_english=np.ones(N, np.float32),
        knowledge=np.full(N, np.nan, np.float32),
    )
    inter = clustered(n_users=600, mixed_time=True)
    full_als, _ = als.train(data.train_interactions(inter), als.AlsParams(8, 0.1, 10, 15))
    hist_als, _ = als.train(ranker.history_only(data.train_interactions(inter)),
                            als.AlsParams(8, 0.1, 10, 15))  # fmt: skip
    return {
        "cat": cat,
        "inter": inter,
        "serve": Signals(cat, content, meta, full_als),
        "train": Signals(cat, content, meta, hist_als),
    }


def test_history_only_keeps_first_80_percent():
    inter = clustered(n_users=4)
    kept = ranker.history_only(inter)
    for u in range(4):
        n = int((inter.user == u).sum())
        assert (kept.user == u).sum() == int(n * 0.8)
        assert kept.ts[kept.user == u].max() < inter.ts[inter.user == u][int(n * 0.8) :].min()


def test_candidates_and_features(world):
    s = world["serve"]
    history = [(1, 5.0, 0), (2, 5.0, 1), (40, 1.0, 2)]
    ctx = user_context(s, history)
    assert list(ctx.positives) == [s.row[2], s.row[1]]  # newest first
    allowed = np.ones(N, dtype=bool)
    rows, ranks = candidates(s, ctx, allowed)
    assert not set(rows) & {s.row[1], s.row[2], s.row[40]}  # seen titles excluded
    x = features(s, ctx, rows, ranks)
    assert x.shape == (len(rows), len(FEATURES)) and x.dtype == np.float32
    col = {f: x[:, i] for i, f in enumerate(FEATURES)}
    same = np.array([group(int(s.ids[r])) == 0 for r in rows])
    assert col["max_keyword"][same].mean() > col["max_keyword"][~same].mean()
    assert np.isnan(col["feel_prof"]).all()  # no feel embeddings in this world


def test_ranker_beats_popular(world):
    s_train, s_serve, inter, cat = world["train"], world["serve"], world["inter"], world["cat"]
    movies = s_train.meta.is_tv == 0
    train_users = data.eval_users(inter, "train", 300)
    valid = data.eval_users(inter, "validation", 60)
    ts = ranker.build_training_set(
        s_train, train_users, movies, ranker.held_out_ratings(inter, train_users)
    )
    vs = ranker.build_training_set(s_serve, valid, movies, ranker.held_out_ratings(inter, valid))
    model = ranker.fit(ts, vs)
    assert len(ts.groups) > 100 and model.meta["train_rows"] == len(ts.y)

    back = ranker.RankerModel.from_bytes(model.to_bytes())
    rec = ranker.RankerRecommender("ranker", cat, s_serve, back)
    users = data.eval_users(inter, "test", 60)
    ours = np.mean(harness.run_t1(rec, users)["ndcg"])
    pop = np.mean(harness.run_t1(Popular(cat), users)["ndcg"])
    assert ours > pop + 0.1
    assert all(group(t) == 0 for t in rec.similar(3, 5) if t not in TV)


def test_rerank(world):
    s = world["serve"]
    rows = np.arange(N)
    scores = np.where(s.meta.is_tv == 1, 10.0, 1.0) + np.linspace(0, 0.1, N)
    picked = ranker.rerank(s, rows, scores, 10, ranker.RerankOptions())
    tv = sum(int(s.ids[r]) in TV for r in picked)
    assert len(picked) == 10 and tv == 7  # capped at 70%
    only_movies = ranker.rerank(s, rows, scores, 5, ranker.RerankOptions(media_type="movie"))
    assert all(int(s.ids[r]) not in TV for r in only_movies)
    ok = np.zeros(N, dtype=bool)
    ok[:3] = True
    assert set(ranker.rerank(s, rows, scores, 10, ranker.RerankOptions(provider_ok=ok))) == {
        0,
        1,
        2,
    }
    # Pure diversity picks from both taste groups.
    diverse = ranker.rerank(s, rows[:50], np.ones(50), 4, ranker.RerankOptions(mmr_lambda=0.0))
    assert {group(int(s.ids[r])) for r in diverse} == {0, 1}


def test_explain(world):
    s = world["serve"]
    positives = np.array([s.row[2], s.row[45]])
    assert ranker.explain(s, positives, s.row[3]) == 2
    assert ranker.explain(s, positives, s.row[44]) == 45
    assert ranker.explain(s, np.array([], dtype=int), s.row[3]) is None


def test_label():
    assert [ranker.label(r) for r in (5.0, 4.5, 4.0, 3.5, 0.0)] == [2, 2, 1, 0, 0]
