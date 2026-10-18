import numpy as np
import pytest

from nextwatch import models_store
from nextwatch.cf import als
from nextwatch.evaluation.data import Interactions
from nextwatch.evaluation.models import Catalogue

N_ITEMS = 60


def clustered(n_users=300, seed=3, mixed_time=False) -> Interactions:
    """Two taste groups: users like items 1-30 or 31-60, and rate the other group low.

    With mixed_time, each user rates in random order, so held-out ratings include positives.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        group = u % 2
        liked = rng.choice(30, size=12, replace=False) + 1 + 30 * group
        disliked = rng.choice(30, size=3, replace=False) + 1 + 30 * (1 - group)
        times = rng.permutation(15) if mixed_time else np.arange(15)
        for i, t in enumerate(liked):
            rows.append((u, int(t), 5.0, int(times[i])))
        for i, t in enumerate(disliked):
            rows.append((u, int(t), 1.0, int(times[12 + i]) + (0 if mixed_time else 100)))
    rows.sort(key=lambda r: (r[0], r[3]))
    u, t, r, ts = zip(*rows, strict=True)
    return Interactions(np.array(u, dtype=np.int32), np.array(t), np.array(r, np.float32),
                        np.array(ts))  # fmt: skip


def catalogue() -> Catalogue:
    ids = np.arange(1, N_ITEMS + 1)
    return Catalogue(
        ids=ids,
        media_type={int(i): "movie" for i in ids},
        tmdb_key={int(i): ("movie", int(i)) for i in ids},
        votes={int(i): 1 for i in ids},
    )


PARAMS = als.AlsParams(factors=8, regularization=0.1, alpha=10, iterations=20)


@pytest.fixture(scope="module")
def trained():
    return als.train(clustered(), PARAMS)


def test_confidence():
    extra = als.confidence_extra(np.array([5.0, 4.0, 2.5, 1.0]), 10)
    assert np.allclose(extra, [25.1, 15.1, 0.1, 0.1])


def test_recommends_within_taste_group(trained):
    model, _ = trained
    rec = als.AlsRecommender(catalogue(), model)
    history = [(t, 5.0, 0) for t in (1, 2, 3, 4, 5)]
    top = rec.recommend(history, {1, 2, 3, 4, 5}, 10)
    assert len(top) == 10 and all(t <= 30 for t in top)
    assert all(t <= 30 for t in rec.similar(1, 5))
    assert rec.recommend([(999, 5.0, 0)], set(), 5) == []  # unknown titles only


def test_fold_in_matches_training(trained):
    """Plan 06: fold-in gives nearly the same result as training with the user included."""
    model, raw = trained
    data = clustered()
    user = 0
    mask = data.user == user
    history = list(zip(data.title[mask].tolist(), data.rating[mask].tolist(), strict=True))
    folded = model.user_vector(history)
    trained_vec = np.asarray(raw.user_factors[user])
    top_folded = set(np.argsort(-(model.item_factors @ folded))[:10])
    top_trained = set(np.argsort(-(model.item_factors @ trained_vec))[:10])
    assert len(top_folded & top_trained) >= 8


def test_serialisation_round_trip(trained):
    model, _ = trained
    back = als.AlsModel.from_bytes(model.to_bytes())
    assert np.array_equal(back.item_ids, model.item_ids)
    assert np.allclose(back.item_factors, model.item_factors)
    assert back.params == model.params


@pytest.mark.db
def test_model_store(conn, trained):
    model, _ = trained
    for _ in range(4):
        models_store.save(conn, als.ARTIFACT, model.to_bytes(), {"n": 1})
    assert models_store.prune(conn, als.ARTIFACT, keep=3) == 1
    version, blob, meta = models_store.load_latest(conn, als.ARTIFACT)
    assert version == 4 and meta == {"n": 1}
    assert np.allclose(als.load(conn).item_factors, model.item_factors)
