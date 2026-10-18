"""LightGBM LambdaRank over candidate features, re-ranking and explanations. See plan 07.

Training data comes from MovieLens train users, split exactly as evaluation splits them. The
collaborative features for those users come from an ALS model trained on their history parts
only: an ALS model that saw their held-out ratings would leak the labels into cf_score.
"""

import json
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from psycopg import Connection

from nextwatch import models_store
from nextwatch.cf import als
from nextwatch.evaluation import data as evaldata
from nextwatch.evaluation.data import Interactions
from nextwatch.evaluation.models import Catalogue, register
from nextwatch.features.content import features_for
from nextwatch.ranking.features import FEATURES, candidates, features, user_context
from nextwatch.ranking.signals import Signals

log = logging.getLogger(__name__)

ARTIFACT = "ranker"


def label(rating: float) -> int:
    return 2 if rating >= 4.5 else 1 if rating >= 4.0 else 0


def history_only(data: Interactions) -> Interactions:
    """Each user's first 80% of ratings by time: what ALS may see when training the ranker."""
    keep = np.zeros(len(data), dtype=bool)
    ids, starts = np.unique(data.user, return_index=True)
    ends = np.append(starts[1:], len(data.user))
    for s, e in zip(starts, ends, strict=True):
        keep[s : s + int((e - s) * evaldata.HISTORY_FRACTION)] = True
    return data.subset(keep)


@dataclass
class TrainingSet:
    x: np.ndarray
    y: np.ndarray
    groups: list[int] = field(default_factory=list)


def build_training_set(
    s: Signals, users: list[evaldata.EvalUser], allowed: np.ndarray, held_ratings: dict
) -> TrainingSet:
    """One query group per user: candidates, features, and labels from held-out ratings.

    held_ratings maps user_id -> {title_id: rating} over the held-out part.
    """
    xs, ys, groups = [], [], []
    for u in users:
        history = list(zip(u.history.tolist(), u.history_ratings.tolist(), u.history_ts.tolist(),
                           strict=True))  # fmt: skip
        ctx = user_context(s, history)
        rows, ranks = candidates(s, ctx, allowed)
        if not len(rows):
            continue
        held = held_ratings[u.user_id]
        y = np.array([label(held.get(int(s.ids[r]), 0.0)) for r in rows], dtype=np.int32)
        if not y.any():
            continue  # no positive among candidates: nothing to learn from this user
        xs.append(features(s, ctx, rows, ranks))
        ys.append(y)
        groups.append(len(rows))
    if not xs:
        raise RuntimeError("no training groups: candidates never contain a held-out positive")
    return TrainingSet(np.vstack(xs), np.concatenate(ys), groups)


def held_out_ratings(data: Interactions, users: list[evaldata.EvalUser]) -> dict:
    """user_id -> {title_id: rating} for each user's held-out (last 20%) ratings."""
    ids, starts = np.unique(data.user, return_index=True)
    ends = np.append(starts[1:], len(data.user))
    bounds = {int(u): (s, e) for u, s, e in zip(ids, starts, ends, strict=True)}
    out = {}
    for u in users:
        start, end = bounds[u.user_id]
        cut = start + len(u.history)
        out[u.user_id] = dict(
            zip(data.title[cut:end].tolist(), data.rating[cut:end].tolist(), strict=True)
        )
    return out


@dataclass
class RankerModel:
    booster_text: str
    feature_names: tuple[str, ...]
    meta: dict

    def __post_init__(self) -> None:
        import lightgbm as lgb

        self.booster = lgb.Booster(model_str=self.booster_text)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(x, num_threads=1)

    def to_bytes(self) -> bytes:
        return json.dumps(
            {"booster": self.booster_text, "features": list(self.feature_names)}
        ).encode()

    @classmethod
    def from_bytes(cls, blob: bytes, meta: dict | None = None) -> "RankerModel":
        d = json.loads(blob)
        if tuple(d["features"]) != FEATURES:
            raise RuntimeError("ranker was trained on different features; retrain it")
        return cls(d["booster"], tuple(d["features"]), meta or {})


PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [20],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "verbose": -1,
}
MAX_ROUNDS = 400


def fit(train: TrainingSet, valid: TrainingSet | None, seed: int = 7) -> RankerModel:
    """LightGBM's native API: lgb.LGBMRanker would pull in scikit-learn."""
    import lightgbm as lgb

    params = {**PARAMS, "seed": seed}
    dtrain = lgb.Dataset(train.x, train.y, group=train.groups, feature_name=list(FEATURES))
    kwargs = {}
    if valid is not None:
        dvalid = lgb.Dataset(valid.x, valid.y, group=valid.groups, reference=dtrain)
        kwargs = {
            "valid_sets": [dvalid],
            "callbacks": [lgb.early_stopping(30, verbose=False)],
        }
    booster = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, **kwargs)
    importance = dict(
        zip(FEATURES, booster.feature_importance("gain").round(1).tolist(), strict=True)
    )
    meta = {
        "best_iteration": booster.best_iteration or MAX_ROUNDS,
        "importance_gain": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
        "train_rows": int(len(train.y)),
        "train_groups": len(train.groups),
    }
    return RankerModel(booster.model_to_string(), FEATURES, meta)


# Re-ranking (plan 07, step 4)


@dataclass(frozen=True)
class RerankOptions:
    mmr_lambda: float = 0.7
    popularity_penalty: float = 0.0
    max_type_share: float = 0.7
    media_type: str = "both"  # "movie", "tv" or "both"
    provider_ok: np.ndarray | None = None  # per-row mask: available on the owner's services


def rerank(
    s: Signals, rows: np.ndarray, scores: np.ndarray, k: int, opts: RerankOptions
) -> list[int]:
    """Pick k rows: MMR for diversity, a popularity penalty, type mix and availability."""
    keep = np.ones(len(rows), dtype=bool)
    if opts.provider_ok is not None:
        keep &= opts.provider_ok[rows]
    if opts.media_type != "both":
        keep &= s.meta.is_tv[rows] == (1.0 if opts.media_type == "tv" else 0.0)
    rows, scores = rows[keep], scores[keep]
    if not len(rows):
        return []
    span = scores.max() - scores.min()
    rel = (scores - scores.min()) / span if span > 0 else np.ones_like(scores)
    if opts.popularity_penalty:
        lv = s.meta.log_votes[rows]
        rel = rel - opts.popularity_penalty * (lv / max(lv.max(), 1e-9))
    emb = s.content.dense
    cap = math.ceil(opts.max_type_share * k) if opts.media_type == "both" else k
    chosen: list[int] = []
    counts = {0.0: 0, 1.0: 0}
    max_sim = np.zeros(len(rows))
    available = np.ones(len(rows), dtype=bool)
    while len(chosen) < k and available.any():
        value = opts.mmr_lambda * rel - (1 - opts.mmr_lambda) * max_sim
        full_types = [t for t, c in counts.items() if c >= cap]
        blocked = np.isin(s.meta.is_tv[rows], full_types) & available
        pick_from = available & ~blocked if (available & ~blocked).any() else available
        i = int(np.argmax(np.where(pick_from, value, -np.inf)))
        chosen.append(int(rows[i]))
        counts[float(s.meta.is_tv[rows[i]])] += 1
        available[i] = False
        if emb is not None:
            max_sim = np.maximum(max_sim, emb[rows] @ emb[rows[i]])
    return chosen


def explain(s: Signals, positives: np.ndarray, row: int) -> int | None:
    """The user's positive that most explains a recommendation, by the strongest similarity."""
    if not len(positives):
        return None
    best = np.zeros(len(positives))
    for f in ("director", "writer", "cast", "keyword"):
        m = s.content.sparse.matrices[f]
        best = np.maximum(best, np.asarray((m[positives] @ m[row].T).todense()).ravel())
    if s.content.dense is not None:
        best = np.maximum(best, s.content.dense[positives] @ s.content.dense[row])
    if s.als_unit is not None and s.als_row[row] >= 0:
        ok = s.als_row[positives] >= 0
        sims = np.zeros(len(positives))
        sims[ok] = s.als_unit[s.als_row[positives[ok]]] @ s.als_unit[s.als_row[row]]
        best = np.maximum(best, sims)
    return int(s.ids[positives[int(np.argmax(best))]])


# Serving and evaluation


class RankerRecommender:
    def __init__(
        self,
        name: str,
        cat: Catalogue,
        s: Signals,
        model: RankerModel,
        rerank_opts: RerankOptions | None = None,
    ):
        self.name = name
        self.s = s
        self.model = model
        self.rerank_opts = rerank_opts
        self.allowed = np.isin(s.ids, cat.ids)

    def rank(self, history, exclude, k):
        ctx = user_context(self.s, history)
        allowed = self.allowed.copy()
        allowed[[self.s.row[t] for t in exclude if t in self.s.row]] = False
        rows, ranks = candidates(self.s, ctx, allowed)
        if not len(rows):
            return ctx, []
        scores = self.model.predict(features(self.s, ctx, rows, ranks))
        if self.rerank_opts is not None:
            return ctx, rerank(self.s, rows, scores, k, self.rerank_opts)
        order = np.argsort(-scores, kind="stable")[:k]
        return ctx, [int(r) for r in rows[order]]

    def recommend(self, history, exclude, k):
        return [int(self.s.ids[r]) for r in self.rank(history, exclude, k)[1]]

    def similar(self, title_id, k):
        return self.recommend([(title_id, 5.0, 0.0)], {title_id}, k)


_signals: dict[int, Signals] = {}


def signals_for(conn: Connection) -> Signals:
    key = id(conn)
    if key not in _signals:
        _signals.clear()
        try:
            als_model = als.load(conn)
        except RuntimeError:
            log.warning("no ALS model; collaborative features are missing")
            als_model = None
        _signals[key] = Signals.load(conn, features_for(conn), als_model)
    return _signals[key]


def load(conn: Connection) -> RankerModel:
    found = models_store.load_latest(conn, ARTIFACT)
    if found is None:
        raise RuntimeError("no ranker model; run `nextwatch train ranker` first")
    version, blob, meta = found
    log.info("loaded ranker version %d", version)
    return RankerModel.from_bytes(blob, meta)


def _factory(name: str, opts: RerankOptions | None):
    def make(cat, _, conn):
        return RankerRecommender(name, cat, signals_for(conn), load(conn), opts)

    return make


register("ranker", _factory("ranker", None))
register("ranker-rerank", _factory("ranker-rerank", RerankOptions()))
