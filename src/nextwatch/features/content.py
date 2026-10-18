"""Content-based recommenders built on sparse and dense features. See plan 04.

A user profile is a weighted mix of the titles they rated: weight (rating - 3), halved for
every year between the rating and the user's newest rating.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from psycopg import Connection

from nextwatch.evaluation.models import Catalogue, register, top_k
from nextwatch.features import embed, sparse

log = logging.getLogger(__name__)

HALF_LIFE_DAYS = 365.0
NEUTRAL = 3.0


@dataclass
class ContentFeatures:
    sparse: sparse.SparseIndex
    dense: np.ndarray | None  # rows aligned to sparse.ids, unit length; None if not embedded

    @classmethod
    def load(cls, conn: Connection) -> "ContentFeatures":
        idx = sparse.build(conn)
        dense = embed.load_matrix(conn, idx.ids)
        if not dense.any():
            log.warning("no %s embeddings; dense similarity is off", embed.MODEL)
            dense = None
        return cls(idx, dense)


def profile_weights(history: Sequence[tuple[int, float, float]]) -> list[tuple[int, float]]:
    """(title_id, weight) for a history of (title_id, rating, unix time)."""
    if not history:
        return []
    newest = max(ts for _, _, ts in history)
    out = []
    for t, r, ts in history:
        age_days = (newest - ts) / 86400
        w = (r - NEUTRAL) * 0.5 ** (age_days / HALF_LIFE_DAYS)
        if w:
            out.append((t, w))
    return out


class ContentModel:
    def __init__(
        self,
        name: str,
        cat: Catalogue,
        features: ContentFeatures,
        use_sparse: bool = True,
        use_dense: bool = True,
        dense_weight: float = 1.0,
    ):
        self.name = name
        self.cat = cat
        self.f = features
        self.use_sparse = use_sparse
        self.use_dense = use_dense and features.dense is not None
        self.dense_weight = dense_weight
        # Catalogue titles that have features, and their feature rows.
        keep = [i for i, t in enumerate(cat.ids) if int(t) in features.sparse.index]
        self.cat_ids = cat.ids[keep]
        self.rows = np.array([features.sparse.index[int(t)] for t in self.cat_ids], dtype=int)

    def score(self, weighted: list[tuple[int, float]]) -> np.ndarray:
        """Score every catalogue title against weighted (title_id, weight) seeds."""
        pairs = [(self.f.sparse.index[t], w) for t, w in weighted if t in self.f.sparse.index]
        scores = np.zeros(len(self.f.sparse))
        if not pairs:
            return scores[self.rows]
        rows = np.array([r for r, _ in pairs])
        weights = np.array([w for _, w in pairs], dtype=np.float32)
        if self.use_sparse:
            scores += self.f.sparse.combined(rows, weights)
        if self.use_dense:
            profile = (weights / np.abs(weights).sum()) @ self.f.dense[rows]
            scores += self.dense_weight * (self.f.dense @ profile)
        return scores[self.rows]

    def recommend(self, history, exclude, k):
        return top_k(self.score(profile_weights(history)), self.cat_ids, exclude, k)

    def similar(self, title_id, k):
        return top_k(self.score([(title_id, 1.0)]), self.cat_ids, {title_id}, k)


_cache: dict[int, ContentFeatures] = {}


def features_for(conn: Connection) -> ContentFeatures:
    """Build features once per connection; eval builds two models from the same data."""
    key = id(conn)
    if key not in _cache:
        _cache.clear()
        _cache[key] = ContentFeatures.load(conn)
    return _cache[key]


register(
    "content-sparse",
    lambda cat, _, conn: ContentModel("content-sparse", cat, features_for(conn), use_dense=False),
)
register(
    "content-dense",
    lambda cat, _, conn: ContentModel("content-dense", cat, features_for(conn), use_sparse=False),
)
register("content", lambda cat, _, conn: ContentModel("content", cat, features_for(conn)))
