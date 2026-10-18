"""Implicit ALS on MovieLens and fold-in of a new user. See plan 06 (collaborative filtering).

Training uses the `implicit` library. Serving needs only numpy: a user vector is solved
from the item factors with the same confidence weighting, so the app never retrains when the
owner rates something.
"""

import io
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from psycopg import Connection
from scipy import sparse

from nextwatch import models_store
from nextwatch.evaluation.data import Interactions
from nextwatch.evaluation.models import Catalogue, register, top_k

log = logging.getLogger(__name__)

ARTIFACT = "als"
# Ratings at or below this count only as a weak "seen" signal.
NEUTRAL = 2.5
# Every rated item gets at least this much extra confidence, so low ratings still mark "seen".
SEEN_FLOOR = 0.1


@dataclass(frozen=True)
class AlsParams:
    factors: int = 128
    regularization: float = 0.05
    alpha: float = 10.0
    iterations: int = 15
    seed: int = 7


def confidence_extra(ratings: np.ndarray, alpha: float) -> np.ndarray:
    """Confidence minus 1: alpha * max(rating - 2.5, 0), floored so every rating counts."""
    return alpha * np.maximum(ratings - NEUTRAL, 0) + SEEN_FLOOR


@dataclass
class AlsModel:
    item_ids: np.ndarray  # title.id per factor row
    item_factors: np.ndarray  # float32, items x factors
    params: AlsParams

    def __post_init__(self) -> None:
        self.index = {int(t): i for i, t in enumerate(self.item_ids)}
        y = self.item_factors.astype(np.float64)
        self._yty = y.T @ y
        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True)
        self._unit = self.item_factors / np.where(norms == 0, 1, norms)

    def user_vector(self, history: Sequence[tuple[int, float]]) -> np.ndarray | None:
        """Least-squares user factors for (title_id, rating) pairs, as implicit ALS solves them.

        x = (Y'Y + Y'(C - I)Y + reg I)^-1  Y'C p, with p = 1 for every rated item.
        """
        rows = [(self.index[t], r) for t, r in history if t in self.index]
        if not rows:
            return None
        idx = np.array([i for i, _ in rows])
        extra = confidence_extra(
            np.array([r for _, r in rows], dtype=np.float64), self.params.alpha
        )
        y = self.item_factors[idx].astype(np.float64)
        a = self._yty + (y.T * extra) @ y + self.params.regularization * np.eye(y.shape[1])
        b = (y.T * (1 + extra)).sum(axis=1)
        return np.linalg.solve(a, b).astype(np.float32)

    def scores(self, user: np.ndarray) -> np.ndarray:
        return self.item_factors @ user

    def similar_scores(self, title_id: int) -> np.ndarray | None:
        i = self.index.get(title_id)
        return None if i is None else self._unit @ self._unit[i]

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            item_ids=self.item_ids,
            item_factors=self.item_factors.astype(np.float32),
            params=np.array(json.dumps(asdict(self.params))),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, blob: bytes) -> "AlsModel":
        z = np.load(io.BytesIO(blob))
        return cls(z["item_ids"], z["item_factors"], AlsParams(**json.loads(str(z["params"]))))


def user_items(data: Interactions, alpha: float) -> tuple[sparse.csr_matrix, np.ndarray]:
    """users x items confidence matrix, and the title id of each item column."""
    users, u_idx = np.unique(data.user, return_inverse=True)
    items, i_idx = np.unique(data.title, return_inverse=True)
    m = sparse.csr_matrix(
        (confidence_extra(data.rating.astype(np.float32), alpha), (u_idx, i_idx)),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )
    m.sum_duplicates()
    return m, items


def train(data: Interactions, params: AlsParams | None = None) -> tuple[AlsModel, object]:
    """Fit ALS. Returns the servable model and the raw implicit model (for tests)."""
    from implicit.cpu.als import AlternatingLeastSquares
    from threadpoolctl import threadpool_limits

    params = params or AlsParams()
    matrix, items = user_items(data, params.alpha)
    log.info("ALS: %d users x %d items, %d ratings", *matrix.shape, matrix.nnz)
    model = AlternatingLeastSquares(
        factors=params.factors,
        regularization=params.regularization,
        alpha=1.0,  # confidence is already in the matrix
        iterations=params.iterations,
        random_state=params.seed,
    )
    # implicit parallelises itself; a threaded BLAS underneath it only fights for the cores.
    with threadpool_limits(1, "blas"):
        model.fit(matrix, show_progress=False)
    return AlsModel(items, np.asarray(model.item_factors, dtype=np.float32), params), model


class AlsRecommender:
    name = "als"

    def __init__(self, cat: Catalogue, model: AlsModel):
        self.cat = cat
        self.model = model
        # Catalogue titles with factors, and their rows. Titles without factors are not ranked.
        keep = [i for i, t in enumerate(cat.ids) if int(t) in model.index]
        self.cat_ids = cat.ids[keep]
        self.rows = np.array([model.index[int(t)] for t in self.cat_ids], dtype=int)

    def recommend(self, history, exclude, k):
        user = self.model.user_vector([(t, r) for t, r, _ in history])
        if user is None:
            return []
        return top_k(self.model.scores(user)[self.rows], self.cat_ids, exclude, k)

    def similar(self, title_id, k):
        s = self.model.similar_scores(title_id)
        return [] if s is None else top_k(s[self.rows], self.cat_ids, {title_id}, k)


def load(conn: Connection) -> AlsModel:
    found = models_store.load_latest(conn, ARTIFACT)
    if found is None:
        raise RuntimeError("no ALS model; run `nextwatch train als` first")
    version, blob, _ = found
    log.info("loaded ALS model version %d", version)
    return AlsModel.from_bytes(blob)


register("als", lambda cat, _, conn: AlsRecommender(cat, load(conn)))
