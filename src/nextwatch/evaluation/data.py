"""MovieLens splits for evaluation and training. See docs/plans/03-evaluation.md.

Users are split once at random (80/10/10). Each evaluated user's ratings are sorted by
time: the first 80% are history the model sees, the last 20% are held out.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from psycopg import Connection

from nextwatch.movielens import RATINGS

SEED = 20261018
POSITIVE = 4.0
HISTORY_FRACTION = 0.8
MIN_HISTORY = 5
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Interactions:
    """Ratings mapped to title ids, sorted by (user, timestamp)."""

    user: np.ndarray  # int32 MovieLens user id
    title: np.ndarray  # int64 title.id
    rating: np.ndarray  # float32
    ts: np.ndarray  # int64 unix seconds

    def __len__(self) -> int:
        return len(self.user)

    def subset(self, mask: np.ndarray) -> "Interactions":
        return Interactions(self.user[mask], self.title[mask], self.rating[mask], self.ts[mask])


@dataclass(frozen=True)
class EvalUser:
    user_id: int
    history: np.ndarray  # title ids, oldest first
    history_ratings: np.ndarray
    history_ts: np.ndarray
    holdout_positives: frozenset[int]

    def positives(self) -> list[int]:
        """History titles rated as positives, oldest first."""
        return [
            int(t) for t, r in zip(self.history, self.history_ratings, strict=True) if r >= POSITIVE
        ]


def movielens_to_title(conn: Connection) -> dict[int, int]:
    """MovieLens movieId -> title.id, for movies in the catalogue."""
    rows = conn.execute(
        """
        SELECT l.movielens_id, t.id FROM movielens_link l
        JOIN title t ON t.media_type = 'movie' AND t.tmdb_id = l.tmdb_id
        """
    ).fetchall()
    return dict(rows)


def load_interactions(data_dir: Path, ml_to_title: dict[int, int]) -> Interactions:
    if not ml_to_title:
        raise RuntimeError("no MovieLens movies are in the catalogue; run ingest and normalise")
    mapping = pl.DataFrame(
        {"movieId": list(ml_to_title), "title": list(ml_to_title.values())},
        schema={"movieId": pl.Int32, "title": pl.Int64},
    )
    df = (
        pl.scan_parquet(data_dir / "movielens" / RATINGS)
        .join(mapping.lazy(), on="movieId", how="inner")
        .sort("userId", "timestamp", "title")
        .select("userId", "title", "rating", "timestamp")
        .collect()
    )
    return Interactions(
        df["userId"].to_numpy(),
        df["title"].to_numpy(),
        df["rating"].to_numpy(),
        df["timestamp"].to_numpy(),
    )


def split_users(users: np.ndarray, seed: int = SEED) -> dict[str, np.ndarray]:
    """Fixed 80/10/10 split of unique user ids."""
    unique = np.unique(users)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    a, b = int(len(perm) * 0.8), int(len(perm) * 0.9)
    return {"train": np.sort(perm[:a]), "validation": np.sort(perm[a:b]), "test": np.sort(perm[b:])}


def _user_bounds(user: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique users and the start/end index of each in a user-sorted array."""
    ids, starts = np.unique(user, return_index=True)
    ends = np.append(starts[1:], len(user))
    return ids, starts, ends


def eval_users(data: Interactions, split: str, n_users: int, seed: int = SEED) -> list[EvalUser]:
    """A fixed sample of users from a split, each with history and held-out positives."""
    members = split_users(data.user, seed)[split]
    ids, starts, ends = _user_bounds(data.user)
    in_split = np.isin(ids, members)
    candidates = []
    for uid, s, e in zip(ids[in_split], starts[in_split], ends[in_split], strict=True):
        cut = s + int((e - s) * HISTORY_FRACTION)
        if cut - s < MIN_HISTORY:
            continue
        held = data.rating[cut:e] >= POSITIVE
        if not held.any():
            continue
        candidates.append((int(uid), s, cut, e))
    rng = np.random.default_rng(seed + 1)
    if len(candidates) > n_users:
        pick = np.sort(rng.choice(len(candidates), n_users, replace=False))
        candidates = [candidates[i] for i in pick]
    out = []
    for uid, s, cut, e in candidates:
        held_titles = data.title[cut:e][data.rating[cut:e] >= POSITIVE]
        out.append(
            EvalUser(
                uid,
                data.title[s:cut],
                data.rating[s:cut],
                data.ts[s:cut],
                frozenset(int(t) for t in held_titles),
            )
        )
    return out


def train_interactions(data: Interactions, seed: int = SEED) -> Interactions:
    """All ratings of train users, for model training."""
    return data.subset(np.isin(data.user, split_users(data.user, seed)["train"]))
