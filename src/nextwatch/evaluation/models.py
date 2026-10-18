"""The recommender interface, the catalogue it ranks, and the baselines."""

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from psycopg import Connection

from nextwatch.evaluation.data import POSITIVE, Interactions


@dataclass
class Catalogue:
    """The titles a model may recommend, and the metadata the baselines use."""

    ids: np.ndarray  # title.id, sorted
    media_type: dict[int, str]
    tmdb_key: dict[int, tuple[str, int]]
    votes: dict[int, int]
    genres: dict[int, frozenset[int]] = field(default_factory=dict)
    tmdb_recs: dict[int, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {int(t): i for i, t in enumerate(self.ids)}
        self.by_key = {v: k for k, v in self.tmdb_key.items()}

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def load(cls, conn: Connection, media_type: str | None = None) -> "Catalogue":
        rows = conn.execute(
            "SELECT id, media_type, tmdb_id, vote_count FROM title"
            " WHERE %(mt)s::text IS NULL OR media_type = %(mt)s ORDER BY id",
            {"mt": media_type},
        ).fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        genres: dict[int, set[int]] = {}
        for tid, gid in conn.execute("SELECT title_id, genre_id FROM title_genre"):
            genres.setdefault(tid, set()).add(gid)
        all_keys = {
            (m, t): i for i, m, t in conn.execute("SELECT id, media_type, tmdb_id FROM title")
        }
        recs: dict[int, list[int]] = {}
        for tid, mt, rec in conn.execute(
            "SELECT title_id, rec_media_type, rec_tmdb_id FROM title_tmdb_rec"
            " ORDER BY title_id, rank"
        ):
            if (target := all_keys.get((mt, rec))) is not None:
                recs.setdefault(tid, []).append(target)
        keep = set(ids.tolist())
        return cls(
            ids=ids,
            media_type={r[0]: r[1] for r in rows},
            tmdb_key={r[0]: (r[1], r[2]) for r in rows},
            votes={r[0]: r[3] for r in rows},
            genres={t: frozenset(g) for t, g in genres.items() if t in keep},
            tmdb_recs={t: [x for x in r if x in keep] for t, r in recs.items() if t in keep},
        )


class Recommender(Protocol):
    name: str

    def recommend(
        self, history: Sequence[tuple[int, float]], exclude: set[int], k: int
    ) -> list[int]:
        """Top k titles for a user with these (title_id, rating) pairs, oldest first."""
        ...

    def similar(self, title_id: int, k: int) -> list[int]:
        """Top k titles like this one, not including it."""
        ...


def top_k(scores: np.ndarray, ids: np.ndarray, exclude: set[int], k: int) -> list[int]:
    """Highest-scoring ids, skipping excluded ones. Ties break on position (stable)."""
    want = min(len(ids), k + len(exclude))
    if want <= 0:
        return []
    part = np.argpartition(-scores, want - 1)[:want] if want < len(ids) else np.arange(len(ids))
    order = part[np.lexsort((part, -scores[part]))]
    out = []
    for i in order:
        t = int(ids[i])
        if t not in exclude:
            out.append(t)
            if len(out) == k:
                break
    return out


class Popular:
    """Most-voted titles the user has not seen."""

    name = "popular"

    def __init__(self, catalogue: Catalogue):
        self.cat = catalogue
        self.scores = np.array([math.log1p(catalogue.votes[int(t)]) for t in catalogue.ids])

    def recommend(self, history, exclude, k):
        return top_k(self.scores, self.cat.ids, exclude, k)

    def similar(self, title_id, k):
        return top_k(self.scores, self.cat.ids, {title_id}, k)


class PopularInGenre:
    """Popularity weighted by how much the user likes each genre."""

    name = "popular-genre"

    def __init__(self, catalogue: Catalogue):
        self.cat = catalogue
        all_genres = sorted({g for gs in catalogue.genres.values() for g in gs})
        self.genre_index = {g: i for i, g in enumerate(all_genres)}
        self.matrix = np.zeros((len(catalogue), len(all_genres)), dtype=np.float32)
        for t, gs in catalogue.genres.items():
            for g in gs:
                self.matrix[catalogue.index[t], self.genre_index[g]] = 1
        self.log_votes = np.array(
            [math.log1p(catalogue.votes[int(t)]) for t in catalogue.ids], dtype=np.float32
        )

    def _weights(self, titles: Sequence[int]) -> np.ndarray:
        w = np.zeros(len(self.genre_index), dtype=np.float32)
        for t in titles:
            for g in self.cat.genres.get(t, ()):
                w[self.genre_index[g]] += 1
        return w / w.sum() if w.sum() else w

    def _rank(self, titles: Sequence[int], exclude: set[int], k: int) -> list[int]:
        scores = self.log_votes * (self.matrix @ self._weights(titles) + 0.05)
        return top_k(scores, self.cat.ids, exclude, k)

    def recommend(self, history, exclude, k):
        liked = [t for t, r in history if r >= POSITIVE] or [t for t, _ in history]
        return self._rank(liked, exclude, k)

    def similar(self, title_id, k):
        return self._rank([title_id], {title_id}, k)


class TmdbRecs:
    """TMDB's own recommendations for the user's positives, most frequent first."""

    name = "tmdb"

    def __init__(self, catalogue: Catalogue):
        self.cat = catalogue
        self.fallback = Popular(catalogue)

    def _rank(self, seeds: Sequence[int], exclude: set[int], k: int) -> list[int]:
        counts: Counter[int] = Counter()
        for s in seeds:
            for rank, t in enumerate(self.cat.tmdb_recs.get(s, [])):
                counts[t] += 1 - rank / 100  # earlier recs break ties
        ranked = [
            t for t, _ in sorted(counts.items(), key=lambda x: (-x[1], -self.cat.votes[x[0]]))
        ]
        out = [t for t in ranked if t not in exclude][:k]
        if len(out) < k:
            out += [t for t in self.fallback.recommend([], exclude | set(out), k - len(out))]
        return out

    def recommend(self, history, exclude, k):
        return self._rank([t for t, r in history if r >= POSITIVE], exclude, k)

    def similar(self, title_id, k):
        return self._rank([title_id], {title_id}, k)


ModelFactory = Callable[[Catalogue, Interactions | None], Recommender]

MODELS: dict[str, ModelFactory] = {
    "popular": lambda cat, _: Popular(cat),
    "popular-genre": lambda cat, _: PopularInGenre(cat),
    "tmdb": lambda cat, _: TmdbRecs(cat),
}
