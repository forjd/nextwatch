"""Everything the ranker reads, loaded once and aligned to one row order. See plan 07."""

import logging
import math
from dataclasses import dataclass

import numpy as np
from psycopg import Connection

from nextwatch.cf.als import AlsModel
from nextwatch.evaluation.models import Catalogue
from nextwatch.features.content import ContentFeatures

log = logging.getLogger(__name__)

KNOWLEDGE = {"overview-only": 0.0, "partial": 1.0, "known": 2.0}


@dataclass
class Meta:
    """Per-title columns, aligned to Signals rows."""

    log_votes: np.ndarray
    vote_average: np.ndarray
    log_popularity: np.ndarray
    year: np.ndarray  # NaN if unknown
    runtime: np.ndarray  # NaN if unknown
    is_tv: np.ndarray
    is_english: np.ndarray
    knowledge: np.ndarray  # vibe knowledge 0-2, NaN if untagged


class Signals:
    """Catalogue, content features, ALS and metadata, all indexed by the same row numbers."""

    def __init__(
        self,
        cat: Catalogue,
        content: ContentFeatures,
        meta: Meta,
        als: AlsModel | None = None,
    ):
        if not np.array_equal(cat.ids, content.sparse.ids):
            raise ValueError("catalogue and content features must cover the same titles")
        self.cat = cat
        self.ids = cat.ids
        self.row = {int(t): i for i, t in enumerate(cat.ids)}
        self.content = content
        self.meta = meta
        self.als = als
        n = len(cat.ids)
        self.als_row = np.array(
            [als.index.get(int(t), -1) for t in cat.ids] if als else [-1] * n, dtype=int
        )
        self.als_unit = als._unit if als else None
        self.recs = [
            np.array([self.row[t] for t in cat.tmdb_recs.get(int(t), []) if t in self.row])
            for t in cat.ids
        ]
        genres = sorted({g for gs in cat.genres.values() for g in gs})
        self.genre_ids = genres
        self.genre_matrix = np.zeros((n, len(genres)), dtype=np.float32)
        gi = {g: j for j, g in enumerate(genres)}
        for t, gs in cat.genres.items():
            for g in gs:
                self.genre_matrix[self.row[t], gi[g]] = 1

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def load(cls, conn: Connection, content: ContentFeatures, als: AlsModel | None) -> "Signals":
        cat = Catalogue.load(conn)
        rows = {
            r[0]: r[1:]
            for r in conn.execute(
                "SELECT id, vote_count, vote_average, popularity, year, runtime, media_type,"
                " original_language FROM title"
            )
        }
        knowledge = dict(
            conn.execute(
                "SELECT DISTINCT ON (title_id) title_id, tags->>'knowledge' FROM title_vibe"
                " ORDER BY title_id, created_at DESC"
            ).fetchall()
        )

        def col(f):
            return np.array([f(*rows[int(t)]) for t in cat.ids], dtype=np.float32)

        meta = Meta(
            log_votes=col(lambda v, *_: math.log1p(v or 0)),
            vote_average=col(lambda _, a, *__: a if a is not None else np.nan),
            log_popularity=col(lambda _, __, p, *___: math.log1p(p or 0)),
            year=col(lambda *r: r[3] if r[3] is not None else np.nan),
            runtime=col(lambda *r: r[4] if r[4] is not None else np.nan),
            is_tv=col(lambda *r: float(r[5] == "tv")),
            is_english=col(lambda *r: float(r[6] == "en")),
            knowledge=np.array(
                [KNOWLEDGE.get(knowledge.get(int(t)), np.nan) for t in cat.ids], dtype=np.float32
            ),
        )
        return cls(cat, content, meta, als)
