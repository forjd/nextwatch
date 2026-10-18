"""Candidate generation and per-candidate features for one user. See plan 07, steps 1 and 2."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nextwatch.evaluation.data import POSITIVE
from nextwatch.features.content import profile_weights
from nextwatch.features.sparse import CONTENT_FAMILIES, FAMILIES
from nextwatch.ranking.signals import Signals

# (source, how many candidates it contributes)
SOURCES = (
    ("als", 200),
    ("embed", 150),
    ("sparse", 150),
    ("feel", 100),
    ("tmdb", 100),
    ("genre", 50),
)
MAX_POSITIVES = 50

FEATURES = (
    "cf_score", "has_cf", "sim_als_max", "sim_als_mean",
    *(f"prof_{f}" for f in FAMILIES), *(f"max_{f}" for f in FAMILIES),
    "embed_prof", "embed_max", "feel_prof", "feel_max",
    "tmdb_count",
    "log_votes", "vote_average", "log_popularity", "year", "runtime", "is_tv", "is_english",
    "knowledge", "genre_affinity", "year_gap",
    *(f"rank_{s}" for s, _ in SOURCES),
)  # fmt: skip


@dataclass
class UserContext:
    """Per-user scores over every title, computed once per request."""

    seen: np.ndarray  # rows the user rated
    positives: np.ndarray  # rows rated >= 4 (or the best-rated ones), newest first
    scores: dict[str, np.ndarray]  # source/family -> score for every row
    tmdb_counts: np.ndarray
    genre_affinity: np.ndarray  # per genre column: mean rating, NaN if never rated
    mean_year: float


def user_context(s: Signals, history: Sequence[tuple[int, float, float]]) -> UserContext:
    known = [(s.row[t], r, ts) for t, r, ts in history if t in s.row]
    n = len(s)
    seen = np.array([r for r, _, _ in known], dtype=int)
    pos = [r for r, rating, _ in reversed(known) if rating >= POSITIVE]
    if not pos and known:
        best = max(rating for _, rating, _ in known)
        pos = [r for r, rating, _ in reversed(known) if rating == best]
    positives = np.array(pos[:MAX_POSITIVES], dtype=int)

    weighted = profile_weights([(row, r, ts) for row, r, ts in known])
    rows = np.array([row for row, _ in weighted], dtype=int)
    w = np.array([wt for _, wt in weighted], dtype=np.float32)
    scores: dict[str, np.ndarray] = {}
    c = s.content
    if len(rows):
        scores.update(c.sparse.profile_scores(rows, w, FAMILIES))
        norm = w / np.abs(w).sum()
        if c.dense is not None:
            scores["embed"] = c.dense @ (norm @ c.dense[rows])
        if c.feel is not None:
            scores["feel"] = c.feel @ (norm @ c.feel[rows])
    else:
        scores.update({f: np.zeros(n) for f in FAMILIES})
    scores["sparse"] = sum(scores[f] for f in CONTENT_FAMILIES)

    if s.als is not None:
        als_hist = [(int(s.ids[row]), r) for row, r, _ in known]
        vec = s.als.user_vector(als_hist)
        if vec is not None:
            raw = s.als.scores(vec)
            scores["als"] = np.where(s.als_row >= 0, raw[np.maximum(s.als_row, 0)], np.nan)

    tmdb_counts = np.zeros(n, dtype=np.float32)
    for p in positives:
        if len(s.recs[p]):
            tmdb_counts[s.recs[p]] += 1

    g_sum = np.zeros(len(s.genre_ids))
    g_n = np.zeros(len(s.genre_ids))
    for row, r, _ in known:
        g_sum += s.genre_matrix[row] * r
        g_n += s.genre_matrix[row]
    with np.errstate(invalid="ignore", divide="ignore"):
        genre_affinity = np.where(g_n > 0, g_sum / g_n, np.nan)
    years = s.meta.year[seen] if len(seen) else np.array([])
    years = years[~np.isnan(years)]
    return UserContext(
        seen=seen,
        positives=positives,
        scores=scores,
        tmdb_counts=tmdb_counts,
        genre_affinity=genre_affinity,
        mean_year=float(years.mean()) if len(years) else np.nan,
    )


def _top(scores: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Rows of the k best allowed scores, best first."""
    s = np.where(allowed & ~np.isnan(scores), scores, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    part = np.argpartition(-s, k - 1)[:k]
    return part[np.argsort(-s[part], kind="stable")]


def candidates(
    s: Signals, ctx: UserContext, allowed: np.ndarray, sizes: dict[str, int] | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Union of each source's top rows. Returns rows and, per source, each row's rank or NaN."""
    sizes = sizes or dict(SOURCES)
    allowed = allowed.copy()
    allowed[ctx.seen] = False
    per_source: dict[str, np.ndarray] = {}
    if "als" in ctx.scores:
        per_source["als"] = _top(ctx.scores["als"], allowed, sizes["als"])
    for name in ("embed", "sparse", "feel"):
        if name in ctx.scores:
            per_source[name] = _top(ctx.scores[name], allowed, sizes[name])
    per_source["tmdb"] = _top(
        np.where(ctx.tmdb_counts > 0, ctx.tmdb_counts + 1e-3 * s.meta.log_votes, np.nan),
        allowed,
        sizes["tmdb"],
    )
    weights = np.nan_to_num(ctx.genre_affinity - 2.5, nan=0).clip(min=0)
    genre_score = s.meta.log_votes * (s.genre_matrix @ weights + 0.05)
    per_source["genre"] = _top(genre_score, allowed, sizes["genre"])

    rows = np.unique(np.concatenate([r for r in per_source.values() if len(r)] or [[]])).astype(int)
    ranks = {}
    for name, _ in SOURCES:
        rank = np.full(len(rows), np.nan, dtype=np.float32)
        if name in per_source and len(per_source[name]):
            pos = {r: i for i, r in enumerate(per_source[name])}
            rank = np.array([pos.get(r, np.nan) for r in rows], dtype=np.float32)
        ranks[name] = rank
    return rows, ranks


def _max_mean(sims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sims.shape[1] == 0:
        nan = np.full(sims.shape[0], np.nan, dtype=np.float32)
        return nan, nan
    return sims.max(axis=1), sims.mean(axis=1)


def features(
    s: Signals, ctx: UserContext, rows: np.ndarray, ranks: dict[str, np.ndarray]
) -> np.ndarray:
    """Feature matrix (len(rows) x len(FEATURES)), float32, NaN where a signal is missing."""
    n = len(rows)
    cols: dict[str, np.ndarray] = {}
    nan = np.full(n, np.nan, dtype=np.float32)
    pos = ctx.positives

    cols["cf_score"] = ctx.scores["als"][rows] if "als" in ctx.scores else nan
    cols["has_cf"] = (s.als_row[rows] >= 0).astype(np.float32)
    cols["sim_als_max"], cols["sim_als_mean"] = nan, nan
    if s.als_unit is not None:
        cand_ok = s.als_row[rows] >= 0
        pos_rows = s.als_row[pos][s.als_row[pos] >= 0]
        if cand_ok.any() and len(pos_rows):
            sims = s.als_unit[s.als_row[rows[cand_ok]]] @ s.als_unit[pos_rows].T
            mx, mn = _max_mean(sims)
            cols["sim_als_max"], cols["sim_als_mean"] = nan.copy(), nan.copy()
            cols["sim_als_max"][cand_ok], cols["sim_als_mean"][cand_ok] = mx, mn

    c = s.content
    for f in FAMILIES:
        cols[f"prof_{f}"] = ctx.scores[f][rows]
        m = c.sparse.matrices[f]
        cols[f"max_{f}"] = (
            np.asarray((m[rows] @ m[pos].T).max(axis=1).todense()).ravel() if len(pos) else nan
        )
    for name, mat in (("embed", c.dense), ("feel", c.feel)):
        if mat is None:
            cols[f"{name}_prof"], cols[f"{name}_max"] = nan, nan
            continue
        cols[f"{name}_prof"] = ctx.scores[name][rows]
        cols[f"{name}_max"] = _max_mean(mat[rows] @ mat[pos].T)[0] if len(pos) else nan

    cols["tmdb_count"] = ctx.tmdb_counts[rows]
    m = s.meta
    for name in ("log_votes", "vote_average", "log_popularity", "year", "runtime", "is_tv",
                 "is_english", "knowledge"):  # fmt: skip
        cols[name] = getattr(m, name)[rows]
    g = s.genre_matrix[rows]
    aff = np.nan_to_num(ctx.genre_affinity, nan=0)
    has = (~np.isnan(ctx.genre_affinity)).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["genre_affinity"] = (g @ aff) / (g @ has)
    cols["year_gap"] = np.abs(m.year[rows] - ctx.mean_year)
    for name, _ in SOURCES:
        cols[f"rank_{name}"] = ranks[name]
    return np.column_stack([np.asarray(cols[f], dtype=np.float32) for f in FEATURES])
