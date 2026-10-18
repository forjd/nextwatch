"""Recommendations for the owner: models, profile, filters and caching. See plan 08.

Uses the best model available: the ranker if trained, else content similarity, else
popularity. A fresh install with an empty catalogue still serves pages.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from psycopg import Connection
from psycopg.types.json import Jsonb

from nextwatch.cf import als
from nextwatch.features.content import ContentFeatures, ContentModel, profile_weights
from nextwatch.ranking import ranker
from nextwatch.ranking.features import candidates, features, user_context
from nextwatch.ranking.signals import Signals

log = logging.getLogger(__name__)

RELOAD_CHECK_SECONDS = 300
ON_SERVICE_KINDS = ("flatrate", "free", "ads")


@dataclass(frozen=True)
class Settings:
    providers: tuple[int, ...] = ()
    media_type: str = "both"
    mmr_lambda: float = 0.7
    popularity_penalty: float = 0.0


@dataclass(frozen=True)
class Filters:
    media_type: str | None = None  # None: use the setting
    on_my_services: bool = False
    max_runtime: int | None = None
    attention: str | None = None  # the most demanding level allowed
    no_gore: bool = False


@dataclass
class Rec:
    title_id: int
    because: int | None = None
    score: float = 0.0


ATTENTION_ORDER = ("background-friendly", "easy-watch", "engaged", "demanding")


@dataclass
class Profile:
    history: list[tuple[int, float, float]]  # (title_id, rating, unix time), oldest first
    hidden: set[int]  # rated, seen or dismissed
    watchlist: list[int]
    version: tuple


def load_profile(conn: Connection) -> Profile:
    history = [
        (t, float(s), at.timestamp())
        for t, s, at in conn.execute(
            "SELECT title_id, score, rated_at FROM rating ORDER BY rated_at, title_id"
        )
    ]
    hidden = {t for t, _, _ in history}
    hidden |= {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT title_id FROM event WHERE kind IN ('seen', 'dismiss')"
            " AND title_id IS NOT NULL"
        )
    }
    watchlist = [
        r[0]
        for r in conn.execute(
            """
            SELECT title_id FROM (
                SELECT DISTINCT ON (title_id) title_id, kind, at FROM event
                WHERE kind IN ('watchlist', 'unwatchlist') AND title_id IS NOT NULL
                ORDER BY title_id, at DESC, id DESC
            ) last WHERE kind = 'watchlist' ORDER BY at DESC
            """
        )
    ]
    version = conn.execute(
        "SELECT (SELECT count(*) FROM rating), (SELECT max(rated_at) FROM rating),"
        " (SELECT max(id) FROM event WHERE kind <> 'impression' AND kind <> 'click')"
    ).fetchone()
    return Profile(history, hidden, watchlist, tuple(version))


def load_settings(conn: Connection) -> Settings:
    raw = dict(conn.execute("SELECT key, value FROM app_setting").fetchall())
    s = Settings()
    return replace(
        s,
        providers=tuple(raw.get("providers", s.providers)),
        media_type=raw.get("media_type", s.media_type),
        mmr_lambda=float(raw.get("mmr_lambda", s.mmr_lambda)),
        popularity_penalty=float(raw.get("popularity_penalty", s.popularity_penalty)),
    )


def save_settings(conn: Connection, s: Settings) -> None:
    with conn.transaction():
        for key, value in (
            ("providers", list(s.providers)),
            ("media_type", s.media_type),
            ("mmr_lambda", s.mmr_lambda),
            ("popularity_penalty", s.popularity_penalty),
        ):
            conn.execute(
                "INSERT INTO app_setting (key, value) VALUES (%s, %s) ON CONFLICT (key)"
                " DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, Jsonb(value)),
            )


@dataclass
class Loaded:
    signals: Signals | None
    ranker: ranker.RankerModel | None
    version: tuple
    title_filters: dict[str, np.ndarray] = field(default_factory=dict)


def _data_version(conn: Connection) -> tuple:
    return tuple(
        conn.execute(
            "SELECT (SELECT max(normalised_at) FROM title),"
            " (SELECT count(*) FROM title),"
            " (SELECT max(created_at) FROM model_artifact),"
            " (SELECT max(created_at) FROM title_embedding),"
            " (SELECT max(created_at) FROM title_vibe)"
        ).fetchone()
    )


def _filter_columns(conn: Connection, s: Signals) -> dict[str, np.ndarray]:
    """Per-row columns the filters need: runtime, attention level, gore, providers."""
    n = len(s)
    attention = np.full(n, np.nan, dtype=np.float32)
    gore = np.zeros(n, dtype=bool)
    for tid, tags in conn.execute(
        "SELECT DISTINCT ON (title_id) title_id, tags FROM title_vibe"
        " ORDER BY title_id, created_at DESC"
    ):
        if (i := s.row.get(tid)) is None:
            continue
        if tags.get("attention") in ATTENTION_ORDER:
            attention[i] = ATTENTION_ORDER.index(tags["attention"])
        gore[i] = tags.get("intensity", {}).get("gore") == "strong"
    providers: dict[int, set[int]] = {}
    for tid, pid in conn.execute(
        "SELECT title_id, provider_id FROM title_provider WHERE kind = ANY(%s)",
        (list(ON_SERVICE_KINDS),),
    ):
        providers.setdefault(tid, set()).add(pid)
    return {
        "attention": attention,
        "gore": gore,
        "providers": np.array([providers.get(int(t), set()) for t in s.ids], dtype=object),
    }


class Engine:
    """Holds the loaded models and answers recommendation requests. Thread-safe for reads."""

    def __init__(self, connect: Callable[[], Connection], embedder_factory=None):
        self.connect = connect
        self.embedder_factory = embedder_factory
        self._embedder = None
        self._loaded: Loaded | None = None
        self._lock = threading.Lock()
        self._checked = 0.0
        self._cache: dict[Any, Any] = {}

    # Loading

    def load(self) -> Loaded:
        with self.connect() as conn:
            version = _data_version(conn)
            n = conn.execute("SELECT count(*) FROM title").fetchone()[0]
            if n == 0:
                log.warning("catalogue is empty; run ingest and normalise")
                return Loaded(None, None, version)
            content = ContentFeatures.load(conn)
            try:
                als_model = als.load(conn)
            except RuntimeError:
                als_model = None
            signals = Signals.load(conn, content, als_model)
            try:
                rank_model = ranker.load(conn)
            except RuntimeError:
                rank_model = None
            loaded = Loaded(signals, rank_model, version, _filter_columns(conn, signals))
        log.info(
            "engine loaded: %d titles, als=%s, ranker=%s, embeddings=%s, feel=%s",
            n, als_model is not None, rank_model is not None,
            content.dense is not None, content.feel is not None,
        )  # fmt: skip
        return loaded

    def loaded(self) -> Loaded:
        """Current models, reloading in place when data or models changed."""
        now = time.monotonic()
        with self._lock:
            if self._loaded is None:
                self._loaded = self.load()
                self._checked = now
            elif now - self._checked > RELOAD_CHECK_SECONDS:
                self._checked = now
                with self.connect() as conn:
                    changed = _data_version(conn) != self._loaded.version
                if changed:
                    self._loaded = self.load()
                    self._cache.clear()
            return self._loaded

    def reload(self) -> None:
        with self._lock:
            self._loaded = self.load()
            self._cache.clear()

    def status(self) -> dict[str, bool]:
        ld = self.loaded()
        s = ld.signals
        return {
            "catalogue": s is not None,
            "als": bool(s and s.als is not None),
            "ranker": ld.ranker is not None,
            "embeddings": bool(s and s.content.dense is not None),
            "feel": bool(s and s.content.feel is not None),
        }

    # Filters

    def _allowed(self, ld: Loaded, profile: Profile, f: Filters, st: Settings) -> np.ndarray:
        s = ld.signals
        ok = np.ones(len(s), dtype=bool)
        hidden = [s.row[t] for t in profile.hidden if t in s.row]
        ok[hidden] = False
        media = f.media_type or st.media_type
        if media in ("movie", "tv"):
            ok &= s.meta.is_tv == (1.0 if media == "tv" else 0.0)
        if f.max_runtime:
            ok &= ~(s.meta.runtime > f.max_runtime)  # unknown runtime passes
        if f.attention in ATTENTION_ORDER:
            level = ATTENTION_ORDER.index(f.attention)
            ok &= ~(ld.title_filters["attention"] > level)
        if f.no_gore:
            ok &= ~ld.title_filters["gore"]
        if f.on_my_services:
            mine = set(st.providers)
            ok &= np.array([bool(p & mine) for p in ld.title_filters["providers"]])
        return ok

    # Recommendations

    def for_you(self, profile: Profile, f: Filters, st: Settings, k: int = 40) -> list[Rec]:
        key = ("for_you", profile.version, f, st, k)
        if key in self._cache:
            return self._cache[key]
        ld = self.loaded()
        if ld.signals is None:
            return []
        out = self._rank(ld, profile.history, self._allowed(ld, profile, f, st), st, k)
        self._cache[key] = out
        return out

    def similar(self, title_id: int, profile: Profile, st: Settings, k: int = 20) -> list[Rec]:
        ld = self.loaded()
        s = ld.signals
        if s is None or title_id not in s.row:
            return []
        allowed = np.ones(len(s), dtype=bool)
        allowed[s.row[title_id]] = False
        recs = self._rank(ld, [(title_id, 5.0, 0.0)], allowed, replace(st, media_type="both"), k)
        for r in recs:
            r.because = None
        return recs

    def _rank(self, ld: Loaded, history, allowed: np.ndarray, st: Settings, k: int) -> list[Rec]:
        s = ld.signals
        ctx = user_context(s, history)
        opts = ranker.RerankOptions(
            mmr_lambda=st.mmr_lambda,
            popularity_penalty=st.popularity_penalty,
            media_type=st.media_type,
        )
        if not history:
            rows = np.flatnonzero(allowed)
            scores = s.meta.log_votes[rows]
        elif ld.ranker is not None:
            rows, ranks = candidates(s, ctx, allowed)
            scores = ld.ranker.predict(features(s, ctx, rows, ranks)) if len(rows) else []
        else:
            model = ContentModel("content", s.cat, s.content, use_vibe=True)
            all_scores = model.score(profile_weights(history))
            rows = np.flatnonzero(allowed)
            top = rows[np.argsort(-all_scores[rows], kind="stable")[: k * 10]]
            rows, scores = top, all_scores[top]
        if not len(rows):
            return []
        picked = ranker.rerank(s, np.asarray(rows), np.asarray(scores, dtype=float), k, opts)
        score_of = dict(zip(np.asarray(rows).tolist(), np.asarray(scores).tolist(), strict=True))
        return [
            Rec(int(s.ids[r]), ranker.explain(s, ctx.positives, r), float(score_of[r]))
            for r in picked
        ]

    def mood(self, query: str, profile: Profile, f: Filters, st: Settings, k: int = 40):
        """Free-text mood search over feel embeddings (or title embeddings), nudged by taste."""
        ld = self.loaded()
        s = ld.signals
        if s is None or not query.strip():
            return []
        matrix = s.content.feel if s.content.feel is not None else s.content.dense
        if matrix is None:
            return []
        if self._embedder is None:
            if self.embedder_factory is None:
                return []
            self._embedder = self.embedder_factory()
        q = self._embedder([query]).astype(np.float32)[0]
        q /= np.linalg.norm(q) or 1
        sim = matrix @ q
        if profile.history:
            taste = ContentModel("content", s.cat, s.content).score(
                profile_weights(profile.history)
            )
            span = np.ptp(taste) or 1
            sim = sim + 0.3 * (taste - taste.min()) / span * np.ptp(sim)
        allowed = self._allowed(ld, profile, f, st)
        rows = np.flatnonzero(allowed)
        top = rows[np.argsort(-sim[rows], kind="stable")[: k * 5]]
        opts = ranker.RerankOptions(mmr_lambda=st.mmr_lambda, media_type="both")
        picked = ranker.rerank(s, top, sim[top].astype(float), k, opts)
        return [Rec(int(s.ids[r]), None, float(sim[r])) for r in picked]

    def predicted(self, title_ids: list[int], profile: Profile) -> dict[int, float]:
        """A taste score for given titles (watchlist ordering). Higher is better."""
        ld = self.loaded()
        s = ld.signals
        if s is None or not profile.history:
            return {}
        scores = ContentModel("content", s.cat, s.content).score(profile_weights(profile.history))
        return {t: float(scores[s.row[t]]) for t in title_ids if t in s.row}

    # Onboarding

    def quick_rate_batch(self, conn: Connection, profile: Profile, skipped: set[int], n=10):
        """Well-known titles that fill gaps in what we know: genres and decades not yet rated."""
        ld = self.loaded()
        s = ld.signals
        if s is None:
            return []
        rated = {t for t, _, _ in profile.history} | profile.hidden | skipped
        pool = [int(t) for t in s.ids[np.argsort(-s.meta.log_votes, kind="stable")[:800]]]
        pool = [t for t in pool if t not in rated]
        known_genres = np.zeros(len(s.genre_ids))
        decades: dict[int, int] = {}
        for t, _, _ in profile.history:
            if t in s.row:
                known_genres += s.genre_matrix[s.row[t]]
                y = s.meta.year[s.row[t]]
                if not np.isnan(y):
                    decades[int(y) // 10] = decades.get(int(y) // 10, 0) + 1
        chosen: list[int] = []
        tv = 0
        for _ in range(n):
            best, best_score = None, -1e9
            for rank, t in enumerate(pool):
                if t in chosen:
                    continue
                row = s.row[t]
                novelty = float((s.genre_matrix[row] / (1 + known_genres)).sum())
                y = s.meta.year[row]
                novelty += 0 if np.isnan(y) else 1 / (1 + decades.get(int(y) // 10, 0))
                is_tv = s.meta.is_tv[row] == 1
                balance = 0.5 if is_tv and tv < (len(chosen) + 1) / 2 else 0
                score = novelty + balance - rank / 400
                if score > best_score:
                    best, best_score = t, score
            if best is None:
                break
            chosen.append(best)
            row = s.row[best]
            known_genres += s.genre_matrix[row]
            tv += int(s.meta.is_tv[row] == 1)
            y = s.meta.year[row]
            if not np.isnan(y):
                decades[int(y) // 10] = decades.get(int(y) // 10, 0) + 1
        return chosen
