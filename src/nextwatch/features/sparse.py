"""Sparse TF-IDF features, one matrix per family. See docs/plans/04-content-similarity.md.

Each family is a titles x tokens matrix with TF-IDF weights and L2-normalised rows, so the
dot product of two rows is their cosine similarity in that family. IDF does the job the old
hand-tuned weights did: a person in three films is a strong link, a common genre a weak one.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from psycopg import Connection
from scipy import sparse

log = logging.getLogger(__name__)

FAMILIES = ("cast", "director", "writer", "crew", "keyword", "genre", "meta")

WRITER_JOBS = ("Screenplay", "Writer", "Story", "Novel", "Characters")
CREW_JOBS = ("Original Music Composer", "Director of Photography", "Editor", "Producer")


def cast_weight(order: int | None) -> float:
    """Billing weight: leads count most."""
    if order is None or order >= 10:
        return 0.3
    if order >= 3:
        return 0.6
    return 1.0


@dataclass
class SparseIndex:
    ids: np.ndarray  # title.id for each row
    matrices: dict[str, sparse.csr_matrix]
    vocab: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        self.index = {int(t): i for i, t in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def profile_scores(
        self, rows: np.ndarray, weights: np.ndarray, families=FAMILIES
    ) -> dict[str, np.ndarray]:
        """Score every title against a weighted mix of rows, per family.

        Weights are normalised by their absolute sum, so families stay on a comparable scale.
        """
        total = np.abs(weights).sum()
        if len(rows) == 0 or total == 0:
            return {f: np.zeros(len(self)) for f in families}
        w = sparse.csr_matrix(weights / total)
        out = {}
        for f in families:
            m = self.matrices[f]
            profile = w @ m[rows]  # 1 x tokens
            out[f] = np.asarray((m @ profile.T).todense()).ravel()
        return out

    def combined(self, rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Equal-weight sum over families. Used for candidates only; the ranker learns weights."""
        return sum(self.profile_scores(rows, weights).values())

    def pair(self, a: int, b: np.ndarray) -> dict[str, np.ndarray]:
        """Cosine per family between row a and each row in b."""
        return {f: np.asarray((m[b] @ m[a].T).todense()).ravel() for f, m in self.matrices.items()}


def _tokens(conn: Connection) -> tuple[np.ndarray, dict[str, dict[int, dict[str, float]]]]:
    """Per family, title id -> {token: term weight}."""
    ids = np.array([r[0] for r in conn.execute("SELECT id FROM title ORDER BY id")], dtype=np.int64)
    fam: dict[str, dict[int, dict[str, float]]] = {f: defaultdict(dict) for f in FAMILIES}

    def add(family: str, tid: int, token: str, weight: float = 1.0) -> None:
        cur = fam[family][tid]
        cur[token] = max(cur.get(token, 0.0), weight)

    for tid, pid, order in conn.execute(
        "SELECT title_id, person_id, cast_order FROM credit WHERE role = 'cast'"
    ):
        add("cast", tid, f"cast:{pid}", cast_weight(order))
    for tid, pid, job in conn.execute(
        "SELECT title_id, person_id, job FROM credit WHERE role = 'crew'"
    ):
        if job in ("Director", "Creator"):
            add("director", tid, f"director:{pid}")
        elif job in WRITER_JOBS:
            add("writer", tid, f"writer:{pid}")
        elif job in CREW_JOBS:
            add("crew", tid, f"crew:{pid}")
    for tid, kid in conn.execute("SELECT title_id, keyword_id FROM title_keyword"):
        add("keyword", tid, f"kw:{kid}")
    for tid, gid in conn.execute("SELECT title_id, genre_id FROM title_genre"):
        add("genre", tid, f"genre:{gid}")
    for tid, lang, year, coll in conn.execute(
        "SELECT id, original_language, year, collection_id FROM title"
    ):
        if lang:
            add("meta", tid, f"lang:{lang}")
        if year:
            add("meta", tid, f"decade:{year // 10 * 10}s")
        if coll:
            add("meta", tid, f"collection:{coll}")
    for tid, country in conn.execute("SELECT title_id, country FROM title_country"):
        add("meta", tid, f"country:{country}")
    for tid, cid in conn.execute("SELECT title_id, company_id FROM title_company"):
        add("meta", tid, f"company:{cid}")
    return ids, fam


def tfidf(rows: list[dict[str, float]], n_docs: int) -> tuple[sparse.csr_matrix, dict[str, int]]:
    """TF-IDF with smooth IDF and L2-normalised rows. Empty rows stay zero."""
    vocab: dict[str, int] = {}
    df: dict[str, int] = defaultdict(int)
    for row in rows:
        for t in row:
            df[t] += 1
            if t not in vocab:
                vocab[t] = len(vocab)
    idf = {t: np.log((1 + n_docs) / (1 + d)) + 1 for t, d in df.items()}
    indptr, indices, data = [0], [], []
    for row in rows:
        vals = {vocab[t]: w * idf[t] for t, w in row.items()}
        norm = np.sqrt(sum(v * v for v in vals.values())) or 1.0
        for j, v in sorted(vals.items()):
            indices.append(j)
            data.append(v / norm)
        indptr.append(len(indices))
    m = sparse.csr_matrix(
        (np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)),
        shape=(len(rows), len(vocab)),
    )
    return m, vocab


def build(conn: Connection) -> SparseIndex:
    ids, fam = _tokens(conn)
    matrices, vocab = {}, {}
    for f in FAMILIES:
        rows = [fam[f].get(int(t), {}) for t in ids]
        matrices[f], vocab[f] = tfidf(rows, len(ids))
        log.info("%s: %d tokens, %d nonzeros", f, len(vocab[f]), matrices[f].nnz)
    return SparseIndex(ids, matrices, vocab)
