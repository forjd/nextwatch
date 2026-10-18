"""Dense text embeddings of each title. See docs/plans/04-content-similarity.md."""

import hashlib
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from psycopg import Connection

log = logging.getLogger(__name__)

MODEL = "bge-small-en-v1.5"
FASTEMBED_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
BATCH = 256

Embedder = Callable[[Sequence[str]], np.ndarray]


def fastembed_embedder(cache_dir: Path) -> Embedder:
    """The real model, on CPU through ONNX. Downloads ~70 MB on first use."""
    from fastembed import TextEmbedding

    model = TextEmbedding(FASTEMBED_NAME, cache_dir=str(cache_dir))

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.vstack(list(model.embed(list(texts), batch_size=64)))

    return embed


def title_text(
    name: str,
    year: int | None,
    genres: Sequence[str],
    tagline: str | None,
    overview: str | None,
    keywords: Sequence[str],
) -> str:
    head = f"{name} ({year})" if year else name
    parts = [f"{head}."]
    if genres:
        parts.append(f"{', '.join(genres)}.")
    if tagline:
        parts.append(tagline)
    if overview:
        parts.append(overview)
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}.")
    return " ".join(parts)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _texts(conn: Connection) -> list[tuple[int, str]]:
    rows = conn.execute(
        """
        SELECT t.id, t.name, t.year, t.tagline, t.overview,
               coalesce((SELECT array_agg(g.name ORDER BY g.name) FROM title_genre tg
                         JOIN genre g ON g.id = tg.genre_id WHERE tg.title_id = t.id), '{}'),
               coalesce((SELECT array_agg(k.name ORDER BY k.name) FROM title_keyword tk
                         JOIN keyword k ON k.id = tk.keyword_id WHERE tk.title_id = t.id), '{}')
        FROM title t ORDER BY t.id
        """
    ).fetchall()
    return [(r[0], title_text(r[1], r[2], r[5], r[3], r[4], r[6])) for r in rows]


def embed_titles(conn: Connection, embed: Embedder, model: str = MODEL) -> int:
    """Embed every title whose text changed since it was last embedded. Returns the count."""
    existing = dict(
        conn.execute(
            "SELECT title_id, source_hash FROM title_embedding WHERE model = %s", (model,)
        ).fetchall()
    )
    todo = [(tid, text) for tid, text in _texts(conn) if existing.get(tid) != text_hash(text)]
    log.info("%d titles to embed", len(todo))
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        vectors = embed([text for _, text in batch]).astype(np.float32)
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO title_embedding (title_id, model, embedding, source_hash)
                VALUES (%s, %s, %s::halfvec, %s)
                ON CONFLICT (title_id, model) DO UPDATE SET embedding = EXCLUDED.embedding,
                    source_hash = EXCLUDED.source_hash, created_at = now()
                """,
                [
                    (tid, model, _vec(v), text_hash(text))
                    for (tid, text), v in zip(batch, vectors, strict=True)
                ],
            )
        if (start // BATCH) % 20 == 0:
            log.info("embedded %d / %d", start + len(batch), len(todo))
    return len(todo)


def _vec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"


def load_matrix(conn: Connection, ids: np.ndarray, model: str = MODEL) -> np.ndarray:
    """Unit-normalised embeddings aligned to ids. Titles without one get a zero row."""
    index = {int(t): i for i, t in enumerate(ids)}
    out = np.zeros((len(ids), DIM), dtype=np.float32)
    for tid, emb in conn.execute(
        "SELECT title_id, embedding::text FROM title_embedding WHERE model = %s", (model,)
    ):
        if (i := index.get(tid)) is not None:
            out[i] = np.array(emb.strip("[]").split(","), dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)
