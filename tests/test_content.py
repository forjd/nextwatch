import numpy as np
import pytest

from nextwatch.evaluation.models import Catalogue
from nextwatch.features import embed, sparse
from nextwatch.features.content import ContentFeatures, ContentModel, profile_weights
from nextwatch.normalise import normalise
from tests.payloads import movie_payload, tv_payload
from tests.test_normalise import store


def test_tfidf_rows_are_unit_and_rare_tokens_weigh_more():
    m, vocab = sparse.tfidf([{"a": 1, "rare": 1}, {"a": 1}, {"a": 1}, {}], 4)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    assert np.allclose(norms, [1, 1, 1, 0])
    row = m[0].toarray().ravel()
    assert row[vocab["rare"]] > row[vocab["a"]]


def test_cast_weight():
    assert [sparse.cast_weight(o) for o in (0, 2, 3, 9, 10, None)] == [1, 1, 0.6, 0.6, 0.3, 0.3]


def test_profile_weights_decay_and_sign():
    day = 86400
    w = dict(profile_weights([(1, 5.0, 0), (2, 1.0, 365 * day), (3, 3.0, 365 * day)]))
    assert w[1] == pytest.approx(1.0)  # +2, one year old: halved
    assert w[2] == -2.0 and 3 not in w


def test_title_text():
    text = embed.title_text("Alien", 1979, ["Horror"], "In space...", "A crew.", ["alien"])
    assert text == "Alien (1979). Horror. In space... A crew. Keywords: alien."
    assert embed.title_text("X", None, [], None, None, []) == "X."


def fake_embedder(texts):
    """Deterministic vectors: titles sharing words get similar vectors."""
    out = np.zeros((len(texts), embed.DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for word in t.lower().split():
            out[i, hash(word.strip(".,")) % embed.DIM] += 1
    return out


@pytest.fixture
def world(conn):
    """Three Matrix-like movies sharing a director, an unrelated TV show, and a lone movie."""
    store(conn, "movie", 603, movie_payload())
    store(conn, "movie", 604, movie_payload(604, title="The Matrix Reloaded"))
    store(conn, "movie", 605, movie_payload(605, title="The Matrix Revolutions"))
    store(conn, "tv", 1396, tv_payload())
    store(conn, "movie", 862, {"id": 862, "title": "Toy Story", "overview": "Toys come alive.",
                               "genres": [{"id": 16, "name": "Animation"}]})  # fmt: skip
    normalise(conn)
    return {k: v for v, k in conn.execute("SELECT id, tmdb_id FROM title")}


@pytest.mark.db
def test_sparse_index_and_similar(conn, world):
    idx = sparse.build(conn)
    assert len(idx) == 5
    a, b = idx.index[world[603]], idx.index[world[604]]
    sims = idx.pair(a, np.array([b, idx.index[world[862]]]))
    assert sims["director"][0] == pytest.approx(1.0) and sims["director"][1] == 0
    model = ContentModel("content-sparse", Catalogue.load(conn), ContentFeatures(idx, None))
    assert set(model.similar(world[603], 2)) == {world[604], world[605]}


@pytest.mark.db
def test_embed_titles_is_incremental(conn, world):
    assert embed.embed_titles(conn, fake_embedder) == 5
    assert embed.embed_titles(conn, fake_embedder) == 0
    conn.execute("UPDATE title SET overview = 'Changed.' WHERE tmdb_id = 862")
    assert embed.embed_titles(conn, fake_embedder) == 1

    ids = np.array(sorted(world.values()))
    matrix = embed.load_matrix(conn, ids)
    assert matrix.shape == (5, embed.DIM)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-3)

    # The HNSW index answers nearest-neighbour queries.
    row = conn.execute(
        """
        SELECT t.tmdb_id FROM title_embedding e JOIN title t ON t.id = e.title_id
        WHERE e.model = %s AND t.tmdb_id <> 603
        ORDER BY e.embedding <=> (SELECT embedding FROM title_embedding
                                  WHERE title_id = %s AND model = %s)
        LIMIT 1
        """,
        (embed.MODEL, world[603], embed.MODEL),
    ).fetchone()
    assert row[0] in (604, 605)


@pytest.mark.db
def test_content_model_recommends_from_profile(conn, world):
    embed.embed_titles(conn, fake_embedder)
    feats = ContentFeatures.load(conn)
    assert feats.dense is not None
    movies = Catalogue.load(conn, "movie")
    model = ContentModel("content", movies, feats)
    recs = model.recommend([(world[603], 5.0, 0), (world[862], 1.0, 0)], {world[603]}, 2)
    assert set(recs) == {world[604], world[605]}
    assert all(movies.media_type[t] == "movie" for t in model.similar(world[603], 4))
