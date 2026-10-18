import pytest
from psycopg.types.json import Jsonb

from nextwatch import backup
from nextwatch.normalise import normalise
from tests.payloads import movie_payload, tv_payload
from tests.test_normalise import store

pytestmark = pytest.mark.db


def test_export_import_round_trip(conn):
    store(conn, "movie", 603, movie_payload())
    store(conn, "tv", 1396, tv_payload())
    normalise(conn)
    movie, show = (r[0] for r in conn.execute("SELECT id FROM title ORDER BY media_type"))
    conn.execute("INSERT INTO rating (title_id, score, source) VALUES (%s, 4.5, 'app')", (movie,))
    conn.execute(
        "INSERT INTO event (title_id, kind, context) VALUES (%s, 'watchlist', %s)",
        (show, Jsonb({"page": "for-you"})),
    )
    blob = backup.dumps(backup.export_data(conn))
    assert blob == backup.dumps(backup.export_data(conn))  # deterministic

    # Rebuild the catalogue from scratch: title ids change, the backup still applies.
    conn.execute("TRUNCATE title, rating, event CASCADE")
    normalise(conn)
    assert backup.import_data(conn, backup.loads(blob)) == (1, 1, 0)
    assert backup.import_data(conn, backup.loads(blob)) == (1, 0, 0)  # events not duplicated
    row = conn.execute(
        "SELECT t.tmdb_id, r.score FROM rating r JOIN title t ON t.id = r.title_id"
    ).fetchone()
    assert row[0] == 603 and float(row[1]) == 4.5
    assert backup.export_data(conn) == backup.loads(blob)
