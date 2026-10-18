"""The nextwatch web app: FastAPI, Jinja2 and HTMX. See docs/plans/08-app.md."""

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from starlette.middleware.sessions import SessionMiddleware

from nextwatch.app import imports
from nextwatch.app.engine import (
    ATTENTION_ORDER,
    Engine,
    Filters,
    Rec,
    Settings,
    load_profile,
    load_settings,
    save_settings,
)

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
POSTER_URL = "https://image.tmdb.org/t/p/w342"
SESSION_DAYS = 30
OPEN_PATHS = ("/login", "/healthz", "/static/")
EVENT_KINDS = ("seen", "dismiss", "watchlist", "unwatchlist")
SCORES = [x / 2 for x in range(1, 11)]


def create_app(
    database_url: str,
    password: str,
    session_secret: str,
    embedder_factory=None,
) -> FastAPI:
    if not password or not session_secret:
        raise RuntimeError("APP_PASSWORD and SESSION_SECRET must be set")
    pool = ConnectionPool(database_url, min_size=1, max_size=5, open=False,
                          kwargs={"autocommit": True})  # fmt: skip
    engine = Engine(pool.connection, embedder_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        pool.open()
        try:
            yield
        finally:
            pool.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.engine = engine
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(poster_url=POSTER_URL, scores=SCORES)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        if path.startswith(OPEN_PATHS) or request.session.get("ok"):
            return await call_next(request)
        if request.headers.get("HX-Request"):
            return Response(status_code=401, headers={"HX-Redirect": "/login"})
        return RedirectResponse("/login", status_code=303)

    # Added after require_login so it runs first and request.session exists.
    app.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        max_age=SESSION_DAYS * 86400,
        same_site="lax",
        https_only=False,
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx)

    # Data helpers

    def cards(conn: Connection, recs: list[Rec]) -> list[dict]:
        ids = [r.title_id for r in recs] + [r.because for r in recs if r.because]
        if not ids:
            return []
        rows = {
            r[0]: r
            for r in conn.execute(
                "SELECT t.id, t.media_type, t.name, t.year, t.poster_path, t.runtime,"
                " r.score FROM title t LEFT JOIN rating r ON r.title_id = t.id"
                " WHERE t.id = ANY(%s)",
                (ids,),
            )
        }
        watch = set(load_profile(conn).watchlist)
        out = []
        for r in recs:
            if r.title_id not in rows:
                continue
            tid, mt, name, year, poster, runtime, score = rows[r.title_id]
            because = rows.get(r.because)
            out.append({
                "id": tid, "media_type": mt, "name": name, "year": year, "poster": poster,
                "runtime": runtime, "score": float(score) if score is not None else None,
                "watchlisted": tid in watch,
                "because": {"id": because[0], "name": because[2]} if because else None,
            })  # fmt: skip
        return out

    def card(conn: Connection, title_id: int) -> dict | None:
        found = cards(conn, [Rec(title_id)])
        return found[0] if found else None

    def log_impressions(conn: Connection, page: str, items: list[dict]) -> None:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO event (title_id, kind, context) VALUES (%s, 'impression', %s)",
                [(c["id"], Jsonb({"page": page, "rank": i})) for i, c in enumerate(items)],
            )

    def parse_filters(request: Request) -> Filters:
        q = request.query_params
        media = q.get("type") if q.get("type") in ("movie", "tv", "both") else None
        runtime = q.get("runtime")
        attention = q.get("attention")
        return Filters(
            media_type=media,
            on_my_services=q.get("services") == "1",
            max_runtime=int(runtime) if runtime and runtime.isdigit() else None,
            attention=attention if attention in ATTENTION_ORDER else None,
            no_gore=q.get("no_gore") == "1",
        )

    # Auth

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        return render(request, "login.html", error=None)

    @app.post("/login")
    def login(request: Request, password_in: Annotated[str, Form(alias="password")]):
        if hmac.compare_digest(password_in.encode(), password.encode()):
            request.session["ok"] = True
            return RedirectResponse("/", status_code=303)
        return render(request, "login.html", error="Wrong password.")

    @app.post("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/healthz")
    def healthz():
        try:
            with pool.connection() as conn:
                conn.execute("SELECT 1")
        except Exception as e:  # noqa: BLE001 - report any database failure
            return JSONResponse({"ok": False, "error": str(e)}, status_code=503)
        return {"ok": True, "models": engine.status()}

    # Pages

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        with pool.connection() as conn:
            rated = conn.execute("SELECT count(*) FROM rating").fetchone()[0]
        return RedirectResponse("/for-you" if rated else "/start", status_code=303)

    @app.get("/for-you", response_class=HTMLResponse)
    def for_you(request: Request):
        f = parse_filters(request)
        with pool.connection() as conn:
            profile, st = load_profile(conn), load_settings(conn)
            items = cards(conn, engine.for_you(profile, f, st))
            log_impressions(conn, "for-you", items)
        return render(request, "for_you.html", items=items, f=f, st=st,
                      attention_levels=ATTENTION_ORDER, status=engine.status())  # fmt: skip

    @app.get("/title/{title_id}", response_class=HTMLResponse)
    def title_page(request: Request, title_id: int):
        with pool.connection() as conn:
            t = conn.execute(
                "SELECT t.id, t.media_type, t.tmdb_id, t.name, t.year, t.overview, t.tagline,"
                " t.runtime, t.number_of_seasons, t.poster_path, t.certification_gb,"
                " t.vote_average, t.vote_count, r.score"
                " FROM title t LEFT JOIN rating r ON r.title_id = t.id WHERE t.id = %s",
                (title_id,),
            ).fetchone()
            if t is None:
                return HTMLResponse("Not found", status_code=404)
            keys = ("id", "media_type", "tmdb_id", "name", "year", "overview", "tagline",
                    "runtime", "seasons", "poster", "cert", "vote_average", "vote_count",
                    "score")  # fmt: skip
            title = dict(zip(keys, t, strict=True))
            title["genres"] = [r[0] for r in conn.execute(
                "SELECT g.name FROM title_genre tg JOIN genre g ON g.id = tg.genre_id"
                " WHERE tg.title_id = %s ORDER BY g.name", (title_id,))]  # fmt: skip
            title["providers"] = conn.execute(
                "SELECT p.name, tp.kind FROM title_provider tp"
                " JOIN provider p ON p.id = tp.provider_id WHERE tp.title_id = %s"
                " ORDER BY tp.kind, tp.display_priority", (title_id,)).fetchall()  # fmt: skip
            title["crew"] = conn.execute(
                "SELECT DISTINCT p.name, c.job FROM credit c JOIN person p ON p.id = c.person_id"
                " WHERE c.title_id = %s AND c.job IN ('Director', 'Creator', 'Screenplay',"
                " 'Writer', 'Novel') ORDER BY c.job, p.name", (title_id,)).fetchall()  # fmt: skip
            title["cast"] = conn.execute(
                "SELECT p.name, c.character FROM credit c JOIN person p ON p.id = c.person_id"
                " WHERE c.title_id = %s AND c.role = 'cast' ORDER BY c.cast_order LIMIT 10",
                (title_id,)).fetchall()  # fmt: skip
            vibe = conn.execute(
                "SELECT tags, feel FROM title_vibe WHERE title_id = %s"
                " ORDER BY created_at DESC LIMIT 1", (title_id,)).fetchone()  # fmt: skip
            title["vibe"] = {"tags": vibe[0], "feel": vibe[1]} if vibe else None
            title["watchlisted"] = title_id in load_profile(conn).watchlist
            conn.execute("INSERT INTO event (title_id, kind) VALUES (%s, 'click')", (title_id,))
            profile, st = load_profile(conn), load_settings(conn)
            similar = cards(conn, engine.similar(title_id, profile, st))
        return render(request, "title.html", t=title, similar=similar)

    @app.get("/mood", response_class=HTMLResponse)
    def mood(request: Request, q: str = ""):
        f = parse_filters(request)
        with pool.connection() as conn:
            profile, st = load_profile(conn), load_settings(conn)
            items = cards(conn, engine.mood(q, profile, f, st)) if q else []
        return render(request, "mood.html", q=q, items=items, f=f,
                      attention_levels=ATTENTION_ORDER)  # fmt: skip

    @app.get("/search", response_class=HTMLResponse)
    def search(request: Request, q: str = ""):
        items: list[dict] = []
        with pool.connection() as conn:
            if q.strip():
                ids = [r[0] for r in conn.execute(
                    "SELECT id FROM title WHERE name %% %(q)s OR name ILIKE %(like)s"
                    " ORDER BY (lower(name) = lower(%(q)s)) DESC, similarity(name, %(q)s) DESC,"
                    " vote_count DESC LIMIT 30",
                    {"q": q.strip(), "like": f"%{q.strip()}%"})]  # fmt: skip
                items = cards(conn, [Rec(i) for i in ids])
        if request.headers.get("HX-Request"):
            return render(request, "_grid.html", items=items)
        return render(request, "search.html", q=q, items=items)

    @app.get("/watchlist", response_class=HTMLResponse)
    def watchlist(request: Request):
        with pool.connection() as conn:
            profile = load_profile(conn)
            pred = engine.predicted(profile.watchlist, profile)
            ordered = sorted(profile.watchlist, key=lambda t: -pred.get(t, 0))
            items = cards(conn, [Rec(t) for t in ordered])
        return render(request, "watchlist.html", items=items)

    @app.get("/ratings", response_class=HTMLResponse)
    def ratings(request: Request):
        with pool.connection() as conn:
            ids = [r[0] for r in conn.execute("SELECT title_id FROM rating ORDER BY rated_at DESC")]
            items = cards(conn, [Rec(i) for i in ids])
        return render(request, "ratings.html", items=items)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, saved: int = 0):
        with pool.connection() as conn:
            st = load_settings(conn)
            providers = conn.execute(
                "SELECT p.id, p.name, count(*) FROM provider p"
                " JOIN title_provider tp ON tp.provider_id = p.id"
                " WHERE tp.kind IN ('flatrate', 'free', 'ads')"
                " GROUP BY p.id, p.name ORDER BY count(*) DESC LIMIT 60"
            ).fetchall()
        return render(request, "settings.html", st=st, providers=providers, saved=saved)

    @app.post("/settings")
    async def save_settings_route(request: Request):
        form = await request.form()
        st = Settings(
            providers=tuple(int(p) for p in form.getlist("providers")),
            media_type=form.get("media_type") if form.get("media_type") in ("movie", "tv")
            else "both",
            mmr_lambda=min(1.0, max(0.0, float(form.get("mmr_lambda") or 0.7))),
            popularity_penalty=min(1.0, max(0.0, float(form.get("popularity_penalty") or 0))),
        )  # fmt: skip
        with pool.connection() as conn:
            save_settings(conn, st)
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.get("/start", response_class=HTMLResponse)
    def start(request: Request):
        skipped = set(request.session.get("skipped", []))
        with pool.connection() as conn:
            profile = load_profile(conn)
            ids = engine.quick_rate_batch(conn, profile, skipped, n=12)
            items = cards(conn, [Rec(i) for i in ids])
        return render(request, "start.html", items=items, rated=len(profile.history))

    @app.post("/start/skip/{title_id}", response_class=HTMLResponse)
    def skip(request: Request, title_id: int):
        request.session["skipped"] = [*request.session.get("skipped", []), title_id][-500:]
        return HTMLResponse("")

    @app.get("/import", response_class=HTMLResponse)
    def import_page(request: Request):
        return render(request, "import.html", result=None, error=None)

    @app.post("/import", response_class=HTMLResponse)
    async def import_upload(request: Request, file: UploadFile):
        text = (await file.read()).decode("utf-8", errors="replace")
        with pool.connection() as conn:
            try:
                result = imports.import_csv(conn, text)
            except ValueError as e:
                return render(request, "import.html", result=None, error=str(e))
        return render(request, "import.html", result=result, error=None)

    # Card actions (HTMX): each returns the updated card

    @app.post("/rate/{title_id}", response_class=HTMLResponse)
    def rate(
        request: Request,
        title_id: int,
        score: Annotated[float, Form()],
        source: Annotated[str, Form()] = "app",
    ):
        if score not in SCORES:
            return HTMLResponse("bad score", status_code=400)
        source = "onboarding" if source == "onboarding" else "app"
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO rating (title_id, score, source) VALUES (%s, %s, %s)"
                " ON CONFLICT (title_id) DO UPDATE SET score = EXCLUDED.score, rated_at = now()",
                (title_id, score, source),
            )
            c = card(conn, title_id)
        return render(request, "_card.html", c=c)

    @app.post("/unrate/{title_id}", response_class=HTMLResponse)
    def unrate(request: Request, title_id: int):
        with pool.connection() as conn:
            conn.execute("DELETE FROM rating WHERE title_id = %s", (title_id,))
            c = card(conn, title_id)
        return render(request, "_card.html", c=c)

    @app.post("/event/{title_id}/{kind}", response_class=HTMLResponse)
    def event(request: Request, title_id: int, kind: str):
        if kind not in EVENT_KINDS:
            return HTMLResponse("bad event", status_code=400)
        with pool.connection() as conn:
            conn.execute("INSERT INTO event (title_id, kind) VALUES (%s, %s)", (title_id, kind))
            c = card(conn, title_id)
        if kind == "dismiss":
            return HTMLResponse("")  # the card disappears
        return render(request, "_card.html", c=c)

    @app.get("/about", response_class=HTMLResponse)
    def about(request: Request):
        return render(request, "about.html", status=engine.status())

    return app


def app_from_env() -> FastAPI:
    """Entry point for uvicorn: `uvicorn --factory nextwatch.app.main:app_from_env`."""
    from nextwatch.config import load_settings as load_config
    from nextwatch.features.embed import fastembed_embedder

    cfg = load_config()
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    return create_app(
        cfg.database_url,
        os.environ.get("APP_PASSWORD", ""),
        os.environ.get("SESSION_SECRET", ""),
        embedder_factory=lambda: fastembed_embedder(data_dir / "models"),
    )
