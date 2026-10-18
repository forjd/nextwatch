"""The prompt for vibe tagging. Bump PROMPT_VERSION whenever this or the vocabulary changes."""

from dataclasses import dataclass

from psycopg import Connection

from nextwatch.vibes.vocab import MAX_FORM, MAX_PLACES, MAX_THEMES, MAX_TONE, load

PROMPT_VERSION = 1


def system_prompt() -> str:
    v = load()
    return f"""You tag movies and TV series by how they feel to watch, for a personal \
recommendation engine. Your tags are compared across hundreds of thousands of titles, so use \
only the allowed values and be consistent.

For a title you know well, use your own knowledge of it, not only the description given. For a \
title you do not know, stay close to the description and do not invent details. Report which \
case applies in "knowledge": "known", "partial", or "overview-only".

Fields:
- tone: up to {MAX_TONE} of: {", ".join(v.tone)}. Strength 1 (a touch), 2 (clear), 3 (defining).
- pacing: one of {", ".join(v.pacing)}.
- attention: how much focus it asks of the viewer; one of {", ".join(v.attention)}.
- themes: up to {MAX_THEMES} of: {", ".join(v.themes)}. Most central first.
- era: one of {", ".join(v.era)}.
- places: up to {MAX_PLACES} of: {", ".join(v.places)}. Empty if none fit.
- form: up to {MAX_FORM} of: {", ".join(v.form)}. Empty if none apply.
- intensity: violence, gore, sexual_content and language, each {", ".join(v.intensity_levels)}.
- feel: one plain sentence of at most 30 words on what it is like to watch. Describe the \
experience, not the plot. No title, no names, no spoilers.

Return only the JSON object."""


@dataclass(frozen=True)
class TitleInput:
    title_id: int
    media_type: str
    text: str


def title_inputs(conn: Connection, title_ids: list[int]) -> list[TitleInput]:
    """The facts we give the model for each title, as a short plain-text block."""
    rows = conn.execute(
        """
        SELECT t.id, t.media_type, t.name, t.year, t.overview, t.tagline, t.runtime,
               t.number_of_seasons, t.original_language,
               coalesce((SELECT array_agg(g.name ORDER BY g.name) FROM title_genre tg
                         JOIN genre g ON g.id = tg.genre_id WHERE tg.title_id = t.id), '{}'),
               coalesce((SELECT array_agg(k.name ORDER BY k.name) FROM title_keyword tk
                         JOIN keyword k ON k.id = tk.keyword_id WHERE tk.title_id = t.id), '{}'),
               coalesce((SELECT array_agg(p.name ORDER BY c.cast_order) FROM credit c
                         JOIN person p ON p.id = c.person_id
                         WHERE c.title_id = t.id AND c.role = 'cast' AND c.cast_order < 5), '{}'),
               coalesce((SELECT array_agg(DISTINCT p.name) FROM credit c
                         JOIN person p ON p.id = c.person_id
                         WHERE c.title_id = t.id AND c.job IN ('Director', 'Creator')), '{}')
        FROM title t WHERE t.id = ANY(%s)
        """,
        (title_ids,),
    ).fetchall()
    out = []
    for (tid, mt, name, year, overview, tagline, runtime, seasons, lang, genres, keywords,
         cast, directors) in rows:  # fmt: skip
        kind = "TV series" if mt == "tv" else "Movie"
        lines = [f"{kind}: {name}" + (f" ({year})" if year else "")]
        if genres:
            lines.append(f"Genres: {', '.join(genres)}")
        if directors:
            lines.append(f"{'Created' if mt == 'tv' else 'Directed'} by: {', '.join(directors)}")
        if cast:
            lines.append(f"Starring: {', '.join(cast)}")
        if mt == "tv" and seasons:
            lines.append(f"Seasons: {seasons}")
        if runtime:
            lines.append(f"Runtime: {runtime} min" + (" per episode" if mt == "tv" else ""))
        if lang:
            lines.append(f"Original language: {lang}")
        if tagline:
            lines.append(f"Tagline: {tagline}")
        if overview:
            lines.append(f"Overview: {overview}")
        if keywords:
            lines.append(f"Keywords: {', '.join(keywords)}")
        out.append(TitleInput(tid, mt, "\n".join(lines)))
    return out
