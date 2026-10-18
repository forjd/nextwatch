"""The vibe vocabulary, the JSON schema built from it, and validation of model output."""

from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import yaml

MAX_TONE = 4
MAX_THEMES = 6
MAX_PLACES = 2
MAX_FORM = 4
MAX_FEEL_WORDS = 40
INTENSITY_KINDS = ("violence", "gore", "sexual_content", "language")


@dataclass(frozen=True)
class Vocabulary:
    tone: tuple[str, ...]
    pacing: tuple[str, ...]
    attention: tuple[str, ...]
    themes: tuple[str, ...]
    era: tuple[str, ...]
    places: tuple[str, ...]
    form: tuple[str, ...]
    intensity_levels: tuple[str, ...]
    knowledge: tuple[str, ...]


@cache
def load() -> Vocabulary:
    raw = yaml.safe_load(resources.files("nextwatch.vibes").joinpath("vocabulary.yaml").read_text())
    return Vocabulary(**{k: tuple(v) for k, v in raw.items()})


def _enum(values: tuple[str, ...]) -> dict:
    return {"type": "string", "enum": list(values)}


def json_schema(v: Vocabulary | None = None) -> dict[str, Any]:
    """Strict JSON schema: every property required, no extra properties."""
    v = v or load()
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "tone",
            "pacing",
            "attention",
            "themes",
            "era",
            "places",
            "form",
            "intensity",
            "knowledge",
            "feel",
        ],  # fmt: skip
        "properties": {
            "tone": {
                "type": "array",
                "description": f"Up to {MAX_TONE} tones, strongest first.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["value", "strength"],
                    "properties": {
                        "value": _enum(v.tone),
                        "strength": {"type": "integer", "enum": [1, 2, 3]},
                    },
                },
            },
            "pacing": _enum(v.pacing),
            "attention": _enum(v.attention),
            "themes": {
                "type": "array",
                "description": f"Up to {MAX_THEMES}, most central first.",
                "items": _enum(v.themes),
            },
            "era": _enum(v.era),
            "places": {
                "type": "array",
                "description": f"Up to {MAX_PLACES}.",
                "items": _enum(v.places),
            },
            "form": {
                "type": "array",
                "description": f"Up to {MAX_FORM}. Empty if none apply.",
                "items": _enum(v.form),
            },
            "intensity": {
                "type": "object",
                "additionalProperties": False,
                "required": list(INTENSITY_KINDS),
                "properties": {k: _enum(v.intensity_levels) for k in INTENSITY_KINDS},
            },
            "knowledge": _enum(v.knowledge),
            "feel": {
                "type": "string",
                "description": "One sentence, at most 30 words: what it is like to watch.",
            },
        },
    }


class InvalidTags(ValueError):
    pass


def _dedupe(values):
    seen, out = set(), []
    for x in values:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def validate(data: Any, v: Vocabulary | None = None) -> dict[str, Any]:
    """Check model output against the vocabulary. Trim over-long lists rather than reject.

    Raises InvalidTags for anything outside the vocabulary or missing.
    """
    v = v or load()
    if not isinstance(data, dict):
        raise InvalidTags("not an object")
    try:
        tone = []
        for t in data["tone"]:
            if t["value"] not in v.tone or t["strength"] not in (1, 2, 3):
                raise InvalidTags(f"bad tone {t}")
            if t["value"] not in {x["value"] for x in tone}:
                tone.append({"value": t["value"], "strength": int(t["strength"])})

        def pick(key: str, allowed: tuple[str, ...], limit: int) -> list[str]:
            values = _dedupe(data[key])
            bad = [x for x in values if x not in allowed]
            if bad:
                raise InvalidTags(f"bad {key}: {bad}")
            return values[:limit]

        def one(key: str, allowed: tuple[str, ...]) -> str:
            if data[key] not in allowed:
                raise InvalidTags(f"bad {key}: {data[key]!r}")
            return data[key]

        intensity = {}
        for k in INTENSITY_KINDS:
            if data["intensity"][k] not in v.intensity_levels:
                raise InvalidTags(f"bad intensity {k}")
            intensity[k] = data["intensity"][k]
        feel = " ".join(str(data["feel"]).split())
        if not feel:
            raise InvalidTags("empty feel")
        words = feel.split(" ")
        if len(words) > MAX_FEEL_WORDS:
            feel = " ".join(words[:MAX_FEEL_WORDS]).rstrip(",;") + "."
        return {
            "tone": tone[:MAX_TONE],
            "pacing": one("pacing", v.pacing),
            "attention": one("attention", v.attention),
            "themes": pick("themes", v.themes, MAX_THEMES),
            "era": one("era", v.era),
            "places": pick("places", v.places, MAX_PLACES),
            "form": pick("form", v.form, MAX_FORM),
            "intensity": intensity,
            "knowledge": one("knowledge", v.knowledge),
            "feel": feel,
        }
    except (KeyError, TypeError) as e:
        raise InvalidTags(f"missing or malformed field: {e!r}") from e
