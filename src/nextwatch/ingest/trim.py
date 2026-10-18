"""Shrink full TMDB payloads before storing them. See "Payload trimming" in plan 01."""

from typing import Any

PROVIDER_REGION = "GB"
RATING_REGIONS = {"GB", "US"}

# Image fields we will never show. We keep poster_path.
_DROP_KEYS = {"backdrop_path", "logo_path", "profile_path", "still_path"}


def _drop_images(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_images(v) for k, v in value.items() if k not in _DROP_KEYS}
    if isinstance(value, list):
        return [_drop_images(v) for v in value]
    return value


def trim_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)

    providers = out.get("watch/providers")
    if isinstance(providers, dict):
        region = providers.get("results", {}).get(PROVIDER_REGION)
        out["watch/providers"] = {"results": {PROVIDER_REGION: region} if region else {}}

    for key in ("release_dates", "content_ratings"):
        block = out.get(key)
        if isinstance(block, dict):
            block = dict(block)
            block["results"] = [
                r for r in block.get("results", []) if r.get("iso_3166_1") in RATING_REGIONS
            ]
            out[key] = block

    recs = out.get("recommendations")
    if isinstance(recs, dict):
        out["recommendations"] = [
            {"id": r["id"], "media_type": r.get("media_type")}
            for r in recs.get("results", [])
            if "id" in r
        ]

    return _drop_images(out)
