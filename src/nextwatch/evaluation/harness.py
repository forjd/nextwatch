"""Run a recommender on the evaluation tasks and record the result. See plan 03."""

import datetime as dt
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from psycopg import Connection
from psycopg.types.json import Jsonb

from nextwatch.evaluation import metrics
from nextwatch.evaluation.data import EvalUser
from nextwatch.evaluation.models import Catalogue, Recommender

log = logging.getLogger(__name__)

K = 20


@dataclass(frozen=True)
class GoldenCase:
    seed: str
    must: tuple[str, ...]
    must_not: tuple[str, ...]


def load_golden(path: Path) -> list[GoldenCase]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or []
    return [
        GoldenCase(c["seed"], tuple(c.get("must", [])), tuple(c.get("must_not", []))) for c in raw
    ]


def _key(ref: str) -> tuple[str, int]:
    media_type, tmdb_id = ref.split("/")
    return media_type, int(tmdb_id)


def run_t1(model: Recommender, users: list[EvalUser], k: int = K) -> dict[str, Any]:
    """ "For you": rank unseen titles from history, score against held-out positives."""
    recall, ndcg, lists = [], [], []
    for u in users:
        history = list(zip(u.history.tolist(), u.history_ratings.tolist(), strict=True))
        ranked = model.recommend(history, set(u.history.tolist()), k)
        lists.append(ranked)
        recall.append(metrics.recall_at_k(ranked, u.holdout_positives, k))
        ndcg.append(metrics.ndcg_at_k(ranked, u.holdout_positives, k))
    return {"recall": recall, "ndcg": ndcg, "lists": lists}


def run_t2(model: Recommender, users: list[EvalUser], k: int = K, seed: int = 0) -> list[float]:
    """ "More like this": seed is a random history positive, targets are held-out positives."""
    rng = np.random.default_rng(seed)
    out = []
    for u in users:
        positives = u.positives()
        if not positives:
            continue
        s = positives[rng.integers(len(positives))]
        out.append(metrics.recall_at_k(model.similar(s, k), u.holdout_positives, k))
    return out


def run_t3(model: Recommender, cat: Catalogue, cases: list[GoldenCase], k: int = K) -> dict:
    """Golden set: each seed must include and exclude given titles in its top k."""
    results = []
    for case in cases:
        seed_id = cat.by_key.get(_key(case.seed))
        must = [cat.by_key.get(_key(r)) for r in case.must]
        must_not = [cat.by_key.get(_key(r)) for r in case.must_not]
        if seed_id is None or None in must or None in must_not:
            results.append({"seed": case.seed, "status": "missing"})
            continue
        top = set(model.similar(seed_id, k))
        missed = [r for r, t in zip(case.must, must, strict=True) if t not in top]
        unwanted = [r for r, t in zip(case.must_not, must_not, strict=True) if t in top]
        results.append({
            "seed": case.seed,
            "status": "pass" if not missed and not unwanted else "fail",
            "missed": missed,
            "unwanted": unwanted,
        })  # fmt: skip
    scored = [r for r in results if r["status"] != "missing"]
    rate = sum(r["status"] == "pass" for r in scored) / len(scored) if scored else None
    return {"pass_rate": rate, "cases": results}


def _summary(values: list[float]) -> dict[str, float]:
    lo, hi = metrics.bootstrap_ci(values)
    return {"mean": float(np.mean(values)) if values else 0.0, "ci_low": lo, "ci_high": hi}


def evaluate(
    model: Recommender,
    users: list[EvalUser],
    movie_cat: Catalogue,
    model_all: Recommender | None = None,
    full_cat: Catalogue | None = None,
    golden: list[GoldenCase] | None = None,
    k: int = K,
) -> dict[str, Any]:
    t1 = run_t1(model, users, k)
    t2 = run_t2(model, users, k)
    held = [t for u in users for t in u.holdout_positives]
    out: dict[str, Any] = {
        "users": len(users),
        "t1": {
            f"recall@{k}": _summary(t1["recall"]),
            f"ndcg@{k}": _summary(t1["ndcg"]),
            "coverage": metrics.coverage(t1["lists"], len(movie_cat), k),
            "mean_log_votes_recommended": metrics.mean_log_votes(
                [t for ranked in t1["lists"] for t in ranked], movie_cat.votes
            ),
            "mean_log_votes_held_out": metrics.mean_log_votes(held, movie_cat.votes),
        },
        "t2": {f"recall@{k}": _summary(t2)},
        "per_user": {"recall": t1["recall"], "ndcg": t1["ndcg"]},
    }
    if golden and model_all is not None and full_cat is not None:
        out["t3"] = run_t3(model_all, full_cat, golden, k)
    return out


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_run(conn: Connection, config: dict, result: dict, report_dir: Path) -> int:
    run_id = conn.execute(
        "INSERT INTO eval_run (git_sha, config, metrics) VALUES (%s, %s, %s) RETURNING id",
        (git_sha(), Jsonb(config), Jsonb(result)),
    ).fetchone()[0]
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    path = report_dir / f"{stamp}-{run_id}-{config['model']}.md"
    path.write_text(render_report(run_id, config, result))
    log.info("eval run %d: %s", run_id, path)
    return run_id


def _fmt(s: dict[str, float]) -> str:
    return f"{s['mean']:.4f} ({s['ci_low']:.4f}–{s['ci_high']:.4f})"


def render_report(run_id: int, config: dict, result: dict) -> str:
    k = config["k"]
    t1, t2 = result["t1"], result["t2"]
    lines = [
        f"# Eval run {run_id}: {config['model']}",
        "",
        f"Split `{config['split']}`, {result['users']} users, k = {k}, git `{config.get('git')}`.",
        "",
        "| Metric | Value (95% CI) |",
        "| --- | --- |",
        f"| T1 nDCG@{k} | {_fmt(t1[f'ndcg@{k}'])} |",
        f"| T1 Recall@{k} | {_fmt(t1[f'recall@{k}'])} |",
        f"| T1 coverage | {t1['coverage']:.4f} |",
        f"| T1 mean log votes, recommended | {t1['mean_log_votes_recommended']:.2f} |",
        f"| T1 mean log votes, held out | {t1['mean_log_votes_held_out']:.2f} |",
        f"| T2 Recall@{k} | {_fmt(t2[f'recall@{k}'])} |",
    ]
    if "t3" in result:
        rate = result["t3"]["pass_rate"]
        lines.append(f"| T3 golden pass rate | {'n/a' if rate is None else f'{rate:.2f}'} |")
        lines += ["", "## Golden set", ""]
        for c in result["t3"]["cases"]:
            detail = ""
            if c["status"] == "fail":
                detail = f" missed {c['missed']}, unwanted {c['unwanted']}"
            lines.append(f"- `{c['seed']}`: {c['status']}{detail}")
    return "\n".join(lines) + "\n"


def compare(conn: Connection, a: int, b: int) -> str:
    rows = {
        r[0]: (r[1], r[2])
        for r in conn.execute(
            "SELECT id, config, metrics FROM eval_run WHERE id = ANY(%s)", ([a, b],)
        )
    }
    if a not in rows or b not in rows:
        raise ValueError("unknown eval run id")
    (ca, ma), (cb, mb) = rows[a], rows[b]
    same_users = all(ca.get(x) == cb.get(x) for x in ("split", "users", "seed"))
    k = ca["k"]
    lines = [
        f"| Metric | #{a} {ca['model']} | #{b} {cb['model']} | Δ (95% CI) |",
        "| --- | --- | --- | --- |",
    ]
    for metric in ("ndcg", "recall"):
        va, vb = ma["t1"][f"{metric}@{k}"]["mean"], mb["t1"][f"{metric}@{k}"]["mean"]
        ci = "different users"
        if same_users:
            lo, hi = metrics.paired_bootstrap_ci(ma["per_user"][metric], mb["per_user"][metric])
            ci = f"{vb - va:+.4f} ({lo:+.4f}, {hi:+.4f})"
        lines.append(f"| T1 {metric}@{k} | {va:.4f} | {vb:.4f} | {ci} |")
    va, vb = ma["t2"][f"recall@{k}"]["mean"], mb["t2"][f"recall@{k}"]["mean"]
    lines.append(f"| T2 recall@{k} | {va:.4f} | {vb:.4f} | {vb - va:+.4f} |")
    ra, rb = (m.get("t3", {}).get("pass_rate") for m in (ma, mb))
    if ra is not None or rb is not None:
        lines.append(f"| T3 pass rate | {ra} | {rb} | |")
    return "\n".join(lines)
