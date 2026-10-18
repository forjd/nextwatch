"""Ranking metrics. All take a ranked list of title ids and a set of relevant ones."""

import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], relevant: set[int] | frozenset[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for t in ranked[:k] if t in relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(ranked: Sequence[int], relevant: set[int] | frozenset[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(i + 2) for i, t in enumerate(ranked[:k]) if t in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def coverage(lists: Sequence[Sequence[int]], catalogue_size: int, k: int) -> float:
    """Share of the catalogue that appears in anyone's top k."""
    if catalogue_size == 0:
        return 0.0
    return len({t for ranked in lists for t in ranked[:k]}) / catalogue_size


def mean_log_votes(titles: Sequence[int], votes: dict[int, int]) -> float:
    if not titles:
        return 0.0
    return float(np.mean([math.log1p(votes.get(t, 0)) for t in titles]))


def bootstrap_ci(
    values: Sequence[float], n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """95% interval for the mean, by resampling users."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(
    a: Sequence[float], b: Sequence[float], n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Interval for mean(b - a) over the same users. Excludes 0 = a real difference."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_ci(diff, n, seed, alpha)
