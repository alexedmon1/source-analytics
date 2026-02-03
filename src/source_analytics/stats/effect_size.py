"""Effect size calculations."""

from __future__ import annotations

import numpy as np


def cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """Compute Cohen's d (pooled standard deviation).

    Positive d means group_a > group_b.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan

    mean_diff = np.mean(a) - np.mean(b)
    pooled_var = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd == 0:
        return 0.0

    return float(mean_diff / pooled_sd)


def hedges_g(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """Compute Hedges' g (bias-corrected Cohen's d for small samples)."""
    d = cohens_d(group_a, group_b)
    if np.isnan(d):
        return np.nan

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n = len(a) + len(b)

    # Hedges' correction factor
    correction = 1 - 3 / (4 * (n - 2) - 1)
    return float(d * correction)
