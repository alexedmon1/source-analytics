"""Multiple comparison correction: Benjamini-Hochberg FDR."""

from __future__ import annotations

import numpy as np


def fdr_correction(
    p_values: np.ndarray | list[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array-like
        Uncorrected p-values.
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    rejected : ndarray of bool
        Whether each test is significant after correction.
    q_values : ndarray
        FDR-adjusted p-values (q-values).
    """
    pvals = np.asarray(p_values, dtype=float)
    n = len(pvals)

    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    # Handle NaNs: keep them, but don't include in correction
    nan_mask = np.isnan(pvals)
    valid_idx = np.where(~nan_mask)[0]
    n_valid = len(valid_idx)

    if n_valid == 0:
        return np.full(n, False), np.full(n, np.nan)

    valid_pvals = pvals[valid_idx]

    # Sort
    sorted_idx = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[sorted_idx]

    # BH procedure
    ranks = np.arange(1, n_valid + 1)
    q_sorted = sorted_pvals * n_valid / ranks

    # Enforce monotonicity (cumulative minimum from the right)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    # Unsort
    q_valid = np.empty(n_valid)
    q_valid[sorted_idx] = q_sorted

    # Build full arrays
    q_values = np.full(n, np.nan)
    q_values[valid_idx] = q_valid

    rejected = np.full(n, False)
    rejected[valid_idx] = q_valid <= alpha

    return rejected, q_values
