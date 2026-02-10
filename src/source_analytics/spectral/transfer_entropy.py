"""Transfer entropy: directed information-theoretic connectivity measure.

TE(X→Y) quantifies how much the past of X reduces uncertainty about the
future of Y, beyond Y's own past.  This is an asymmetric measure:
TE(X→Y) ≠ TE(Y→X).

Algorithm (binned TE):
    1. Band-pass filter both signals (4th-order Butterworth, same as AEC).
    2. Discretize into equal-probability bins (quantile-based).
    3. Build lagged vectors: Y_future = y[lag:], Y_past = y[:-lag], X_past = x[:-lag].
    4. Compute joint entropies from histogram counts.
    5. TE = H(Y_future, Y_past) + H(Y_past, X_past) − H(Y_past) − H(Y_future, Y_past, X_past).

References:
    Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85(2), 461.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)


def compute_transfer_entropy(
    roi_timeseries: dict[str, np.ndarray],
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    *,
    lag: int = 1,
    n_bins: int = 5,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Compute directed transfer entropy between all ROI pairs.

    Parameters
    ----------
    roi_timeseries : dict[str, ndarray]
        Mapping of ROI name → 1-D time course (signed, phase-preserving).
    sfreq : float
        Sampling frequency in Hz.
    bands : dict[str, tuple[float, float]]
        Frequency band definitions, e.g. ``{"low_gamma": (30, 55)}``.
    lag : int
        Number of samples for the history/prediction lag (default: 1).
    n_bins : int
        Number of equal-probability bins for discretization (default: 5).

    Returns
    -------
    band_results : dict[str, dict[str, ndarray]]
        ``band_results[band]["te"]`` is an (n_rois, n_rois) **asymmetric** matrix
        where ``te[i, j]`` = TE(roi_i → roi_j).
        ``band_results[band]["net_te"]`` = ``te[i, j] − te[j, i]``.
    roi_names : list[str]
        Ordered list of ROI names (row/column indices of matrices).
    """
    roi_names = sorted(roi_timeseries.keys())
    n_rois = len(roi_names)
    ts_list = [roi_timeseries[name] for name in roi_names]
    min_len = min(len(ts) for ts in ts_list)

    band_results: dict[str, dict[str, np.ndarray]] = {}
    nyq = sfreq / 2.0

    for band_name, (fmin, fmax) in bands.items():
        lo = max(fmin / nyq, 1e-5)
        hi = min(fmax / nyq, 0.9999)
        sos = butter(4, [lo, hi], btype="band", output="sos")

        # Band-pass filter all ROIs
        filtered: list[np.ndarray] = []
        for ts in ts_list:
            filtered.append(sosfiltfilt(sos, ts[:min_len]))

        te_mat = np.zeros((n_rois, n_rois), dtype=np.float64)

        for i in range(n_rois):
            for j in range(n_rois):
                if i == j:
                    continue
                te_mat[i, j] = _transfer_entropy_pair(
                    filtered[i], filtered[j], lag=lag, n_bins=n_bins,
                )

        net_te = te_mat - te_mat.T

        band_results[band_name] = {
            "te": te_mat,
            "net_te": net_te,
        }

    return band_results, roi_names


def _transfer_entropy_pair(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    n_bins: int = 5,
) -> float:
    """Compute TE(X → Y) for a single pair of signals.

    TE(X→Y) = H(Y_future, Y_past) + H(Y_past, X_past) − H(Y_past) − H(Y_future, Y_past, X_past)

    Parameters
    ----------
    x, y : 1-D arrays
        Source and target signals (same length, already band-pass filtered).
    lag : int
        Lag in samples.
    n_bins : int
        Number of quantile-based bins.

    Returns
    -------
    te : float
        Transfer entropy in nats (non-negative by construction, though
        finite-sample estimation can yield small negative values which are
        clipped to 0).
    """
    n = len(x)
    if n <= lag:
        return 0.0

    # Discretize
    x_d = _discretize(x, n_bins)
    y_d = _discretize(y, n_bins)

    # Build lagged vectors
    y_future = y_d[lag:]
    y_past = y_d[:n - lag]
    x_past = x_d[:n - lag]

    # Joint entropies
    h_yf_yp = _joint_entropy_2(y_future, y_past, n_bins)
    h_yp_xp = _joint_entropy_2(y_past, x_past, n_bins)
    h_yp = _entropy_1d(y_past, n_bins)
    h_yf_yp_xp = _joint_entropy_3(y_future, y_past, x_past, n_bins)

    te = h_yf_yp + h_yp_xp - h_yp - h_yf_yp_xp
    return max(te, 0.0)


def _discretize(signal: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize signal into equal-probability bins (quantile-based).

    Parameters
    ----------
    signal : 1-D array
        Continuous signal.
    n_bins : int
        Number of bins.

    Returns
    -------
    binned : 1-D int array
        Bin indices in [0, n_bins-1].
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(signal, quantiles)
    # np.digitize with right=False puts values into [edge[i-1], edge[i])
    # Subtract 1 to get 0-based and clip to handle edge cases
    binned = np.digitize(signal, edges[1:-1], right=False)
    return np.clip(binned, 0, n_bins - 1)


def _entropy_1d(a: np.ndarray, n_bins: int) -> float:
    """Shannon entropy of a single discrete variable (nats)."""
    counts = np.bincount(a, minlength=n_bins).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def _joint_entropy_2(a: np.ndarray, b: np.ndarray, n_bins: int) -> float:
    """Joint Shannon entropy of two discrete variables (nats)."""
    # Encode pair as single integer for fast histogram
    combined = a * n_bins + b
    counts = np.bincount(combined, minlength=n_bins * n_bins).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def _joint_entropy_3(a: np.ndarray, b: np.ndarray, c: np.ndarray, n_bins: int) -> float:
    """Joint Shannon entropy of three discrete variables (nats).

    Encodes (a, b, c) as a single integer index for histogram counting.
    """
    combined = (a * n_bins + b) * n_bins + c
    counts = np.bincount(combined, minlength=n_bins ** 3).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))
