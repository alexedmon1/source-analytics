"""Welch PSD computation for ROI time series."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def compute_psd(
    timeseries: np.ndarray,
    sfreq: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    fmin: float = 0.5,
    fmax: float | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method.

    Parameters
    ----------
    timeseries : ndarray, shape (n_times,)
        Continuous time series for a single ROI.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment. Default: 2 * sfreq (2-second windows).
    noverlap : int, optional
        Overlap between segments. Default: nperseg // 2 (50%).
    fmin, fmax : float
        Frequency range to return.
    window : str
        Window function (default: Hann).

    Returns
    -------
    freqs : ndarray
        Frequency vector in Hz.
    psd : ndarray
        Power spectral density (V^2/Hz or power/Hz).
    """
    if nperseg is None:
        nperseg = int(2 * sfreq)
    nperseg = min(nperseg, len(timeseries))

    if noverlap is None:
        noverlap = nperseg // 2

    if fmax is None:
        fmax = sfreq / 2.0

    freqs, psd = welch(
        timeseries,
        fs=sfreq,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Crop to requested frequency range
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]


def compute_psd_multiroi(
    roi_timeseries: dict[str, np.ndarray],
    sfreq: float,
    **kwargs,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute PSD for all ROIs.

    Returns
    -------
    dict[str, tuple[ndarray, ndarray]]
        Mapping of ROI name -> (freqs, psd).
    """
    results = {}
    freqs = None
    for roi_name, ts in roi_timeseries.items():
        f, p = compute_psd(ts, sfreq, **kwargs)
        results[roi_name] = (f, p)
        freqs = f
    return results
