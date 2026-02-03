"""Functional connectivity: coherence and imaginary coherence matrices."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch, csd


def compute_connectivity_matrix(
    roi_timeseries: dict[str, np.ndarray],
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    *,
    nperseg: int | None = None,
    window: str = "hann",
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Compute coherence and imaginary coherence matrices for all ROI pairs.

    Uses Welch's method for auto-spectra and CSD to compute
    magnitude-squared coherence and imaginary coherence per frequency band.

    Parameters
    ----------
    roi_timeseries : dict[str, ndarray]
        Mapping of ROI name -> 1-D time course (signed, phase-preserving).
    sfreq : float
        Sampling frequency in Hz.
    bands : dict[str, tuple[float, float]]
        Frequency band definitions, e.g. ``{"alpha": (8, 13)}``.
    nperseg : int, optional
        Segment length for Welch/CSD. Default: ``2 * sfreq`` (2-second windows).
    window : str
        Window function (default: Hann).

    Returns
    -------
    band_results : dict[str, dict[str, ndarray]]
        ``band_results[band_name]["coherence"]`` is an (n_rois, n_rois) symmetric
        matrix of magnitude-squared coherence averaged within the band.
        ``band_results[band_name]["imag_coherence"]`` is the mean absolute
        imaginary coherence within the band.
    roi_names : list[str]
        Ordered list of ROI names (rows/columns of matrices).
    """
    roi_names = sorted(roi_timeseries.keys())
    n_rois = len(roi_names)
    ts_list = [roi_timeseries[name] for name in roi_names]

    if nperseg is None:
        nperseg = int(2 * sfreq)
    # Clamp to shortest timeseries
    min_len = min(len(ts) for ts in ts_list)
    nperseg = min(nperseg, min_len)
    noverlap = nperseg // 2

    # Pre-compute all auto-spectra (PSD)
    auto_spectra: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for ts in ts_list:
        f, pxx = welch(ts, fs=sfreq, window=window, nperseg=nperseg,
                        noverlap=noverlap)
        auto_spectra.append(pxx)
        if freqs is None:
            freqs = f

    assert freqs is not None

    # Build frequency masks per band
    band_masks: dict[str, np.ndarray] = {}
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not mask.any():
            continue
        band_masks[band_name] = mask

    # Initialize result matrices
    band_results: dict[str, dict[str, np.ndarray]] = {}
    for band_name in band_masks:
        band_results[band_name] = {
            "coherence": np.eye(n_rois, dtype=np.float64),
            "imag_coherence": np.zeros((n_rois, n_rois), dtype=np.float64),
        }

    # Compute CSD for each unique pair and derive coherence
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            _, pxy = csd(ts_list[i], ts_list[j], fs=sfreq, window=window,
                         nperseg=nperseg, noverlap=noverlap)

            pxx_i = auto_spectra[i]
            pxx_j = auto_spectra[j]

            for band_name, mask in band_masks.items():
                csd_band = pxy[mask]
                pxx_i_band = pxx_i[mask]
                pxx_j_band = pxx_j[mask]

                # Magnitude-squared coherence: |Pxy|^2 / (Pxx * Pyy)
                denom = pxx_i_band * pxx_j_band
                coh_freq = np.abs(csd_band) ** 2 / np.where(denom > 0, denom, 1.0)
                coh_mean = float(np.mean(coh_freq))

                # Imaginary coherence: |Im(Pxy / sqrt(Pxx * Pyy))|
                norm = np.sqrt(np.where(denom > 0, denom, 1.0))
                icoh_freq = np.abs(np.imag(csd_band / norm))
                icoh_mean = float(np.mean(icoh_freq))

                band_results[band_name]["coherence"][i, j] = coh_mean
                band_results[band_name]["coherence"][j, i] = coh_mean
                band_results[band_name]["imag_coherence"][i, j] = icoh_mean
                band_results[band_name]["imag_coherence"][j, i] = icoh_mean

    return band_results, roi_names
