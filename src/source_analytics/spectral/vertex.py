"""Vectorized PSD and feature extraction for vertex-level source data.

Operates on (n_vertices, n_times) arrays — all vertices processed at once
via scipy.signal.welch with axis=-1 broadcasting.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


def compute_psd_vertices(
    stc_data: np.ndarray,
    sfreq: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    fmin: float = 0.5,
    fmax: float | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD for all vertices simultaneously using Welch's method.

    Parameters
    ----------
    stc_data : ndarray, shape (n_vertices, n_times)
        Source time courses.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Segment length. Default: 2 * sfreq (2-second windows).
    noverlap : int, optional
        Overlap between segments. Default: nperseg // 2.
    fmin, fmax : float
        Frequency range to return.
    window : str
        Window function.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector in Hz.
    psd : ndarray, shape (n_vertices, n_freqs)
        Power spectral density per vertex.
    """
    n_vertices, n_times = stc_data.shape

    if nperseg is None:
        nperseg = int(2 * sfreq)
    nperseg = min(nperseg, n_times)

    if noverlap is None:
        noverlap = nperseg // 2

    if fmax is None:
        fmax = sfreq / 2.0

    freqs, psd = welch(
        stc_data,
        fs=sfreq,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )

    # Crop to requested frequency range
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[:, mask]


def extract_band_power_vertices(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: dict[str, tuple[float, float]],
    noise_exclude: tuple[float, float] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract band power metrics for all vertices at once.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector.
    psd : ndarray, shape (n_vertices, n_freqs)
        PSD per vertex.
    bands : dict
        Band name -> (fmin, fmax).
    noise_exclude : tuple, optional
        Frequency range to exclude from total power computation
        (e.g., (55, 65) for line noise).

    Returns
    -------
    dict[str, dict[str, ndarray]]
        band_name -> {"absolute": (n_vertices,), "relative": (n_vertices,),
        "dB": (n_vertices,)}
    """
    # Total power (optionally excluding noise band)
    if noise_exclude is not None:
        lo, hi = noise_exclude
        total_mask = ~((freqs >= lo) & (freqs <= hi))
        total_power = trapezoid(psd[:, total_mask], freqs[total_mask], axis=-1)
    else:
        total_power = trapezoid(psd, freqs, axis=-1)

    total_power = np.maximum(total_power, np.finfo(float).eps)

    result = {}
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            n = psd.shape[0]
            result[band_name] = {
                "absolute": np.zeros(n),
                "relative": np.zeros(n),
                "dB": np.full(n, -np.inf),
            }
            continue

        band_psd = psd[:, mask]
        band_freqs = freqs[mask]
        abs_power = trapezoid(band_psd, band_freqs, axis=-1)

        rel_power = abs_power / total_power
        db_power = np.where(abs_power > 0, 10.0 * np.log10(abs_power), -np.inf)

        result[band_name] = {
            "absolute": abs_power,
            "relative": rel_power,
            "dB": db_power,
        }

    return result


def compute_falff(
    freqs: np.ndarray,
    psd: np.ndarray,
    gamma_range: tuple[float, float] = (65, 100),
    total_range: tuple[float, float] = (1, 100),
) -> np.ndarray:
    """Compute fractional amplitude of low-frequency fluctuations (fALFF).

    Here defined as high-gamma power / total broadband power, identifying
    vertices with disproportionate high-frequency activity.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
    psd : ndarray, shape (n_vertices, n_freqs)
    gamma_range : tuple
        Frequency range for numerator (high gamma).
    total_range : tuple
        Frequency range for denominator (total).

    Returns
    -------
    ndarray, shape (n_vertices,)
        fALFF ratio per vertex.
    """
    gamma_mask = (freqs >= gamma_range[0]) & (freqs <= gamma_range[1])
    total_mask = (freqs >= total_range[0]) & (freqs <= total_range[1])

    gamma_power = trapezoid(psd[:, gamma_mask], freqs[gamma_mask], axis=-1)
    total_power = trapezoid(psd[:, total_mask], freqs[total_mask], axis=-1)
    total_power = np.maximum(total_power, np.finfo(float).eps)

    return gamma_power / total_power


def compute_spectral_slope(
    freqs: np.ndarray,
    psd: np.ndarray,
    fit_range: tuple[float, float] = (2, 50),
) -> np.ndarray:
    """Compute 1/f spectral slope via log-log linear regression.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
    psd : ndarray, shape (n_vertices, n_freqs)
    fit_range : tuple
        Frequency range for fitting.

    Returns
    -------
    ndarray, shape (n_vertices,)
        Spectral slope (exponent) per vertex. Negative values indicate
        typical 1/f decay; steeper negative = more aperiodic dominance.
    """
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    log_f = np.log10(freqs[mask])
    log_psd = np.log10(np.maximum(psd[:, mask], np.finfo(float).eps))

    # Vectorized least-squares: slope = cov(x, y) / var(x) for each vertex
    n = log_f.shape[0]
    mean_f = log_f.mean()
    mean_psd = log_psd.mean(axis=-1, keepdims=True)

    f_centered = log_f - mean_f  # (n_freqs,)
    psd_centered = log_psd - mean_psd  # (n_vertices, n_freqs)

    var_f = np.sum(f_centered ** 2)
    cov = np.sum(psd_centered * f_centered, axis=-1)  # (n_vertices,)

    slopes = cov / var_f
    return slopes


def compute_peak_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    search_range: tuple[float, float] = (6, 13),
) -> np.ndarray:
    """Find peak frequency in a specified range per vertex.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
    psd : ndarray, shape (n_vertices, n_freqs)
    search_range : tuple
        Frequency range to search for peak.

    Returns
    -------
    ndarray, shape (n_vertices,)
        Peak frequency in Hz per vertex.
    """
    mask = (freqs >= search_range[0]) & (freqs <= search_range[1])
    search_freqs = freqs[mask]
    search_psd = psd[:, mask]

    peak_idx = np.argmax(search_psd, axis=-1)
    return search_freqs[peak_idx]
