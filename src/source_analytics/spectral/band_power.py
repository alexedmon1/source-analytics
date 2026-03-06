"""Frequency band power extraction from PSD."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


def extract_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: dict[str, tuple[float, float]],
) -> dict[str, dict[str, float]]:
    """Extract absolute (dB) and relative band power from a PSD.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector in Hz.
    psd : ndarray
        Power spectral density values.
    bands : dict
        Band name -> (fmin, fmax) mapping.

    Returns
    -------
    dict[str, dict[str, float]]
        For each band: {"absolute": ..., "relative": ...}
        where absolute is 10*log10(integrated power) in dB.
    """
    total_power = trapezoid(psd, freqs)
    if total_power <= 0:
        total_power = np.finfo(float).eps

    result = {}
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            result[band_name] = {"absolute": -np.inf, "relative": 0.0}
            continue

        band_freqs = freqs[mask]
        band_psd = psd[mask]
        abs_power = float(trapezoid(band_psd, band_freqs))

        rel_power = abs_power / total_power
        db_power = 10 * np.log10(abs_power) if abs_power > 0 else -np.inf

        result[band_name] = {
            "absolute": float(db_power),
            "relative": rel_power,
        }

    return result


def extract_band_power_multiroi(
    roi_psds: dict[str, tuple[np.ndarray, np.ndarray]],
    bands: dict[str, tuple[float, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Extract band power for all ROIs.

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        roi_name -> band_name -> {"absolute", "relative"}
    """
    return {
        roi_name: extract_band_power(freqs, psd, bands)
        for roi_name, (freqs, psd) in roi_psds.items()
    }
