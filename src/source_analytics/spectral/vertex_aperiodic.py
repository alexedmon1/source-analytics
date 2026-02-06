"""Vertex-level spectral parameterization (aperiodic + oscillatory peaks).

Wraps the existing fit_aperiodic() in a vectorized loop over all source
vertices, extracting aperiodic parameters (exponent, offset) and detecting
oscillatory peaks (especially gamma) at each spatial location.
"""

from __future__ import annotations

import logging

import numpy as np

from .aperiodic import fit_aperiodic

logger = logging.getLogger(__name__)


def fit_aperiodic_vertices(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple[float, float] = (1, 100),
    max_n_peaks: int = 6,
    peak_width_limits: tuple[float, float] = (1.0, 12.0),
    gamma_range: tuple[float, float] = (30, 100),
) -> dict[str, np.ndarray]:
    """Fit aperiodic (1/f) model at each vertex.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector.
    psd : ndarray, shape (n_vertices, n_freqs)
        PSD per vertex.
    freq_range : tuple
        Frequency range for fitting.
    max_n_peaks : int
        Maximum number of peaks to detect per vertex.
    peak_width_limits : tuple
        Min and max peak width in Hz.
    gamma_range : tuple
        Frequency range to search for gamma peaks.

    Returns
    -------
    dict[str, ndarray]
        Keys: exponent, offset, r_squared, n_peaks, method,
              has_gamma_peak, gamma_peak_freq, gamma_peak_power
        All arrays have shape (n_vertices,) except method which is a list.
    """
    n_vertices = psd.shape[0]

    exponents = np.zeros(n_vertices)
    offsets = np.zeros(n_vertices)
    r_squareds = np.zeros(n_vertices)
    n_peaks_arr = np.zeros(n_vertices, dtype=int)
    methods = []
    has_gamma = np.zeros(n_vertices, dtype=bool)
    gamma_freq = np.full(n_vertices, np.nan)
    gamma_power = np.full(n_vertices, np.nan)

    for vi in range(n_vertices):
        try:
            result = fit_aperiodic(
                freqs, psd[vi], freq_range=freq_range, max_n_peaks=max_n_peaks,
            )

            exponents[vi] = result["exponent"]
            offsets[vi] = result["offset"]
            r_squareds[vi] = result["r_squared"]
            n_peaks_arr[vi] = result.get("n_peaks", 0)
            methods.append(result.get("method", "unknown"))

            # Check for gamma peaks if specparam was used
            peaks = result.get("peaks", [])
            if peaks:
                for peak in peaks:
                    cf = peak.get("center_frequency", 0)
                    if gamma_range[0] <= cf <= gamma_range[1]:
                        has_gamma[vi] = True
                        # Keep the strongest gamma peak
                        pw = peak.get("power", 0)
                        if np.isnan(gamma_power[vi]) or pw > gamma_power[vi]:
                            gamma_freq[vi] = cf
                            gamma_power[vi] = pw

        except Exception as e:
            logger.debug("Vertex %d fit failed: %s", vi, e)
            methods.append("failed")

    n_specparam = sum(1 for m in methods if m == "specparam")
    n_linreg = sum(1 for m in methods if m == "linreg")
    n_gamma = int(has_gamma.sum())
    logger.info(
        "Specparam fit: %d specparam, %d linreg, %d gamma peaks detected",
        n_specparam, n_linreg, n_gamma,
    )

    return {
        "exponent": exponents,
        "offset": offsets,
        "r_squared": r_squareds,
        "n_peaks": n_peaks_arr,
        "method": methods,
        "has_gamma_peak": has_gamma,
        "gamma_peak_freq": gamma_freq,
        "gamma_peak_power": gamma_power,
    }
