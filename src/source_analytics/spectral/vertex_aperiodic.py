"""Vertex-level spectral parameterization (aperiodic + oscillatory peaks).

Wraps the existing fit_aperiodic() in a vectorized loop over all source
vertices, extracting aperiodic parameters (exponent, offset) and detecting
oscillatory peaks across all configured frequency bands at each spatial
location.
"""

from __future__ import annotations

import logging

import numpy as np

from .aperiodic import fit_aperiodic

logger = logging.getLogger(__name__)


def _safe_band_key(band_name: str) -> str:
    """Sanitize band name for use as a dict/column key."""
    return band_name.lower().replace(" ", "_")


def fit_aperiodic_vertices(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple[float, float] = (1, 100),
    max_n_peaks: int = 6,
    peak_width_limits: tuple[float, float] = (1.0, 12.0),
    bands: dict[str, tuple[float, float]] | None = None,
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
    bands : dict mapping band name to (fmin, fmax), optional
        Frequency bands for peak detection.  When *None*, defaults to
        ``{"Gamma": (30, 100)}`` for backward compatibility.

    Returns
    -------
    dict[str, ndarray]
        Always contains: exponent, offset, r_squared, n_peaks, method.
        Per-band keys: has_{key}_peak, {key}_peak_freq, {key}_peak_power
        where {key} is the lower-cased, underscore-separated band name.
        All arrays have shape (n_vertices,) except method which is a list.
    """
    if bands is None:
        bands = {"Gamma": (30, 100)}

    n_vertices = psd.shape[0]

    exponents = np.zeros(n_vertices)
    offsets = np.zeros(n_vertices)
    r_squareds = np.zeros(n_vertices)
    n_peaks_arr = np.zeros(n_vertices, dtype=int)
    methods: list[str] = []

    # Per-band peak arrays
    band_keys = {name: _safe_band_key(name) for name in bands}
    band_has_peak = {key: np.zeros(n_vertices, dtype=bool) for key in band_keys.values()}
    band_peak_freq = {key: np.full(n_vertices, np.nan) for key in band_keys.values()}
    band_peak_power = {key: np.full(n_vertices, np.nan) for key in band_keys.values()}

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

            # Match detected peaks to frequency bands
            peaks = result.get("peaks", [])
            if peaks:
                for peak in peaks:
                    cf = peak.get("center_frequency", 0)
                    pw = peak.get("power", 0)
                    for band_name, (flo, fhi) in bands.items():
                        key = band_keys[band_name]
                        if flo <= cf <= fhi:
                            if np.isnan(band_peak_power[key][vi]) or pw > band_peak_power[key][vi]:
                                band_has_peak[key][vi] = True
                                band_peak_freq[key][vi] = cf
                                band_peak_power[key][vi] = pw

        except Exception as e:
            logger.debug("Vertex %d fit failed: %s", vi, e)
            methods.append("failed")

    n_specparam = sum(1 for m in methods if m == "specparam")
    n_linreg = sum(1 for m in methods if m == "linreg")
    logger.info("Specparam fit: %d specparam, %d linreg", n_specparam, n_linreg)
    for band_name in bands:
        key = band_keys[band_name]
        n_det = int(band_has_peak[key].sum())
        logger.info("  %s peaks detected: %d/%d vertices", band_name, n_det, n_vertices)

    result_dict: dict[str, np.ndarray | list[str]] = {
        "exponent": exponents,
        "offset": offsets,
        "r_squared": r_squareds,
        "n_peaks": n_peaks_arr,
        "method": methods,
    }
    for key in band_keys.values():
        result_dict[f"has_{key}_peak"] = band_has_peak[key]
        result_dict[f"{key}_peak_freq"] = band_peak_freq[key]
        result_dict[f"{key}_peak_power"] = band_peak_power[key]

    return result_dict
