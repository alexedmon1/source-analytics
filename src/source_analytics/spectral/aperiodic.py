"""Aperiodic (1/f) spectral fitting for ROI PSDs.

Uses specparam (FOOOF) when available; falls back to simple log-log
linear regression for a rough exponent estimate.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Try to import specparam; set flag for fallback
try:
    from specparam import SpectralModel

    _HAS_SPECPARAM = True
except ImportError:
    _HAS_SPECPARAM = False


def fit_aperiodic(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple[float, float] = (2, 50),
    max_n_peaks: int = 6,
) -> dict:
    """Fit aperiodic parameters to a single PSD.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector in Hz.
    psd : ndarray
        Power spectral density values.
    freq_range : tuple
        (fmin, fmax) in Hz for the fitting range.
    max_n_peaks : int
        Maximum number of periodic peaks for specparam.

    Returns
    -------
    dict with keys:
        exponent : float — aperiodic exponent (slope of 1/f)
        offset : float — aperiodic offset (broadband power)
        r_squared : float — goodness of fit
        n_peaks : int — number of detected periodic peaks
        error : float — model fitting error
        method : str — "specparam" or "linreg"
    """
    if _HAS_SPECPARAM:
        return _fit_specparam(freqs, psd, freq_range, max_n_peaks)
    else:
        logger.debug("specparam not installed; using log-log linear regression fallback")
        return _fit_linreg_fallback(freqs, psd, freq_range)


def _fit_specparam(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple[float, float],
    max_n_peaks: int,
) -> dict:
    """Fit using specparam (FOOOF).

    Compatible with specparam v2.x API (get_params / results.metrics).
    """
    sm = SpectralModel(
        peak_width_limits=[1.0, 12.0],
        max_n_peaks=max_n_peaks,
        min_peak_height=0.1,
        aperiodic_mode="fixed",
    )
    sm.fit(freqs, psd, freq_range)

    # v2 API: get_params returns [offset, exponent] for fixed mode
    ap = sm.get_params("aperiodic")
    n_peaks = int(sm.results.n_peaks)

    # Metrics are in results.metrics.results dict
    metrics = sm.results.metrics.results
    r_squared = float(metrics.get("gof_rsquared", float("nan")))
    error = float(metrics.get("error_mae", float("nan")))

    return {
        "exponent": float(ap[1]),
        "offset": float(ap[0]),
        "r_squared": r_squared,
        "n_peaks": n_peaks,
        "error": error,
        "method": "specparam",
    }


def _fit_linreg_fallback(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple[float, float],
) -> dict:
    """Fallback: linear regression in log-log space."""
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1]) & (freqs > 0) & (psd > 0)
    if mask.sum() < 3:
        return {
            "exponent": float("nan"),
            "offset": float("nan"),
            "r_squared": float("nan"),
            "n_peaks": 0,
            "error": float("nan"),
            "method": "linreg",
        }

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])

    # y = offset + slope * x  =>  slope is negative exponent
    coeffs = np.polyfit(log_f, log_p, 1)
    slope, intercept = coeffs

    # R-squared
    predicted = np.polyval(coeffs, log_f)
    ss_res = np.sum((log_p - predicted) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "exponent": float(-slope),  # convention: positive exponent
        "offset": float(intercept),
        "r_squared": float(r_squared),
        "n_peaks": 0,
        "error": float(np.sqrt(ss_res / mask.sum())),
        "method": "linreg",
    }


def fit_aperiodic_multiroi(
    roi_psds: dict[str, tuple[np.ndarray, np.ndarray]],
    freq_range: tuple[float, float] = (2, 50),
    max_n_peaks: int = 6,
) -> dict[str, dict]:
    """Fit aperiodic parameters for all ROIs.

    Parameters
    ----------
    roi_psds : dict
        Mapping of ROI name -> (freqs, psd).
    freq_range : tuple
        Fitting frequency range.
    max_n_peaks : int
        Maximum periodic peaks (specparam only).

    Returns
    -------
    dict[str, dict]
        Mapping of ROI name -> aperiodic parameter dict.
    """
    results = {}
    for roi_name, (freqs, psd) in roi_psds.items():
        try:
            results[roi_name] = fit_aperiodic(freqs, psd, freq_range, max_n_peaks)
        except Exception as e:
            logger.warning("Aperiodic fit failed for ROI %s: %s", roi_name, e)
            results[roi_name] = {
                "exponent": float("nan"),
                "offset": float("nan"),
                "r_squared": float("nan"),
                "n_peaks": 0,
                "error": float("nan"),
                "method": "failed",
            }
    return results
