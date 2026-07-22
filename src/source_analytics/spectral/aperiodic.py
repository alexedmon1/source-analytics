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


# Package-wide default aperiodic fit range. Override per analysis with a
# ``freq_range: [fmin, fmax]`` key in that analysis's config block.
#
# WHY 12-45 Hz — see docs/APERIODIC_FIT_WINDOW.md for the full derivation.
# There is no universal standard range: Gerster et al. (2022) survey 0.01-100 Hz
# in the literature and deliberately decline to name one, prescribing a PROCEDURE
# instead. This default applies that procedure to notched, high-pass-filtered
# rodent EEG:
#
#   1. "Oscillations crossing the fitting range borders must be avoided for all
#      investigated power spectra" (Gerster 2022) — a peak sitting on a border
#      produces large exponent error. The theta/alpha peak occupies ~5-11 Hz, so
#      the lower border sits above it at 12 Hz.
#   2. The lower border must clear the high-pass roll-off, where power RISES with
#      frequency and so cannot be aperiodic. Compare the rodent-hippocampus
#      convention of starting at 4 Hz "to avoid delta rhythm" (Bhatt 2026).
#   3. The upper border must sit below the spectral plateau and "as low as
#      possible to increase SNR" (Gerster 2022), and below line noise. 45 Hz
#      clears a 57-63 Hz notch with margin.
#
# Trade-off, stated honestly: 12-45 Hz is ~1.9 octaves. The specparam docs warn
# that narrow ranges make the aperiodic component harder to estimate, so this
# buys unbiased slope at the cost of precision. Data whose spectra are NOT
# hemmed in this way should widen it via config.
#
# References
# ----------
# Donoghue T, Haller M, Peterson EJ, et al. (2020). Parameterizing neural power
#   spectra into periodic and aperiodic components. Nature Neuroscience 23,
#   1655-1665. doi:10.1038/s41593-020-00744-x  [the specparam algorithm]
# Gerster M, Waterstraat G, Litvak V, et al. (2022). Separating Neural
#   Oscillations from Aperiodic 1/f Activity: Challenges and Recommendations.
#   Neuroinformatics 20, 991-1012. doi:10.1007/s12021-022-09581-8  [border rules,
#   plateau-onset definition]
# Bhatt N, et al. (2026). Aperiodicity in Mouse CA1 and DG Power Spectra.
#   eNeuro 13(3). doi:10.1523/ENEURO.0136-25.2026  [rodent practice: start above
#   delta; multi-exponent/knee structure at 28-70 Hz]
# Kozhemiako N, et al. (2024). The aperiodic exponent of neural activity varies
#   with vigilance state in mice and men. PLOS ONE 19(4): e0301406.
#   doi:10.1371/journal.pone.0301406  [mouse EEG exponent reference values
#   0.737-1.25 — the 1-3 range often quoted is a HUMAN benchmark]
DEFAULT_FREQ_RANGE: tuple[float, float] = (12.0, 45.0)

# Frequencies at/above which a fit window starts colliding with mains noise and
# its notch. Used only to warn; studies outside 50/60 Hz mains may override.
_LINE_NOISE_GUARD_HZ = 50.0


def resolve_freq_range(cfg: dict | None, key: str = "freq_range") -> tuple[float, float]:
    """Resolve an aperiodic fit range from a config block.

    Falls back to :data:`DEFAULT_FREQ_RANGE` when unset. Emits a warning for the
    two failure modes that silently corrupt an aperiodic fit rather than raising
    an obvious error: a border in the line-noise/notch region, and a lower border
    low enough to sit in a high-pass roll-off. Both are warnings, not errors — a
    study with different filtering may legitimately want a wider window.
    """
    rng = tuple((cfg or {}).get(key) or DEFAULT_FREQ_RANGE)
    if len(rng) != 2:
        raise ValueError(f"{key} must be [fmin, fmax], got {rng!r}")
    fmin, fmax = float(rng[0]), float(rng[1])
    if not np.isfinite([fmin, fmax]).all() or fmin <= 0 or fmax <= fmin:
        raise ValueError(
            f"{key} must satisfy 0 < fmin < fmax, got ({fmin}, {fmax})")
    if fmax > _LINE_NOISE_GUARD_HZ:
        logger.warning(
            "Aperiodic fit range %.4g-%.4g Hz reaches past %.4g Hz, into mains "
            "noise / its notch and the high-frequency plateau. A notch inside "
            "the window flattens the exponent and collapses r^2. Default is %s "
            "— see docs/APERIODIC_FIT_WINDOW.md.",
            fmin, fmax, _LINE_NOISE_GUARD_HZ, DEFAULT_FREQ_RANGE,
        )
    if fmin < 4.0:
        logger.warning(
            "Aperiodic fit range starts at %.4g Hz, inside the band where a "
            "high-pass roll-off makes power RISE with frequency (not aperiodic) "
            "and where delta/theta peaks cross the border. Gerster et al. 2022: "
            "borders must not cross oscillatory peaks. Default is %s.",
            fmin, DEFAULT_FREQ_RANGE,
        )
    return (fmin, fmax)


def centered_offset(offset: float, exponent: float,
                    freq_range: tuple[float, float]) -> float:
    """Offset re-referenced to the geometric centre of the fit window.

    specparam's ``offset`` is the intercept extrapolated to 1 Hz, typically far
    BELOW the fit window, so a steeper slope mechanically forces a higher
    intercept — on FORGE this alone produced r(offset, exponent) ~= 0.96, about
    half of it artifactual. Re-referencing to the window centre removes that
    lever arm (r drops to ~0.60), leaving genuine covariance.

    The narrower/higher the window, the more this matters: from a 12-45 Hz
    window the 1 Hz extrapolation spans ~1.37 decades.

    Report this alongside ``offset`` whenever offset and exponent effects are
    discussed together; never present the two as independent findings.
    """
    f_centre = float(np.sqrt(freq_range[0] * freq_range[1]))
    return float(offset - exponent * np.log10(f_centre))


def fit_aperiodic(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple[float, float] = DEFAULT_FREQ_RANGE,
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
        offset : float — aperiodic offset (broadband power, intercept at 1 Hz)
        offset_centered : float — offset re-referenced to the fit-window centre;
            use this when reporting offset alongside exponent (see
            :func:`centered_offset`)
        r_squared : float — goodness of fit
        n_peaks : int — number of detected periodic peaks
        error : float — model fitting error
        method : str — "specparam" or "linreg"
        fit_fmin, fit_fmax : float — the window actually used, carried alongside
            the estimates so downstream tables record their own provenance
    """
    if _HAS_SPECPARAM:
        out = _fit_specparam(freqs, psd, freq_range, max_n_peaks)
    else:
        logger.debug("specparam not installed; using log-log linear regression fallback")
        out = _fit_linreg_fallback(freqs, psd, freq_range)
    return _annotate_fit(out, freq_range)


def _annotate_fit(out: dict, freq_range: tuple[float, float]) -> dict:
    """Attach the centred offset and the fit-window provenance to a fit result."""
    out["offset_centered"] = centered_offset(
        out["offset"], out["exponent"], freq_range)
    out["fit_fmin"] = float(freq_range[0])
    out["fit_fmax"] = float(freq_range[1])
    return out


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

    # Extract periodic (peak) parameters: (n_peaks, 3) -> [CF, PW, BW]
    peaks_list: list[dict] = []
    if n_peaks > 0:
        try:
            peak_params = sm.get_params("peak")  # (n_peaks, 3)
            if peak_params.ndim == 1:
                # Single peak returned as 1-D array
                peak_params = peak_params.reshape(1, -1)
            for row in peak_params:
                peaks_list.append({
                    "center_frequency": float(row[0]),
                    "power": float(row[1]),
                    "bandwidth": float(row[2]),
                })
        except Exception as e:
            logger.debug("Could not extract peak params: %s", e)

    return {
        "exponent": float(ap[1]),
        "offset": float(ap[0]),
        "r_squared": r_squared,
        "n_peaks": n_peaks,
        "error": error,
        "method": "specparam",
        "peaks": peaks_list,
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
            "peaks": [],
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
        "peaks": [],
    }


def fit_aperiodic_multiroi(
    roi_psds: dict[str, tuple[np.ndarray, np.ndarray]],
    freq_range: tuple[float, float] = DEFAULT_FREQ_RANGE,
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
                "peaks": [],
            }
    return results
