"""Vertex-level spectral parameterization (aperiodic + oscillatory peaks).

Wraps the existing fit_aperiodic() in a vectorized loop over all source
vertices, extracting aperiodic parameters (exponent, offset) and detecting
oscillatory peaks across all configured frequency bands at each spatial
location.
"""

from __future__ import annotations

import logging

import numpy as np

from .aperiodic import DEFAULT_FREQ_RANGE, band_peak_reachability, fit_aperiodic

logger = logging.getLogger(__name__)


def _safe_band_key(band_name: str) -> str:
    """Sanitize band name for use as a dict/column key."""
    return band_name.lower().replace(" ", "_")


def fit_aperiodic_vertices(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple[float, float] = DEFAULT_FREQ_RANGE,
    max_n_peaks: int = 6,
    peak_width_limits: tuple[float, float] = (1.0, 12.0),
    bands: dict[str, tuple[float, float]] | None = None,
    peak_freq_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Fit aperiodic (1/f) model at each vertex.

    Aperiodic estimation and peak detection want *different* windows and this
    function can use two. The aperiodic window (``freq_range``) is deliberately
    narrow — borders clear of oscillatory peaks, roll-off and line noise — which
    is what makes the exponent unbiased, but it also makes every band outside it
    structurally undetectable. ``peak_freq_range`` runs a second, wider fit whose
    ONLY job is to locate oscillations, so that the choice of aperiodic window
    can be checked against where the peaks actually are (Gerster et al. 2022:
    fit borders must not cross oscillatory peaks) rather than merely asserted.

    Peaks from the wide fit are worse-constrained than the narrow fit's aperiodic
    parameters — a wide window is a worse 1/f model, and specparam finds peaks by
    subtracting that model. They are a diagnostic and an interpretive guard on
    band power, not a precision measurement.

    Parameters
    ----------
    freqs : ndarray, shape (n_freqs,)
        Frequency vector.
    psd : ndarray, shape (n_vertices, n_freqs)
        PSD per vertex.
    freq_range : tuple
        Frequency range for the APERIODIC fit (exponent/offset/r^2).
    max_n_peaks : int
        Maximum number of peaks to detect per vertex.
    peak_width_limits : tuple
        Min and max peak width in Hz.
    bands : dict mapping band name to (fmin, fmax), optional
        Frequency bands for peak detection.  When *None*, defaults to
        ``{"Gamma": (30, 100)}`` for backward compatibility.
    peak_freq_range : tuple, optional
        Separate, usually wider window for PEAK detection. When *None* or equal
        to ``freq_range`` a single fit is performed (no extra cost) and peaks
        come from it, preserving the previous behaviour.

    Returns
    -------
    dict[str, ndarray]
        Always contains: exponent, offset, offset_centered, r_squared, n_peaks,
        method, peaks_all (per-vertex list of every detected peak), and the
        reachability/window metadata under ``peak_window`` / ``band_reach``.
        Per-band keys ``has_{key}_peak``, ``{key}_peak_freq``,
        ``{key}_peak_power`` are emitted ONLY for bands the peak window can
        reach — an unreachable band gets no column rather than a fabricated
        ``False``.  All arrays have shape (n_vertices,) except method/peaks_all.
    """
    if bands is None:
        bands = {"Gamma": (30, 100)}

    peak_range = tuple(peak_freq_range) if peak_freq_range else tuple(freq_range)
    two_fit = peak_range != tuple(freq_range)

    reach = band_peak_reachability(bands, peak_range)
    detectable = {n: b for n, b in bands.items() if reach[n]["reachable"]}
    dropped = [n for n in bands if not reach[n]["reachable"]]
    if dropped:
        logger.info(
            "Peak window %.4g-%.4g Hz cannot reach %s — no has_*_peak columns "
            "emitted for these (absence would be structural, not measured).",
            peak_range[0], peak_range[1], ", ".join(dropped),
        )
    censored = [n for n in detectable if reach[n]["censored"]]
    if censored:
        logger.warning(
            "Peak window %.4g-%.4g Hz only partially covers %s — detection "
            "rates are a LOWER bound and peak frequencies are truncated.",
            peak_range[0], peak_range[1], ", ".join(censored),
        )

    n_vertices = psd.shape[0]

    exponents = np.zeros(n_vertices)
    offsets = np.zeros(n_vertices)
    offsets_centered = np.zeros(n_vertices)
    r_squareds = np.zeros(n_vertices)
    n_peaks_arr = np.zeros(n_vertices, dtype=int)
    methods: list[str] = []

    n_peaks_wide = np.zeros(n_vertices, dtype=int)

    # Per-band peak arrays — DETECTABLE bands only (see band_peak_reachability)
    band_keys = {name: _safe_band_key(name) for name in detectable}
    band_has_peak = {key: np.zeros(n_vertices, dtype=bool) for key in band_keys.values()}
    band_peak_freq = {key: np.full(n_vertices, np.nan) for key in band_keys.values()}
    band_peak_power = {key: np.full(n_vertices, np.nan) for key in band_keys.values()}
    peaks_all: list[list[dict]] = [[] for _ in range(n_vertices)]

    for vi in range(n_vertices):
        try:
            result = fit_aperiodic(
                freqs, psd[vi], freq_range=freq_range, max_n_peaks=max_n_peaks,
            )

            exponents[vi] = result["exponent"]
            offsets[vi] = result["offset"]
            offsets_centered[vi] = result["offset_centered"]
            r_squareds[vi] = result["r_squared"]
            # n_peaks stays the APERIODIC fit's peak count: it is QC on that
            # model (many peaks in a narrow window = peaks papering over a bad
            # 1/f fit), not the oscillation inventory. That is n_peaks_wide.
            n_peaks_arr[vi] = result.get("n_peaks", 0)
            methods.append(result.get("method", "unknown"))

            if two_fit:
                peak_fit = fit_aperiodic(
                    freqs, psd[vi], freq_range=peak_range, max_n_peaks=max_n_peaks,
                )
            else:
                peak_fit = result
            peaks = peak_fit.get("peaks", [])
            n_peaks_wide[vi] = peak_fit.get("n_peaks", 0)

            # Match detected peaks to frequency bands
            for peak in peaks:
                cf = peak.get("center_frequency", 0)
                pw = peak.get("power", 0)
                peaks_all[vi].append({
                    "vertex_idx": vi,
                    "center_frequency": float(cf),
                    "power": float(pw),
                    "bandwidth": float(peak.get("bandwidth", np.nan)),
                })
                for band_name, (flo, fhi) in detectable.items():
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
    if two_fit:
        logger.info(
            "Two-fit: aperiodic %.4g-%.4g Hz, peaks %.4g-%.4g Hz",
            freq_range[0], freq_range[1], peak_range[0], peak_range[1],
        )
    for band_name in detectable:
        key = band_keys[band_name]
        n_det = int(band_has_peak[key].sum())
        logger.info(
            "  %s peaks detected: %d/%d vertices%s",
            band_name, n_det, n_vertices,
            " (CENSORED window)" if reach[band_name]["censored"] else "",
        )

    result_dict: dict[str, np.ndarray | list[str]] = {
        "exponent": exponents,
        "offset": offsets,
        "offset_centered": offsets_centered,
        "r_squared": r_squareds,
        "n_peaks": n_peaks_arr,
        "n_peaks_wide": n_peaks_wide,
        "method": methods,
        "peaks_all": peaks_all,
        "peak_window": peak_range,
        "aperiodic_window": tuple(freq_range),
        "band_reach": reach,
    }
    for key in band_keys.values():
        result_dict[f"has_{key}_peak"] = band_has_peak[key]
        result_dict[f"{key}_peak_freq"] = band_peak_freq[key]
        result_dict[f"{key}_peak_power"] = band_peak_power[key]

    return result_dict
