"""Frequency band power extraction from PSD."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


def relative_power_kwargs(rel_cfg: dict | None) -> dict:
    """Turn a study-config ``relative_power`` block into ``extract_band_power``
    kwargs. ``{fmin, fmax, exclude: [[lo, hi], ...]}`` → ``{rel_fmin, rel_fmax,
    rel_exclude}``. Empty/None → ``{}`` (full-spectrum total, legacy)."""
    if not rel_cfg:
        return {}
    exclude = rel_cfg.get("exclude")
    return {
        "rel_fmin": rel_cfg.get("fmin"),
        "rel_fmax": rel_cfg.get("fmax"),
        "rel_exclude": [tuple(x) for x in exclude] if exclude else None,
    }


def delta_reference_kwargs(dref_cfg: dict | None) -> dict:
    """Turn a study-config ``delta_reference`` block into ``extract_band_power``
    kwargs. ``{fmin, fmax, agg, scale, exclude_bands}`` → ``{dref_fmin, dref_fmax,
    dref_agg, dref_scale}``. Empty/None → ``{}`` (no ``delta_ref`` DV is computed).

    ``exclude_bands`` is consumed R-side (the reference band is ~constant by
    construction and is dropped from delta-ref testing/plots), not here."""
    if not dref_cfg:
        return {}
    return {
        "dref_fmin": dref_cfg.get("fmin"),
        "dref_fmax": dref_cfg.get("fmax"),
        "dref_agg": dref_cfg.get("agg", "mean"),
        "dref_scale": dref_cfg.get("scale", "db"),
    }


def _delta_anchor(
    freqs: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
    agg: str,
) -> float:
    """Anchor power for the delta-referenced denominator: the mean (or median)
    PSD *density* over ``[fmin, fmax]`` (default 1–1.5 Hz). A mean over the window
    rather than a single bin avoids the noise floor at the 0.5 Hz spectral edge."""
    amask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(amask):
        return float("nan")
    vals = np.asarray(psd, dtype=float)[amask]
    return float(np.median(vals)) if agg == "median" else float(np.mean(vals))


def _relative_total_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    rel_fmin: float | None,
    rel_fmax: float | None,
    rel_exclude: list[tuple[float, float]] | None,
) -> float:
    """Total power for the relative-power denominator: integrate the PSD over
    ``[rel_fmin, rel_fmax]`` with every excluded internal ``(lo, hi)`` range
    (e.g. the line-noise notch gap) **zeroed** rather than removed — removing
    the points would let ``trapezoid`` bridge the gap with a phantom segment
    and over-count the denominator. All bounds ``None`` → full-spectrum total."""
    rmask = np.ones(len(freqs), dtype=bool)
    if rel_fmin is not None:
        rmask &= freqs >= rel_fmin
    if rel_fmax is not None:
        rmask &= freqs <= rel_fmax
    f = freqs[rmask]
    p = np.asarray(psd, dtype=float)[rmask].copy()
    for lo, hi in (rel_exclude or ()):
        p[(f >= lo) & (f <= hi)] = 0.0
    return float(trapezoid(p, f))


def extract_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: dict[str, tuple[float, float]],
    *,
    rel_fmin: float | None = None,
    rel_fmax: float | None = None,
    rel_exclude: list[tuple[float, float]] | None = None,
    dref_fmin: float | None = None,
    dref_fmax: float | None = None,
    dref_agg: str = "mean",
    dref_scale: str = "db",
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
    rel_fmin, rel_fmax : float, optional
        Restrict the relative-power *denominator* to this frequency range. A
        band whose lower edge is ``>= rel_fmax`` sits entirely outside the
        denominator and gets ``relative = nan`` (e.g. an Epsilon band above an
        80 Hz cap). Both ``None`` → full-spectrum total (legacy behaviour).
    rel_exclude : list of (lo, hi), optional
        Frequency ranges excluded from the denominator (e.g. ``[(55, 65)]`` for
        the line-noise notch gap).
    dref_fmin, dref_fmax : float, optional
        Delta-referenced power (R2) anchor window. When both are given, each band
        gets a ``delta_ref`` value = the band's mean power density ÷ the anchor
        power (:func:`_delta_anchor` over ``[dref_fmin, dref_fmax]``). Both ``None``
        → no ``delta_ref`` key is emitted (the default/exploratory DV set).
    dref_agg : {"mean", "median"}
        Aggregator for the anchor window (default "mean").
    dref_scale : {"db", "ratio"}
        "db" → ``delta_ref = 10*log10(density / anchor)`` (0 dB = equal to the
        anchor; the default, matching the ``absolute`` dB convention). "ratio" →
        the raw linear ``density / anchor`` (× the delta anchor).

    Returns
    -------
    dict[str, dict[str, float]]
        For each band: {"absolute": ..., "relative": ...(, "delta_ref": ...)} where
        **absolute** is the mean power *density* over the band,
        10*log10(∫power ÷ bandwidth), in **dB/Hz** (bandwidth-normalized),
        **relative** is the integrated band power ÷ the restricted-range total (nan
        for bands above ``rel_fmax``), and **delta_ref** (present only when a
        ``dref`` window is given) is the band mean density referenced to the
        1–1.5 Hz anchor, in dB (or linear ratio).
    """
    total_power = _relative_total_power(freqs, psd, rel_fmin, rel_fmax, rel_exclude)
    if total_power <= 0:
        total_power = np.finfo(float).eps

    dref_on = dref_fmin is not None and dref_fmax is not None
    anchor = _delta_anchor(freqs, psd, dref_fmin, dref_fmax, dref_agg) if dref_on else None

    result = {}
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            result[band_name] = {"absolute": -np.inf, "relative": 0.0}
            if dref_on:
                result[band_name]["delta_ref"] = -np.inf if dref_scale == "db" else 0.0
            continue

        band_freqs = freqs[mask]
        band_psd = psd[mask]
        abs_power = float(trapezoid(band_psd, band_freqs))

        # Relative uses the *integrated* band power over the restricted total.
        # A band above the relative-power range has no meaningful relative value.
        if rel_fmax is not None and fmin >= rel_fmax:
            rel_power = float("nan")
        else:
            rel_power = abs_power / total_power

        # "absolute" is the mean power *density* over the band (integrated power
        # ÷ bandwidth), in dB/Hz — bandwidth-normalized so the across-band 1/f
        # shape is visible rather than the width-confounded integrated total.
        bandwidth = fmax - fmin
        density = abs_power / bandwidth if bandwidth > 0 else abs_power
        db_power = 10 * np.log10(density) if density > 0 else -np.inf

        result[band_name] = {
            "absolute": float(db_power),
            "relative": rel_power,
        }

        # Delta-referenced power (R2): band mean density ÷ the delta anchor. Both
        # are densities → a dimensionless ratio; emit it in dB (default) or linear.
        if dref_on:
            if anchor is not None and anchor > 0 and density > 0:
                ratio = density / anchor
                dref = 10 * np.log10(ratio) if dref_scale == "db" else ratio
            else:
                dref = float("nan")
            result[band_name]["delta_ref"] = float(dref)

    return result


def extract_band_power_multiroi(
    roi_psds: dict[str, tuple[np.ndarray, np.ndarray]],
    bands: dict[str, tuple[float, float]],
    **rel_kwargs,
) -> dict[str, dict[str, dict[str, float]]]:
    """Extract band power for all ROIs. ``rel_kwargs`` (rel_fmin/rel_fmax/
    rel_exclude) are forwarded to :func:`extract_band_power`.

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        roi_name -> band_name -> {"absolute", "relative"}
    """
    return {
        roi_name: extract_band_power(freqs, psd, bands, **rel_kwargs)
        for roi_name, (freqs, psd) in roi_psds.items()
    }
