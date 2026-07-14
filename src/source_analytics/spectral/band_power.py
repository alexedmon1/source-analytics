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

    Returns
    -------
    dict[str, dict[str, float]]
        For each band: {"absolute": ..., "relative": ...} where **absolute** is
        the mean power *density* over the band, 10*log10(∫power ÷ bandwidth), in
        **dB/Hz** (bandwidth-normalized), and **relative** is the integrated band
        power ÷ the restricted-range total (nan for bands above ``rel_fmax``).
    """
    total_power = _relative_total_power(freqs, psd, rel_fmin, rel_fmax, rel_exclude)
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
