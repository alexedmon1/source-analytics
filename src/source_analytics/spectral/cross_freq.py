"""Cross-frequency coupling beyond PAC: amplitude–amplitude (AAC) and n:m phase–phase (PPC).

These complement the Tort-2010 phase–amplitude MI in ``pac.py``. All operate on
a 2-D ``(n_signals, n_times)`` array so the same kernel serves ROI time courses
and vertex source time courses (all-pairs N×N output per band pair).

- **AAC** (cross-frequency amplitude envelope correlation): correlation between
  the band-X amplitude envelope of signal i and the band-Y envelope of signal j.
  When band_x == band_y this is amplitude envelope correlation (≈ the same-band
  AEC, minus the leakage orthogonalization that lives in ``connectivity.py``).
- **PPC** (n:m phase–phase coupling): |⟨exp(i(n·φ_X − m·φ_Y))⟩|, the consistency
  of an n:m harmonic phase relationship (n·f_X ≈ m·f_Y).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import hilbert, sosfiltfilt

from .pac import _design_bandpass

logger = logging.getLogger(__name__)


def _band_analytic(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """Bandpass (zero-phase) + Hilbert → analytic signal.

    Parameters
    ----------
    data : ndarray (n_signals, n_times)
    fmin, fmax : float
        Band edges in Hz.

    Returns
    -------
    analytic : ndarray (n_times, n_signals) complex
        Transposed so columns are signals — convenient for column-wise corr.
    """
    data = np.atleast_2d(data)
    sos = _design_bandpass(fmin, fmax, sfreq)
    filt = sosfiltfilt(sos, data, axis=1)
    analytic = hilbert(filt, axis=1)
    return analytic.T  # (n_times, n_signals)


def compute_aac(
    data: np.ndarray,
    sfreq: float,
    band_x: tuple[float, float],
    band_y: tuple[float, float],
) -> np.ndarray:
    """Cross-frequency amplitude–amplitude coupling.

    Returns the N×N matrix ``M[i, j] = pearson_corr(env_X(i), env_Y(j))`` where
    ``env_X`` is the Hilbert amplitude envelope in ``band_x`` and ``env_Y`` in
    ``band_y``. Asymmetric when ``band_x != band_y`` (M[i,j] pairs X@i with Y@j);
    symmetric and ≈ envelope correlation when the bands are equal. Range [-1, 1].
    """
    ax = np.abs(_band_analytic(data, sfreq, *band_x))  # (T, N)
    ay = np.abs(_band_analytic(data, sfreq, *band_y))  # (T, N)

    axc = ax - ax.mean(axis=0)
    ayc = ay - ay.mean(axis=0)
    sx = np.sqrt((axc ** 2).sum(axis=0))
    sy = np.sqrt((ayc ** 2).sum(axis=0))

    cov = axc.T @ ayc                      # cov[i,j] = Σ_t axc[t,i]·ayc[t,j]
    denom = np.outer(sx, sy)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(denom > 0, cov / denom, 0.0)
    return m


def compute_ppc(
    data: np.ndarray,
    sfreq: float,
    band_x: tuple[float, float],
    band_y: tuple[float, float],
    n: int = 1,
    m: int = 1,
) -> np.ndarray:
    """n:m phase–phase coupling.

    Returns the N×N matrix
    ``PPC[i, j] = |mean_t exp(i·(n·φ_X(i) − m·φ_Y(j)))|`` where ``φ_X`` is the
    instantaneous phase in ``band_x`` and ``φ_Y`` in ``band_y``. ``n:m`` is the
    harmonic ratio (n·f_X ≈ m·f_Y; n = m = 1 is the same-frequency PLV). Range
    [0, 1]; 1 = perfect n:m phase locking. The diagonal is within-signal n:m
    coupling.
    """
    px = np.angle(_band_analytic(data, sfreq, *band_x))  # (T, N)
    py = np.angle(_band_analytic(data, sfreq, *band_y))  # (T, N)

    ex = np.exp(1j * n * px)               # (T, N)
    ey = np.exp(1j * m * py)               # (T, N)

    n_times = px.shape[0]
    # cross[i,j] = mean_t exp(i(n φ_X(i) − m φ_Y(j))) = mean_t ex[t,i]·conj(ey[t,j])
    cross = (ex.T @ np.conj(ey)) / n_times
    return np.abs(cross)
