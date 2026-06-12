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
    """Cross-frequency amplitude–amplitude (power–power) coupling.

    Returns the N×N matrix ``M[i, j] = pearson_corr(P_X(i), P_Y(j))`` where
    ``P_X`` is the band-X **power** envelope (squared Hilbert amplitude) and
    ``P_Y`` the band-Y power envelope. Asymmetric when ``band_x != band_y``
    (M[i,j] pairs X@i with Y@j); symmetric when the bands are equal. Range [-1, 1].

    Design choices (cross-frequency AAC has no single canonical paper; see
    CONNECTIVITY_METHODS.md): **power** (squared) envelopes — the FFT/comodulogram
    lineage (Masimore et al. 2004, *J Neurosci Methods*); **Pearson**; **raw**
    (not orthogonalized) envelopes. Conceptual primary: Bruns et al. 2000,
    *NeuroReport* (amplitude envelope correlation among incoherent signals).
    """
    ax = np.abs(_band_analytic(data, sfreq, *band_x)) ** 2  # (T, N) band-X power
    ay = np.abs(_band_analytic(data, sfreq, *band_y)) ** 2  # (T, N) band-Y power

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
    *,
    n_surrogates: int = 0,
    min_shift_sec: float = 1.0,
    seed: int | None = None,
):
    """n:m phase–phase coupling (Palva et al. 2005 phase-locking factor).

    Returns the N×N matrix
    ``PPC[i, j] = |mean_t exp(i·(n·φ_X(i) − m·φ_Y(j)))|`` (the n:m PLF) where
    ``φ_X`` is the instantaneous phase in ``band_x`` and ``φ_Y`` in ``band_y``.
    ``n:m`` is the harmonic ratio (n·f_X ≈ m·f_Y; n = m = 1 is the same-frequency
    PLV). Range [0, 1]; 1 = perfect n:m phase locking. The diagonal is
    within-signal n:m coupling.

    The raw PLF is positively biased at finite N (E[PLF] ≈ √π/(2√N) for random
    phases). With ``n_surrogates > 0`` a **surrogate z-score** is returned
    alongside the PLF: the band-Y phase is circularly time-shifted by random
    amounts (preserving each signal's own phase statistics while destroying the
    cross-frequency relationship), and ``z = (PLF − mean_surr) / std_surr``.

    Returns
    -------
    plf : ndarray (N, N)                      if n_surrogates == 0
    (plf, z) : tuple of ndarray (N, N), (N, N)  if n_surrogates > 0
    """
    px = np.angle(_band_analytic(data, sfreq, *band_x))  # (T, N)
    py = np.angle(_band_analytic(data, sfreq, *band_y))  # (T, N)

    ex = np.exp(1j * n * px)               # (T, N)
    ey = np.exp(1j * m * py)               # (T, N)

    n_times = px.shape[0]
    # cross[i,j] = mean_t exp(i(n φ_X(i) − m φ_Y(j))) = mean_t ex[t,i]·conj(ey[t,j])
    plf = np.abs((ex.T @ np.conj(ey)) / n_times)

    if n_surrogates <= 0:
        return plf

    rng = np.random.default_rng(seed)
    min_shift = max(int(min_shift_sec * sfreq), 1)
    max_shift = n_times - min_shift
    # Welford online mean/variance over surrogate PLF matrices
    s_mean = np.zeros_like(plf)
    s_m2 = np.zeros_like(plf)
    for k in range(n_surrogates):
        shift = (int(rng.integers(min_shift, max_shift))
                 if max_shift > min_shift else min_shift)
        ey_shift = np.roll(ey, shift, axis=0)
        surr = np.abs((ex.T @ np.conj(ey_shift)) / n_times)
        delta = surr - s_mean
        s_mean += delta / (k + 1)
        s_m2 += delta * (surr - s_mean)
    s_std = np.sqrt(s_m2 / max(n_surrogates - 1, 1))
    z = np.where(s_std > 0, (plf - s_mean) / s_std, 0.0)
    return plf, z
