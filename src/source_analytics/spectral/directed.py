"""Directed connectivity via MVAR: the directed transfer function (DTF).

From a multivariate autoregressive model fit ``X(t) = Σ_k A_k X(t-k) + E(t)``
the spectral transfer matrix is ``H(f) = A(f)^-1`` with
``A(f) = I - Σ_k A_k exp(-i 2π f k / fs)``. The DTF from node ``j`` to node ``i``
at frequency ``f`` is

    γ_ij(f) = |H_ij(f)| / sqrt( Σ_m |H_im(f)|² )      ∈ [0, 1]

— the fraction of the inflow to ``i`` that originates from ``j``
(Kaminski & Blinowska 1991, *Biol Cybern*).

Source/EEG signals are strongly collinear (volume conduction / source leakage:
real FORGE source data has mean inter-node |corr| ≈ 0.64). Plain least-squares
MVAR is then ill-conditioned and the fitted model is non-stationary (companion
spectral radius ≫ 1). The fit is therefore **ridge-regularized** (Tikhonov), the
standard remedy for collinear multichannel AR estimation — empirically this turns
an explosive fit (radius 11–30) into a stable one (radius ≈ 0.96) while still
recovering true directionality. See ``fit_mvar``.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ORDER = 8      # higher orders destabilize the collinear source MVAR
DEFAULT_RIDGE = 0.05   # Tikhonov penalty as a fraction of mean Gram diagonal


def fit_mvar(
    data: np.ndarray, order: int = DEFAULT_ORDER, ridge: float = DEFAULT_RIDGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares MVAR(``order``) fit with ridge regularization.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    order : int
        AR model order (lags).
    ridge : float
        Tikhonov penalty added to the Gram matrix diagonal as a fraction of its
        mean diagonal (0 disables regularization → plain least squares).

    Returns
    -------
    A : ndarray, shape (n_channels, n_channels, order)
        AR coefficient tensor; ``A[:, :, k]`` is the lag-(k+1) coefficient matrix.
    rescov : ndarray, shape (n_channels, n_channels)
        Residual covariance.
    """
    n, T = data.shape
    data = data - data.mean(axis=1, keepdims=True)
    Y = data[:, order:]                                            # (n, T-order)
    X = np.vstack([data[:, order - k:T - k] for k in range(1, order + 1)])  # (n*order, T-order)
    G = X @ X.T
    if ridge > 0:
        G = G + ridge * (np.trace(G) / G.shape[0]) * np.eye(G.shape[0])
        B = Y @ X.T @ np.linalg.inv(G)
    else:
        B = Y @ X.T @ np.linalg.pinv(G)
    resid = Y - B @ X
    A = B.reshape(n, order, n).transpose(0, 2, 1)                 # (n, n, order)
    return A, np.cov(resid)


def mvar_spectral_radius(A: np.ndarray) -> float:
    """Companion-matrix spectral radius; ``< 1`` ⟺ the fitted VAR is stable."""
    n, _, p = A.shape
    C = np.zeros((n * p, n * p))
    C[:n] = np.hstack([A[:, :, k] for k in range(p)])
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def dtf_spectrum(A: np.ndarray, sfreq: float, freqs: np.ndarray) -> np.ndarray:
    """Frequency-resolved DTF. Returns (n, n, n_freqs); ``out[i, j, f]`` = j→i."""
    n, _, p = A.shape
    out = np.empty((n, n, len(freqs)))
    for fi, f in enumerate(freqs):
        Af = np.eye(n, dtype=complex)
        for k in range(p):
            Af -= A[:, :, k] * np.exp(-2j * np.pi * f * (k + 1) / sfreq)
        H = np.linalg.inv(Af)
        Hmag2 = np.abs(H) ** 2
        out[:, :, fi] = np.sqrt(Hmag2 / Hmag2.sum(axis=1, keepdims=True))
    return out


def compute_dtf(
    node_ts: dict[str, np.ndarray],
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    order: int = DEFAULT_ORDER,
    ridge: float = DEFAULT_RIDGE,
    n_freqs: int = 128,
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Band-resolved DTF from a single ridge-MVAR fit over all nodes.

    Mirrors the interface of :func:`..transfer_entropy.compute_transfer_entropy`.

    Parameters
    ----------
    node_ts : dict[str, ndarray]
        Node name -> 1-D signed time course (same length per node).
    sfreq : float
    bands : dict[str, (lo, hi)]
    order, ridge : MVAR parameters.
    n_freqs : int
        Frequency grid resolution from 0 to Nyquist.

    Returns
    -------
    (results, node_names) where ``results[band]["dtf"][i, j]`` is the directed
    influence **source ``i`` → target ``j``**, averaged over the band (matching
    the source/target convention of the transfer-entropy matrices).
    """
    node_names = list(node_ts.keys())
    data = np.vstack([node_ts[k] for k in node_names])
    A, _ = fit_mvar(data, order=order, ridge=ridge)

    radius = mvar_spectral_radius(A)
    if radius >= 1.0:
        logger.warning(
            "MVAR unstable (spectral radius %.2f >= 1) at order=%d ridge=%.3g — "
            "DTF may be unreliable; raise ridge or lower order.", radius, order, ridge,
        )

    freqs = np.linspace(0, sfreq / 2, n_freqs)
    dtf = dtf_spectrum(A, sfreq, freqs)  # out[i, j] = j -> i

    results: dict[str, dict[str, np.ndarray]] = {}
    for band, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs <= hi)
        if not idx.any():
            continue
        # Transpose to source->target (mat[i, j] = i -> j) to match TE matrices.
        mat = dtf[:, :, idx].mean(axis=2).T.copy()
        np.fill_diagonal(mat, 0.0)
        results[band] = {"dtf": mat}
    return results, node_names
