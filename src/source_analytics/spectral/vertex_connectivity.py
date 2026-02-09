"""Vertex-level functional connectivity: imaginary coherence and FCD.

Computes all-to-all imaginary coherence between source vertices using
vectorized STFT + matrix multiply, then derives Functional Connectivity
Density (FCD) maps showing how connected each vertex is to the rest of
the brain.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import stft, get_window

logger = logging.getLogger(__name__)


def compute_vertex_connectivity_matrix(
    stc_data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    metric: str = "imag_coherence",
    nperseg: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """Compute all-to-all connectivity matrix for source vertices.

    Uses vectorized STFT + matrix multiply for O(n_seg) complexity instead
    of O(n_vertices^2) individual CSD calls.

    Parameters
    ----------
    stc_data : ndarray, shape (n_vertices, n_times)
        Source time courses (signed, not magnitude).
    sfreq : float
        Sampling frequency.
    band : tuple[float, float]
        Frequency band (fmin, fmax) to average connectivity over.
    metric : str
        Connectivity metric. Currently only "imag_coherence" supported.
    nperseg : int, optional
        Welch segment length. Default: 2 * sfreq.
    window : str
        Window function.

    Returns
    -------
    conn_matrix : ndarray, shape (n_vertices, n_vertices)
        Symmetric connectivity matrix.
    """
    if metric != "imag_coherence":
        raise ValueError(f"Unknown metric: {metric}")

    n_vertices, n_times = stc_data.shape
    fmin, fmax = band

    if nperseg is None:
        nperseg = int(2 * sfreq)
    nperseg = min(nperseg, n_times)
    noverlap = nperseg // 2

    # Compute STFT for all vertices at once
    # boundary=None, padded=False to match welch/csd segment boundaries
    freqs, _, Zxx = stft(
        stc_data, fs=sfreq, window=window,
        nperseg=nperseg, noverlap=noverlap,
        boundary=None, padded=False,
    )
    # Zxx: (n_vertices, n_freq, n_segments)

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not band_mask.any():
        return np.zeros((n_vertices, n_vertices))

    # Upcast to complex128 for numerical precision — float32 input yields
    # complex64 STFT coefficients (~1e-10), and the imaginary part of the
    # cross-spectrum is lost in complex64 matrix multiply.
    Zxx_band = Zxx[:, band_mask, :].astype(np.complex128)
    n_band_freq = Zxx_band.shape[1]
    n_seg = Zxx_band.shape[2]
    total = n_band_freq * n_seg

    # Accumulate cross-spectra and auto-spectra via segment loop
    # Memory: O(n_vertices * n_band_freq) per iteration
    csd_sum = np.zeros((n_vertices, n_vertices), dtype=np.complex128)
    auto_sum = np.zeros(n_vertices, dtype=np.float64)

    for seg_idx in range(n_seg):
        Z_seg = Zxx_band[:, :, seg_idx]  # (n_vertices, n_band_freq)
        # Cross-spectral matrix for this segment (summed over band freqs)
        csd_sum += Z_seg @ np.conj(Z_seg).T
        # Auto-spectra
        auto_sum += np.sum(np.abs(Z_seg) ** 2, axis=1)

    # Mean cross-spectrum and auto-spectrum over band freqs and segments
    csd_mean = csd_sum / total
    auto_mean = np.maximum(auto_sum / total, np.finfo(float).tiny)

    # Normalize to coherency and extract imaginary coherence
    norm = np.sqrt(np.outer(auto_mean, auto_mean))
    coherency = csd_mean / norm
    conn_matrix = np.abs(np.imag(coherency))
    np.fill_diagonal(conn_matrix, 0.0)

    return conn_matrix


def compute_vertex_connectivity_matrix_epochs(
    epochs: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    metric: str = "imag_coherence",
    nperseg: int | None = None,
) -> np.ndarray:
    """Compute connectivity matrix averaged over epochs.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_vertices, n_times)
        Epoched source time courses.
    sfreq : float
        Sampling frequency.
    band : tuple[float, float]
        Frequency band.
    metric : str
        Connectivity metric.
    nperseg : int, optional
        Welch segment length.

    Returns
    -------
    conn_matrix : ndarray, shape (n_vertices, n_vertices)
        Epoch-averaged connectivity matrix.
    """
    n_epochs = epochs.shape[0]
    matrices = []
    for ep in range(n_epochs):
        mat = compute_vertex_connectivity_matrix(
            epochs[ep], sfreq, band, metric=metric, nperseg=nperseg,
        )
        matrices.append(mat)
    return np.mean(matrices, axis=0)


def compute_fcd(
    conn_matrix: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Compute Functional Connectivity Density (FCD) per vertex.

    FCD counts the number of connections above threshold for each vertex,
    normalized by the total number of possible connections.

    Parameters
    ----------
    conn_matrix : ndarray, shape (n_vertices, n_vertices)
        Connectivity matrix.
    threshold : float
        Minimum connectivity value to count as a connection.

    Returns
    -------
    fcd : ndarray, shape (n_vertices,)
        Normalized FCD per vertex (0 to 1).
    """
    n = conn_matrix.shape[0]
    above_thresh = conn_matrix > threshold
    np.fill_diagonal(above_thresh, False)
    degree = above_thresh.sum(axis=1).astype(float)
    return degree / (n - 1)


def compute_seed_connectivity(
    stc_data: np.ndarray,
    sfreq: float,
    seed_idx: int,
    band: tuple[float, float],
    metric: str = "imag_coherence",
    nperseg: int | None = None,
) -> np.ndarray:
    """Compute connectivity of one seed vertex to all others.

    Parameters
    ----------
    stc_data : ndarray, shape (n_vertices, n_times)
    sfreq : float
    seed_idx : int
        Index of seed vertex.
    band : tuple[float, float]
    metric : str
    nperseg : int, optional

    Returns
    -------
    connectivity : ndarray, shape (n_vertices,)
    """
    # Use the full matrix computation (now vectorized and fast)
    conn_mat = compute_vertex_connectivity_matrix(
        stc_data, sfreq, band, metric=metric, nperseg=nperseg,
    )
    return conn_mat[seed_idx]
