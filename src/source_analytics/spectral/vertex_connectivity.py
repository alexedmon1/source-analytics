"""Vertex-level functional connectivity: imaginary coherence and FCD.

Computes all-to-all imaginary coherence between source vertices using
CSD-based methods, then derives Functional Connectivity Density (FCD)
maps showing how connected each vertex is to the rest of the brain.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import csd, welch

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
    n_vertices, n_times = stc_data.shape
    fmin, fmax = band

    if nperseg is None:
        nperseg = int(2 * sfreq)
    nperseg = min(nperseg, n_times)
    noverlap = nperseg // 2

    # Compute auto-spectra for all vertices
    freqs, pxx = welch(stc_data, fs=sfreq, window=window,
                       nperseg=nperseg, noverlap=noverlap, axis=-1)
    band_mask = (freqs >= fmin) & (freqs <= fmax)

    # Average auto-spectra in band
    pxx_band = pxx[:, band_mask].mean(axis=1)  # (n_vertices,)
    pxx_band = np.maximum(pxx_band, np.finfo(float).eps)

    conn_matrix = np.zeros((n_vertices, n_vertices))

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            # Cross-spectral density
            f_csd, pxy = csd(
                stc_data[i], stc_data[j],
                fs=sfreq, window=window,
                nperseg=nperseg, noverlap=noverlap,
            )
            csd_mask = (f_csd >= fmin) & (f_csd <= fmax)

            if metric == "imag_coherence":
                # Imaginary coherence: |Im(Pxy / sqrt(Pxx * Pyy))|
                norm = np.sqrt(pxx_band[i] * pxx_band[j])
                coherency = pxy[csd_mask].mean() / norm
                val = abs(coherency.imag)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            conn_matrix[i, j] = val
            conn_matrix[j, i] = val

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
    n_vertices, n_times = stc_data.shape
    fmin, fmax = band

    if nperseg is None:
        nperseg = int(2 * sfreq)
    nperseg = min(nperseg, n_times)
    noverlap = nperseg // 2

    # Seed auto-spectrum
    freqs, pxx_seed = welch(stc_data[seed_idx], fs=sfreq, window="hann",
                            nperseg=nperseg, noverlap=noverlap)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    pxx_seed_band = max(pxx_seed[band_mask].mean(), np.finfo(float).eps)

    # All auto-spectra
    _, pxx_all = welch(stc_data, fs=sfreq, window="hann",
                       nperseg=nperseg, noverlap=noverlap, axis=-1)
    pxx_all_band = np.maximum(pxx_all[:, band_mask].mean(axis=1), np.finfo(float).eps)

    connectivity = np.zeros(n_vertices)
    for j in range(n_vertices):
        if j == seed_idx:
            continue
        f_csd, pxy = csd(
            stc_data[seed_idx], stc_data[j],
            fs=sfreq, window="hann", nperseg=nperseg, noverlap=noverlap,
        )
        csd_mask = (f_csd >= fmin) & (f_csd <= fmax)
        norm = np.sqrt(pxx_seed_band * pxx_all_band[j])
        coherency = pxy[csd_mask].mean() / norm
        connectivity[j] = abs(coherency.imag)

    return connectivity
