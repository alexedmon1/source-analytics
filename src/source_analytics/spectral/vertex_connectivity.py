"""Vertex-level functional connectivity: multiple metrics and FCD.

Computes all-to-all connectivity between source vertices using
vectorized STFT + matrix multiply, then derives Functional Connectivity
Density (FCD) maps showing how connected each vertex is to the rest of
the brain.

Supported metrics:
- coherence: magnitude-squared coherence |CSD|^2 / (PSD_i * PSD_j)
- imag_coherence: |Im(coherency)| — volume-conduction resistant
- pli: Phase Lag Index |mean(sign(Im(CSD)))| (Stam 2007)
- dwpli: Debiased weighted PLI (Vinck 2011)
- wpli: Weighted PLI, non-debiased (Vinck 2011)
- dpli: Directed PLI (Stam & van Straaten 2012) — asymmetric; dpli[a,b]>0.5 ⇒ a leads b
- aec: Orthogonalized amplitude envelope correlation (Hipp 2012)
- partial_corr: Partial correlation via precision matrix
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import stft, hilbert, butter, sosfiltfilt

logger = logging.getLogger(__name__)

_SPECTRAL_METRICS = {"coherence", "imag_coherence", "pli", "dwpli", "wpli", "dpli"}
_ALL_METRICS = _SPECTRAL_METRICS | {"aec", "partial_corr"}


def _stft_cross_spectra(
    stc_data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    nperseg: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute STFT and accumulate cross/auto spectra over band.

    Returns
    -------
    csd_sum : (n_vertices, n_vertices) complex128
    auto_sum : (n_vertices,) float64
    Zxx_band : (n_vertices, n_band_freq, n_seg) complex128
    total : int — n_band_freq * n_seg
    """
    n_vertices, n_times = stc_data.shape
    fmin, fmax = band

    if nperseg is None:
        nperseg = int(2 * sfreq)
    nperseg = min(nperseg, n_times)
    noverlap = nperseg // 2

    freqs, _, Zxx = stft(
        stc_data, fs=sfreq, window=window,
        nperseg=nperseg, noverlap=noverlap,
        boundary=None, padded=False,
    )

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not band_mask.any():
        return (
            np.zeros((n_vertices, n_vertices), dtype=np.complex128),
            np.zeros(n_vertices, dtype=np.float64),
            np.zeros((n_vertices, 0, 0), dtype=np.complex128),
            0,
        )

    Zxx_band = Zxx[:, band_mask, :].astype(np.complex128)
    n_band_freq = Zxx_band.shape[1]
    n_seg = Zxx_band.shape[2]
    total = n_band_freq * n_seg

    csd_sum = np.zeros((n_vertices, n_vertices), dtype=np.complex128)
    auto_sum = np.zeros(n_vertices, dtype=np.float64)

    for seg_idx in range(n_seg):
        Z_seg = Zxx_band[:, :, seg_idx]
        csd_sum += Z_seg @ np.conj(Z_seg).T
        auto_sum += np.sum(np.abs(Z_seg) ** 2, axis=1)

    return csd_sum, auto_sum, Zxx_band, total


def _compute_coherence(csd_sum, auto_sum, total, n_vertices):
    """Magnitude-squared coherence from accumulated cross-spectra."""
    csd_mean = csd_sum / max(total, 1)
    auto_mean = np.maximum(auto_sum / max(total, 1), np.finfo(float).tiny)
    norm = np.outer(auto_mean, auto_mean)
    coh = np.abs(csd_mean) ** 2 / norm
    np.fill_diagonal(coh, 0.0)
    return coh


def _compute_imag_coherence(csd_sum, auto_sum, total, n_vertices):
    """Imaginary coherence |Im(coherency)|."""
    csd_mean = csd_sum / max(total, 1)
    auto_mean = np.maximum(auto_sum / max(total, 1), np.finfo(float).tiny)
    norm = np.sqrt(np.outer(auto_mean, auto_mean))
    coherency = csd_mean / norm
    conn = np.abs(np.imag(coherency))
    np.fill_diagonal(conn, 0.0)
    return conn


def _compute_pli(Zxx_band, n_vertices):
    """Phase Lag Index: |mean_segments(sign(Im(CSD)))| per freq, averaged over band.

    Uses vectorized matrix operations per segment.
    """
    n_band_freq = Zxx_band.shape[1]
    n_seg = Zxx_band.shape[2]
    if n_seg == 0 or n_band_freq == 0:
        return np.zeros((n_vertices, n_vertices))

    # Accumulate sign(Im(CSD)) across segments per freq
    # For each freq bin, compute sign of Im of cross-spectrum matrix
    sign_sum = np.zeros((n_vertices, n_vertices, n_band_freq), dtype=np.float64)

    for seg_idx in range(n_seg):
        Z_seg = Zxx_band[:, :, seg_idx]  # (n_vertices, n_band_freq)
        for fi in range(n_band_freq):
            z = Z_seg[:, fi]  # (n_vertices,)
            csd_mat = np.outer(z, np.conj(z))
            sign_sum[:, :, fi] += np.sign(np.imag(csd_mat))

    # PLI: |mean over segments| per freq, then mean over band
    pli_per_freq = np.abs(sign_sum / n_seg)  # (n_v, n_v, n_band_freq)
    pli = np.mean(pli_per_freq, axis=2)
    np.fill_diagonal(pli, 0.0)
    return pli


def _compute_dwpli(Zxx_band, n_vertices):
    """Debiased weighted PLI (Vinck 2011), vectorized.

    dwPLI_f = (sum_t(Im)^2 - sum_t(Im^2)) / (sum_t(|Im|)^2 - sum_t(Im^2))
    per frequency, then averaged over band.
    """
    tiny = np.finfo(float).tiny
    n_band_freq = Zxx_band.shape[1]
    n_seg = Zxx_band.shape[2]
    if n_seg == 0 or n_band_freq == 0:
        return np.zeros((n_vertices, n_vertices))

    # Accumulate per-freq statistics across segments
    sum_im = np.zeros((n_vertices, n_vertices, n_band_freq), dtype=np.float64)
    sum_im_sq = np.zeros_like(sum_im)
    sum_abs_im = np.zeros_like(sum_im)

    for seg_idx in range(n_seg):
        Z_seg = Zxx_band[:, :, seg_idx]  # (n_vertices, n_band_freq)
        for fi in range(n_band_freq):
            z = Z_seg[:, fi]
            im_csd = np.imag(np.outer(z, np.conj(z)))
            sum_im[:, :, fi] += im_csd
            sum_im_sq[:, :, fi] += im_csd ** 2
            sum_abs_im[:, :, fi] += np.abs(im_csd)

    numer = sum_im ** 2 - sum_im_sq
    denom = sum_abs_im ** 2 - sum_im_sq
    valid = np.abs(denom) > tiny

    dwpli_per_freq = np.zeros_like(numer)
    dwpli_per_freq[valid] = numer[valid] / denom[valid]
    dwpli_per_freq = np.clip(dwpli_per_freq, 0.0, 1.0)

    dwpli = np.mean(dwpli_per_freq, axis=2)
    np.fill_diagonal(dwpli, 0.0)
    return dwpli


def _compute_wpli(Zxx_band, n_vertices):
    """Weighted PLI, non-debiased (Vinck 2011), vectorized.

    wPLI_f = |sum_t Im(CSD)| / sum_t |Im(CSD)| per frequency, then averaged
    over band. Symmetric. (dwPLI is the bias-corrected variant; both are kept
    so the debiasing effect can be compared.)
    """
    tiny = np.finfo(float).tiny
    n_band_freq = Zxx_band.shape[1]
    n_seg = Zxx_band.shape[2]
    if n_seg == 0 or n_band_freq == 0:
        return np.zeros((n_vertices, n_vertices))

    sum_im = np.zeros((n_vertices, n_vertices, n_band_freq), dtype=np.float64)
    sum_abs_im = np.zeros_like(sum_im)

    for seg_idx in range(n_seg):
        Z_seg = Zxx_band[:, :, seg_idx]
        for fi in range(n_band_freq):
            z = Z_seg[:, fi]
            im_csd = np.imag(np.outer(z, np.conj(z)))
            sum_im[:, :, fi] += im_csd
            sum_abs_im[:, :, fi] += np.abs(im_csd)

    valid = sum_abs_im > tiny
    wpli_per_freq = np.zeros_like(sum_im)
    wpli_per_freq[valid] = np.abs(sum_im[valid]) / sum_abs_im[valid]
    wpli = np.mean(wpli_per_freq, axis=2)
    np.fill_diagonal(wpli, 0.0)
    return wpli


def _compute_dpli(Zxx_band, n_vertices):
    """Directed PLI (Stam & van Straaten 2012), vectorized.

    dPLI_f = mean_t H(Im(CSD)) per frequency, averaged over band, where H is
    the Heaviside step (H(0)=0.5). **Asymmetric**: with CSD[a,b]=z_a·conj(z_b),
    Im is antisymmetric, so dPLI[a,b] = 1 − dPLI[b,a]. dPLI[a,b] > 0.5 ⇒ vertex
    a phase-leads vertex b; 0.5 ⇒ no preferred lead/lag. The returned matrix is
    NOT symmetric — callers that symmetrize (NBS, graph) must handle dPLI
    explicitly as ordered a→b edges.
    """
    n_band_freq = Zxx_band.shape[1]
    n_seg = Zxx_band.shape[2]
    if n_seg == 0 or n_band_freq == 0:
        return np.zeros((n_vertices, n_vertices))

    h_sum = np.zeros((n_vertices, n_vertices, n_band_freq), dtype=np.float64)
    for seg_idx in range(n_seg):
        Z_seg = Zxx_band[:, :, seg_idx]
        for fi in range(n_band_freq):
            z = Z_seg[:, fi]
            im_csd = np.imag(np.outer(z, np.conj(z)))
            h_sum[:, :, fi] += 0.5 * (np.sign(im_csd) + 1.0)

    dpli = np.mean(h_sum / n_seg, axis=2)
    np.fill_diagonal(dpli, 0.0)
    return dpli


def _orth_log_power(z_ref, z_other, eps):
    """Hipp-2012 orthogonalized log-power correlation, one direction.

    ``Y_⊥X = imag(Y · X*/|X|)`` then Pearson of ``log|X|²`` vs ``log(Y_⊥X)²``.
    """
    y_orth = np.imag(z_other * np.conj(z_ref) / (np.abs(z_ref) + eps))
    log_pow_ref = np.log(np.abs(z_ref) ** 2 + eps)
    log_pow_orth = np.log(y_orth ** 2 + eps)
    return float(np.corrcoef(log_pow_ref, log_pow_orth)[0, 1])


def _compute_aec(stc_data, sfreq, band, n_vertices):
    """Orthogonalized amplitude envelope correlation (Hipp et al. 2012, Nat Neurosci).

    Hipp orthogonalization ``Y_⊥X = imag(Y·X*/|X|)`` (removes zero-lag volume
    conduction), then Pearson of the log-power envelopes (square → log →
    correlate), averaged over both directions. See CONNECTIVITY_METHODS.md.
    """
    eps = np.finfo(float).tiny
    fmin, fmax = band
    nyq = sfreq / 2
    lo = max(fmin / nyq, 1e-5)
    hi = min(fmax / nyq, 0.9999)
    sos = butter(4, [lo, hi], btype="band", output="sos")

    # Filter and compute analytic signal for each vertex
    n_times = stc_data.shape[1]
    analytic = np.empty((n_times, n_vertices), dtype=np.complex128)
    for i in range(n_vertices):
        filtered = sosfiltfilt(sos, stc_data[i])
        analytic[:, i] = hilbert(filtered)

    aec = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            zi = analytic[:, i]
            zj = analytic[:, j]
            r1 = _orth_log_power(zi, zj, eps)   # j orthogonalized w.r.t. i
            r2 = _orth_log_power(zj, zi, eps)   # i orthogonalized w.r.t. j
            val = (r1 + r2) / 2.0
            aec[i, j] = val
            aec[j, i] = val

    return aec


def _compute_partial_corr(stc_data, sfreq, band, n_vertices):
    """Partial correlation via precision matrix on band-filtered data."""
    fmin, fmax = band
    nyq = sfreq / 2
    lo = max(fmin / nyq, 1e-5)
    hi = min(fmax / nyq, 0.9999)
    sos = butter(4, [lo, hi], btype="band", output="sos")

    n_times = stc_data.shape[1]
    filtered = np.empty((n_times, n_vertices))
    for i in range(n_vertices):
        filtered[:, i] = sosfiltfilt(sos, stc_data[i])

    cov = np.cov(filtered, rowvar=False)
    trace = np.trace(cov) / n_vertices
    alpha = 0.01
    cov_reg = (1 - alpha) * cov + alpha * trace * np.eye(n_vertices)

    try:
        prec = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        logger.warning("Singular covariance matrix — returning zeros")
        return np.zeros((n_vertices, n_vertices))

    d = np.sqrt(np.diag(prec))
    d[d == 0] = 1.0
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 0.0)
    # Take absolute value for unsigned connectivity strength
    return np.abs(pcorr)


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
        Connectivity metric: "coherence", "imag_coherence", "pli", "dwpli",
        "aec", or "partial_corr".
    nperseg : int, optional
        Welch segment length. Default: 2 * sfreq.
    window : str
        Window function.

    Returns
    -------
    conn_matrix : ndarray, shape (n_vertices, n_vertices)
        Symmetric connectivity matrix.
    """
    if metric not in _ALL_METRICS:
        raise ValueError(
            f"Unknown metric: {metric}. Supported: {sorted(_ALL_METRICS)}"
        )

    n_vertices = stc_data.shape[0]

    # AEC and partial_corr don't use the STFT path
    if metric == "aec":
        return _compute_aec(stc_data, sfreq, band, n_vertices)
    if metric == "partial_corr":
        return _compute_partial_corr(stc_data, sfreq, band, n_vertices)

    # Spectral metrics: coherence, imag_coherence, pli, dwpli
    csd_sum, auto_sum, Zxx_band, total = _stft_cross_spectra(
        stc_data, sfreq, band, nperseg=nperseg, window=window,
    )

    if total == 0:
        return np.zeros((n_vertices, n_vertices))

    if metric == "coherence":
        return _compute_coherence(csd_sum, auto_sum, total, n_vertices)
    elif metric == "imag_coherence":
        return _compute_imag_coherence(csd_sum, auto_sum, total, n_vertices)
    elif metric == "pli":
        return _compute_pli(Zxx_band, n_vertices)
    elif metric == "dwpli":
        return _compute_dwpli(Zxx_band, n_vertices)
    elif metric == "wpli":
        return _compute_wpli(Zxx_band, n_vertices)
    elif metric == "dpli":
        return _compute_dpli(Zxx_band, n_vertices)

    raise ValueError(f"Unhandled metric: {metric}")  # pragma: no cover


def compute_vertex_connectivity_matrix_multi(
    stc_data: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    metrics: list[str],
    nperseg: int | None = None,
    window: str = "hann",
) -> dict[str, np.ndarray]:
    """Compute multiple connectivity metrics in a single pass where possible.

    Spectral metrics (coherence, imag_coherence, pli, dwpli) share one
    STFT computation. AEC and partial_corr are computed independently.

    Parameters
    ----------
    stc_data : ndarray, shape (n_vertices, n_times)
        Source time courses (signed, not magnitude).
    sfreq : float
        Sampling frequency.
    band : tuple[float, float]
        Frequency band (fmin, fmax).
    metrics : list[str]
        List of metrics to compute.
    nperseg : int, optional
        Welch segment length.
    window : str
        Window function.

    Returns
    -------
    results : dict[str, ndarray]
        Mapping of metric name to (n_vertices, n_vertices) matrix.
    """
    for m in metrics:
        if m not in _ALL_METRICS:
            raise ValueError(f"Unknown metric: {m}. Supported: {sorted(_ALL_METRICS)}")

    n_vertices = stc_data.shape[0]
    results: dict[str, np.ndarray] = {}

    # Compute STFT once for all spectral metrics
    spectral_requested = [m for m in metrics if m in _SPECTRAL_METRICS]
    if spectral_requested:
        csd_sum, auto_sum, Zxx_band, total = _stft_cross_spectra(
            stc_data, sfreq, band, nperseg=nperseg, window=window,
        )

        if total == 0:
            for m in spectral_requested:
                results[m] = np.zeros((n_vertices, n_vertices))
        else:
            if "coherence" in spectral_requested:
                results["coherence"] = _compute_coherence(
                    csd_sum, auto_sum, total, n_vertices,
                )
            if "imag_coherence" in spectral_requested:
                results["imag_coherence"] = _compute_imag_coherence(
                    csd_sum, auto_sum, total, n_vertices,
                )
            if "pli" in spectral_requested:
                results["pli"] = _compute_pli(Zxx_band, n_vertices)
            if "dwpli" in spectral_requested:
                results["dwpli"] = _compute_dwpli(Zxx_band, n_vertices)
            if "wpli" in spectral_requested:
                results["wpli"] = _compute_wpli(Zxx_band, n_vertices)
            if "dpli" in spectral_requested:
                results["dpli"] = _compute_dpli(Zxx_band, n_vertices)

    # Non-spectral metrics
    if "aec" in metrics:
        results["aec"] = _compute_aec(stc_data, sfreq, band, n_vertices)
    if "partial_corr" in metrics:
        results["partial_corr"] = _compute_partial_corr(
            stc_data, sfreq, band, n_vertices,
        )

    return results


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


def compute_vertex_connectivity_matrix_epochs_multi(
    epochs: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    metrics: list[str],
    nperseg: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute multiple connectivity metrics averaged over epochs.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_vertices, n_times)
    sfreq : float
    band : tuple[float, float]
    metrics : list[str]
    nperseg : int, optional

    Returns
    -------
    results : dict[str, ndarray]
        Each metric -> epoch-averaged (n_vertices, n_vertices) matrix.
    """
    n_epochs = epochs.shape[0]
    # Accumulate per metric
    accum: dict[str, list[np.ndarray]] = {m: [] for m in metrics}

    for ep in range(n_epochs):
        ep_results = compute_vertex_connectivity_matrix_multi(
            epochs[ep], sfreq, band, metrics=metrics, nperseg=nperseg,
        )
        for m in metrics:
            accum[m].append(ep_results[m])

    return {m: np.mean(accum[m], axis=0) for m in metrics}


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
    conn_mat = compute_vertex_connectivity_matrix(
        stc_data, sfreq, band, metric=metric, nperseg=nperseg,
    )
    return conn_mat[seed_idx]
