"""Functional connectivity: coherence, imaginary coherence, PLI, dwPLI, AEC, and partial correlation."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import welch, csd, stft, hilbert, butter, sosfiltfilt

logger = logging.getLogger(__name__)


def compute_connectivity_matrix(
    roi_timeseries: dict[str, np.ndarray],
    sfreq: float,
    bands: dict[str, tuple[float, float]],
    *,
    nperseg: int | None = None,
    window: str = "hann",
) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Compute connectivity matrices for all ROI pairs.

    Metrics computed:
    - **coherence**: magnitude-squared coherence (Welch CSD)
    - **imag_coherence**: mean |Im(Cxy)| (volume-conduction resistant)
    - **pli**: phase lag index |mean(sign(Im(Pxy)))| (Stam 2007)
    - **dwpli**: debiased weighted PLI (Vinck 2011) — per-segment STFT
    - **aec**: orthogonalized amplitude envelope correlation (Hipp 2012)
    - **partial_corr**: partial correlation via precision matrix

    Parameters
    ----------
    roi_timeseries : dict[str, ndarray]
        Mapping of ROI name -> 1-D time course (signed, phase-preserving).
    sfreq : float
        Sampling frequency in Hz.
    bands : dict[str, tuple[float, float]]
        Frequency band definitions, e.g. ``{"alpha": (8, 13)}``.
    nperseg : int, optional
        Segment length for Welch/CSD/STFT. Default: ``2 * sfreq`` (2-second windows).
    window : str
        Window function (default: Hann).

    Returns
    -------
    band_results : dict[str, dict[str, ndarray]]
        Each metric is an (n_rois, n_rois) symmetric matrix per band.
    roi_names : list[str]
        Ordered list of ROI names (rows/columns of matrices).
    """
    roi_names = sorted(roi_timeseries.keys())
    n_rois = len(roi_names)
    ts_list = [roi_timeseries[name] for name in roi_names]

    if nperseg is None:
        nperseg = int(2 * sfreq)
    # Clamp to shortest timeseries
    min_len = min(len(ts) for ts in ts_list)
    nperseg = min(nperseg, min_len)
    noverlap = nperseg // 2

    # Pre-compute all auto-spectra (PSD)
    auto_spectra: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for ts in ts_list:
        f, pxx = welch(ts, fs=sfreq, window=window, nperseg=nperseg,
                        noverlap=noverlap)
        auto_spectra.append(pxx)
        if freqs is None:
            freqs = f

    assert freqs is not None

    # Build frequency masks per band
    band_masks: dict[str, np.ndarray] = {}
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not mask.any():
            continue
        band_masks[band_name] = mask

    # Initialize result matrices
    band_results: dict[str, dict[str, np.ndarray]] = {}
    for band_name in band_masks:
        band_results[band_name] = {
            "coherence": np.eye(n_rois, dtype=np.float64),
            "imag_coherence": np.zeros((n_rois, n_rois), dtype=np.float64),
            "pli": np.zeros((n_rois, n_rois), dtype=np.float64),
        }

    # Compute CSD for each unique pair and derive coherence
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            _, pxy = csd(ts_list[i], ts_list[j], fs=sfreq, window=window,
                         nperseg=nperseg, noverlap=noverlap)

            pxx_i = auto_spectra[i]
            pxx_j = auto_spectra[j]

            for band_name, mask in band_masks.items():
                csd_band = pxy[mask]
                pxx_i_band = pxx_i[mask]
                pxx_j_band = pxx_j[mask]

                # Magnitude-squared coherence: |Pxy|^2 / (Pxx * Pyy)
                denom = pxx_i_band * pxx_j_band
                coh_freq = np.abs(csd_band) ** 2 / np.where(denom > 0, denom, 1.0)
                coh_mean = float(np.mean(coh_freq))

                # Imaginary coherence: |Im(Pxy / sqrt(Pxx * Pyy))|
                norm = np.sqrt(np.where(denom > 0, denom, 1.0))
                icoh_freq = np.abs(np.imag(csd_band / norm))
                icoh_mean = float(np.mean(icoh_freq))

                # Phase Lag Index: |mean(sign(Im(Pxy)))|
                pli_val = float(np.abs(np.mean(np.sign(np.imag(csd_band)))))

                band_results[band_name]["coherence"][i, j] = coh_mean
                band_results[band_name]["coherence"][j, i] = coh_mean
                band_results[band_name]["imag_coherence"][i, j] = icoh_mean
                band_results[band_name]["imag_coherence"][j, i] = icoh_mean
                band_results[band_name]["pli"][i, j] = pli_val
                band_results[band_name]["pli"][j, i] = pli_val

    # --- Debiased Weighted PLI (Vinck et al. 2011) via per-segment STFT ---
    _compute_dwpli(ts_list, sfreq, n_rois, nperseg, noverlap, window,
                   band_masks, band_results)

    # --- Orthogonalized AEC (Hipp et al. 2012) ---
    for band_name, (fmin, fmax) in bands.items():
        if band_name not in band_results:
            continue
        band_results[band_name]["aec"] = _band_orthogonalized_aec(
            ts_list, sfreq, fmin, fmax,
        )

    # --- Partial correlation on band-filtered time series ---
    for band_name, (fmin, fmax) in bands.items():
        if band_name not in band_results:
            continue
        pcorr = _band_partial_correlation(ts_list, sfreq, fmin, fmax)
        band_results[band_name]["partial_corr"] = pcorr

    return band_results, roi_names


def _compute_dwpli(
    ts_list: list[np.ndarray],
    sfreq: float,
    n_rois: int,
    nperseg: int,
    noverlap: int,
    window: str,
    band_masks: dict[str, np.ndarray],
    band_results: dict[str, dict[str, np.ndarray]],
) -> None:
    """Compute debiased weighted PLI (Vinck et al. 2011) in-place.

    Uses per-segment STFT cross-spectra. For each frequency bin within a band,
    dwPLI is computed across segments, then averaged over the band.

    dwPLI_f = (sum_t(Im)^2 - sum_t(Im^2)) / (sum_t(|Im|)^2 - sum_t(Im^2))
    """
    tiny = np.finfo(float).tiny

    # Compute STFT for all ROIs: each is (n_freqs, n_segments) complex
    stft_data: list[np.ndarray] = []
    for ts in ts_list:
        _, _, Zxx = stft(ts, fs=sfreq, window=window, nperseg=nperseg,
                         noverlap=noverlap, boundary=None, padded=False)
        stft_data.append(Zxx.astype(np.complex128))

    # Initialize dwpli matrices
    for band_name in band_masks:
        band_results[band_name]["dwpli"] = np.zeros(
            (n_rois, n_rois), dtype=np.float64,
        )

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            # Per-segment cross-spectrum: (n_freqs, n_segments)
            csd_seg = stft_data[i] * np.conj(stft_data[j])

            for band_name, mask in band_masks.items():
                im_csd = np.imag(csd_seg[mask, :])  # (n_band_freqs, n_segments)

                # dwPLI per frequency bin (across segments), then average over band
                sum_im = np.sum(im_csd, axis=1)
                sum_im_sq = np.sum(im_csd ** 2, axis=1)
                sum_abs_im_sq = np.sum(np.abs(im_csd), axis=1) ** 2

                numer = sum_im ** 2 - sum_im_sq
                denom = sum_abs_im_sq - sum_im_sq

                valid = np.abs(denom) > tiny
                dwpli_freq = np.zeros(len(sum_im))
                dwpli_freq[valid] = numer[valid] / denom[valid]
                dwpli_freq = np.clip(dwpli_freq, 0.0, 1.0)

                val = float(np.mean(dwpli_freq))
                band_results[band_name]["dwpli"][i, j] = val
                band_results[band_name]["dwpli"][j, i] = val


def _band_orthogonalized_aec(
    ts_list: list[np.ndarray],
    sfreq: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Orthogonalized amplitude envelope correlation (Hipp et al. 2012).

    Band-pass filters each ROI, computes the analytic signal via Hilbert
    transform, then for each pair orthogonalizes one signal w.r.t. the other
    (regression in complex domain) to remove zero-lag volume conduction.
    The AEC is the Pearson correlation of the envelopes, symmetrized over
    both directions.

    Returns
    -------
    aec : ndarray (n_rois, n_rois)
        Symmetric orthogonalized AEC matrix, diagonal = 1.
    """
    tiny = np.finfo(float).tiny
    n = len(ts_list)
    min_len = min(len(ts) for ts in ts_list)

    # Band-pass filter
    nyq = sfreq / 2
    lo = max(fmin / nyq, 1e-5)
    hi = min(fmax / nyq, 0.9999)
    sos = butter(4, [lo, hi], btype="band", output="sos")

    # Filter and compute analytic signal for each ROI
    analytic = np.empty((min_len, n), dtype=np.complex128)
    for i, ts in enumerate(ts_list):
        filtered = sosfiltfilt(sos, ts[:min_len])
        analytic[:, i] = hilbert(filtered)

    envelopes = np.abs(analytic)
    aec = np.eye(n, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            zi = analytic[:, i]
            zj = analytic[:, j]

            # Direction 1: orthogonalize j w.r.t. i
            #   Remove component of zj in phase with zi (zero-lag coupling)
            beta_ji = np.real(zj * np.conj(zi)) / (np.abs(zi) ** 2 + tiny)
            zj_orth = zj - beta_ji * zi
            r1 = np.corrcoef(envelopes[:, i], np.abs(zj_orth))[0, 1]

            # Direction 2: orthogonalize i w.r.t. j
            beta_ij = np.real(zi * np.conj(zj)) / (np.abs(zj) ** 2 + tiny)
            zi_orth = zi - beta_ij * zj
            r2 = np.corrcoef(envelopes[:, j], np.abs(zi_orth))[0, 1]

            # Symmetrize
            val = (r1 + r2) / 2.0
            aec[i, j] = val
            aec[j, i] = val

    return aec


def _band_partial_correlation(
    ts_list: list[np.ndarray],
    sfreq: float,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Partial correlation matrix from band-filtered time series.

    Band-pass filters each ROI, then computes partial correlations
    via the precision matrix (inverse covariance) with Ledoit-Wolf
    shrinkage for numerical stability.

    Parameters
    ----------
    ts_list : list of 1-D arrays
        ROI time courses.
    sfreq : float
        Sampling frequency.
    fmin, fmax : float
        Band edges in Hz.

    Returns
    -------
    pcorr : ndarray (n_rois, n_rois)
        Symmetric partial correlation matrix, diagonal = 1.
    """
    n = len(ts_list)
    min_len = min(len(ts) for ts in ts_list)

    # Band-pass filter
    nyq = sfreq / 2
    lo = max(fmin / nyq, 1e-5)
    hi = min(fmax / nyq, 0.9999)
    sos = butter(4, [lo, hi], btype="band", output="sos")

    filtered = np.empty((min_len, n))
    for i, ts in enumerate(ts_list):
        filtered[:, i] = sosfiltfilt(sos, ts[:min_len])

    # Covariance with Ledoit-Wolf shrinkage
    cov = np.cov(filtered, rowvar=False)
    # Shrink toward diagonal for numerical stability
    trace = np.trace(cov) / n
    alpha = 0.01
    cov_reg = (1 - alpha) * cov + alpha * trace * np.eye(n)

    try:
        prec = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        logger.warning("Singular covariance matrix — returning zeros")
        return np.zeros((n, n))

    # Convert precision to partial correlation: -P[i,j] / sqrt(P[i,i]*P[j,j])
    d = np.sqrt(np.diag(prec))
    d[d == 0] = 1.0
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)

    return pcorr
