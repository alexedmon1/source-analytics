"""Phase-Amplitude Coupling (PAC) via Modulation Index with surrogate z-scoring.

Implements the Tort et al. (2010) Modulation Index:
1. Bandpass filter for phase and amplitude bands (Butterworth, zero-phase)
2. Hilbert transform -> instantaneous phase and amplitude envelope
3. Bin phase into N bins, compute mean amplitude per bin
4. MI = KL divergence of amplitude distribution from uniform / log(N)
5. Surrogate distribution via circular time-shifts of amplitude envelope
6. z-score = (observed MI - mean(surrogates)) / std(surrogates)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

logger = logging.getLogger(__name__)


def _design_bandpass(
    fmin: float,
    fmax: float,
    sfreq: float,
    order: int = 4,
) -> np.ndarray:
    """Design a Butterworth bandpass filter (SOS format).

    Automatically reduces order to 2 for narrow bands (< 5 Hz bandwidth)
    to avoid instability.

    Parameters
    ----------
    fmin, fmax : float
        Band edges in Hz.
    sfreq : float
        Sampling frequency in Hz.
    order : int
        Filter order. Reduced to 2 if bandwidth < 5 Hz.

    Returns
    -------
    sos : ndarray
        Second-order sections representation.
    """
    bandwidth = fmax - fmin
    if bandwidth < 5.0:
        order = min(order, 2)

    nyq = sfreq / 2.0
    low = fmin / nyq
    high = fmax / nyq

    # Clamp to valid range
    low = max(low, 1e-5)
    high = min(high, 1.0 - 1e-5)

    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def _compute_mi_from_phase_amp(
    phase: np.ndarray,
    amplitude: np.ndarray,
    n_bins: int,
    bin_edges: np.ndarray,
) -> float:
    """Compute Modulation Index from pre-extracted phase and amplitude.

    Uses vectorized np.digitize + np.bincount for speed.

    Parameters
    ----------
    phase : ndarray, shape (n_samples,)
        Instantaneous phase in radians (-pi to pi).
    amplitude : ndarray, shape (n_samples,)
        Amplitude envelope.
    n_bins : int
        Number of phase bins.
    bin_edges : ndarray, shape (n_bins + 1,)
        Phase bin edges in radians.

    Returns
    -------
    mi : float
        Modulation Index (0 = uniform, higher = stronger coupling).
    """
    # Assign each sample to a phase bin (1-indexed from digitize)
    bin_idx = np.digitize(phase, bin_edges) - 1
    # Clamp to valid range [0, n_bins - 1]
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Mean amplitude per bin using bincount
    bin_sums = np.bincount(bin_idx, weights=amplitude, minlength=n_bins).astype(float)
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)

    # Avoid division by zero
    mask = bin_counts > 0
    mean_amp = np.zeros(n_bins)
    mean_amp[mask] = bin_sums[mask] / bin_counts[mask]

    # Normalize to a probability distribution
    total = mean_amp.sum()
    if total == 0:
        return 0.0
    p = mean_amp / total

    # KL divergence from uniform
    uniform = 1.0 / n_bins
    # Avoid log(0) by replacing zeros with tiny value
    p_safe = np.where(p > 0, p, 1e-20)
    kl = np.sum(p_safe * np.log(p_safe / uniform))

    # Normalize by log(N) so MI in [0, 1]
    mi = kl / np.log(n_bins)
    return float(mi)


def _compute_mi_batch(
    bin_idx: np.ndarray,
    amplitude: np.ndarray,
    n_bins: int,
    shifts: np.ndarray,
) -> np.ndarray:
    """Compute MI for multiple circular shifts of the amplitude envelope.

    Vectorized over shifts to avoid Python loop overhead.

    Parameters
    ----------
    bin_idx : ndarray, shape (n_samples,)
        Phase bin indices (0-indexed).
    amplitude : ndarray, shape (n_samples,)
        Amplitude envelope.
    n_bins : int
        Number of phase bins.
    shifts : ndarray, shape (n_surrogates,)
        Circular shift amounts in samples.

    Returns
    -------
    mis : ndarray, shape (n_surrogates,)
        Modulation Index for each surrogate shift.
    """
    n_samples = len(amplitude)
    log_n = np.log(n_bins)
    uniform = 1.0 / n_bins

    # Pre-compute bin counts (same for all surrogates since phase doesn't shift)
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    mask = bin_counts > 0

    mis = np.empty(len(shifts))
    for k in range(len(shifts)):
        # Circular shift amplitude
        amp_shifted = np.roll(amplitude, shifts[k])

        # Binned mean amplitude
        bin_sums = np.bincount(bin_idx, weights=amp_shifted, minlength=n_bins).astype(float)
        mean_amp = np.zeros(n_bins)
        mean_amp[mask] = bin_sums[mask] / bin_counts[mask]

        total = mean_amp.sum()
        if total == 0:
            mis[k] = 0.0
            continue

        p = mean_amp / total
        p_safe = np.where(p > 0, p, 1e-20)
        kl = np.sum(p_safe * np.log(p_safe / uniform))
        mis[k] = kl / log_n

    return mis


def compute_pac(
    timeseries: np.ndarray,
    sfreq: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_bins: int = 18,
) -> float:
    """Compute raw Modulation Index for a single signal and frequency pair.

    Parameters
    ----------
    timeseries : ndarray, shape (n_samples,)
        1-D time course (signed signal).
    sfreq : float
        Sampling frequency in Hz.
    phase_band : tuple of (fmin, fmax)
        Frequency band for phase extraction.
    amp_band : tuple of (fmin, fmax)
        Frequency band for amplitude envelope extraction.
    n_bins : int
        Number of phase bins (default 18 = 20 degrees each).

    Returns
    -------
    mi : float
        Modulation Index.
    """
    # Bandpass filter for phase
    sos_phase = _design_bandpass(phase_band[0], phase_band[1], sfreq)
    x_phase = sosfiltfilt(sos_phase, timeseries)

    # Bandpass filter for amplitude
    sos_amp = _design_bandpass(amp_band[0], amp_band[1], sfreq)
    x_amp = sosfiltfilt(sos_amp, timeseries)

    # Hilbert transform
    phase = np.angle(hilbert(x_phase))
    amplitude = np.abs(hilbert(x_amp))

    # Phase bin edges
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    return _compute_mi_from_phase_amp(phase, amplitude, n_bins, bin_edges)


def compute_pac_zscore(
    timeseries: np.ndarray,
    sfreq: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_bins: int = 18,
    n_surrogates: int = 200,
    min_shift_sec: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute z-scored PAC via surrogate circular time-shifts.

    Parameters
    ----------
    timeseries : ndarray, shape (n_samples,)
        1-D time course (signed signal).
    sfreq : float
        Sampling frequency in Hz.
    phase_band : tuple of (fmin, fmax)
        Frequency band for phase extraction.
    amp_band : tuple of (fmin, fmax)
        Frequency band for amplitude envelope extraction.
    n_bins : int
        Number of phase bins.
    n_surrogates : int
        Number of surrogate MIs to generate.
    min_shift_sec : float
        Minimum circular shift in seconds.

    Returns
    -------
    mi : float
        Observed Modulation Index.
    z_score : float
        Z-scored MI: (mi - mean(surrogates)) / std(surrogates).
    surr_mean : float
        Mean of surrogate MI distribution.
    surr_std : float
        Std of surrogate MI distribution.
    """
    n_samples = len(timeseries)

    # Filter once (reuse for surrogates)
    sos_phase = _design_bandpass(phase_band[0], phase_band[1], sfreq)
    x_phase = sosfiltfilt(sos_phase, timeseries)

    sos_amp = _design_bandpass(amp_band[0], amp_band[1], sfreq)
    x_amp = sosfiltfilt(sos_amp, timeseries)

    # Hilbert transform
    phase = np.angle(hilbert(x_phase))
    amplitude = np.abs(hilbert(x_amp))

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    # Precompute bin indices (same for observed and all surrogates)
    bin_idx = np.digitize(phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Observed MI
    mi = _compute_mi_from_phase_amp(phase, amplitude, n_bins, bin_edges)

    # Surrogate distribution via circular shifts of amplitude envelope
    min_shift = max(int(min_shift_sec * sfreq), 1)
    max_shift = n_samples - min_shift
    if max_shift <= min_shift:
        # Signal too short for meaningful surrogates
        return mi, 0.0, mi, 0.0

    rng = np.random.default_rng()
    shifts = rng.integers(min_shift, max_shift, size=n_surrogates)

    surr_mis = _compute_mi_batch(bin_idx, amplitude, n_bins, shifts)

    surr_mean = float(np.mean(surr_mis))
    surr_std = float(np.std(surr_mis))

    if surr_std > 0:
        z_score = (mi - surr_mean) / surr_std
    else:
        z_score = 0.0

    return mi, float(z_score), surr_mean, surr_std


def get_valid_pac_pairs(
    bands: dict[str, tuple[float, float]],
    min_ratio: float = 2.5,
) -> list[tuple[str, str]]:
    """Generate valid phase-amplitude frequency pairs from config bands.

    A pair is valid when the amplitude band's center frequency is at least
    `min_ratio` times the phase band's center frequency, and the amplitude
    band's lower edge is above the phase band's upper edge.

    Parameters
    ----------
    bands : dict
        Band name -> (fmin, fmax) mapping from study config.
    min_ratio : float
        Minimum ratio of amplitude center to phase center frequency.

    Returns
    -------
    pairs : list of (phase_band_name, amp_band_name)
        Valid frequency pairs for PAC analysis.
    """
    band_names = list(bands.keys())
    pairs = []

    for phase_name in band_names:
        phase_lo, phase_hi = bands[phase_name]
        phase_center = (phase_lo + phase_hi) / 2.0

        for amp_name in band_names:
            amp_lo, amp_hi = bands[amp_name]
            amp_center = (amp_lo + amp_hi) / 2.0

            # Amplitude band must be well above phase band
            if amp_lo <= phase_hi:
                continue
            if amp_center / phase_center < min_ratio:
                continue

            pairs.append((phase_name, amp_name))

    return pairs


def compute_pac_multiroi(
    roi_timeseries: dict[str, np.ndarray],
    sfreq: float,
    pac_pairs: list[tuple[str, str]],
    bands: dict[str, tuple[float, float]],
    n_bins: int = 18,
    n_surrogates: int = 200,
    min_shift_sec: float = 1.0,
) -> list[dict[str, Any]]:
    """Compute PAC z-scores for all ROIs and all frequency pairs.

    Caches bandpass-filtered and Hilbert-transformed signals per band per ROI
    to avoid redundant filtering when a band appears in multiple pairs.

    Parameters
    ----------
    roi_timeseries : dict[str, ndarray]
        ROI name -> 1-D time course (signed).
    sfreq : float
        Sampling frequency in Hz.
    pac_pairs : list of (phase_band_name, amp_band_name)
        Frequency pairs to compute.
    bands : dict
        Band name -> (fmin, fmax) mapping.
    n_bins : int
        Number of phase bins.
    n_surrogates : int
        Number of surrogates for z-scoring.
    min_shift_sec : float
        Minimum circular shift in seconds.

    Returns
    -------
    results : list of dict
        Each dict has keys: roi, phase_band, amp_band, freq_pair, mi,
        z_score, surr_mean, surr_std.
    """
    results = []
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    log_n = np.log(n_bins)
    uniform = 1.0 / n_bins

    # Precompute SOS filters for each unique band
    sos_cache: dict[str, np.ndarray] = {}
    needed_bands = set()
    for p, a in pac_pairs:
        needed_bands.add(p)
        needed_bands.add(a)
    for band_name in needed_bands:
        fmin, fmax = bands[band_name]
        sos_cache[band_name] = _design_bandpass(fmin, fmax, sfreq)

    rng = np.random.default_rng()

    for roi_name, ts in roi_timeseries.items():
        n_samples = len(ts)
        min_shift = max(int(min_shift_sec * sfreq), 1)
        max_shift = n_samples - min_shift
        can_surrogate = max_shift > min_shift

        # Cache filtered phase/amplitude per band for this ROI
        phase_cache: dict[str, np.ndarray] = {}     # band -> phase array
        bin_idx_cache: dict[str, np.ndarray] = {}    # band -> bin indices
        amp_cache: dict[str, np.ndarray] = {}        # band -> amplitude envelope

        for phase_name, amp_name in pac_pairs:
            freq_pair = f"{phase_name}-{amp_name}"

            try:
                # Get or compute phase
                if phase_name not in phase_cache:
                    x_filt = sosfiltfilt(sos_cache[phase_name], ts)
                    ph = np.angle(hilbert(x_filt))
                    phase_cache[phase_name] = ph
                    bidx = np.digitize(ph, bin_edges) - 1
                    bidx = np.clip(bidx, 0, n_bins - 1)
                    bin_idx_cache[phase_name] = bidx

                # Get or compute amplitude
                if amp_name not in amp_cache:
                    x_filt = sosfiltfilt(sos_cache[amp_name], ts)
                    amp_cache[amp_name] = np.abs(hilbert(x_filt))

                phase = phase_cache[phase_name]
                amplitude = amp_cache[amp_name]
                bin_idx = bin_idx_cache[phase_name]

                # Observed MI
                mi = _compute_mi_from_phase_amp(phase, amplitude, n_bins, bin_edges)

                # Surrogate z-scoring
                if can_surrogate:
                    shifts = rng.integers(min_shift, max_shift, size=n_surrogates)
                    surr_mis = _compute_mi_batch(bin_idx, amplitude, n_bins, shifts)
                    surr_mean = float(np.mean(surr_mis))
                    surr_std = float(np.std(surr_mis))
                    z_score = (mi - surr_mean) / surr_std if surr_std > 0 else 0.0
                else:
                    surr_mean = mi
                    surr_std = 0.0
                    z_score = 0.0

                results.append({
                    "roi": roi_name,
                    "phase_band": phase_name,
                    "amp_band": amp_name,
                    "freq_pair": freq_pair,
                    "mi": mi,
                    "z_score": z_score,
                    "surr_mean": surr_mean,
                    "surr_std": surr_std,
                })
            except Exception as e:
                logger.warning(
                    "PAC failed for ROI=%s pair=%s-%s: %s",
                    roi_name, phase_name, amp_name, e,
                )

    return results
