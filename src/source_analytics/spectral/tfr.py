"""Time-frequency analysis: Morlet wavelet TFR, ITC, ERSP, STP.

Uses MNE-Python's validated tfr_array_morlet() for the core wavelet
decomposition (±5σ wavelets, Tallon-Baudry 1997 normalization, pre-cached
FFTs, generator-based memory management for averaged output modes).
"""

from __future__ import annotations

import logging

import numpy as np
from mne.time_frequency import tfr_array_morlet

logger = logging.getLogger(__name__)


def _safe_n_cycles(
    freqs: np.ndarray,
    n_cycles: float | np.ndarray,
    sfreq: float,
    n_times: int,
) -> np.ndarray:
    """Cap n_cycles per frequency so wavelets fit within the epoch.

    MNE's Morlet wavelets extend ±5σ (10σ total) where σ = n_cycles/(2π·f).
    Wavelet length in samples ≈ 10 · n_cycles · sfreq / (2π · f).
    We require this to be < n_times.

    Parameters
    ----------
    freqs : ndarray
        Frequencies in Hz.
    n_cycles : float or ndarray
        Requested n_cycles (scalar or per-frequency).
    sfreq : float
        Sampling frequency.
    n_times : int
        Epoch length in samples.

    Returns
    -------
    ndarray
        Per-frequency n_cycles, capped where necessary.
    """
    freqs = np.asarray(freqs, dtype=float)
    if np.isscalar(n_cycles):
        cycles = np.full(len(freqs), float(n_cycles))
    else:
        cycles = np.asarray(n_cycles, dtype=float).copy()

    # Maximum n_cycles for each frequency: wavelet_len < n_times
    # wavelet_len = 10 * sigma_t * sfreq = 10 * nc / (2π * f) * sfreq
    # Require: 10 * nc * sfreq / (2π * f) < n_times
    # => nc < n_times * 2π * f / (10 * sfreq)
    # Use 0.95 safety margin to avoid edge-case rounding issues
    max_cycles = 0.95 * n_times * 2.0 * np.pi * freqs / (10.0 * sfreq)

    capped = cycles > max_cycles
    if np.any(capped):
        n_capped = int(np.sum(capped))
        freq_lo = freqs[capped].min()
        freq_hi = freqs[capped].max()
        logger.info(
            "Capping n_cycles for %d frequencies (%.1f-%.1f Hz) to fit "
            "within %d-sample epoch (sfreq=%.0f Hz)",
            n_capped, freq_lo, freq_hi, n_times, sfreq,
        )
        cycles[capped] = max_cycles[capped]

    # Enforce minimum of 1 cycle
    cycles = np.maximum(cycles, 1.0)

    return cycles


def morlet_tfr(
    epochs: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray = 7.0,
    decim: int = 1,
    n_jobs: int = 1,
) -> np.ndarray:
    """Morlet wavelet time-frequency decomposition via MNE.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_times)
        Epoched time series for a single ROI.
    sfreq : float
        Sampling frequency in Hz.
    freqs : ndarray, shape (n_freqs,)
        Frequencies of interest in Hz.
    n_cycles : float or ndarray
        Number of wavelet cycles. If scalar, same for all frequencies.
        If array, must match len(freqs).
    decim : int
        Decimation factor applied after TFR (default 1 = no decimation).
    n_jobs : int
        Number of parallel jobs (across channels).

    Returns
    -------
    ndarray, shape (n_epochs, n_freqs, n_times), complex
        Complex-valued time-frequency representation.
    """
    n_times = epochs.shape[1]
    cycles = _safe_n_cycles(freqs, n_cycles, sfreq, n_times)

    # MNE expects (n_epochs, n_channels, n_times) — add channel dim
    data = epochs[:, np.newaxis, :]

    result = tfr_array_morlet(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=cycles,
        zero_mean=True,
        use_fft=True,
        decim=decim,
        output="complex",
        n_jobs=n_jobs,
        verbose=False,
    )

    # Remove channel dim: (n_epochs, 1, n_freqs, n_times) -> (n_epochs, n_freqs, n_times)
    return result[:, 0, :, :]


def morlet_tfr_avg_power_itc(
    epochs: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray = 7.0,
    decim: int = 1,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute average power and ITC in a single pass via MNE.

    This is more memory-efficient than computing full complex TFR
    and then deriving power and ITC separately, because MNE uses
    generator-based accumulation (never materializes the full
    (n_epochs, n_freqs, n_times) complex array).

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_times)
        Epoched time series for a single ROI.
    sfreq : float
        Sampling frequency in Hz.
    freqs : ndarray, shape (n_freqs,)
        Frequencies of interest in Hz.
    n_cycles : float or ndarray
        Number of wavelet cycles.
    decim : int
        Decimation factor applied after TFR.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    avg_power : ndarray, shape (n_freqs, n_times)
        Average power across trials.
    itc : ndarray, shape (n_freqs, n_times)
        Inter-trial coherence, values in [0, 1].
    """
    n_times = epochs.shape[1]
    cycles = _safe_n_cycles(freqs, n_cycles, sfreq, n_times)

    data = epochs[:, np.newaxis, :]

    result = tfr_array_morlet(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=cycles,
        zero_mean=True,
        use_fft=True,
        decim=decim,
        output="avg_power_itc",
        n_jobs=n_jobs,
        verbose=False,
    )

    # result is complex: real = avg_power, imag = ITC
    # Shape: (1, n_freqs, n_times) — remove channel dim
    avg_power = result.real[0, :, :]
    itc = result.imag[0, :, :]

    return avg_power, itc


def compute_itc(tfr_complex: np.ndarray) -> np.ndarray:
    """Inter-trial coherence from complex TFR.

    ITC = |mean(exp(j * phase))| across trials.

    Parameters
    ----------
    tfr_complex : ndarray, shape (n_epochs, n_freqs, n_times)
        Complex-valued TFR.

    Returns
    -------
    ndarray, shape (n_freqs, n_times)
        ITC values in [0, 1].
    """
    phase = tfr_complex / np.abs(tfr_complex)
    phase = np.nan_to_num(phase, nan=0.0)
    itc = np.abs(np.mean(phase, axis=0))
    return itc


def compute_ersp(
    avg_power: np.ndarray,
    sfreq: float,
    baseline: tuple[float, float],
    xmin: float = 0.0,
) -> np.ndarray:
    """Event-related spectral perturbation.

    ERSP = 10 * log10(power / baseline_power).

    Parameters
    ----------
    avg_power : ndarray, shape (n_freqs, n_times)
        Average power across trials (from morlet_tfr_avg_power_itc or
        manual computation).
    sfreq : float
        Sampling frequency in Hz.
    baseline : tuple (tmin, tmax)
        Baseline window in seconds (relative to epoch start at xmin).
    xmin : float
        Start time of epoch in seconds (e.g., -0.5).

    Returns
    -------
    ndarray, shape (n_freqs, n_times)
        ERSP in dB.
    """
    n_times = avg_power.shape[1]

    bl_start = int(round((baseline[0] - xmin) * sfreq))
    bl_end = int(round((baseline[1] - xmin) * sfreq))
    bl_start = max(0, bl_start)
    bl_end = min(n_times, bl_end)

    baseline_power = np.mean(avg_power[:, bl_start:bl_end], axis=1, keepdims=True)
    # Guard the denominator without flooring legitimate small baselines.
    baseline_power = np.where(
        baseline_power > 0, baseline_power, np.finfo(float).tiny
    )
    ersp = 10.0 * np.log10(avg_power / baseline_power)
    return ersp


def compute_stp(tfr_complex: np.ndarray) -> np.ndarray:
    """Single-trial power: mean |TFR|^2 across trials (no baseline correction).

    Parameters
    ----------
    tfr_complex : ndarray, shape (n_epochs, n_freqs, n_times)
        Complex-valued TFR.

    Returns
    -------
    ndarray, shape (n_freqs, n_times)
        Mean power across trials.
    """
    return np.mean((tfr_complex * np.conj(tfr_complex)).real, axis=0)


def extract_measure_in_band(
    measure_2d: np.ndarray,
    freqs: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    time_window: tuple[float, float],
    xmin: float = 0.0,
) -> float:
    """Extract mean value within a frequency band and time window.

    Parameters
    ----------
    measure_2d : ndarray, shape (n_freqs, n_times)
        2-D TF map (ITC, ERSP, or STP).
    freqs : ndarray, shape (n_freqs,)
        Frequency vector.
    sfreq : float
        Sampling frequency in Hz.
    band : tuple (fmin, fmax)
        Frequency band in Hz.
    time_window : tuple (tmin, tmax)
        Time window in seconds.
    xmin : float
        Start time of epoch in seconds.

    Returns
    -------
    float
        Mean value in the specified band x time window.
    """
    n_times = measure_2d.shape[1]

    freq_mask = (freqs >= band[0]) & (freqs <= band[1])

    t_start = int(round((time_window[0] - xmin) * sfreq))
    t_end = int(round((time_window[1] - xmin) * sfreq))
    t_start = max(0, t_start)
    t_end = min(n_times, t_end)

    if not np.any(freq_mask) or t_start >= t_end:
        return np.nan

    return float(np.mean(measure_2d[freq_mask, t_start:t_end]))
