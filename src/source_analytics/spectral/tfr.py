"""Time-frequency analysis: Morlet wavelet TFR, ITC, ERSP, STP."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def _morlet_wavelet(sfreq: float, freq: float, n_cycles: float) -> np.ndarray:
    """Create a Morlet wavelet for a given frequency.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Center frequency in Hz.
    n_cycles : float
        Number of wavelet cycles (controls time-frequency trade-off).

    Returns
    -------
    ndarray, shape (n_wavelet,)
        Complex Morlet wavelet, normalized to unit energy.
    """
    sigma_t = n_cycles / (2.0 * np.pi * freq)
    # Wavelet duration: +/- 3 standard deviations
    n_samples = int(np.ceil(6.0 * sigma_t * sfreq))
    # Ensure odd length for symmetry
    if n_samples % 2 == 0:
        n_samples += 1
    t = np.arange(-n_samples // 2, n_samples // 2 + 1) / sfreq
    wavelet = np.exp(2j * np.pi * freq * t) * np.exp(-t**2 / (2.0 * sigma_t**2))
    # Normalize to unit energy
    wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
    return wavelet


def morlet_tfr(
    epochs: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray = 7.0,
) -> np.ndarray:
    """Morlet wavelet time-frequency decomposition.

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

    Returns
    -------
    ndarray, shape (n_epochs, n_freqs, n_times), complex
        Complex-valued time-frequency representation.
    """
    n_epochs, n_times = epochs.shape
    n_freqs = len(freqs)

    if np.isscalar(n_cycles):
        cycles = np.full(n_freqs, n_cycles)
    else:
        cycles = np.asarray(n_cycles)

    tfr = np.zeros((n_epochs, n_freqs, n_times), dtype=np.complex128)

    for fi, (freq, nc) in enumerate(zip(freqs, cycles)):
        wavelet = _morlet_wavelet(sfreq, freq, nc)
        for ei in range(n_epochs):
            # FFT-based convolution for speed
            conv = fftconvolve(epochs[ei], wavelet, mode="same")
            tfr[ei, fi, :] = conv

    return tfr


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
    # Normalize to unit magnitude (extract phase)
    phase = tfr_complex / np.abs(tfr_complex)
    # Handle any zeros
    phase = np.nan_to_num(phase, nan=0.0)
    itc = np.abs(np.mean(phase, axis=0))
    return itc


def compute_ersp(
    tfr_complex: np.ndarray,
    sfreq: float,
    baseline: tuple[float, float],
    xmin: float = 0.0,
) -> np.ndarray:
    """Event-related spectral perturbation.

    ERSP = 10 * log10(mean_power / baseline_power).

    Parameters
    ----------
    tfr_complex : ndarray, shape (n_epochs, n_freqs, n_times)
        Complex-valued TFR.
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
    power = np.mean(np.abs(tfr_complex) ** 2, axis=0)  # (n_freqs, n_times)
    n_times = power.shape[1]

    # Convert baseline times to sample indices
    bl_start = int(round((baseline[0] - xmin) * sfreq))
    bl_end = int(round((baseline[1] - xmin) * sfreq))
    bl_start = max(0, bl_start)
    bl_end = min(n_times, bl_end)

    baseline_power = np.mean(power[:, bl_start:bl_end], axis=1, keepdims=True)
    # Avoid log of zero
    baseline_power = np.maximum(baseline_power, np.finfo(float).eps)
    ersp = 10.0 * np.log10(power / baseline_power)
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
    return np.mean(np.abs(tfr_complex) ** 2, axis=0)


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

    # Frequency mask
    freq_mask = (freqs >= band[0]) & (freqs <= band[1])

    # Time mask
    t_start = int(round((time_window[0] - xmin) * sfreq))
    t_end = int(round((time_window[1] - xmin) * sfreq))
    t_start = max(0, t_start)
    t_end = min(n_times, t_end)

    if not np.any(freq_mask) or t_start >= t_end:
        return np.nan

    return float(np.mean(measure_2d[freq_mask, t_start:t_end]))
