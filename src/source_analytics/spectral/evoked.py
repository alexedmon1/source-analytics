"""Evoked-response measures: ERP amplitude and latency, induced vs evoked power.

Neither existed in this package or in the lab's EEGLAB scripts. Everything on
both sides was ITC, ERSP or trial-averaged power — all of which discard sign,
which is why the absence went unnoticed.

Sign is now worth having. Under a fixed-orientation inverse a source's time
course is its projection onto the cortical normal, so its polarity is
anatomically meaningful rather than an arbitrary SVD convention. An N1 that
inverts between two regions means something.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def evoked_average(epochs: np.ndarray) -> np.ndarray:
    """Phase-locked average across trials.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs, n_times)

    Returns
    -------
    ndarray, shape (n_times,)
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 2:
        raise ValueError(f"expected (n_epochs, n_times), got {epochs.shape}")
    return epochs.mean(axis=0)


def baseline_correct(
    waveform: np.ndarray,
    sfreq: float,
    baseline: tuple[float, float],
    xmin: float,
) -> np.ndarray:
    """Subtract the mean of a baseline window.

    Subtractive, not divisive: an ERP is measured in the same units as the
    signal, so the dB ratio used for ERSP would be meaningless here and would
    also destroy the sign.
    """
    waveform = np.asarray(waveform, dtype=float)
    b0 = max(0, int(round((baseline[0] - xmin) * sfreq)))
    b1 = min(len(waveform), int(round((baseline[1] - xmin) * sfreq)))
    if b0 >= b1:
        raise ValueError(
            f"baseline {baseline} is empty for an epoch starting at {xmin}"
        )
    return waveform - waveform[b0:b1].mean()


def erp_peak(
    waveform: np.ndarray,
    sfreq: float,
    time_window: tuple[float, float],
    xmin: float = 0.0,
    polarity: str = "abs",
) -> dict:
    """Peak amplitude and latency within a window.

    Parameters
    ----------
    polarity : {'abs', 'positive', 'negative'}
        ``positive`` and ``negative`` find a signed extremum, which is what a
        named component needs — an N1 is the most negative point, not the
        largest excursion. ``abs`` takes the largest absolute deflection and is
        the safe default when the polarity convention is not established.

    Returns
    -------
    dict
        ``amplitude`` (signed, in input units), ``latency`` (seconds),
        ``polarity``. Amplitude and latency are NaN if the window is empty.
    """
    waveform = np.asarray(waveform, dtype=float)
    n = len(waveform)
    t0 = max(0, int(round((time_window[0] - xmin) * sfreq)))
    t1 = min(n, int(round((time_window[1] - xmin) * sfreq)))

    if t0 >= t1:
        return {"amplitude": np.nan, "latency": np.nan, "polarity": polarity}

    seg = waveform[t0:t1]
    if polarity == "positive":
        idx = int(np.argmax(seg))
    elif polarity == "negative":
        idx = int(np.argmin(seg))
    elif polarity == "abs":
        idx = int(np.argmax(np.abs(seg)))
    else:
        raise ValueError(
            f"polarity must be 'abs', 'positive' or 'negative'; got {polarity!r}"
        )

    return {
        "amplitude": float(seg[idx]),
        "latency": float(xmin + (t0 + idx) / sfreq),
        "polarity": polarity,
    }


def erp_mean_amplitude(
    waveform: np.ndarray,
    sfreq: float,
    time_window: tuple[float, float],
    xmin: float = 0.0,
) -> float:
    """Mean amplitude over a window.

    Less sensitive to noise than a peak, and the measure to prefer when the
    component's timing is not the question — a peak search always finds a peak,
    including in noise.
    """
    waveform = np.asarray(waveform, dtype=float)
    t0 = max(0, int(round((time_window[0] - xmin) * sfreq)))
    t1 = min(len(waveform), int(round((time_window[1] - xmin) * sfreq)))
    if t0 >= t1:
        return np.nan
    return float(waveform[t0:t1].mean())


def subtract_evoked(epochs: np.ndarray) -> np.ndarray:
    """Remove the phase-locked average from every trial.

    What remains is the induced response: activity time-locked to the stimulus
    but not phase-locked, which a total-power measure cannot separate from the
    evoked part. Both `stp` and `ersp` in this package are computed on total
    power, so neither distinguishes them today.

    Note the trade: subtracting an average estimated from the same trials
    removes a 1/n_epochs share of the induced power along with the evoked
    part, so induced power is slightly underestimated at low trial counts.
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 2:
        raise ValueError(f"expected (n_epochs, n_times), got {epochs.shape}")
    if len(epochs) < 2:
        raise ValueError("induced power needs at least 2 trials")
    return epochs - epochs.mean(axis=0, keepdims=True)


def erp_measures(
    epochs: np.ndarray,
    sfreq: float,
    xmin: float,
    baseline: tuple[float, float] | None,
    specs,
) -> dict:
    """Compute a set of ERP measures from one unit's epochs.

    Parameters
    ----------
    specs : sequence of dict
        Each with ``name``, ``time_window``, optional ``polarity``
        (default ``abs``) and optional ``type`` — ``peak`` (default) or
        ``mean``.

    Returns
    -------
    dict
        ``{name: value}``; peak measures also emit ``{name}_latency``.
    """
    wave = evoked_average(epochs)
    if baseline is not None:
        wave = baseline_correct(wave, sfreq, baseline, xmin)

    out: dict[str, float] = {}
    for spec in specs:
        name = spec["name"]
        window = tuple(spec["time_window"])
        kind = spec.get("type", "peak")
        if kind == "mean":
            out[name] = erp_mean_amplitude(wave, sfreq, window, xmin)
        elif kind == "peak":
            peak = erp_peak(wave, sfreq, window, xmin,
                            polarity=spec.get("polarity", "abs"))
            out[name] = peak["amplitude"]
            out[f"{name}_latency"] = peak["latency"]
        else:
            raise ValueError(
                f"ERP measure type must be 'peak' or 'mean'; got {kind!r}"
            )
    return out
