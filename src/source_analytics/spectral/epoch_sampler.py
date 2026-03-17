"""Random epoch sampling for whole-brain analyses.

Instead of computing PSD/connectivity on full continuous recordings,
randomly sample non-overlapping epochs of fixed duration. Benefits:
- Reduces non-stationarity effects
- Enables bootstrap variance estimates
- Matches analysis windows across subjects
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def sample_epochs(
    data: np.ndarray,
    sfreq: float,
    epoch_duration_sec: float = 2.0,
    n_epochs: int = 100,
    seed: int | None = None,
    overlap: float = 0.0,
) -> np.ndarray:
    """Randomly sample non-overlapping epochs from continuous data.

    Parameters
    ----------
    data : ndarray, shape (n_channels_or_vertices, n_times)
        Continuous time-series data.
    sfreq : float
        Sampling frequency in Hz.
    epoch_duration_sec : float
        Duration of each epoch in seconds.
    n_epochs : int
        Number of epochs to sample. If 0 or None, returns data unchanged
        in a (1, n_channels, n_times) array for backward compatibility.
    seed : int, optional
        Random seed for reproducibility.
    overlap : float
        Fraction of overlap allowed between epochs (0.0 = no overlap).
        Currently only 0.0 is supported.

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, epoch_length)
        Sampled epochs.
    """
    if n_epochs is None or n_epochs == 0:
        return data[np.newaxis, :, :]  # (1, n_channels, n_times)

    n_channels, n_times = data.shape
    epoch_len = int(epoch_duration_sec * sfreq)

    if epoch_len > n_times:
        logger.warning(
            "Epoch length (%d samples) exceeds data length (%d). "
            "Using full data as single epoch.",
            epoch_len, n_times,
        )
        return data[np.newaxis, :, :]

    # Maximum number of non-overlapping epochs
    max_epochs = n_times // epoch_len
    if n_epochs > max_epochs:
        logger.warning(
            "Requested %d epochs but only %d non-overlapping epochs fit. "
            "Using %d.",
            n_epochs, max_epochs, max_epochs,
        )
        n_epochs = max_epochs

    rng = np.random.default_rng(seed)

    # Generate all possible non-overlapping start positions
    # and randomly select n_epochs of them
    all_starts = np.arange(0, n_times - epoch_len + 1, epoch_len)
    selected_starts = rng.choice(all_starts, size=n_epochs, replace=False)
    selected_starts.sort()

    epochs = np.empty((n_epochs, n_channels, epoch_len), dtype=data.dtype)
    for i, start in enumerate(selected_starts):
        epochs[i] = data[:, start : start + epoch_len]

    logger.info(
        "Sampled %d epochs of %.1fs (%.0f samples) from %.1fs recording",
        n_epochs, epoch_duration_sec, epoch_len, n_times / sfreq,
    )

    return epochs


def sample_roi_epochs(
    roi_ts: dict[str, np.ndarray],
    sfreq: float,
    epoch_duration_sec: float = 2.0,
    n_epochs: int = 80,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Randomly sample epochs from ROI timeseries and concatenate back.

    Equalizes recording duration across subjects by randomly selecting
    *n_epochs* non-overlapping windows of *epoch_duration_sec*, then
    concatenating them into a continuous array per ROI.

    Parameters
    ----------
    roi_ts : dict[str, ndarray]
        Mapping of ROI name -> 1-D time course.
    sfreq : float
        Sampling frequency in Hz.
    epoch_duration_sec : float
        Duration of each epoch in seconds.
    n_epochs : int
        Number of epochs to sample.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, ndarray]
        Same keys, values truncated to ``n_epochs * epoch_len`` samples.
    """
    if not roi_ts:
        return roi_ts

    # Stack into (n_rois, n_times) matrix
    roi_names = list(roi_ts.keys())
    data = np.stack([roi_ts[name] for name in roi_names])  # (n_rois, n_times)

    # Sample epochs: (n_epochs, n_rois, epoch_len)
    epochs = sample_epochs(
        data, sfreq,
        epoch_duration_sec=epoch_duration_sec,
        n_epochs=n_epochs,
        seed=seed,
    )

    # Concatenate back to continuous: (n_rois, n_epochs * epoch_len)
    n_ep, n_rois, epoch_len = epochs.shape
    continuous = epochs.transpose(1, 0, 2).reshape(n_rois, n_ep * epoch_len)

    return {name: continuous[i] for i, name in enumerate(roi_names)}


def get_epoch_config(config_dict: dict) -> dict | None:
    """Extract epoch sampling config from vertex config.

    Parameters
    ----------
    config_dict : dict
        The vertex config section.

    Returns
    -------
    dict or None
        Epoch config dict if enabled, None otherwise.
    """
    epoch_cfg = config_dict.get("epoch_sampling", {})
    if not epoch_cfg.get("enabled", False):
        return None
    return epoch_cfg
