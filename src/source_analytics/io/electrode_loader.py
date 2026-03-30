"""Load raw EEGLAB .set/.fdt files for electrode-level analysis.

Uses scipy.io.loadmat for .set metadata and numpy for .fdt binary data.
No MNE dependency required.

Supports both wrapped (``mat["EEG"]``) and unwrapped (top-level fields)
EEGLAB .set formats.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from math import gcd

logger = logging.getLogger(__name__)


def _get_field(container, name):
    """Get a field from either a mat_struct or a dict."""
    if isinstance(container, dict):
        return container[name]
    return getattr(container, name)


def _has_field(container, name):
    """Check if a field exists in either a mat_struct or a dict."""
    if isinstance(container, dict):
        return name in container
    return hasattr(container, name)


def _resample_2d(data: np.ndarray, sfreq: float, target_sfreq: float) -> np.ndarray:
    """Resample 2-D array (channels x samples) using polyphase filtering.

    Uses scipy.signal.resample_poly with an integer up/down ratio derived
    from the GCD of the two rates.  An anti-aliasing low-pass filter is
    applied automatically by resample_poly.
    """
    ratio = target_sfreq / sfreq
    up = int(target_sfreq)
    down = int(sfreq)
    g = gcd(up, down)
    up, down = up // g, down // g

    resampled = resample_poly(data, up, down, axis=-1)
    logger.info(
        "Resampled: %.0f -> %.0f Hz (up=%d, down=%d), %d -> %d samples",
        sfreq, target_sfreq, up, down, data.shape[-1], resampled.shape[-1],
    )
    return resampled


def load_eeglab_set(
    set_path: str | Path,
    target_sfreq: float | None = None,
) -> tuple[np.ndarray, float, list[str], np.ndarray | None]:
    """Load an EEGLAB .set/.fdt file pair.

    Parameters
    ----------
    set_path : Path
        Path to the ``.set`` file. The corresponding ``.fdt`` file is
        expected in the same directory.
    target_sfreq : float, optional
        If provided, resample the data to this sampling rate using
        polyphase filtering (anti-aliased).

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
        Continuous EEG data (epochs concatenated along time axis).
    sfreq : float
        Sampling frequency in Hz (after resampling if applied).
    ch_names : list[str]
        Channel names (e.g. ``["E1", "E2", ...]``).
    ch_coords : ndarray or None
        3-D electrode coordinates, shape ``(n_channels, 3)``, or *None*
        if coordinates are not available.
    """
    set_path = Path(set_path)
    if not set_path.exists():
        raise FileNotFoundError(f"EEG .set file not found: {set_path}")

    # Load .set metadata
    mat = loadmat(str(set_path), squeeze_me=True, struct_as_record=False)

    # Handle both wrapped ("EEG" struct) and unwrapped (top-level) formats
    if "EEG" in mat:
        eeg = mat["EEG"]
    else:
        eeg = mat  # top-level fields

    sfreq = float(_get_field(eeg, "srate"))
    n_channels = int(_get_field(eeg, "nbchan"))
    n_points = int(_get_field(eeg, "pnts"))
    n_trials = int(_get_field(eeg, "trials")) if _has_field(eeg, "trials") else 1

    # Extract channel names and coordinates
    ch_names = []
    ch_coords_list = []
    chanlocs = _get_field(eeg, "chanlocs")
    if not hasattr(chanlocs, "__len__"):
        chanlocs = [chanlocs]

    for ch in chanlocs:
        label = str(ch.labels) if hasattr(ch, "labels") else f"Ch{len(ch_names)+1}"
        ch_names.append(label)

        try:
            x = float(ch.X) if hasattr(ch, "X") and ch.X is not None else np.nan
            y = float(ch.Y) if hasattr(ch, "Y") and ch.Y is not None else np.nan
            z = float(ch.Z) if hasattr(ch, "Z") and ch.Z is not None else np.nan
            ch_coords_list.append([x, y, z])
        except (TypeError, ValueError):
            ch_coords_list.append([np.nan, np.nan, np.nan])

    ch_coords = np.array(ch_coords_list)
    if np.all(np.isnan(ch_coords)):
        ch_coords = None

    # Load data — may be inline or in a .fdt file
    data = None
    data_field = _get_field(eeg, "data")

    if isinstance(data_field, np.ndarray) and data_field.size > 0:
        # Data is stored inline in the .set file
        data = np.array(data_field, dtype=np.float64)
    else:
        # Data is in a separate .fdt file
        fdt_name = str(data_field) if isinstance(data_field, str) else set_path.stem + ".fdt"
        fdt_path = set_path.parent / fdt_name
        if not fdt_path.exists():
            fdt_path = set_path.with_suffix(".fdt")
        if not fdt_path.exists():
            raise FileNotFoundError(
                f"EEG .fdt data file not found: tried {set_path.parent / fdt_name} "
                f"and {set_path.with_suffix('.fdt')}"
            )

        data = np.fromfile(str(fdt_path), dtype=np.float32).astype(np.float64)

    # Reshape to (n_channels, n_points * n_trials)
    # EEGLAB .fdt format: data stored as (n_channels, n_points, n_trials) in column-major (Fortran) order
    total_samples = n_points * n_trials
    expected_size = n_channels * total_samples

    if data.size == expected_size:
        # Reshape: EEGLAB stores as (channels, points, trials) in column-major
        data = data.reshape((n_channels, n_points, n_trials), order="F")
        # Concatenate epochs along time axis
        data = data.reshape(n_channels, total_samples, order="F")
    else:
        logger.warning(
            "Data size %d does not match expected %d x %d x %d = %d. "
            "Attempting best-effort reshape.",
            data.size, n_channels, n_points, n_trials, expected_size,
        )
        n_samples_actual = data.size // n_channels
        data = data[: n_channels * n_samples_actual].reshape(n_channels, n_samples_actual)

    logger.info(
        "Loaded %s: %d channels, %d samples (%.1f s, %d epochs), sfreq=%.0f Hz",
        set_path.name, n_channels, data.shape[1],
        data.shape[1] / sfreq, n_trials, sfreq,
    )

    if target_sfreq is not None and sfreq > target_sfreq:
        data = _resample_2d(data, sfreq, target_sfreq)
        sfreq = target_sfreq
    elif target_sfreq is not None and sfreq < target_sfreq:
        logger.info(
            "Skipping resample: file sfreq (%.0f Hz) < target (%.0f Hz)",
            sfreq, target_sfreq,
        )

    return data, sfreq, ch_names, ch_coords


def load_eeglab_epochs(
    set_path: str | Path,
    target_sfreq: float | None = None,
) -> tuple[np.ndarray, float, list[str], int]:
    """Load an EEGLAB .set/.fdt file and return data with epoch structure preserved.

    Parameters
    ----------
    set_path : Path
        Path to the ``.set`` file.
    target_sfreq : float, optional
        If provided, resample each epoch to this sampling rate.

    Returns
    -------
    epochs : ndarray, shape (n_epochs, n_channels, epoch_samples)
        Epoched EEG data (after resampling if applied).
    sfreq : float
        Sampling frequency in Hz (after resampling if applied).
    ch_names : list[str]
        Channel names.
    epoch_samples : int
        Number of samples per epoch (after resampling if applied).
    """
    set_path = Path(set_path)
    if not set_path.exists():
        raise FileNotFoundError(f"EEG .set file not found: {set_path}")

    mat = loadmat(str(set_path), squeeze_me=True, struct_as_record=False)

    if "EEG" in mat:
        eeg = mat["EEG"]
    else:
        eeg = mat

    sfreq = float(_get_field(eeg, "srate"))
    n_channels = int(_get_field(eeg, "nbchan"))
    n_points = int(_get_field(eeg, "pnts"))
    n_trials = int(_get_field(eeg, "trials")) if _has_field(eeg, "trials") else 1

    # Extract channel names
    ch_names = []
    chanlocs = _get_field(eeg, "chanlocs")
    if not hasattr(chanlocs, "__len__"):
        chanlocs = [chanlocs]
    for ch in chanlocs:
        label = str(ch.labels) if hasattr(ch, "labels") else f"Ch{len(ch_names)+1}"
        ch_names.append(label)

    # Load data
    data_field = _get_field(eeg, "data")
    if isinstance(data_field, np.ndarray) and data_field.size > 0:
        data = np.array(data_field, dtype=np.float64)
    else:
        fdt_name = str(data_field) if isinstance(data_field, str) else set_path.stem + ".fdt"
        fdt_path = set_path.parent / fdt_name
        if not fdt_path.exists():
            fdt_path = set_path.with_suffix(".fdt")
        if not fdt_path.exists():
            raise FileNotFoundError(
                f"EEG .fdt data file not found: tried {set_path.parent / fdt_name} "
                f"and {set_path.with_suffix('.fdt')}"
            )
        data = np.fromfile(str(fdt_path), dtype=np.float32).astype(np.float64)

    # Reshape to (n_channels, n_points, n_trials) preserving epoch structure
    total_samples = n_points * n_trials
    expected_size = n_channels * total_samples

    if data.size != expected_size:
        raise ValueError(
            f"Data size {data.size} does not match expected "
            f"{n_channels} x {n_points} x {n_trials} = {expected_size}"
        )

    # EEGLAB stores as (channels, points, trials) in column-major (Fortran) order
    data = data.reshape((n_channels, n_points, n_trials), order="F")
    # Transpose to (n_trials, n_channels, n_points)
    epochs = np.transpose(data, (2, 0, 1))

    logger.info(
        "Loaded epochs from %s: %d epochs x %d channels x %d samples, sfreq=%.0f Hz",
        set_path.name, n_trials, n_channels, n_points, sfreq,
    )

    if target_sfreq is not None and sfreq > target_sfreq:
        # Resample each epoch: reshape to (n_epochs * n_channels, n_points),
        # resample along time axis, then reshape back
        orig_shape = epochs.shape  # (n_epochs, n_channels, n_points)
        flat = epochs.reshape(-1, n_points)
        flat = _resample_2d(flat, sfreq, target_sfreq)
        new_n_points = flat.shape[-1]
        epochs = flat.reshape(orig_shape[0], orig_shape[1], new_n_points)
        sfreq = target_sfreq
        n_points = new_n_points
    elif target_sfreq is not None and sfreq < target_sfreq:
        logger.info(
            "Skipping resample: file sfreq (%.0f Hz) < target (%.0f Hz)",
            sfreq, target_sfreq,
        )

    return epochs, sfreq, ch_names, n_points
