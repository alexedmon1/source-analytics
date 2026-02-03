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


def load_eeglab_set(
    set_path: str | Path,
) -> tuple[np.ndarray, float, list[str], np.ndarray | None]:
    """Load an EEGLAB .set/.fdt file pair.

    Parameters
    ----------
    set_path : Path
        Path to the ``.set`` file. The corresponding ``.fdt`` file is
        expected in the same directory.

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
        Continuous EEG data (epochs concatenated along time axis).
    sfreq : float
        Sampling frequency in Hz.
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

    return data, sfreq, ch_names, ch_coords
