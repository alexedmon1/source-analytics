"""SubjectLoader: reads pipeline output files for a single subject."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


class SubjectLoader:
    """Load source localization pipeline outputs for one subject.

    Parameters
    ----------
    data_dir : Path
        The ``data/`` directory inside a subject's pipeline output.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Subject data directory not found: {self.data_dir}")

    def _load_pkl(self, filename: str) -> Any:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_npy(self, filename: str) -> np.ndarray:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return np.load(path)

    def load_roi_timeseries(self, signed: bool = False) -> dict[str, np.ndarray]:
        """Load ROI time series (magnitude or signed).

        Returns
        -------
        dict[str, ndarray]
            Mapping of ROI name -> 1-D time course array.
        """
        suffix = "signed" if signed else "magnitude"
        for fname in [
            f"step6_roi_timeseries_{suffix}.pkl",
            "step6_roi_timeseries.pkl",
        ]:
            try:
                return self._load_pkl(fname)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"No ROI timeseries file found in {self.data_dir} "
            f"(tried step6_roi_timeseries_{suffix}.pkl, step6_roi_timeseries.pkl)"
        )

    def load_sfreq(self) -> float:
        """Extract sampling frequency.

        Reads from the .set file (EEGLAB MAT format) first, which does
        not require MNE.  Falls back to step1_info.pkl if no .set file
        is present.
        """
        # Primary: read from .set file (scipy only, no MNE needed)
        for set_name in ["roi_timeseries_magnitude.set", "roi_timeseries_signed.set"]:
            set_path = self.data_dir / set_name
            if set_path.exists():
                try:
                    mat = loadmat(str(set_path), squeeze_me=False, variable_names=["srate"])
                    srate = mat["srate"]
                    return float(np.squeeze(srate))
                except Exception as e:
                    logger.debug("Could not read sfreq from %s: %s", set_path, e)

        # Fallback: step1_info.pkl (requires MNE to unpickle)
        try:
            info = self._load_pkl("step1_info.pkl")
            return float(info["sfreq"])
        except Exception as e:
            raise RuntimeError(
                f"Cannot determine sfreq for {self.data_dir}. "
                f"No .set file found and step1_info.pkl failed: {e}"
            ) from e

    def load_info(self) -> Any:
        """Load the MNE Info object (step1_info.pkl).

        Note: requires MNE to be installed.
        """
        return self._load_pkl("step1_info.pkl")

    def load_source_coords(self) -> np.ndarray:
        """Load source coordinates in mm (n_sources, 3)."""
        return self._load_npy("step3_source_coords_mm.npy")

    def load_band_power(self) -> dict[str, dict[str, float]] | None:
        """Load pre-computed band power if available (step7_band_power.pkl)."""
        try:
            return self._load_pkl("step7_band_power.pkl")
        except FileNotFoundError:
            return None

    @property
    def available_files(self) -> list[str]:
        """List all files present in the data directory."""
        return sorted(f.name for f in self.data_dir.iterdir() if f.is_file())

    def has_file(self, filename: str) -> bool:
        return (self.data_dir / filename).exists()
