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
        # Cache for on-the-fly ROI extraction (avoids re-extracting per analysis)
        self._roi_cache: dict[str, dict[str, np.ndarray]] = {}

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

    def load_roi_timeseries(self, signed: bool = True) -> dict[str, np.ndarray]:
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
        for set_name in ["roi_timeseries_signed.set", "roi_timeseries_magnitude.set"]:
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

    def load_roi_epochs(
        self,
        epoch_samples: int,
        signed: bool = True,
        atlas_dir: str | Path | None = None,
    ) -> dict[str, np.ndarray]:
        """Load ROI time series reshaped into epochs.

        Source-localization concatenates epochs into continuous 1-D arrays.
        This method recovers epoch structure by reshaping based on
        ``epoch_samples`` (samples per epoch).

        Parameters
        ----------
        epoch_samples : int
            Number of samples per epoch.
        signed : bool
            If True, load signed timeseries; otherwise magnitude.
        atlas_dir : str or Path, optional
            Atlas directory for on-the-fly extraction (passed through to
            :meth:`load_or_extract_roi_timeseries`).

        Returns
        -------
        dict[str, ndarray]
            Mapping of ROI name -> array of shape (n_epochs, epoch_samples).

        Raises
        ------
        ValueError
            If total samples is not evenly divisible by epoch_samples.
        """
        roi_ts = self.load_or_extract_roi_timeseries(signed=signed, atlas_dir=atlas_dir)
        roi_epochs = {}
        for roi_name, ts in roi_ts.items():
            n_total = len(ts)
            if n_total % epoch_samples != 0:
                raise ValueError(
                    f"ROI '{roi_name}': total samples ({n_total}) not divisible "
                    f"by epoch_samples ({epoch_samples}). Cannot recover epochs."
                )
            roi_epochs[roi_name] = ts.reshape(-1, epoch_samples)
        return roi_epochs

    def load_source_epochs(
        self,
        epoch_samples: int,
        magnitude: bool = False,
    ) -> np.ndarray:
        """Load vertex source time courses reshaped into epochs.

        The source-localization pipeline concatenates trial epochs into
        continuous arrays; this recovers epoch structure by reshaping on
        ``epoch_samples`` (the vertex analogue of :meth:`load_roi_epochs`).

        Parameters
        ----------
        epoch_samples : int
            Number of samples per epoch.
        magnitude : bool
            If True, rectified amplitudes; otherwise signed (default).

        Returns
        -------
        ndarray, shape (n_vertices, n_epochs, epoch_samples)

        Raises
        ------
        ValueError
            If total samples is not evenly divisible by epoch_samples.
        """
        stc = self.load_source_timecourses(magnitude=magnitude)
        n_total = stc.shape[1]
        if n_total % epoch_samples != 0:
            raise ValueError(
                f"Source time courses: total samples ({n_total}) not divisible "
                f"by epoch_samples ({epoch_samples}). Cannot recover epochs."
            )
        n_vertices = stc.shape[0]
        return stc.reshape(n_vertices, -1, epoch_samples)

    def load_or_extract_roi_timeseries(
        self,
        signed: bool = True,
        atlas_dir: str | Path | None = None,
        method: str = "nearest",
        proximity_radius_mm: float = 2.0,
    ) -> dict[str, np.ndarray]:
        """Load ROI time series, extracting on-the-fly from step5 if needed.

        Tries pre-extracted step6 files first (backward compatibility).
        If not found, falls back to on-the-fly ROI extraction from
        step5_stc.pkl + step3_source_coords_mm.npy using the atlas.

        Parameters
        ----------
        signed : bool
            If True, use signed time courses; otherwise magnitude.
        atlas_dir : str or Path, optional
            Atlas directory for on-the-fly extraction. If None, auto-detected
            via :func:`~source_analytics.atlas.find_atlas_dir`.
        method : str
            Source-to-ROI assignment: ``"nearest"`` or ``"proximity"``.
        proximity_radius_mm : float
            Radius for proximity method.

        Returns
        -------
        dict[str, ndarray]
            Mapping of ROI name -> 1-D time course array.
        """
        # 1. Try loading pre-extracted step6 files
        try:
            return self.load_roi_timeseries(signed=signed)
        except FileNotFoundError:
            pass

        # 2. Fall back to on-the-fly extraction from step5 (cached)
        cache_key = "signed" if signed else "magnitude"
        if cache_key in self._roi_cache:
            return self._roi_cache[cache_key]

        logger.info(
            "No step6 ROI files in %s — extracting on-the-fly from step5",
            self.data_dir,
        )
        from ..atlas.atlas_utils import extract_roi_timeseries, find_atlas_dir

        stc_data = self.load_source_timecourses(magnitude=not signed)
        coords = self.load_source_coords()

        if atlas_dir is None:
            atlas_dir = find_atlas_dir()

        result = extract_roi_timeseries(
            stc_data,
            coords,
            atlas_dir,
            method=method,
            proximity_radius_mm=proximity_radius_mm,
        )
        self._roi_cache[cache_key] = result
        return result

    def load_source_timecourses(self, magnitude: bool = False) -> np.ndarray:
        """Load full source time courses.

        Reads ``step5_stc_signed.pkl`` by default (phase-preserving signed
        amplitudes) or ``step5_stc_magnitude.pkl`` when ``magnitude=True``.

        Falls back to legacy ``step5_stc.pkl`` ONLY when ``magnitude=True``
        is explicitly requested — historical pipeline output wrote the
        unsuffixed file as magnitude, so loading it as "signed" would
        silently return rectified data and corrupt phase-based analyses.

        Parameters
        ----------
        magnitude : bool
            If True, return rectified (absolute) source amplitudes.
            If False (default), return signed source amplitudes.

        Returns
        -------
        ndarray, shape (n_sources, n_times)
            Source-space time courses.

        Raises
        ------
        FileNotFoundError
            If neither ``step5_stc_{signed,magnitude}.pkl`` exists. Re-run
            source-localization to produce the variant-suffixed outputs.
        """
        suffix = "magnitude" if magnitude else "signed"
        primary = f"step5_stc_{suffix}.pkl"
        if (self.data_dir / primary).exists():
            stc = self._load_pkl(primary)
        elif magnitude and (self.data_dir / "step5_stc.pkl").exists():
            # Legacy pipeline output: step5_stc.pkl was saved as a magnitude
            # duplicate. Safe to read only when magnitude is requested.
            stc = self._load_pkl("step5_stc.pkl")
        else:
            raise FileNotFoundError(
                f"{primary} not found in {self.data_dir}. "
                f"Re-run source-localization to produce signed/magnitude "
                f"variants (legacy step5_stc.pkl is magnitude-only and "
                f"cannot be used for signed analyses)."
            )
        # MNE SourceEstimate stores data as (n_sources, n_times)
        return stc.data if hasattr(stc, "data") else np.asarray(stc)
