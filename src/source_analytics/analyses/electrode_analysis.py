"""Electrode-level PSD and band power analysis.

Mirrors the PSD analysis module but operates on raw scalp EEG channels
instead of source-localized ROI time courses.  Uses ``subject_roster.csv``
to map each discovered subject to its raw ``.set/.fdt`` file.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.electrode_loader import load_eeglab_set
from ..spectral.psd import compute_psd
from ..spectral.band_power import extract_band_power
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    """Locate the R/ directory relative to this package."""
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find R/ scripts directory. Expected at: " + str(pkg_root / "R")
    )


class ElectrodeAnalysis(BaseAnalysis):
    """Electrode-level power spectral density analysis.

    Python computes per-channel PSD and band power, exports CSVs.
    R handles LMM statistics (group * channel) and visualization.

    Requires ``electrode.subject_roster`` in the study config pointing
    to a CSV with columns: ``subject_id, group, eeg_filename, eeg_dir``.
    """

    name = "electrode"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._subject_band_power: list[dict] = []
        self._subject_psd_curves: list[dict] = []
        self._sfreq: float | None = None
        self._roster: pd.DataFrame | None = None

    def setup(self) -> None:
        self._subject_band_power.clear()
        self._subject_psd_curves.clear()

        roster_path = self.config.electrode.get("subject_roster")
        if not roster_path:
            raise ValueError(
                "electrode.subject_roster not set in study config. "
                "Add 'electrode: {subject_roster: /path/to/subject_roster.csv}' "
                "to analysis.yaml."
            )
        roster_path = Path(roster_path)
        if not roster_path.exists():
            raise FileNotFoundError(f"Subject roster not found: {roster_path}")

        self._roster = pd.read_csv(roster_path)
        logger.info("Loaded subject roster: %d entries from %s", len(self._roster), roster_path)

        # Build lookup: subject_id -> row
        required_cols = {"subject_id", "eeg_filename", "eeg_dir"}
        missing = required_cols - set(self._roster.columns)
        if missing:
            raise ValueError(
                f"Subject roster missing required columns: {missing}. "
                f"Available: {list(self._roster.columns)}"
            )

    def _find_eeg_path(self, subject: SubjectInfo) -> Path | None:
        """Look up the raw EEG file path for a subject from the roster."""
        # Match on subject_id AND group to avoid collisions when the same
        # base ID exists in multiple groups (e.g., Dsbpro_0 in KO and WT).
        roster_group = subject.pipeline_dir.parent.name  # e.g., "KO ICV"
        matches = self._roster[
            (self._roster["subject_id"] == subject.subject_id)
            & (self._roster["group"] == roster_group)
        ]
        if matches.empty:
            # Fallback: match by subject_id only (safe when IDs are unique)
            matches = self._roster[self._roster["subject_id"] == subject.subject_id]
        if matches.empty:
            # Try matching by pipeline_dir basename
            matches = self._roster[
                self._roster["subject_id"] == subject.pipeline_dir.name
            ]
        if matches.empty:
            logger.warning(
                "Subject %s not found in roster, skipping electrode analysis",
                subject.subject_id,
            )
            return None

        if len(matches) > 1:
            logger.warning(
                "Multiple roster matches for %s in group %s, using first",
                subject.subject_id, roster_group,
            )

        row = matches.iloc[0]
        eeg_path = Path(row["eeg_dir"]) / row["eeg_filename"]
        if not eeg_path.exists():
            logger.warning("Raw EEG file not found: %s", eeg_path)
            return None
        return eeg_path

    def process_subject(self, subject: SubjectInfo) -> None:
        eeg_path = self._find_eeg_path(subject)
        if eeg_path is None:
            return

        data, sfreq, ch_names, _ = load_eeglab_set(eeg_path)

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        uid = f"{subject.group}_{subject.subject_id}"

        # Compute PSD and band power for each channel
        fmax = max(hi for _, hi in self.config.bands.values()) + 10

        for ch_idx, ch_name in enumerate(ch_names):
            ch_data = data[ch_idx, :]

            # Skip channels with all zeros or NaN
            if np.all(ch_data == 0) or np.any(np.isnan(ch_data)):
                logger.warning(
                    "Subject %s channel %s has bad data, skipping",
                    subject.subject_id, ch_name,
                )
                continue

            freqs, psd = compute_psd(ch_data, sfreq, fmax=fmax)

            # PSD curves
            for i, freq in enumerate(freqs):
                self._subject_psd_curves.append({
                    "subject": uid,
                    "group": subject.group,
                    "channel": ch_name,
                    "freq_hz": float(freq),
                    "psd": float(psd[i]),
                })

            # Band power
            bp = extract_band_power(freqs, psd, self.config.bands)
            for band_name, power_vals in bp.items():
                self._subject_band_power.append({
                    "subject": uid,
                    "group": subject.group,
                    "channel": ch_name,
                    "band": band_name,
                    "absolute": power_vals["absolute"],
                    "relative": power_vals["relative"],
                    "dB": power_vals["dB"],
                })

    def aggregate(self) -> None:
        """Export CSVs for R consumption."""
        data_dir = self.output_dir / "data"

        band_df = pd.DataFrame(self._subject_band_power)
        if band_df.empty:
            logger.warning("No electrode band power data collected")
            return

        band_df.to_csv(data_dir / "electrode_band_power.csv", index=False)
        logger.info("Exported electrode_band_power.csv (%d rows)", len(band_df))

        psd_df = pd.DataFrame(self._subject_psd_curves)
        if not psd_df.empty:
            psd_df.to_csv(data_dir / "electrode_psd_curves.csv", index=False)
            logger.info("Exported electrode_psd_curves.csv (%d rows)", len(psd_df))

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Delegated to R."""
        pass

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary."""
        data_dir = self.output_dir / "data"

        if not (data_dir / "electrode_band_power.csv").exists():
            logger.error("electrode_band_power.csv not found — skipping R analysis")
            return

        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "electrode_analysis.R"
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return

        # Write study config YAML for R
        config_path = data_dir / "study_config.yaml"
        import yaml

        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        cmd = [
            "Rscript", str(r_script),
            "--data-dir", str(data_dir),
            "--config", str(config_path),
            "--output-dir", str(self.output_dir),
        ]

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    logger.info("[R] %s", line)
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        logger.info("[R] %s", line)
            if result.returncode != 0:
                logger.error("R script failed with exit code %d", result.returncode)
        except FileNotFoundError:
            logger.error("Rscript not found. Install R to enable statistics and visualization.")
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 600 seconds")
