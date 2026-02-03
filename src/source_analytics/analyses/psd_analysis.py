"""PSD Analysis module: computes PSD, exports CSVs, calls R for stats/viz."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.psd import compute_psd_multiroi
from ..spectral.band_power import extract_band_power_multiroi
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    """Locate the R/ directory relative to this package."""
    # Walk up from this file to find the R/ directory
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent  # src/../..
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    # Fallback: check common locations
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find R/ scripts directory. Expected at: " + str(pkg_root / "R")
    )


class PSDAnalysis(BaseAnalysis):
    """Power spectral density analysis with group comparisons.

    Python computes PSD and band power, exports CSVs.
    R (lme4, ggplot2) handles statistics and visualization.
    """

    name = "psd"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._subject_band_power: list[dict] = []
        self._subject_psd_curves: list[dict] = []
        self._subject_groups: dict[str, str] = {}
        self._sfreq: float | None = None

    def setup(self) -> None:
        self._subject_band_power.clear()
        self._subject_psd_curves.clear()
        self._subject_groups.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)

        roi_ts = loader.load_roi_timeseries(signed=False)
        sfreq = loader.load_sfreq()

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        uid = f"{subject.group}_{subject.subject_id}"

        # Compute PSD for all ROIs
        fmax = max(hi for _, hi in self.config.bands.values()) + 10
        roi_psds = compute_psd_multiroi(roi_ts, sfreq, fmax=fmax)
        self._subject_groups[uid] = subject.group

        # Collect PSD curves for export
        for roi_name, (freqs, psd) in roi_psds.items():
            for i, freq in enumerate(freqs):
                self._subject_psd_curves.append({
                    "subject": uid,
                    "group": subject.group,
                    "roi": roi_name,
                    "freq_hz": float(freq),
                    "psd": float(psd[i]),
                })

        # Extract band power
        band_power = extract_band_power_multiroi(roi_psds, self.config.bands)

        for roi_name, bp_dict in band_power.items():
            for band_name, power_vals in bp_dict.items():
                self._subject_band_power.append({
                    "subject": uid,
                    "group": subject.group,
                    "roi": roi_name,
                    "band": band_name,
                    "absolute": power_vals["absolute"],
                    "relative": power_vals["relative"],
                    "dB": power_vals["dB"],
                })

    def aggregate(self) -> None:
        """Export CSVs for R consumption."""
        data_dir = self.output_dir / "data"

        # Band power CSV
        band_df = pd.DataFrame(self._subject_band_power)
        if band_df.empty:
            logger.warning("No band power data collected")
            return

        band_df.to_csv(data_dir / "band_power.csv", index=False)
        logger.info("Exported band_power.csv (%d rows)", len(band_df))

        # PSD curves CSV
        psd_df = pd.DataFrame(self._subject_psd_curves)
        if not psd_df.empty:
            psd_df.to_csv(data_dir / "psd_curves.csv", index=False)
            logger.info("Exported psd_curves.csv (%d rows)", len(psd_df))

    def statistics(self) -> None:
        """Delegated to R — this is a no-op in Python."""
        pass

    def figures(self) -> None:
        """Delegated to R — this is a no-op in Python."""
        pass

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary."""
        data_dir = self.output_dir / "data"

        # Verify CSVs exist
        if not (data_dir / "band_power.csv").exists():
            logger.error("band_power.csv not found — skipping R analysis")
            return

        # Find R scripts
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "psd_analysis.R"
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return

        # Find the study config YAML path
        # Copy config to data dir so R can read it
        config_path = data_dir / "study_config.yaml"
        import yaml
        # Always write config so sfreq is up-to-date
        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Call Rscript
        cmd = [
            "Rscript", str(r_script),
            "--data-dir", str(data_dir),
            "--config", str(config_path),
            "--output-dir", str(self.output_dir),
        ]

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
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
