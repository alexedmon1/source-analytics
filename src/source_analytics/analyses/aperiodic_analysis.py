"""Aperiodic analysis module: fits 1/f spectral parameters, exports CSV, calls R."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.psd import compute_psd_multiroi
from ..spectral.aperiodic import fit_aperiodic_multiroi
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


class AperiodicAnalysis(BaseAnalysis):
    """Aperiodic (1/f) spectral analysis with group comparisons.

    Python fits specparam/FOOOF per ROI per subject, exports CSV.
    R runs LMM statistics (exponent, offset) and generates figures.
    """

    name = "aperiodic"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._subject_aperiodic: list[dict] = []
        self._subject_groups: dict[str, str] = {}
        self._sfreq: float | None = None

    def setup(self) -> None:
        self._subject_aperiodic.clear()
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

        # Compute PSD for all ROIs (wide range for aperiodic fitting)
        roi_psds = compute_psd_multiroi(roi_ts, sfreq, fmin=1.0, fmax=100.0)
        self._subject_groups[uid] = subject.group

        # Fit aperiodic parameters per ROI
        aperiodic_params = fit_aperiodic_multiroi(roi_psds, freq_range=(2, 50))

        for roi_name, params in aperiodic_params.items():
            self._subject_aperiodic.append({
                "subject": uid,
                "group": subject.group,
                "roi": roi_name,
                "exponent": params["exponent"],
                "offset": params["offset"],
                "r_squared": params["r_squared"],
                "n_peaks": params["n_peaks"],
                "error": params["error"],
                "method": params["method"],
            })

    def aggregate(self) -> None:
        """Export aperiodic_params.csv for R consumption."""
        data_dir = self.output_dir / "data"

        df = pd.DataFrame(self._subject_aperiodic)
        if df.empty:
            logger.warning("No aperiodic data collected")
            return

        df.to_csv(data_dir / "aperiodic_params.csv", index=False)
        logger.info("Exported aperiodic_params.csv (%d rows)", len(df))

        # Log summary
        method = df["method"].iloc[0] if len(df) > 0 else "unknown"
        logger.info("Fitting method: %s", method)
        valid = df.dropna(subset=["exponent"])
        if len(valid) > 0:
            logger.info(
                "Exponent: mean=%.3f, std=%.3f",
                valid["exponent"].mean(), valid["exponent"].std(),
            )
            logger.info(
                "Offset: mean=%.3f, std=%.3f",
                valid["offset"].mean(), valid["offset"].std(),
            )

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Delegated to R."""
        pass

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary."""
        data_dir = self.output_dir / "data"

        if not (data_dir / "aperiodic_params.csv").exists():
            logger.error("aperiodic_params.csv not found -- skipping R analysis")
            return

        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "aperiodic_analysis.R"
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return

        # Write study config YAML with sfreq
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
