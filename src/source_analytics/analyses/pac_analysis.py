"""Phase-Amplitude Coupling (PAC) Analysis: Modulation Index with surrogate z-scoring.

Uses signed (phase-preserving) ROI timeseries to compute PAC for all ROIs
across valid cross-frequency pairs. Z-scored MI normalizes for spectral
differences across subjects.

Python computes PAC z-scores and exports per-subject CSV.
R (lme4, ggplot2) handles global t-tests, region-level LMM, figures,
and summary report.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.pac import compute_pac_multiroi, get_valid_pac_pairs
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    """Locate the R/ directory relative to this package."""
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent  # src/../..
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find R/ scripts directory. Expected at: " + str(pkg_root / "R")
    )


class PACAnalysis(BaseAnalysis):
    """Phase-Amplitude Coupling analysis via Modulation Index with surrogate z-scoring.

    Computes z-scored MI for all ROIs across valid cross-frequency pairs
    (e.g., theta-gamma, delta-gamma, alpha-gamma). Uses signed timeseries
    to preserve oscillatory phase information.

    Python computes PAC values and exports CSV. R handles statistics
    (global t-tests, region-level LMM), figures (bar charts, comodulograms,
    forest plots), and summary report.
    """

    name = "pac"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._pac_rows: list[dict] = []
        self._sfreq: float | None = None

    def setup(self) -> None:
        self._pac_rows.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)

        # Use signed timeseries to preserve oscillatory phase
        roi_ts = loader.load_roi_timeseries(signed=True)
        sfreq = loader.load_sfreq()

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        uid = f"{subject.group}_{subject.subject_id}"

        # Build valid PAC pairs from config bands
        pac_pairs = get_valid_pac_pairs(self.config.bands)

        if not pac_pairs:
            logger.warning(
                "No valid PAC pairs found for subject %s with bands: %s",
                subject.subject_id, list(self.config.bands.keys()),
            )
            return

        logger.info(
            "Computing PAC for %s: %d ROIs x %d pairs",
            subject.subject_id, len(roi_ts), len(pac_pairs),
        )

        # Compute PAC z-scores for all ROIs x all pairs
        results = compute_pac_multiroi(
            roi_ts, sfreq, pac_pairs, self.config.bands,
        )

        # Append subject/group info to each row
        for row in results:
            row["subject"] = uid
            row["group"] = subject.group
            self._pac_rows.append(row)

    def aggregate(self) -> None:
        """Export PAC values CSV for R consumption."""
        data_dir = self.output_dir / "data"

        pac_df = pd.DataFrame(self._pac_rows)
        if pac_df.empty:
            logger.warning("No PAC data collected")
            return

        # Reorder columns for clarity
        col_order = [
            "subject", "group", "roi", "phase_band", "amp_band",
            "freq_pair", "mi", "z_score", "surr_mean", "surr_std",
        ]
        pac_df = pac_df[[c for c in col_order if c in pac_df.columns]]

        pac_df.to_csv(data_dir / "pac_values.csv", index=False)
        logger.info("Exported pac_values.csv (%d rows)", len(pac_df))

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Delegated to R."""
        pass

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary report."""
        data_dir = self.output_dir / "data"

        if not (data_dir / "pac_values.csv").exists():
            logger.error("pac_values.csv not found -- skipping R analysis")
            return

        # Find R scripts
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "pac_analysis.R"
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return

        # Write study config YAML for R
        config_path = data_dir / "study_config.yaml"
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
            logger.error(
                "Rscript not found. Install R to enable statistics and visualization."
            )
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 600 seconds")
