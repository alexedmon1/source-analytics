"""Connectivity Analysis: coherence and imaginary coherence between ROI pairs."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.connectivity import compute_connectivity_matrix
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


class ConnectivityAnalysis(BaseAnalysis):
    """Functional connectivity analysis using coherence and imaginary coherence.

    Uses **signed** (phase-preserving) ROI timeseries to compute coherence
    and imaginary coherence for all 1035 unique ROI pairs (46 ROIs).

    Python computes connectivity matrices and exports edge-level CSV.
    R (lme4, ggplot2) handles global t-tests, region-pair LMM, figures,
    and summary report.
    """

    name = "connectivity"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._edge_rows: list[dict] = []
        self._sfreq: float | None = None

    def setup(self) -> None:
        self._edge_rows.clear()

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

        # Compute connectivity matrices for all bands
        band_results, roi_names = compute_connectivity_matrix(
            roi_ts, sfreq, self.config.bands,
        )
        n_rois = len(roi_names)

        # Flatten upper triangle to edge rows
        for band_name, metrics in band_results.items():
            coh_mat = metrics["coherence"]
            icoh_mat = metrics["imag_coherence"]

            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    self._edge_rows.append({
                        "subject": uid,
                        "group": subject.group,
                        "band": band_name,
                        "roi1": roi_names[i],
                        "roi2": roi_names[j],
                        "coherence": float(coh_mat[i, j]),
                        "imag_coherence": float(icoh_mat[i, j]),
                    })

    def aggregate(self) -> None:
        """Export edge-level CSV for R consumption."""
        data_dir = self.output_dir / "data"

        edge_df = pd.DataFrame(self._edge_rows)
        if edge_df.empty:
            logger.warning("No connectivity data collected")
            return

        edge_df.to_csv(data_dir / "connectivity_edges.csv", index=False)
        logger.info("Exported connectivity_edges.csv (%d rows)", len(edge_df))

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Delegated to R."""
        pass

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary report."""
        data_dir = self.output_dir / "data"

        if not (data_dir / "connectivity_edges.csv").exists():
            logger.error("connectivity_edges.csv not found -- skipping R analysis")
            return

        # Find R scripts
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "connectivity_analysis.R"
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
