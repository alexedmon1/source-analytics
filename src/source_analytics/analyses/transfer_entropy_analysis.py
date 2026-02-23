"""Transfer Entropy Analysis: directed information-theoretic connectivity between ROI pairs."""

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
from ..spectral.transfer_entropy import compute_transfer_entropy
from ..viz.constants import CC_ROIS
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


class TransferEntropyAnalysis(BaseAnalysis):
    """Directed transfer entropy analysis between ROI pairs.

    Uses **signed** (phase-preserving) ROI timeseries to compute binned
    transfer entropy for all n*(n-1) directed ROI pairs (40 brain ROIs →
    1,560 directed pairs; 6 corpus callosum white matter tracts excluded).

    Python computes TE matrices and exports directed edge-level CSV.
    R (lme4, ggplot2) handles global t-tests, directional paired t-tests,
    region-pair LMM, and summary report.
    """

    name = "transfer_entropy"

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
        # Exclude corpus callosum white matter tracts
        roi_ts = {k: v for k, v in roi_ts.items() if k not in CC_ROIS}
        sfreq = loader.load_sfreq()

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        uid = f"{subject.group}_{subject.subject_id}"

        # Compute transfer entropy matrices for all bands
        band_results, roi_names = compute_transfer_entropy(
            roi_ts, sfreq, self.config.bands,
        )
        n_rois = len(roi_names)

        # Flatten to directed edge rows (all n*(n-1) pairs)
        for band_name, metrics in band_results.items():
            te_mat = metrics["te"]
            net_te_mat = metrics["net_te"]

            for i in range(n_rois):
                for j in range(n_rois):
                    if i == j:
                        continue
                    row = {
                        "subject": uid,
                        "group": subject.group,
                        "band": band_name,
                        "source_roi": roi_names[i],
                        "target_roi": roi_names[j],
                        "te": float(te_mat[i, j]),
                        "net_te": float(net_te_mat[i, j]),
                    }
                    self._edge_rows.append(row)

    def aggregate(self) -> None:
        """Export directed edge-level CSV for R consumption."""
        data_dir = self.output_dir / "data"

        edge_df = pd.DataFrame(self._edge_rows)
        if edge_df.empty:
            logger.warning("No transfer entropy data collected")
            return

        edge_df.to_csv(data_dir / "transfer_entropy_edges.csv", index=False)
        logger.info("Exported transfer_entropy_edges.csv (%d rows)", len(edge_df))

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Regenerate R figures from existing data/tables."""
        self._call_r_figures_only(
            "transfer_entropy_analysis.R", "transfer_entropy_edges.csv",
        )

    def summary(self) -> None:
        """Call Rscript for statistics and summary report."""
        data_dir = self.output_dir / "data"

        if not (data_dir / "transfer_entropy_edges.csv").exists():
            logger.error("transfer_entropy_edges.csv not found -- skipping R analysis")
            return

        # Find R scripts
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "transfer_entropy_analysis.R"
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
            "--fig-dir", str(self.fig_dir),
            "--tbl-dir", str(self.tbl_dir),
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
