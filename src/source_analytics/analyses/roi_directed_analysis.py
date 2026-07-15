"""ROI directed connectivity: information-theoretic / directed influence between ROI pairs.

The home for directed connectivity metrics: transfer entropy (``te``, ``net_te``)
and the directed transfer function (``dtf``, MVAR-based). ``--metric`` selects
which to compute; both share the signed-ROI front-end.
"""

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
from ..spectral.directed import compute_dtf, DEFAULT_ORDER, DEFAULT_RIDGE
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


class ROIDirectedAnalysis(BaseAnalysis):
    """ROI-level directed connectivity (transfer entropy; DTF planned).

    Uses **signed** (phase-preserving) ROI timeseries to compute binned
    transfer entropy for all n*(n-1) directed ROI pairs (40 brain ROIs →
    1,560 directed pairs; 6 corpus callosum white matter tracts excluded).

    Python computes directed matrices and exports directed edge-level CSV.
    R (lme4, ggplot2) handles global t-tests, directional paired t-tests,
    region-pair LMM, and summary report.

    ``--metric`` selects which directed measure(s) to compute:
    ``te`` (transfer entropy, which also yields ``net_te``) and/or ``dtf``
    (directed transfer function via ridge-MVAR).
    """

    name = "roi_directed"
    SELECTABLE = {"metric": "directed measure", "band": "frequency band",
                  "hypothesis": "declared hypothesis"}

    # Directed measures this module can produce (shared signed-ROI front-end).
    # `te` additionally emits the derived `net_te`.
    _DIRECTED_METRICS = ["te", "dtf"]

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._edge_rows: list[dict] = []
        self._sfreq: float | None = None
        self._metrics: list[str] = list(self._DIRECTED_METRICS)
        cfg = config.raw.get(self.name, {})
        self._mvar_order = int(cfg.get("mvar_order", DEFAULT_ORDER))
        self._mvar_ridge = float(cfg.get("mvar_ridge", DEFAULT_RIDGE))

    def setup(self) -> None:
        self._metrics = self._select("metric", self._DIRECTED_METRICS)
        self._edge_rows.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)

        # Use signed timeseries to preserve oscillatory phase
        roi_ts = loader.load_or_extract_roi_timeseries(
            signed=True, atlas_dir=self._atlas_dir, rois=self.config.rois,
        )
        # Exclude corpus callosum white matter tracts
        roi_ts = {k: v for k, v in roi_ts.items() if k not in CC_ROIS}
        sfreq = loader.load_sfreq()
        draws = self._equalize_roi_timeseries(roi_ts, sfreq)

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        uid = f"{subject.group}_{subject.subject_id}"

        # Compute the selected directed measures per bootstrap draw, averaging
        # the directed matrices. Each kernel returns {band: {metric: matrix}}
        # with matrix[i, j] = source i -> target j.
        avg_results: dict[str, dict[str, np.ndarray]] | None = None
        roi_names: list[str] | None = None
        n_draws = len(draws)

        for draw_ts in draws:
            draw_results: dict[str, dict[str, np.ndarray]] = {}
            if "te" in self._metrics:
                te_res, roi_names = compute_transfer_entropy(
                    draw_ts, sfreq, self._selected_bands(),
                )
                for band, mets in te_res.items():
                    draw_results.setdefault(band, {}).update(mets)  # te, net_te
            if "dtf" in self._metrics:
                dtf_res, roi_names = compute_dtf(
                    draw_ts, sfreq, self._selected_bands(),
                    order=self._mvar_order, ridge=self._mvar_ridge,
                )
                for band, mets in dtf_res.items():
                    draw_results.setdefault(band, {}).update(mets)  # dtf

            if avg_results is None:
                avg_results = {
                    band: {m: mat.copy() for m, mat in mets.items()}
                    for band, mets in draw_results.items()
                }
            else:
                for band, mets in draw_results.items():
                    for m, mat in mets.items():
                        avg_results[band][m] += mat

        if avg_results is None:
            return
        if n_draws > 1:
            for band in avg_results:
                for m in avg_results[band]:
                    avg_results[band][m] /= n_draws

        n_rois = len(roi_names)

        # Flatten to directed edge rows (all n*(n-1) pairs), emitting whichever
        # measure columns were computed.
        for band_name, mets in avg_results.items():
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
                    }
                    for m, mat in mets.items():
                        row[m] = float(mat[i, j])
                    self._edge_rows.append(row)

    def aggregate(self) -> None:
        """Export directed edge-level CSV for R consumption."""
        data_dir = self.output_dir / "data"

        edge_df = pd.DataFrame(self._edge_rows)
        if edge_df.empty:
            logger.warning("No transfer entropy data collected")
            return

        edge_df.to_csv(data_dir / "roi_transfer_entropy_edges.csv", index=False)
        logger.info("Exported roi_transfer_entropy_edges.csv (%d rows)", len(edge_df))

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Regenerate R figures from existing data/tables."""
        self._call_r_figures_only(
            "roi_transfer_entropy_analysis.R", "roi_transfer_entropy_edges.csv",
        )

    def summary(self) -> None:
        """Call Rscript for statistics and summary report.

        The R stats currently cover transfer entropy; a DTF-only run skips R
        (the ``dtf`` column is still exported in the edge CSV for downstream use).
        """
        data_dir = self.output_dir / "data"

        if "te" not in self._metrics:
            logger.info(
                "roi_directed: metrics=%s — DTF has no R stats yet; edge CSV written, "
                "skipping R.", self._metrics,
            )
            return

        if not (data_dir / "roi_transfer_entropy_edges.csv").exists():
            logger.error("roi_transfer_entropy_edges.csv not found -- skipping R analysis")
            return

        # Find R scripts
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "roi_transfer_entropy_analysis.R"
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
        cmd.extend(self._r_no_figures_flags())
        cmd.extend(self._r_roi_categories_flags())

        # Manual hypothesis selection: pass --hypothesis NAME[,NAME] through to R
        # (mirrors roi_cross_freq; the R script honours it across all three tiers).
        wanted_hyp = self._selection.get("hypothesis")
        if wanted_hyp:
            cmd.extend(["--hypothesis", ",".join(sorted(wanted_hyp))])

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
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
