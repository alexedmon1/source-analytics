"""ROI Connectivity Analysis: coherence and imaginary coherence between ROI pairs."""

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
from ..viz.constants import CC_ROIS, METRIC_LABELS
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
    and imaginary coherence for all 780 unique ROI pairs (40 brain ROIs;
    6 corpus callosum white matter tracts excluded).

    Python computes connectivity matrices and exports edge-level CSV.
    R (lme4, ggplot2) handles global t-tests, region-pair LMM, figures,
    and summary report.
    """

    name = "roi_connectivity"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._edge_rows: list[dict] = []
        self._sfreq: float | None = None

    def setup(self) -> None:
        self._edge_rows.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)

        # Use signed timeseries to preserve oscillatory phase
        roi_ts = loader.load_or_extract_roi_timeseries(
            signed=True, atlas_dir=self._atlas_dir,
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

        # Compute connectivity per bootstrap draw and average matrices
        avg_results: dict[str, dict[str, np.ndarray]] | None = None
        roi_names: list[str] | None = None
        n_draws = len(draws)

        for draw_ts in draws:
            band_results, roi_names = compute_connectivity_matrix(
                draw_ts, sfreq, self.config.bands,
            )
            if avg_results is None:
                # First draw — initialize accumulators
                avg_results = {
                    band: {metric: mat.copy() for metric, mat in metrics.items()}
                    for band, metrics in band_results.items()
                }
            else:
                # Accumulate
                for band, metrics in band_results.items():
                    for metric, mat in metrics.items():
                        avg_results[band][metric] += mat

        # Divide by number of draws to get mean
        if n_draws > 1:
            for band in avg_results:
                for metric in avg_results[band]:
                    avg_results[band][metric] /= n_draws

        n_rois = len(roi_names)

        # Flatten upper triangle to edge rows
        for band_name, metrics in avg_results.items():
            coh_mat = metrics["coherence"]
            icoh_mat = metrics["imag_coherence"]
            pli_mat = metrics.get("pli")
            dwpli_mat = metrics.get("dwpli")
            aec_mat = metrics.get("aec")
            pcorr_mat = metrics.get("partial_corr")

            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    row = {
                        "subject": uid,
                        "group": subject.group,
                        "band": band_name,
                        "roi1": roi_names[i],
                        "roi2": roi_names[j],
                        "coherence": float(coh_mat[i, j]),
                        "imag_coherence": float(icoh_mat[i, j]),
                    }
                    if pli_mat is not None:
                        row["pli"] = float(pli_mat[i, j])
                    if dwpli_mat is not None:
                        row["dwpli"] = float(dwpli_mat[i, j])
                    if aec_mat is not None:
                        row["aec"] = float(aec_mat[i, j])
                    if pcorr_mat is not None:
                        row["partial_corr"] = float(pcorr_mat[i, j])
                    self._edge_rows.append(row)

    def aggregate(self) -> None:
        """Export edge-level CSV for R consumption."""
        data_dir = self.output_dir / "data"

        edge_df = pd.DataFrame(self._edge_rows)
        if edge_df.empty:
            logger.warning("No connectivity data collected")
            return

        edge_df.to_csv(data_dir / "roi_connectivity_edges.csv", index=False)
        logger.info("Exported roi_connectivity_edges.csv (%d rows)", len(edge_df))

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Generate circos and heatmap figures for ROI-level connectivity.

        Produces one multi-row figure per metric × band × plot_type, with
        all contrasts combined as rows (Group A | Group B | Difference per row).
        """
        from ..viz.connectivity_plots import (
            build_roi_matrix,
            build_significance_matrix,
            plot_connectivity_multicontrast,
            plot_significance_circos,
        )

        data_dir = self.output_dir / "data"
        fig_dir = self.fig_dir
        tables_dir = self.tbl_dir

        # Try both filenames (aggregate() writes roi_connectivity_edges.csv;
        # older runs may have connectivity_edges.csv)
        csv_path = data_dir / "roi_connectivity_edges.csv"
        if not csv_path.exists():
            csv_path = data_dir / "connectivity_edges.csv"
        if not csv_path.exists():
            logger.warning("No connectivity edge CSV found — skipping figures")
            return

        edges_df = pd.read_csv(csv_path)
        roi_cats = self.config.roi_categories
        if not roi_cats:
            logger.warning("No roi_categories in config — skipping connectivity figures")
            return

        # Load posthoc p-values if available (for significance overlays)
        posthoc_path = tables_dir / "connectivity_posthoc_region_pair.csv"
        posthoc_df = None
        if posthoc_path.exists():
            posthoc_df = pd.read_csv(posthoc_path)
            logger.info("Loaded posthoc results: %d rows", len(posthoc_df))

        metrics = ["coherence", "imag_coherence"]
        if "pli" in edges_df.columns:
            metrics.append("pli")
        if "dwpli" in edges_df.columns:
            metrics.append("dwpli")
        if "aec" in edges_df.columns:
            metrics.append("aec")
        if "partial_corr" in edges_df.columns:
            metrics.append("partial_corr")
        bands = list(self.config.bands.keys())
        contrasts = self.config.contrasts

        if not contrasts:
            logger.warning("No contrasts defined — skipping connectivity figures")
            return

        n_figs = 0
        for band in bands:
            band_df = edges_df[edges_df["band"] == band]
            if band_df.empty:
                continue
            band_safe = band.replace(" ", "_").lower()

            for metric in metrics:
                metric_label = METRIC_LABELS.get(metric, metric.replace('_', ' ').title())

                # Build matrices for all contrasts
                contrast_data = []
                roi_labels = region_names = region_sizes = None
                for contrast in contrasts:
                    ga, gb = contrast.group_a, contrast.group_b
                    label_a = self.config.get_group_label(ga)
                    label_b = self.config.get_group_label(gb)

                    mat_a, roi_labels, region_names, region_sizes = build_roi_matrix(
                        band_df, roi_cats, metric, group=ga,
                    )
                    mat_b, _, _, _ = build_roi_matrix(
                        band_df, roi_cats, metric, group=gb,
                    )
                    contrast_data.append((mat_a, mat_b, f"{label_a} vs {label_b}"))

                if not contrast_data or roi_labels is None:
                    continue

                for plot_type in ("circos", "heatmap"):
                    out = fig_dir / f"{plot_type}_{metric}_{band_safe}.png"
                    plot_connectivity_multicontrast(
                        contrast_data,
                        roi_labels, region_names, region_sizes,
                        out,
                        plot_type=plot_type,
                        title=f"{band} — {metric_label}",
                        show_roi_labels=(plot_type != "circos"),
                    )
                    n_figs += 1

                # Significance circos per contrast (these stay per-contrast
                # since they depend on specific posthoc results)
                if posthoc_df is not None:
                    for contrast in contrasts:
                        ga, gb = contrast.group_a, contrast.group_b
                        label_a = self.config.get_group_label(ga)
                        label_b = self.config.get_group_label(gb)

                        mat_a, rl, rn, rs = build_roi_matrix(
                            band_df, roi_cats, metric, group=ga,
                        )
                        mat_b, _, _, _ = build_roi_matrix(
                            band_df, roi_cats, metric, group=gb,
                        )

                        sig_mask = build_significance_matrix(
                            posthoc_df, rl, rn, rs, band, metric,
                        )
                        if sig_mask is not None and sig_mask.any():
                            out = fig_dir / f"circos_sig_{contrast.name}_{metric}_{band_safe}.png"
                            plot_significance_circos(
                                mat_a, mat_b,
                                rl, rn, rs, sig_mask, out,
                                group_labels=(label_a, label_b),
                                title=f"{band} — {metric_label} (uncorrected p < 0.05)",
                            )
                            n_figs += 1

        logger.info("Generated %d connectivity figures", n_figs)

        # Also regenerate R-generated figures (bar/boxplots, heatmaps)
        self._call_r_figures_only(
            "roi_connectivity_analysis.R", "roi_connectivity_edges.csv",
        )

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary report."""
        data_dir = self.output_dir / "data"

        if not (data_dir / "roi_connectivity_edges.csv").exists():
            logger.error("roi_connectivity_edges.csv not found -- skipping R analysis")
            return

        # Find R scripts
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "roi_connectivity_analysis.R"
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
