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
from ..results_spec import ResultsSpec, SpecTable
from ..spectral.connectivity import DPLI_DIRECTION_NOTE, compute_connectivity_matrix
from ..viz.constants import CC_ROIS, METRIC_LABELS
from .base import BaseAnalysis

logger = logging.getLogger(__name__)

#: The kernel's metric names -> the results specification's measure vocabulary.
CONNECTIVITY_MEASURES = {"coherence": "coherence", "imag_coherence": "imcoh", "pli": "pli", "dwpli": "dwpli",
                         "wpli": "wpli", "dpli": "dpli", "aec": "aec", "partial_corr": "partial_correlation"}


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
    SELECTABLE = {"metric": "connectivity metric", "band": "frequency band"}

    _DESCRIPTIVE_COLUMNS = (
        ("group", {"Description": "the subjects' group (the design factor)"}),
        ("n", {"Description": "subjects in the group with a value"}),
        ("mean", {"Description": "the group's mean, in the measure's units (the per-subject values are the "
                                 "subject's value for the ROI pair, or its mean over every ROI pair)"}),
        ("sd", {"Description": "the SD across the group's subjects"}),
        ("roi1", {"Description": "the pair's first ROI"}),
        ("roi2", {"Description": "the pair's second ROI"}),
        ("spatial", {"Description": "the ROI pair, 'A -- B'", "Standard": "element", "ElementName": "ROI pair"}),
    )
    RESULTS_SPEC = ResultsSpec(
        domain="connectivity",
        measures=CONNECTIVITY_MEASURES,
        units={"coherence": "magnitude-squared coherence (0 to 1)", "imag_coherence": "|imaginary coherence|",
               "pli": "phase lag index (0 to 1)", "dwpli": "debiased weighted PLI", "wpli": "weighted PLI (0 to 1)",
               "dpli": "directed PLI (0.5 = no lead)", "aec": "amplitude envelope correlation (r)",
               "partial_corr": "partial correlation (r)"},
        input_file="roi_connectivity_edges.csv",
        role="descriptive",
        method="per subject, band and connectivity measure, the connectivity of every ROI pair from the signed "
               "ROI time series; the groups' means and SDs, per ROI pair and over every pair. No tests: the "
               "group comparisons of these networks are roi_nbs (sub-networks) and roi_graph (graph metrics)",
        tables=(
            SpecTable("_group_summary.csv", "descriptives",
                      "one group's mean and SD of one connectivity measure in one band, over every ROI pair",
                      "group means of each connectivity measure, over the whole network", element=False,
                      headline=True),
            SpecTable("_group_edges.csv", "descriptives",
                      "one group's mean and SD of one connectivity measure in one band and one ROI pair",
                      "group means per ROI pair (the matrices the figures draw)", element=False),
        ),
        extra_columns=_DESCRIPTIVE_COLUMNS,
        correction={"p_kind": "none", "method": "none", "family": "none: descriptive, no tests",
                    "statement": "descriptive: group means and SDs, no tests (see roi_nbs and roi_graph)"},
        effect={"measure": None,
                "reason": "descriptive: group means of connectivity, no contrast; the tests and their effects "
                          "are in roi_nbs and roi_graph"},
        caveats=(DPLI_DIRECTION_NOTE,
                 ("Zero-lag measures (coherence, AEC) are inflated by volume conduction and field spread "
                  "between nearby ROIs; the lagged ones (imaginary coherence, PLI, wPLI, dwPLI) are less so."),),
    )

    # Metrics the ROI kernel emits. Only the selected ones are computed
    # (compute_connectivity_matrix(metrics=...)); values equal a full pass.
    _ROI_METRICS = [
        "coherence", "imag_coherence", "pli", "dwpli", "wpli", "dpli",
        "aec", "partial_corr",
    ]

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._edge_rows: list[dict] = []
        self._sfreq: float | None = None
        self._metrics: list[str] = list(self._ROI_METRICS)

    def setup(self) -> None:
        # Configured set: the module's `metrics:` list (under
        # paradigms.<p>.analyses.roi_connectivity), else every ROI metric — the
        # same contract vertex_connectivity honours. --metric / --select then
        # narrows within that set.
        cfg_metrics = (self.config.raw.get(self.name) or {}).get("metrics")
        if cfg_metrics:
            unknown = [m for m in cfg_metrics if m not in self._ROI_METRICS]
            if unknown:
                raise ValueError(
                    f"roi_connectivity: unknown metrics in config {unknown}; "
                    f"known: {list(self._ROI_METRICS)}"
                )
            configured = [m for m in self._ROI_METRICS if m in set(cfg_metrics)]
        else:
            configured = list(self._ROI_METRICS)
        self._metrics = self._select("metric", configured)
        if "dpli" in self._metrics:
            logger.warning("%s: %s", self.name, DPLI_DIRECTION_NOTE)
        self._edge_rows.clear()

    def _compute_subject(self, subject: SubjectInfo):
        """Pure per-subject ROI connectivity compute (parallel-safe)."""
        loader = SubjectLoader(subject.data_dir)

        # Use signed timeseries to preserve oscillatory phase
        roi_ts = loader.load_or_extract_roi_timeseries(
            signed=True, atlas_dir=self._atlas_dir, rois=self.config.rois,
        )
        # Exclude corpus callosum white matter tracts
        roi_ts = {k: v for k, v in roi_ts.items() if k not in CC_ROIS}
        sfreq = loader.load_sfreq()
        draws = self._equalize_roi_timeseries(roi_ts, sfreq)

        uid = f"{subject.group}_{subject.subject_id}"
        edge_rows: list[dict] = []

        # Compute connectivity per bootstrap draw and average matrices
        avg_results: dict[str, dict[str, np.ndarray]] | None = None
        roi_names: list[str] | None = None
        n_draws = len(draws)

        for draw_ts in draws:
            band_results, roi_names = compute_connectivity_matrix(
                draw_ts, sfreq, self._selected_bands(), metrics=self._metrics,
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

        # Flatten upper triangle to edge rows (the selected metrics only).
        for band_name, metrics in avg_results.items():
            sel_mats = {
                m: metrics[m] for m in self._metrics
                if metrics.get(m) is not None
            }
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    row = {
                        "subject": uid,
                        "group": subject.group,
                        "band": band_name,
                        "roi1": roi_names[i],
                        "roi2": roi_names[j],
                    }
                    for m, mat in sel_mats.items():
                        row[m] = float(mat[i, j])
                    edge_rows.append(row)

        return {"uid": uid, "sfreq": float(sfreq), "edge_rows": edge_rows}

    def _merge_subject(self, payload) -> None:
        if self._sfreq is None:
            self._sfreq = payload["sfreq"]
        elif payload["sfreq"] != self._sfreq:
            logger.warning("Subject %s has sfreq=%.0f, expected %.0f",
                           payload["uid"], payload["sfreq"], self._sfreq)
        self._edge_rows.extend(payload["edge_rows"])

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
        """The groups' descriptive tables (the R report reads the edges itself): per group, band and
        connectivity measure, the mean and SD across subjects of each ROI pair and of each subject's
        mean over every pair, from ``data/roi_connectivity_edges.csv``."""
        path = self.output_dir / "data" / "roi_connectivity_edges.csv"
        if not path.exists():
            logger.warning("No connectivity edge CSV: no group tables")
            return
        edges = pd.read_csv(path)
        metrics = [m for m in self._metrics if m in edges.columns]
        if not metrics:
            return
        long = edges.melt(id_vars=["subject", "group", "band", "roi1", "roi2"], value_vars=metrics,
                          var_name="dv", value_name="value").dropna(subset=["value"])
        long["measure"] = long["dv"].map(CONNECTIVITY_MEASURES)

        def describe(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
            out = df.groupby(by, sort=False)["value"].agg(n="count", mean="mean", sd="std").reset_index()
            return out

        per_subject = long.groupby(["subject", "group", "band", "dv", "measure"], sort=False)["value"].mean() \
            .reset_index()
        summary = describe(per_subject, ["group", "band", "dv", "measure"])
        edges_tbl = describe(long, ["group", "band", "dv", "measure", "roi1", "roi2"])
        edges_tbl.insert(edges_tbl.columns.get_loc("roi1"), "spatial",
                         edges_tbl["roi1"] + " -- " + edges_tbl["roi2"])
        self.tbl_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(self.tbl_dir / f"{self.name}_group_summary.csv", index=False)
        edges_tbl.to_csv(self.tbl_dir / f"{self.name}_group_edges.csv", index=False)
        logger.info("Exported %s_group_summary.csv (%d rows) and _group_edges.csv (%d rows)",
                    self.name, len(summary), len(edges_tbl))

    def figures(self) -> None:
        """Generate circos and heatmap figures for ROI-level connectivity.

        Produces one multi-row figure per metric × band × plot_type, with
        all contrasts combined as rows (Group A | Group B | Difference per row).
        """
        from ..figure_record import record_figure
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

        # Plot the selected metrics that are actually present in the CSV.
        metrics = [m for m in self._metrics if m in edges_df.columns]
        bands = list(self._selected_bands().keys())
        contrasts = self._pairwise_contrasts()

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
                    record_figure(fig_dir, out, measure=metric, facets={"band": band}, kind="matrix",
                                  shows=f"{self.name}_group_edges.csv")
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
                            record_figure(fig_dir, out, measure=metric, contrast=contrast.name,
                                          facets={"band": band}, kind="matrix")
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
        config_data = self._r_config_data()
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
                timeout=self._r_timeout,
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    logger.info("[R] %s", line)
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        logger.info("[R] %s", line)
            if result.returncode != 0:
                self._r_step_failed("R script failed with exit code %d", result.returncode)
        except FileNotFoundError:
            logger.error(
                "Rscript not found. Install R to enable statistics and visualization."
            )
        except subprocess.TimeoutExpired:
            self._r_step_failed("R script timed out after %s s", self._r_timeout)
