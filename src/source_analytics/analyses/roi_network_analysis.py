"""ROI Network Analysis: graph-theoretic metrics on ROI connectivity matrices.

Reads pre-computed ROI connectivity edges (from roi_connectivity analysis),
builds 40x40 symmetric connectivity matrices, computes graph metrics
(degree, clustering, betweenness, global efficiency, modularity,
small-worldness) and runs the Network-Based Statistic (NBS) for
subnetwork identification.
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
from ..stats.graph_metrics import compute_graph_metrics, nbs_permutation_test
from ..viz.constants import CC_ROIS, METRIC_LABELS
from .base import BaseAnalysis, find_r_script_dir

logger = logging.getLogger(__name__)


def _build_subject_matrix(
    edges_df: pd.DataFrame,
    roi_labels: list[str],
    metric: str,
) -> np.ndarray:
    """Build a symmetric N×N connectivity matrix from edge rows."""
    roi_idx = {name: i for i, name in enumerate(roi_labels)}
    n = len(roi_labels)
    mat = np.zeros((n, n), dtype=np.float64)

    for _, row in edges_df.iterrows():
        r1, r2 = row["roi1"], row["roi2"]
        i = roi_idx.get(r1)
        j = roi_idx.get(r2)
        if i is None or j is None:
            continue
        val = row[metric]
        mat[i, j] = val
        mat[j, i] = val

    return mat


class ROINetworkAnalysis(BaseAnalysis):
    """Graph-theoretic network analysis on ROI connectivity (40 brain ROIs)."""

    name = "roi_network"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._nodal_rows: list[dict] = []
        self._global_rows: list[dict] = []
        self._roi_labels: list[str] = []
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}
        self._conn_matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self._edges_df: pd.DataFrame | None = None

        # Config
        net_cfg = config.raw.get("roi_network", {})
        self._threshold_method = net_cfg.get("threshold_method", "proportional")
        self._threshold_value = float(net_cfg.get("threshold_value", 0.15))
        self._nbs_threshold = float(net_cfg.get("nbs_threshold", 2.5))
        self._nbs_permutations = int(net_cfg.get("nbs_permutations", 5000))
        self._connectivity_metrics = net_cfg.get(
            "connectivity_metrics", ["imag_coherence"]
        )

        self._nbs_results: dict = {}

    def setup(self) -> None:
        self._nodal_rows.clear()
        self._global_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._conn_matrices.clear()
        self._nbs_results.clear()

        # Locate the roi_connectivity edges CSV
        # Check sibling directory (same parent output dir)
        parent_dir = self.config.output_dir
        csv_path = parent_dir / "roi_connectivity" / "data" / "roi_connectivity_edges.csv"
        if not csv_path.exists():
            # Fallback: check within our own output
            csv_path = self.output_dir / "data" / "roi_connectivity_edges.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"roi_connectivity_edges.csv not found. "
                f"Run roi_connectivity analysis first. "
                f"Checked: {parent_dir / 'roi_connectivity' / 'data'}"
            )

        self._edges_df = pd.read_csv(csv_path)
        logger.info(
            "Loaded roi_connectivity_edges.csv: %d rows", len(self._edges_df),
        )

        # Build ordered ROI list (excluding CC white matter tracts)
        all_rois = set(self._edges_df["roi1"]) | set(self._edges_df["roi2"])
        brain_rois = sorted(r for r in all_rois if r not in CC_ROIS)
        self._roi_labels = brain_rois
        logger.info("ROI network: %d brain ROIs", len(brain_rois))

    def process_subject(self, subject: SubjectInfo) -> None:
        uid = f"{subject.group}_{subject.subject_id}"
        self._subject_groups[uid] = subject.group

        if self._edges_df is None:
            return

        subj_df = self._edges_df[self._edges_df["subject"] == uid]
        if subj_df.empty:
            logger.warning("No connectivity edges for %s — skipping", uid)
            return

        subject_metrics = {}
        subject_conn = {}

        for band_name in self.config.bands:
            band_df = subj_df[subj_df["band"] == band_name]
            if band_df.empty:
                continue

            band_metrics = {}
            band_conn = {}

            for metric in self._connectivity_metrics:
                if metric not in band_df.columns:
                    logger.warning(
                        "Metric '%s' not in edges CSV — skipping", metric,
                    )
                    continue

                # Build N×N matrix for this subject × band × metric
                conn_mat = _build_subject_matrix(
                    band_df, self._roi_labels, metric,
                )
                band_conn[metric] = conn_mat

                # Compute graph metrics
                gm = compute_graph_metrics(
                    conn_mat,
                    threshold_method=self._threshold_method,
                    threshold_value=self._threshold_value,
                )
                band_metrics[metric] = gm

                # Global metrics row
                self._global_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "band": band_name,
                    "metric": metric,
                    "global_efficiency": gm.global_efficiency,
                    "modularity": gm.modularity,
                    "small_worldness": gm.small_worldness,
                    "n_edges": gm.n_edges,
                })

                # Nodal metrics rows
                for ri, roi_name in enumerate(self._roi_labels):
                    self._nodal_rows.append({
                        "subject": uid,
                        "group": subject.group,
                        "band": band_name,
                        "metric": metric,
                        "roi": roi_name,
                        "degree": int(gm.degree[ri]),
                        "clustering": float(gm.clustering[ri]),
                        "betweenness": float(gm.betweenness[ri]),
                    })

            subject_metrics[band_name] = band_metrics
            subject_conn[band_name] = band_conn

        self._subject_data[uid] = subject_metrics
        self._conn_matrices[uid] = subject_conn

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        nodal_df = pd.DataFrame(self._nodal_rows)
        if not nodal_df.empty:
            nodal_df.to_csv(data_dir / "roi_network_nodal_metrics.csv", index=False)
            logger.info(
                "Exported roi_network_nodal_metrics.csv (%d rows)", len(nodal_df),
            )

        global_df = pd.DataFrame(self._global_rows)
        if not global_df.empty:
            global_df.to_csv(data_dir / "roi_network_global_metrics.csv", index=False)
            logger.info(
                "Exported roi_network_global_metrics.csv (%d rows)", len(global_df),
            )

        # Save ROI labels
        roi_df = pd.DataFrame({"roi": self._roi_labels})
        roi_df.to_csv(data_dir / "roi_labels.csv", index=False)

    def statistics(self) -> None:
        tbl_dir = self.tbl_dir
        all_stats = []

        for contrast in self.config.contrasts:
            group_a_uids = [
                uid for uid, g in self._subject_groups.items()
                if g == contrast.group_a
            ]
            group_b_uids = [
                uid for uid, g in self._subject_groups.items()
                if g == contrast.group_b
            ]

            if not group_a_uids or not group_b_uids:
                continue

            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)

            for band_name in self.config.bands:
                for metric in self._connectivity_metrics:
                    # Collect nodal metrics for t-tests
                    for nodal_metric in ["degree", "clustering", "betweenness"]:
                        vals_a = []
                        vals_b = []
                        for uid in group_a_uids:
                            bm = self._subject_data.get(uid, {}).get(band_name, {})
                            gm = bm.get(metric)
                            if gm is not None:
                                vals_a.append(
                                    getattr(gm, nodal_metric).astype(float)
                                )
                        for uid in group_b_uids:
                            bm = self._subject_data.get(uid, {}).get(band_name, {})
                            gm = bm.get(metric)
                            if gm is not None:
                                vals_b.append(
                                    getattr(gm, nodal_metric).astype(float)
                                )

                        if not vals_a or not vals_b:
                            continue

                        data_a = np.array(vals_a)
                        data_b = np.array(vals_b)

                        # Per-ROI Welch t-tests
                        from scipy.stats import ttest_ind

                        n_rois = data_a.shape[1]
                        for ri in range(n_rois):
                            t_stat, p_val = ttest_ind(
                                data_a[:, ri], data_b[:, ri],
                                equal_var=False,
                            )
                            all_stats.append({
                                "contrast": contrast.name,
                                "band": band_name,
                                "conn_metric": metric,
                                "graph_metric": nodal_metric,
                                "roi": self._roi_labels[ri],
                                "mean_a": float(data_a[:, ri].mean()),
                                "mean_b": float(data_b[:, ri].mean()),
                                "t": float(t_stat),
                                "p": float(p_val),
                            })

                    # NBS on connectivity matrices
                    matrices_a = []
                    matrices_b = []
                    for uid in group_a_uids:
                        bc = self._conn_matrices.get(uid, {}).get(band_name, {})
                        mat = bc.get(metric)
                        if mat is not None:
                            matrices_a.append(mat)
                    for uid in group_b_uids:
                        bc = self._conn_matrices.get(uid, {}).get(band_name, {})
                        mat = bc.get(metric)
                        if mat is not None:
                            matrices_b.append(mat)

                    if matrices_a and matrices_b:
                        nbs_result = nbs_permutation_test(
                            matrices_a, matrices_b,
                            nbs_threshold=self._nbs_threshold,
                            n_permutations=self._nbs_permutations,
                            seed=42,
                        )
                        key = f"{contrast.name}_{band_name}_{metric}"
                        self._nbs_results[key] = nbs_result

        # Apply FDR correction to nodal stats
        if all_stats:
            stats_df = pd.DataFrame(all_stats)

            # Handle NaN p-values (from zero-variance ROIs, e.g. constant
            # degree/betweenness in disconnected regions)
            nan_mask = np.isnan(stats_df["p"].values)
            n_nan = nan_mask.sum()
            if n_nan > 0:
                logger.warning(
                    "%d / %d nodal t-tests produced NaN p-values "
                    "(likely zero-variance ROIs); setting to 1.0 for FDR",
                    n_nan, len(stats_df),
                )
                stats_df.loc[nan_mask, "p"] = 1.0

            from scipy.stats import false_discovery_control
            try:
                stats_df["p_fdr"] = false_discovery_control(stats_df["p"].values)
            except (AttributeError, TypeError):
                # Fallback for older scipy: Benjamini-Hochberg manual
                p_vals = stats_df["p"].values
                n_tests = len(p_vals)
                sorted_idx = np.argsort(p_vals)
                ranks = np.empty(n_tests, dtype=float)
                ranks[sorted_idx] = np.arange(1, n_tests + 1)
                p_fdr = np.minimum(p_vals * n_tests / ranks, 1.0)
                # Enforce monotonicity
                p_fdr_sorted = p_fdr[sorted_idx]
                for i in range(n_tests - 2, -1, -1):
                    p_fdr_sorted[i] = min(p_fdr_sorted[i], p_fdr_sorted[i + 1])
                p_fdr[sorted_idx] = p_fdr_sorted
                stats_df["p_fdr"] = p_fdr

            stats_df.to_csv(tbl_dir / "roi_network_stats.csv", index=False)
            logger.info(
                "Exported roi_network_stats.csv (%d rows)", len(stats_df),
            )

        # NBS summary
        nbs_rows = []
        for key, nbs in self._nbs_results.items():
            for i, (size, pval) in enumerate(
                zip(nbs.component_sizes, nbs.component_pvalues)
            ):
                nbs_rows.append({
                    "key": key,
                    "component": i + 1,
                    "n_edges": size,
                    "p_corrected": pval,
                })
        if nbs_rows:
            nbs_df = pd.DataFrame(nbs_rows)
            nbs_df.to_csv(tbl_dir / "roi_nbs_results.csv", index=False)
            logger.info("Exported roi_nbs_results.csv")

    def figures(self) -> None:
        """Generate NBS circos overlays and global metric bar charts."""
        fig_dir = self.fig_dir

        # Global metric bar charts
        global_csv = self.output_dir / "data" / "roi_network_global_metrics.csv"
        if global_csv.exists():
            self._plot_global_metrics(global_csv, fig_dir)

    def _plot_global_metrics(self, csv_path: Path, fig_dir: Path) -> None:
        """Bar charts of global graph metrics by group, with significance asterisks."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats

        df = pd.read_csv(csv_path)
        global_metrics = ["global_efficiency", "modularity", "small_worldness"]

        # Compute pairwise Welch t-tests for significance annotations
        contrasts = self.config.raw.get("contrasts", [])
        sig_results = []

        for metric in self._connectivity_metrics:
            sub = df[df["metric"] == metric] if "metric" in df.columns else df
            if sub.empty:
                continue

            for gm_name in global_metrics:
                for contrast in contrasts:
                    cname = contrast["name"]
                    ga, gb = contrast["group_a"], contrast["group_b"]
                    for band in sub["band"].unique():
                        vals_a = sub[(sub["group"] == ga) & (sub["band"] == band)][gm_name].dropna()
                        vals_b = sub[(sub["group"] == gb) & (sub["band"] == band)][gm_name].dropna()
                        if len(vals_a) < 2 or len(vals_b) < 2:
                            continue
                        t_stat, p_val = sp_stats.ttest_ind(vals_a, vals_b, equal_var=False)
                        sig_results.append({
                            "metric": metric, "gm_name": gm_name,
                            "contrast": cname, "band": band,
                            "p_value": p_val,
                        })

        # FDR correction (Benjamini-Hochberg) across all tests
        sig_df = pd.DataFrame(sig_results)
        if not sig_df.empty:
            p_vals = sig_df["p_value"].values
            n_tests = len(p_vals)
            ranked = np.argsort(p_vals)
            q_vals = np.empty(n_tests)
            for i, rank_i in enumerate(np.argsort(ranked)):
                q_vals[ranked[rank_i]] = p_vals[ranked[rank_i]] * n_tests / (rank_i + 1)
            # Enforce monotonicity (step-up)
            q_vals_sorted = q_vals[np.argsort(p_vals)]
            for i in range(n_tests - 2, -1, -1):
                q_vals_sorted[i] = min(q_vals_sorted[i], q_vals_sorted[i + 1])
            q_vals[np.argsort(p_vals)] = q_vals_sorted
            q_vals = np.minimum(q_vals, 1.0)
            sig_df["q_value"] = q_vals
            sig_df["significant"] = sig_df["q_value"] < 0.05
            sig_df["sig_label"] = sig_df["q_value"].apply(
                lambda q: "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else ""))
            )

            # Save to tables
            tbl_dir = self.tbl_dir
            sig_df.to_csv(tbl_dir / "roi_network_global_pairwise.csv", index=False)
            logger.info("Exported roi_network_global_pairwise.csv")

        for metric in self._connectivity_metrics:
            sub = df[df["metric"] == metric] if "metric" in df.columns else df
            if sub.empty:
                continue

            metric_label = METRIC_LABELS.get(metric, metric)

            for gm_name in global_metrics:
                fig, ax = plt.subplots(figsize=(10, 5))
                groups = sub["group"].unique()
                bands = sub["band"].unique()
                x = np.arange(len(bands))
                width = 0.8 / max(len(groups), 1)

                bar_tops = {}  # band -> max bar top for annotation positioning
                for gi, group in enumerate(sorted(groups)):
                    group_data = sub[sub["group"] == group]
                    means = []
                    sems = []
                    for band in bands:
                        vals = group_data[group_data["band"] == band][gm_name]
                        m = vals.mean() if len(vals) > 0 else 0
                        s = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
                        means.append(m)
                        sems.append(s)
                        top = m + s
                        if band not in bar_tops or top > bar_tops[band]:
                            bar_tops[band] = top
                    label = self.config.get_group_label(group)
                    ax.bar(
                        x + gi * width, means, width,
                        yerr=sems, label=label, capsize=3,
                    )

                # Add significance asterisks
                if not sig_df.empty:
                    metric_sig = sig_df[
                        (sig_df["metric"] == metric) &
                        (sig_df["gm_name"] == gm_name) &
                        (sig_df["significant"])
                    ]
                    for _, row in metric_sig.iterrows():
                        band = row["band"]
                        bi = list(bands).index(band) if band in bands else -1
                        if bi < 0:
                            continue
                        y_pos = bar_tops.get(band, 0) * 1.05
                        x_pos = x[bi] + width * (len(groups) - 1) / 2
                        ax.text(
                            x_pos, y_pos, row["sig_label"],
                            ha="center", va="bottom",
                            fontsize=12, fontweight="bold", color="black",
                        )

                ax.set_xticks(x + width * (len(groups) - 1) / 2)
                ax.set_xticklabels(bands, rotation=45, ha="right")
                ax.set_ylabel(gm_name.replace("_", " ").title())
                ax.set_title(f"{gm_name.replace('_', ' ').title()} — {metric_label}")
                ax.legend()
                fig.tight_layout()

                safe = f"{metric}_{gm_name}".lower()
                fig.savefig(fig_dir / f"roi_network_{safe}.png", dpi=150)
                plt.close(fig)

        logger.info("Generated ROI network global metric figures")

    def summary(self) -> None:
        data_dir = self.output_dir / "data"

        config_path = data_dir / "study_config.yaml"
        config_data = dict(self.config.raw)
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        try:
            r_dir = find_r_script_dir()
            r_script = r_dir / "roi_network_analysis.R"
            if r_script.exists():
                cmd = [
                    "Rscript", str(r_script),
                    "--data-dir", str(data_dir),
                    "--config", str(config_path),
                    "--output-dir", str(self.output_dir),
                    "--fig-dir", str(self.fig_dir),
                    "--tbl-dir", str(self.tbl_dir),
                ]
                cmd.extend(self._r_no_figures_flags())
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        self._write_python_summary()

    def _write_python_summary(self) -> None:
        tbl_dir = self.tbl_dir

        lines = [
            "# ROI Network Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Graph-theoretic metrics on ROI connectivity (40 brain ROIs)",
            f"**Connectivity metrics**: {', '.join(self._connectivity_metrics)}",
            f"**Threshold method**: {self._threshold_method} ({self._threshold_value})",
            f"**NBS threshold**: t = {self._nbs_threshold}",
            f"**NBS permutations**: {self._nbs_permutations}",
            "",
            "## Methods",
            "",
            "Graph-theoretic metrics (degree, clustering coefficient, betweenness "
            "centrality, global efficiency, modularity, small-worldness) were computed "
            "from thresholded ROI connectivity matrices (40 brain ROIs, excluding "
            "corpus callosum white matter tracts). Group differences in nodal metrics "
            "were tested with per-ROI Welch t-tests (FDR-corrected). The Network-Based "
            "Statistic (Zalesky et al., 2010) was used to identify subnetworks with "
            "significant group differences.",
            "",
        ]

        # Global metrics summary
        global_csv = self.output_dir / "data" / "roi_network_global_metrics.csv"
        if global_csv.exists():
            global_df = pd.read_csv(global_csv)
            lines.append("## Global Metrics")
            lines.append("")
            lines.append(
                "| Band | Metric | Group | Efficiency | Modularity | Small-World | Edges |"
            )
            lines.append(
                "|------|--------|-------|------------|------------|-------------|-------|"
            )
            for _, row in (
                global_df.groupby(["band", "metric", "group"])
                .mean(numeric_only=True)
                .reset_index()
                .iterrows()
            ):
                lines.append(
                    f"| {row['band']} | {row['metric']} | {row['group']} | "
                    f"{row['global_efficiency']:.3f} | {row['modularity']:.3f} | "
                    f"{row['small_worldness']:.2f} | {row['n_edges']:.0f} |"
                )
            lines.append("")

        # NBS results
        nbs_csv = tbl_dir / "roi_nbs_results.csv"
        if nbs_csv.exists():
            nbs_df = pd.read_csv(nbs_csv)
            lines.append("## NBS Results")
            lines.append("")
            sig_nbs = nbs_df[nbs_df["p_corrected"] < 0.05]
            if len(sig_nbs) > 0:
                lines.append(f"**{len(sig_nbs)} significant subnetworks** (p<0.05)")
                for _, row in sig_nbs.iterrows():
                    lines.append(
                        f"- {row['key']}: {row['n_edges']} edges, "
                        f"p = {row['p_corrected']:.4f}"
                    )
            else:
                lines.append("No significant NBS subnetworks at p < 0.05.")
            lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "- `data/roi_network_nodal_metrics.csv` — per-ROI graph metrics",
            "- `data/roi_network_global_metrics.csv` — global graph metrics per subject",
            "- `tables/roi_network_stats.csv` — per-ROI t-tests with FDR correction",
            "- `tables/roi_nbs_results.csv` — NBS subnetwork results",
            "- `figures/roi_network_*.png` — global metric bar charts",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
