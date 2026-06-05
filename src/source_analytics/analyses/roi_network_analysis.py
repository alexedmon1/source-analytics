"""ROI network layer: graph-theoretic metrics + NBS on ROI connectivity.

Reads pre-computed ROI connectivity edges (from roi_connectivity), builds
symmetric connectivity matrices per band × connectivity metric, and runs two
independent analyses that share those matrices:

* :class:`ROIGraphAnalysis` (``roi_graph``) — nodal graph metrics (degree,
  clustering, betweenness) with per-ROI group t-tests, plus global metrics.
* :class:`ROINBSAnalysis` (``roi_nbs``) — the Network-Based Statistic.

:class:`ROINetworkAnalysis` (``roi_network``) is the back-compat combined alias
that runs both, with the original output filenames.
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
from ..stats.graph_metrics import compute_graph_metrics
from ..viz.constants import CC_ROIS, METRIC_LABELS
from ._network_base import NetworkAnalysisBase
from .base import find_r_script_dir

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


class _ROINetworkBase(NetworkAnalysisBase):
    """Shared ROI machinery: edge loading, matrix building, graph metrics, NBS.

    Leaf classes (graph / nbs / combined) wire up which lifecycle steps call
    which of these methods.
    """

    _default_nbs_threshold = 2.5
    _nbs_results_filename = "roi_nbs_results.csv"
    _fallback_config_key = "roi_network"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._nodal_rows: list[dict] = []
        self._global_rows: list[dict] = []
        self._roi_labels: list[str] = []
        self._subject_data: dict[str, dict] = {}
        self._edges_df: pd.DataFrame | None = None

        self._init_network_config()
        self._threshold_method = self._net_cfg.get("threshold_method", "proportional")
        self._threshold_value = float(self._net_cfg.get("threshold_value", 0.15))

    # ------------------------------------------------------------- setup --- #
    def setup(self) -> None:
        self._nodal_rows.clear()
        self._global_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._conn_matrices.clear()
        self._nbs_results.clear()

        parent_dir = self.config.output_dir
        csv_path = parent_dir / "roi_connectivity" / "data" / "roi_connectivity_edges.csv"
        if not csv_path.exists():
            csv_path = self.output_dir / "data" / "roi_connectivity_edges.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                "roi_connectivity_edges.csv not found. Run roi_connectivity "
                f"analysis first. Checked: {parent_dir / 'roi_connectivity' / 'data'}"
            )

        self._edges_df = pd.read_csv(csv_path)
        logger.info("Loaded roi_connectivity_edges.csv: %d rows", len(self._edges_df))

        all_rois = set(self._edges_df["roi1"]) | set(self._edges_df["roi2"])
        self._roi_labels = sorted(r for r in all_rois if r not in CC_ROIS)
        logger.info("ROI network: %d brain ROIs", len(self._roi_labels))

    # ----------------------------------------------------- process steps --- #
    def _process_matrices(self, subject: SubjectInfo) -> None:
        """Build per band × metric connectivity matrices (needed by graph + NBS)."""
        uid = f"{subject.group}_{subject.subject_id}"
        self._subject_groups[uid] = subject.group
        if self._edges_df is None:
            return

        subj_df = self._edges_df[self._edges_df["subject"] == uid]
        if subj_df.empty:
            logger.warning("No connectivity edges for %s — skipping", uid)
            return

        subject_conn: dict[str, dict[str, np.ndarray]] = {}
        for band_name in self.config.bands:
            band_df = subj_df[subj_df["band"] == band_name]
            if band_df.empty:
                continue
            band_conn: dict[str, np.ndarray] = {}
            for metric in self._connectivity_metrics:
                if metric not in band_df.columns:
                    logger.warning("Metric '%s' not in edges CSV — skipping", metric)
                    continue
                band_conn[metric] = _build_subject_matrix(band_df, self._roi_labels, metric)
            if band_conn:
                subject_conn[band_name] = band_conn
        self._conn_matrices[uid] = subject_conn

    def _process_graph(self, subject: SubjectInfo) -> None:
        """Compute graph metrics from the already-built matrices (graph only)."""
        uid = f"{subject.group}_{subject.subject_id}"
        subject_metrics: dict[str, dict] = {}
        for band_name, band_conn in self._conn_matrices.get(uid, {}).items():
            band_metrics = {}
            for metric, conn_mat in band_conn.items():
                gm = compute_graph_metrics(
                    conn_mat,
                    threshold_method=self._threshold_method,
                    threshold_value=self._threshold_value,
                )
                band_metrics[metric] = gm

                self._global_rows.append({
                    "subject": uid, "group": subject.group, "band": band_name,
                    "metric": metric,
                    "global_efficiency": gm.global_efficiency,
                    "modularity": gm.modularity,
                    "small_worldness": gm.small_worldness,
                    "n_edges": gm.n_edges,
                })
                for ri, roi_name in enumerate(self._roi_labels):
                    self._nodal_rows.append({
                        "subject": uid, "group": subject.group, "band": band_name,
                        "metric": metric, "roi": roi_name,
                        "degree": int(gm.degree[ri]),
                        "clustering": float(gm.clustering[ri]),
                        "betweenness": float(gm.betweenness[ri]),
                    })
            subject_metrics[band_name] = band_metrics
        self._subject_data[uid] = subject_metrics

    # --------------------------------------------------- graph aggregate --- #
    def _graph_aggregate(self) -> None:
        data_dir = self.output_dir / "data"
        if self._nodal_rows:
            pd.DataFrame(self._nodal_rows).to_csv(
                data_dir / f"{self.name}_nodal_metrics.csv", index=False)
        if self._global_rows:
            pd.DataFrame(self._global_rows).to_csv(
                data_dir / f"{self.name}_global_metrics.csv", index=False)
        pd.DataFrame({"roi": self._roi_labels}).to_csv(data_dir / "roi_labels.csv", index=False)

    # --------------------------------------------------- graph statistics -- #
    def _graph_statistics(self) -> None:
        from scipy.stats import false_discovery_control, ttest_ind

        from ..stats.cluster_permutation import hedges_g

        all_stats = []
        for contrast in self.config.contrasts:
            group_a_uids = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            group_b_uids = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not group_a_uids or not group_b_uids:
                continue

            for band_name in self.config.bands:
                for metric in self._connectivity_metrics:
                    for nodal_metric in ["degree", "clustering", "betweenness"]:
                        vals_a, vals_b = [], []
                        for uid in group_a_uids:
                            gm = self._subject_data.get(uid, {}).get(band_name, {}).get(metric)
                            if gm is not None:
                                vals_a.append(getattr(gm, nodal_metric).astype(float))
                        for uid in group_b_uids:
                            gm = self._subject_data.get(uid, {}).get(band_name, {}).get(metric)
                            if gm is not None:
                                vals_b.append(getattr(gm, nodal_metric).astype(float))
                        if not vals_a or not vals_b:
                            continue

                        data_a = np.array(vals_a)
                        data_b = np.array(vals_b)
                        # Per-ROI Hedges' g (positive = group_a higher), matching
                        # the vertex_graph stats schema so the gallery digest fires.
                        g_vec = hedges_g(data_a, data_b)
                        for ri in range(data_a.shape[1]):
                            t_stat, p_val = ttest_ind(
                                data_a[:, ri], data_b[:, ri], equal_var=False)
                            all_stats.append({
                                "contrast": contrast.name, "band": band_name,
                                "conn_metric": metric, "graph_metric": nodal_metric,
                                "roi": self._roi_labels[ri],
                                "mean_a": float(data_a[:, ri].mean()),
                                "mean_b": float(data_b[:, ri].mean()),
                                "hedges_g": float(g_vec[ri]),
                                "t": float(t_stat), "p": float(p_val),
                            })

        if not all_stats:
            return
        stats_df = pd.DataFrame(all_stats)
        nan_mask = np.isnan(stats_df["p"].values)
        if nan_mask.sum() > 0:
            logger.warning(
                "%d / %d nodal t-tests produced NaN p-values (zero-variance ROIs); "
                "setting to 1.0 for FDR", int(nan_mask.sum()), len(stats_df))
            stats_df.loc[nan_mask, "p"] = 1.0
        try:
            stats_df["p_fdr"] = false_discovery_control(stats_df["p"].values)
        except (AttributeError, TypeError):
            p_vals = stats_df["p"].values
            n_tests = len(p_vals)
            sorted_idx = np.argsort(p_vals)
            ranks = np.empty(n_tests, dtype=float)
            ranks[sorted_idx] = np.arange(1, n_tests + 1)
            p_fdr = np.minimum(p_vals * n_tests / ranks, 1.0)
            p_fdr_sorted = p_fdr[sorted_idx]
            for i in range(n_tests - 2, -1, -1):
                p_fdr_sorted[i] = min(p_fdr_sorted[i], p_fdr_sorted[i + 1])
            p_fdr[sorted_idx] = p_fdr_sorted
            stats_df["p_fdr"] = p_fdr

        # FDR-corrected significance flag (per-ROI multiplicity), matching the
        # gold-standard `significant` column the gallery summary digest reads.
        stats_df["significant"] = stats_df["p_fdr"] < 0.05

        stats_df.to_csv(self.tbl_dir / f"{self.name}_stats.csv", index=False)
        logger.info("Exported %s_stats.csv (%d rows)", self.name, len(stats_df))

    # ------------------------------------------------------ graph figures -- #
    def _graph_figures(self) -> None:
        global_csv = self.output_dir / "data" / f"{self.name}_global_metrics.csv"
        if global_csv.exists():
            self._plot_global_metrics(global_csv, self.fig_dir)

    def _plot_global_metrics(self, csv_path: Path, fig_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats

        df = pd.read_csv(csv_path)
        global_metrics = ["global_efficiency", "modularity", "small_worldness"]
        contrasts = self.config.raw.get("contrasts", [])
        sig_results = []

        for metric in self._connectivity_metrics:
            sub = df[df["metric"] == metric] if "metric" in df.columns else df
            if sub.empty:
                continue
            for gm_name in global_metrics:
                for contrast in contrasts:
                    ga, gb = contrast["group_a"], contrast["group_b"]
                    for band in sub["band"].unique():
                        vals_a = sub[(sub["group"] == ga) & (sub["band"] == band)][gm_name].dropna()
                        vals_b = sub[(sub["group"] == gb) & (sub["band"] == band)][gm_name].dropna()
                        if len(vals_a) < 2 or len(vals_b) < 2:
                            continue
                        _, p_val = sp_stats.ttest_ind(vals_a, vals_b, equal_var=False)
                        sig_results.append({
                            "metric": metric, "gm_name": gm_name,
                            "contrast": contrast["name"], "band": band, "p_value": p_val,
                        })

        sig_df = pd.DataFrame(sig_results)
        if not sig_df.empty:
            p_vals = sig_df["p_value"].values
            n_tests = len(p_vals)
            ranked = np.argsort(p_vals)
            q_vals = np.empty(n_tests)
            for i, rank_i in enumerate(np.argsort(ranked)):
                q_vals[ranked[rank_i]] = p_vals[ranked[rank_i]] * n_tests / (rank_i + 1)
            q_vals_sorted = q_vals[np.argsort(p_vals)]
            for i in range(n_tests - 2, -1, -1):
                q_vals_sorted[i] = min(q_vals_sorted[i], q_vals_sorted[i + 1])
            q_vals[np.argsort(p_vals)] = q_vals_sorted
            q_vals = np.minimum(q_vals, 1.0)
            sig_df["q_value"] = q_vals
            sig_df["significant"] = sig_df["q_value"] < 0.05
            sig_df["sig_label"] = sig_df["q_value"].apply(
                lambda q: "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "")))
            sig_df.to_csv(self.tbl_dir / f"{self.name}_global_pairwise.csv", index=False)

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
                bar_tops = {}
                for gi, group in enumerate(sorted(groups)):
                    group_data = sub[sub["group"] == group]
                    means, sems = [], []
                    for band in bands:
                        vals = group_data[group_data["band"] == band][gm_name]
                        m = vals.mean() if len(vals) > 0 else 0
                        s = vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0
                        means.append(m); sems.append(s)
                        if band not in bar_tops or (m + s) > bar_tops[band]:
                            bar_tops[band] = m + s
                    ax.bar(x + gi * width, means, width, yerr=sems,
                           label=self.config.get_group_label(group), capsize=3)
                if not sig_df.empty:
                    metric_sig = sig_df[(sig_df["metric"] == metric) &
                                        (sig_df["gm_name"] == gm_name) & (sig_df["significant"])]
                    for _, row in metric_sig.iterrows():
                        band = row["band"]
                        bi = list(bands).index(band) if band in bands else -1
                        if bi < 0:
                            continue
                        ax.text(x[bi] + width * (len(groups) - 1) / 2,
                                bar_tops.get(band, 0) * 1.05, row["sig_label"],
                                ha="center", va="bottom", fontsize=12, fontweight="bold")
                ax.set_xticks(x + width * (len(groups) - 1) / 2)
                ax.set_xticklabels(bands, rotation=45, ha="right")
                ax.set_ylabel(gm_name.replace("_", " ").title())
                ax.set_title(f"{gm_name.replace('_', ' ').title()} — {metric_label}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(fig_dir / f"{self.name}_{metric}_{gm_name}".lower() + ".png", dpi=150)
                plt.close(fig)

    # --------------------------------------------------------- summaries --- #
    def _write_summary(self, graph: bool = True, nbs: bool = True) -> None:
        lines = [
            f"# {self.name} Analysis Summary", "",
            f"**Study**: {self.config.name}",
            f"**Connectivity metrics**: {', '.join(self._connectivity_metrics)}",
        ]
        if graph:
            lines.append(
                f"**Threshold**: {self._threshold_method} ({self._threshold_value})")
            global_csv = self.output_dir / "data" / f"{self.name}_global_metrics.csv"
            if global_csv.exists():
                gdf = pd.read_csv(global_csv)
                lines += ["", "## Global Metrics (group means)", ""]
                for _, row in (gdf.groupby(["band", "metric", "group"]).mean(numeric_only=True)
                               .reset_index().iterrows()):
                    lines.append(
                        f"- {row['band']} / {row['metric']} / {row['group']}: "
                        f"eff={row['global_efficiency']:.3f}, mod={row['modularity']:.3f}, "
                        f"sw={row['small_worldness']:.2f}")
        if nbs:
            lines += ["", f"**NBS threshold**: t = {self._nbs_threshold}",
                      f"**NBS permutations**: {self._nbs_permutations}"]
            nbs_csv = self.tbl_dir / self._nbs_results_filename
            if nbs_csv.exists():
                ndf = pd.read_csv(nbs_csv)
                sig = ndf[ndf["p_corrected"] < 0.05]
                lines += ["", "## NBS Results", ""]
                if len(sig) > 0:
                    lines.append(f"**{len(sig)} significant subnetworks** (p<0.05)")
                    for _, row in sig.iterrows():
                        lines.append(f"- {row['key']}: {row['n_edges']} edges, p={row['p_corrected']:.4f}")
                else:
                    lines.append("No significant NBS subnetworks at p < 0.05.")
        lines.append("")
        (self.output_dir / "ANALYSIS_SUMMARY.md").write_text("\n".join(lines))


class ROIGraphAnalysis(_ROINetworkBase):
    """ROI graph-theoretic metrics (nodal + global), no NBS."""

    name = "roi_graph"

    def process_subject(self, subject: SubjectInfo) -> None:
        self._process_matrices(subject)
        self._process_graph(subject)

    def aggregate(self) -> None:
        self._graph_aggregate()

    def statistics(self) -> None:
        self._graph_statistics()

    def figures(self) -> None:
        self._graph_figures()

    def summary(self) -> None:
        self._write_summary(graph=True, nbs=False)


class ROINBSAnalysis(_ROINetworkBase):
    """ROI Network-Based Statistic, no graph metrics."""

    name = "roi_nbs"

    def process_subject(self, subject: SubjectInfo) -> None:
        self._process_matrices(subject)

    def aggregate(self) -> None:
        pass

    def statistics(self) -> None:
        self._run_nbs()

    def figures(self) -> None:
        pass

    def summary(self) -> None:
        self._write_summary(graph=False, nbs=True)


class ROINetworkAnalysis(_ROINetworkBase):
    """Combined alias: graph metrics + NBS (back-compat filenames)."""

    name = "roi_network"
    _fallback_config_key = None  # reads its own block

    def process_subject(self, subject: SubjectInfo) -> None:
        self._process_matrices(subject)
        self._process_graph(subject)

    def aggregate(self) -> None:
        self._graph_aggregate()

    def statistics(self) -> None:
        self._graph_statistics()
        self._run_nbs()

    def figures(self) -> None:
        self._graph_figures()

    def summary(self) -> None:
        # Keep the richer R report when available; else the python summary.
        data_dir = self.output_dir / "data"
        config_path = data_dir / "study_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(dict(self.config.raw), f, default_flow_style=False)
        try:
            r_script = find_r_script_dir() / "roi_network_analysis.R"
            if r_script.exists():
                cmd = ["Rscript", str(r_script), "--data-dir", str(data_dir),
                       "--config", str(config_path), "--output-dir", str(self.output_dir),
                       "--fig-dir", str(self.fig_dir), "--tbl-dir", str(self.tbl_dir)]
                cmd.extend(self._r_no_figures_flags())
                cmd.extend(self._r_roi_categories_flags())
                if subprocess.run(cmd, capture_output=True, text=True, timeout=3600).returncode == 0:
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        self._write_summary(graph=True, nbs=True)
