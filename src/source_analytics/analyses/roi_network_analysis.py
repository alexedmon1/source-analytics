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
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..results_spec import ResultsSpec, SpecTable, permutation_columns
from ..stats.graph_metrics import compute_graph_metrics
from ..viz.constants import CC_ROIS, METRIC_LABELS
from ._network_base import NetworkAnalysisBase

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
        for band_name in self._selected_bands():
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
        for contrast in self._pairwise_contrasts():
            group_a_uids = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            group_b_uids = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not group_a_uids or not group_b_uids:
                continue

            for band_name in self._selected_bands():
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
            # --steps statistics computes no graphs: the legacy table needs them in memory, the
            # declared hypotheses can still come from the saved nodal metrics.
            self._write_graph_hypotheses()
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

        self._write_graph_hypotheses()

    def _write_graph_hypotheses(self) -> None:
        """Additive declarative-hypothesis CSV (the nodal tabular contract).

        Per-ROI between-subjects contrast for every declared hypothesis, faceted by
        connectivity × graph metric, with declarative FDR. Written alongside the
        legacy ``_stats.csv``; consumes the same in-memory per-subject nodal metrics,
        or, when this run computed none (``--steps statistics``), the per-subject
        nodal metrics an earlier run saved in ``data/<name>_nodal_metrics.csv``.
        """
        spec = self.config.design_spec
        if spec is None or not spec.hypotheses:
            return
        from ..hypothesis import write_module_hypotheses_tabular

        long_rows: list[dict] = []
        if not self._subject_data:
            long_rows = self._saved_nodal_rows(spec.factor)
        for uid, group in self._subject_groups.items():
            sd = self._subject_data.get(uid, {})
            for band_name in self._selected_bands():
                for metric in self._connectivity_metrics:
                    gm = sd.get(band_name, {}).get(metric)
                    if gm is None:
                        continue
                    for graph_metric in ("degree", "clustering", "betweenness"):
                        vec = getattr(gm, graph_metric).astype(float)
                        for ri, roi in enumerate(self._roi_labels):
                            long_rows.append({
                                "subject": uid, spec.factor: group,
                                "band": band_name, "conn_metric": metric,
                                "graph_metric": graph_metric, "roi": roi,
                                "value": float(vec[ri]),
                            })
        if not long_rows:
            return
        wanted = self._selection.get("hypothesis")
        write_module_hypotheses_tabular(
            pd.DataFrame(long_rows), self.config, self.tbl_dir, prefix=self.name,
            value_col="value", spatial_col="roi",
            facet_cols=("conn_metric", "graph_metric"), band_col="band",
            hypothesis=",".join(sorted(wanted)) if wanted else None,
        )

    def _saved_nodal_rows(self, factor: str) -> list[dict]:
        """The long nodal rows from ``data/<name>_nodal_metrics.csv`` (written by the
        aggregate step), limited to this run's bands and connectivity metrics."""
        path = self.output_dir / "data" / f"{self.name}_nodal_metrics.csv"
        if not path.exists():
            logger.warning("No per-subject nodal metrics in memory or at %s: no graph hypotheses", path)
            return []
        df = pd.read_csv(path)
        bands, metrics = set(self._selected_bands()), set(self._connectivity_metrics)
        df = df[df["band"].isin(bands) & df["metric"].isin(metrics)]
        logger.info("Graph hypotheses from the saved nodal metrics: %s (%d rows)", path.name, len(df))
        return [{"subject": r.subject, factor: r.group, "band": r.band, "conn_metric": r.metric,
                 "graph_metric": gname, "roi": r.roi, "value": float(getattr(r, gname))}
                for r in df.itertuples(index=False)
                for gname in ("degree", "clustering", "betweenness")]

    # ------------------------------------------------------ graph figures -- #
    def _graph_figures(self) -> None:
        global_csv = self.output_dir / "data" / f"{self.name}_global_metrics.csv"
        if global_csv.exists():
            self._plot_global_metrics(global_csv, self.fig_dir)
        hyp_csv = self.tbl_dir / f"{self.name}_hypotheses.csv"
        if hyp_csv.exists():
            self._plot_graph_hypotheses(pd.read_csv(hyp_csv), self.fig_dir)

    def _plot_graph_hypotheses(self, hyps: pd.DataFrame, fig_dir: Path) -> None:
        """Per graph metric and declared contrast: ROI x band heatmaps of the test's t, one panel
        per connectivity measure, * where the declared test is significant (results plan 5b.4,
        moved here from source-lightbox, which showed one contrast and measure)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..figure_record import record_figure
        from ..viz import hypothesis_heatmaps as hh

        needed = {"hypothesis", "kind", "graph_metric", "conn_metric", "spatial", "band", "stat"}
        if not needed <= set(hyps.columns):
            return
        tested = hyps[(hyps["kind"] == "contrast") & hyps["spatial"].notna()
                      & (hyps["spatial"].astype(str) != "")]
        labels = {h.name: (h.label or h.name) for h in self._declared()}
        order = [h.name for h in self._declared() if h.name in set(tested["hypothesis"])]
        order += [h for h in dict.fromkeys(tested["hypothesis"]) if h not in order]
        conns = [m for m in self._connectivity_metrics if m in set(tested["conn_metric"])]
        conns += [m for m in dict.fromkeys(tested["conn_metric"]) if m not in conns]
        bands = self._selected_bands()
        n_figs = 0
        for gm in dict.fromkeys(tested["graph_metric"]):
            for hyp in order:
                sub = tested[(tested["graph_metric"] == gm) & (tested["hypothesis"] == hyp)]
                if sub.empty:
                    continue
                top = float(np.nanmax(np.abs(pd.to_numeric(sub["stat"], errors="coerce")))) or 1.0
                fig, axes = plt.subplots(1, len(conns), figsize=(2.3 * len(conns) + 2.5, 8.5),
                                         sharey=True, squeeze=False)
                im = None
                for ax, conn in zip(axes[0], conns):
                    vals, sig, rows, cols = hh.grid(sub[sub["conn_metric"] == conn], "spatial",
                                                    "band", "stat", col_order=bands)
                    im = hh.draw(ax, vals, sig, rows, cols, vmax=top, annotate=False)
                    ax.set_title(METRIC_LABELS.get(conn, conn), fontsize=10)
                fig.suptitle(f"{gm.replace('_', ' ').title()}: {labels.get(hyp, hyp)} "
                             "(t per region; * significant, BH within the declared family)")
                if im is not None:
                    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, label="t (group A - group B)")
                out = hh.save(fig, fig_dir / f"{self.name}_hypotheses__{gm}__{hyp}.png")
                record_figure(fig_dir, out, measure=gm, contrast=hyp, kind="summary",
                              shows=f"{self.name}_hypotheses.csv",
                              caption="A blank cell had no test: the measure did not vary there.")
                n_figs += 1
        logger.info("%s: %d hypothesis heatmaps", self.name, n_figs)

    def _plot_global_metrics(self, csv_path: Path, fig_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats as sp_stats

        from ..figure_record import record_figure

        df = pd.read_csv(csv_path)
        global_metrics = ["global_efficiency", "modularity", "small_worldness"]
        contrasts = self._pairwise_contrasts()
        sig_results = []

        for metric in self._connectivity_metrics:
            sub = df[df["metric"] == metric] if "metric" in df.columns else df
            if sub.empty:
                continue
            for gm_name in global_metrics:
                for contrast in contrasts:
                    ga, gb = contrast.group_a, contrast.group_b
                    for band in sub["band"].unique():
                        vals_a = sub[(sub["group"] == ga) & (sub["band"] == band)][gm_name].dropna()
                        vals_b = sub[(sub["group"] == gb) & (sub["band"] == band)][gm_name].dropna()
                        if len(vals_a) < 2 or len(vals_b) < 2:
                            continue
                        _, p_val = sp_stats.ttest_ind(vals_a, vals_b, equal_var=False)
                        sig_results.append({
                            "metric": metric, "gm_name": gm_name,
                            "contrast": contrast.name, "band": band, "p_value": p_val,
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
                out = fig_dir / (f"{self.name}_{metric}_{gm_name}".lower() + ".png")
                fig.savefig(out, dpi=150)
                plt.close(fig)
                # the stars are Welch t-tests BH-corrected over every bar (_global_pairwise.csv), not the
                # declared hypotheses, which test the nodal metrics
                record_figure(fig_dir, out, measure=gm_name, facets={"connectivity": metric},
                              kind="distribution")

    # -------------------------------------------------------- NBS figures -- #
    def _nbs_figures(self) -> None:
        """The declared hypotheses' NBS results, drawn from the module's own tables: per
        connectivity measure, a heatmap of the largest component per hypothesis and band; and a
        circos of each significant component of a two-group contrast (results plan 5b.4, moved
        here from source-lightbox)."""
        hyp_csv = self.tbl_dir / f"{self.name}_hypotheses.csv"
        if not hyp_csv.exists():
            logger.info("%s: no hypotheses table, so no NBS figures", self.name)
            return
        hyps = pd.read_csv(hyp_csv)
        self._plot_nbs_heatmaps(hyps, self.fig_dir)
        edges_csv = self.tbl_dir / f"{self.name}_subnetwork_edges.csv"
        if edges_csv.exists():
            self._plot_nbs_circos(hyps, pd.read_csv(edges_csv), self.fig_dir)

    def _declared(self) -> list:
        spec = getattr(self.config, "design_spec", None)
        return list(spec.hypotheses) if spec is not None else []

    def _plot_nbs_heatmaps(self, hyps: pd.DataFrame, fig_dir: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..figure_record import record_figure

        tested = hyps[hyps["kind"].isin(["contrast", "omnibus"])].copy()
        if tested.empty:
            return
        tested["component_id"] = pd.to_numeric(tested["component_id"], errors="coerce").fillna(0)
        tested["n_edges"] = pd.to_numeric(tested["n_edges"], errors="coerce").fillna(0)
        tested["sig"] = tested["significant"].astype(str).str.lower().isin(["true", "1"])
        order = [h.name for h in self._declared() if h.name in set(tested["hypothesis"])]
        order += [h for h in dict.fromkeys(tested["hypothesis"]) if h not in order]
        labels = {h.name: (h.label or h.name) for h in self._declared()}
        bands = [b for b in self._selected_bands() if b in set(tested["band"])]
        bands += [b for b in dict.fromkeys(tested["band"]) if b not in bands]
        for metric in dict.fromkeys(tested["dv"]):
            sub = tested[tested["dv"] == metric]
            size = np.zeros((len(order), len(bands)))
            star = np.zeros((len(order), len(bands)), dtype=bool)
            for (h, b), cell in sub.groupby(["hypothesis", "band"]):
                i, j = order.index(h), bands.index(b)
                formed = cell[cell["component_id"] > 0]
                size[i, j] = formed["n_edges"].max() if not formed.empty else 0
                star[i, j] = bool(formed["sig"].any())
            fig, ax = plt.subplots(figsize=(1.1 * len(bands) + 4.5, 0.55 * len(order) + 2))
            im = ax.imshow(size, cmap="Blues", aspect="auto", vmin=0)
            top = size.max() or 1
            for i in range(len(order)):
                for j in range(len(bands)):
                    text = f"{int(size[i, j])}" + ("*" if star[i, j] else "")
                    ax.text(j, i, text, ha="center", va="center", fontsize=9,
                            fontweight="bold" if star[i, j] else "normal",
                            color="white" if size[i, j] > 0.6 * top else "black")
            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels(bands, rotation=45, ha="right")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([labels.get(h, h) for h in order])
            metric_label = METRIC_LABELS.get(metric, metric)
            ax.set_title(f"NBS: largest component per hypothesis and band, {metric_label}")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("edges in the largest component; * p < 0.05 (FWE)")
            fig.tight_layout()
            out = fig_dir / f"{self.name}_components__{metric}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            record_figure(fig_dir, out, measure=metric, kind="summary",
                          shows=f"{self.name}_hypotheses.csv",
                          caption="Equivalence hypotheses are summarized per edge, not by components, "
                                  "and are not shown.")

    def _plot_nbs_circos(self, hyps: pd.DataFrame, edges: pd.DataFrame, fig_dir: Path) -> None:
        from ..figure_record import record_figure
        from ..viz.connectivity_plots import build_roi_matrix, plot_significance_circos

        roi_cats = self.config.roi_categories
        if not roi_cats or self._edges_df is None:
            logger.warning("%s: no roi_categories or connectivity edges, so no NBS circos", self.name)
            return
        declared = {h.name: h for h in self._declared()}
        sig = hyps[hyps["significant"].astype(str).str.lower().isin(["true", "1"])
                   & (hyps["kind"] == "contrast")]
        n_figs = 0
        for _, row in sig.iterrows():
            hyp = declared.get(str(row["hypothesis"]))
            ends = hyp.pairwise_endpoints() if hyp is not None else None
            if ends is None:
                continue  # a weighted contrast of more than two groups has no two-group picture
            ga, gb = ends
            band, metric, cid = str(row["band"]), str(row["dv"]), int(row["component_id"])
            members = edges[(edges["hypothesis"] == row["hypothesis"]) & (edges["band"] == band)
                            & (edges["dv"] == metric)
                            & (pd.to_numeric(edges["component_id"], errors="coerce") == cid)]
            band_df = self._edges_df[self._edges_df["band"] == band]
            if members.empty or band_df.empty or metric not in band_df.columns:
                continue
            mat_a, rl, rn, rs = build_roi_matrix(band_df, roi_cats, metric, group=ga)
            mat_b, *_ = build_roi_matrix(band_df, roi_cats, metric, group=gb)
            index = {r: i for i, r in enumerate(rl)}
            mask = np.zeros((len(rl), len(rl)), dtype=bool)
            for a, b in zip(members["roi_i"], members["roi_j"]):
                if a in index and b in index:
                    mask[index[a], index[b]] = mask[index[b], index[a]] = True
            if not mask.any():
                continue
            same_test = sig[(sig["hypothesis"] == row["hypothesis"]) & (sig["band"] == band)
                            & (sig["dv"] == metric)]
            several = len(same_test) > 1
            out = fig_dir / (f"circos__{metric}__{band.replace(' ', '_')}__{row['hypothesis']}"
                             + (f"__c{cid}" if several else "") + ".png")
            label = hyp.label or hyp.name
            plot_significance_circos(
                mat_a, mat_b, rl, rn, rs, mask, out,
                group_labels=(self.config.get_group_label(ga), self.config.get_group_label(gb)),
                title=f"{label}: {band} ({METRIC_LABELS.get(metric, metric)}), NBS component "
                      f"of {int(row['n_edges'])} edges, p = {float(row['component_p']):.3g}",
                sig_label="edges of an NBS component significant at p < 0.05 (FWE)",
            )
            record_figure(fig_dir, out, measure=metric, contrast=hyp.name,
                          facets={"band": band}, kind="matrix",
                          shows=f"{self.name}_subnetwork_edges.csv")
            n_figs += 1
        logger.info("%s: %d NBS circos", self.name, n_figs)

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
    RESULTS_SPEC = ResultsSpec(
        domain="connectivity",
        measures={"degree": "degree", "clustering": "clustering", "betweenness": "betweenness"},
        units={"degree": "edges (the ROI's degree in the binarized graph)",
               "clustering": "the ROI's clustering coefficient (0 to 1)",
               "betweenness": "the ROI's normalized betweenness centrality (0 to 1)"},
        input_file="roi_graph_nodal_metrics.csv",
        measure_col="graph_metric",
        extra_facets=(("conn_metric", "connectivity",
                       ("the connectivity measure the graph was built from (coherence, imaginary "
                        "coherence, PLI, dwPLI, AEC)")),),
        method="per subject, band and connectivity measure, the ROI connectivity matrix is binarized by a "
               "proportional threshold (network.threshold_value of the strongest edges, 0.15 unless set) "
               "and each ROI's degree, clustering and betweenness computed (networkx); per ROI, band, "
               "connectivity measure and graph metric, a one-way linear model on group with contrasts of "
               "the declared hypotheses; equivalence by two one-sided tests",
        tables=(
            SpecTable("_hypotheses.csv", "elements",
                      "one declared hypothesis on one graph metric of one ROI, in one band and one "
                      "connectivity measure's graph",
                      "the declared hypotheses per ROI, band, connectivity measure and graph metric",
                      element=True, headline=True),
        ),
        caveats=(("Graph metrics depend on the threshold: each graph keeps the same proportion of the "
                  "strongest edges, so a group difference in overall connectivity strength does not "
                  "change the edge count, but it can change which edges survive."),
                 ("Five connectivity measures are each turned into graphs and tested separately; they are "
                  "correlated, and each is corrected only within its own family."),
                 ("The global metrics (global efficiency, modularity, small-worldness) are compared only "
                  "in the legacy roi_graph_global_pairwise.csv (Welch t-tests), not part of this analysis.")),
    )

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
    RESULTS_SPEC = ResultsSpec(
        domain="connectivity",
        measures={"coherence": "coherence", "imag_coherence": "imcoh", "pli": "pli", "wpli": "wpli",
                  "dwpli": "dwpli", "dpli": "dpli", "aec": "aec"},
        units={"coherence": "magnitude-squared coherence (0 to 1)", "imag_coherence": "|imaginary coherence| (0 to 1)",
               "pli": "phase lag index (0 to 1)", "wpli": "weighted PLI (0 to 1)",
               "dwpli": "debiased weighted PLI (squared, can be slightly negative)",
               "dpli": "directed PLI (0.5 = no lead)", "aec": "amplitude envelope correlation (Pearson r)"},
        input_file="roi_connectivity_edges.csv (roi_connectivity's)",
        method="the network-based statistic (Zalesky et al. 2010) per band and connectivity measure: a "
               "statistic per ROI-pair edge for each declared hypothesis (t for two-group contrasts, F for the "
               "omnibus and weighted contrasts), thresholded (network.nbs_threshold, 2.5 unless set); the "
               "connected components of the supra-threshold edges are tested against the largest component "
               "in each label permutation (network.nbs_permutations, 5000 unless set). Equivalence rows "
               "summarize per-edge two one-sided tests",
        tables=(
            SpecTable("_hypotheses.csv", "elements",
                      "one component (sub-network) of one declared hypothesis in one band and connectivity "
                      "measure; component 0 is the row of a test that formed none, or an equivalence summary",
                      "the declared hypotheses: the sub-networks the NBS finds", element=False, headline=True),
            SpecTable("_subnetwork_edges.csv", "other",
                      "one supra-threshold edge of one component",
                      "the edges of every component, for drawing the sub-networks", element=False),
        ),
        extra_columns=(
            *permutation_columns("component", "edges", "component_p",
                                 "the share of label permutations whose largest component has at least as "
                                 "many edges, within the test (FWE over the network)"),
            ("component_id", {"Description": "the component's number within its test (0: none formed, or an "
                                             "equivalence summary)", "Standard": "element",
                              "ElementName": "component"}),
            ("n_edges", {"Description": "the component's supra-threshold edges (equivalence rows: the edges "
                                        "tested)"}),
            ("n_edges_increase", {"Description": "t rows: the component's edges where group_a is higher"}),
            ("n_edges_decrease", {"Description": "t rows: the component's edges where group_a is lower"}),
            ("mean_stat", {"Description": "t rows: the mean signed t over the component's edges"}),
            ("direction", {"Description": "t rows: which way the component's edges go",
                           "Levels": {"increase": "every edge higher in the contrast's positive side",
                                      "decrease": "every edge lower", "mixed": "both"}}),
            ("node_i", {"Description": "the edge's first ROI, by its index in the connectivity matrix"}),
            ("node_j", {"Description": "the edge's second ROI, by its index"}),
            ("roi_i", {"Description": "the edge's first ROI"}),
            ("roi_j", {"Description": "the edge's second ROI"}),
            ("stat", {"Description": "the edge's statistic (signed t, or F)"}),
        ),
        correction={"p_kind": "fwe", "method": "nbs",
                    "family": "the components of one test: one hypothesis, band and connectivity measure",
                    "statement": "NBS: each component's p is the share of label permutations whose largest "
                                 "component is at least as large (in edges), so p < 0.05 controls the "
                                 "family-wise error over the network within each test; tests are not "
                                 "corrected across hypotheses, bands or connectivity measures",
                    "alpha": 0.05},
        effect={"measure": None,
                "reason": "a component is a set of edges chosen by thresholding the statistic, so it has no "
                          "single estimate with an interval; its size (edges), mass and peak statistic "
                          "describe it, and its edges' statistics are in the subnetwork edges table"},
        caveats=(("NBS controls the error of declaring a component, not of any one edge in it: an edge "
                  "inside a significant component is not itself significant."),
                 ("The component threshold is arbitrary: a lower one finds larger, weaker components."),
                 ("Five connectivity measures are each tested; they are correlated, and nothing is corrected "
                  "across them.")),
    )

    def process_subject(self, subject: SubjectInfo) -> None:
        self._process_matrices(subject)

    def aggregate(self) -> None:
        pass

    def statistics(self) -> None:
        self._run_nbs()
        self._run_nbs_hypotheses()

    def figures(self) -> None:
        self._nbs_figures()

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
        self._run_nbs_hypotheses()

    def figures(self) -> None:
        self._graph_figures()
        self._nbs_figures()

    def summary(self) -> None:
        # The combined alias has no R report of its own (graph + NBS statistics
        # run in Python via the hypothesis layer); write the Python summary.
        data_dir = self.output_dir / "data"
        config_path = data_dir / "study_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._r_config_data(), f, default_flow_style=False)
        self._write_summary(graph=True, nbs=True)
