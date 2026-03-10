"""Vertex Network Analysis: graph-theoretic metrics on shell-based vertex connectivity.

Loads shell-based source timecourses (154 vertices, filtered to ~66 dorsal),
computes all-to-all vertex connectivity, then runs graph metrics and NBS
for subnetwork identification at vertex resolution.
"""

from __future__ import annotations

import logging
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.vertex_connectivity import (
    compute_vertex_connectivity_matrix,
    compute_vertex_connectivity_matrix_epochs,
)
from ..spectral.epoch_sampler import sample_epochs, get_epoch_config
from ..stats.graph_metrics import compute_graph_metrics, nbs_permutation_test
from ..stats.cluster_permutation import cluster_permutation_test, hedges_g
from ..viz.glass_brain import plot_glass_brain, plot_band_comparison, plot_glass_brain_edges
from .base import BaseAnalysis, find_r_script_dir

logger = logging.getLogger(__name__)


def _generate_vertex_labels(
    coords: np.ndarray,
    atlas_labels: list[str] | None = None,
) -> list[str]:
    """Generate descriptive labels for vertices.

    For vertices with atlas ROI labels, use those. For unassigned/Exterior
    vertices, generate spatial labels from coordinates.

    Parameters
    ----------
    coords : ndarray, shape (n_vertices, 3)
        Vertex coordinates in mm.
    atlas_labels : list[str], optional
        Atlas ROI abbreviations per vertex.

    Returns
    -------
    labels : list[str]
        Descriptive label per vertex.
    """
    n = len(coords)
    labels = []

    # Compute centroid for relative positioning
    centroid = coords.mean(axis=0)

    for i in range(n):
        # Use atlas label if available and meaningful
        if atlas_labels and atlas_labels[i] not in ("Exterior", "Unknown_0"):
            labels.append(atlas_labels[i])
            continue

        # Generate spatial label
        x, y, z = coords[i]
        parts = []

        # Anterior/Posterior (y-axis)
        if y > centroid[1] + 1.0:
            parts.append("Anterior")
        elif y < centroid[1] - 1.0:
            parts.append("Posterior")
        else:
            parts.append("Central_AP")

        # Left/Right (x-axis)
        if x < centroid[0] - 0.5:
            parts.append("Left")
        elif x > centroid[0] + 0.5:
            parts.append("Right")
        else:
            parts.append("Midline")

        # Dorsal/Ventral (z-axis)
        if z > centroid[2] + 0.5:
            parts.append("Dorsal")
        elif z < centroid[2] - 0.5:
            parts.append("Ventral")
        else:
            parts.append("Central_DV")

        labels.append("_".join(parts))

    return labels


class VertexNetworkAnalysis(BaseAnalysis):
    """Graph-theoretic network analysis on shell vertex connectivity."""

    name = "vertex_network"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._nodal_rows: list[dict] = []
        self._global_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._vertex_indices: np.ndarray | None = None
        self._vertex_labels: list[str] = []
        self._sfreq: float | None = None
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}
        self._conn_matrices: dict[str, dict[str, np.ndarray]] = {}

        # Config
        net_cfg = config.raw.get("vertex_network", {})
        self._threshold_method = net_cfg.get("threshold_method", "proportional")
        self._threshold_value = float(net_cfg.get("threshold_value", 0.1))
        self._nbs_threshold = float(net_cfg.get("nbs_threshold", 3.0))
        self._nbs_permutations = int(net_cfg.get("nbs_permutations", 5000))
        self._metric = net_cfg.get("metric", "imag_coherence")

        wb_cfg = config.wholebrain
        self._n_permutations = int(wb_cfg.get("n_permutations", 1000))
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))

        self._epoch_config = get_epoch_config(wb_cfg)
        self._cluster_results: dict = {}
        self._nbs_results: dict = {}

    def setup(self) -> None:
        self._nodal_rows.clear()
        self._global_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._conn_matrices.clear()
        self._source_coords = None
        self._vertex_indices = None
        self._vertex_labels.clear()
        self._cluster_results.clear()
        self._nbs_results.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        stc_data = loader.load_source_timecourses()
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()

        if self._sfreq is None:
            self._sfreq = sfreq

        # Apply vertex filter (compute mask once from first subject)
        if self._vertex_indices is None:
            mask = self.config.get_vertex_mask(coords)
            self._vertex_indices = np.where(mask)[0]
            self._source_coords = coords[mask]
            if self.config.has_vertex_filter:
                logger.info(
                    "Vertex filter: %d/%d vertices retained",
                    len(self._vertex_indices), len(coords),
                )

            # Generate vertex labels
            try:
                from ..atlas import find_atlas_dir, load_vertex_roi_labels
                atlas_dir = find_atlas_dir()
                atlas_labels = load_vertex_roi_labels(self._source_coords, atlas_dir)
            except Exception:
                atlas_labels = None

            self._vertex_labels = _generate_vertex_labels(
                self._source_coords, atlas_labels,
            )

        stc_data = stc_data[self._vertex_indices]

        self._subject_groups[uid] = subject.group
        subject_metrics = {}
        subject_conn = {}

        for band_name, (fmin, fmax) in self.config.bands.items():
            logger.info(
                "  Computing %s connectivity + graph metrics...", band_name,
            )

            # Check for pre-computed connectivity matrices
            vc_pkl = (
                self.config.output_dir / "vertex_connectivity" / "data"
                / "vertex_connectivity_matrices.pkl"
            )
            conn_mat = None
            if vc_pkl.exists():
                try:
                    with open(vc_pkl, "rb") as f:
                        all_conn = pickle.load(f)
                    # Try metric-keyed format first, then legacy band-only format
                    subj_conn = all_conn.get(uid, {})
                    if band_name in subj_conn:
                        val = subj_conn[band_name]
                        if isinstance(val, dict):
                            conn_mat = val.get(self._metric)
                        else:
                            # Legacy format: band -> matrix (imag_coherence)
                            if self._metric == "imag_coherence":
                                conn_mat = val
                        if conn_mat is not None:
                            logger.info(
                                "    Loaded pre-computed connectivity for %s",
                                band_name,
                            )
                except Exception:
                    pass

            if conn_mat is None:
                if self._epoch_config is not None:
                    epochs = sample_epochs(
                        stc_data, sfreq,
                        epoch_duration_sec=self._epoch_config.get(
                            "epoch_duration_sec", 2.0,
                        ),
                        n_epochs=self._epoch_config.get("n_epochs", 80),
                        seed=self._epoch_config.get("seed", 42),
                    )
                    conn_mat = compute_vertex_connectivity_matrix_epochs(
                        epochs, sfreq, (fmin, fmax), metric=self._metric,
                    )
                else:
                    conn_mat = compute_vertex_connectivity_matrix(
                        stc_data, sfreq, (fmin, fmax), metric=self._metric,
                    )

            subject_conn[band_name] = conn_mat

            # Compute graph metrics
            gm = compute_graph_metrics(
                conn_mat,
                threshold_method=self._threshold_method,
                threshold_value=self._threshold_value,
            )

            subject_metrics[band_name] = gm

            # Global metrics
            self._global_rows.append({
                "subject": uid,
                "group": subject.group,
                "band": band_name,
                "global_efficiency": gm.global_efficiency,
                "modularity": gm.modularity,
                "small_worldness": gm.small_worldness,
                "n_edges": gm.n_edges,
            })

            # Nodal metrics
            n_vertices = len(gm.degree)
            for vi in range(n_vertices):
                label = (
                    self._vertex_labels[vi]
                    if vi < len(self._vertex_labels)
                    else f"v{self._vertex_indices[vi]}"
                )
                self._nodal_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "vertex_idx": int(self._vertex_indices[vi]),
                    "vertex_label": label,
                    "band": band_name,
                    "degree": int(gm.degree[vi]),
                    "clustering": float(gm.clustering[vi]),
                    "betweenness": float(gm.betweenness[vi]),
                })

        self._subject_data[uid] = subject_metrics
        self._conn_matrices[uid] = subject_conn

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        nodal_df = pd.DataFrame(self._nodal_rows)
        if not nodal_df.empty:
            nodal_df.to_csv(
                data_dir / "vertex_network_nodal_metrics.csv", index=False,
            )
            logger.info(
                "Exported vertex_network_nodal_metrics.csv (%d rows)",
                len(nodal_df),
            )

        global_df = pd.DataFrame(self._global_rows)
        if not global_df.empty:
            global_df.to_csv(
                data_dir / "vertex_network_global_metrics.csv", index=False,
            )
            logger.info(
                "Exported vertex_network_global_metrics.csv (%d rows)",
                len(global_df),
            )

        if self._source_coords is not None:
            coords_df = pd.DataFrame(
                self._source_coords, columns=["x", "y", "z"],
            )
            if self._vertex_labels:
                coords_df["label"] = self._vertex_labels
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

    def statistics(self) -> None:
        if self._source_coords is None:
            logger.error("No source coordinates")
            return

        coords = self._source_coords
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
                # Cluster permutation on nodal metrics
                for metric_name in ["degree", "clustering", "betweenness"]:
                    data_a = np.array([
                        getattr(
                            self._subject_data[uid][band_name], metric_name,
                        )
                        for uid in group_a_uids
                        if band_name in self._subject_data.get(uid, {})
                    ]).astype(float)
                    data_b = np.array([
                        getattr(
                            self._subject_data[uid][band_name], metric_name,
                        )
                        for uid in group_b_uids
                        if band_name in self._subject_data.get(uid, {})
                    ]).astype(float)

                    if data_a.size == 0 or data_b.size == 0:
                        continue

                    result = cluster_permutation_test(
                        data_a, data_b, coords,
                        n_perms=self._n_permutations,
                        threshold=self._cluster_threshold,
                        distance_mm=self._adjacency_distance,
                        seed=42,
                    )

                    g_map = hedges_g(data_a, data_b)

                    key = f"{contrast.name}_{band_name}_{metric_name}"
                    self._cluster_results[key] = {
                        "result": result,
                        "mean_a": data_a.mean(axis=0),
                        "mean_b": data_b.mean(axis=0),
                        "group_labels": (label_a, label_b),
                        "band": band_name,
                        "metric": metric_name,
                    }

                    for vi in range(len(result.t_map)):
                        all_stats.append({
                            "contrast": contrast.name,
                            "band": band_name,
                            "metric": metric_name,
                            "vertex_idx": vi,
                            "t": float(result.t_map[vi]),
                            "p": float(result.p_map[vi]),
                            "hedges_g": float(g_map[vi]),
                            "cluster_id": int(result.cluster_labels[vi]),
                        })

                # NBS on connectivity matrices
                matrices_a = [
                    self._conn_matrices[uid][band_name]
                    for uid in group_a_uids
                    if band_name in self._conn_matrices.get(uid, {})
                ]
                matrices_b = [
                    self._conn_matrices[uid][band_name]
                    for uid in group_b_uids
                    if band_name in self._conn_matrices.get(uid, {})
                ]

                if matrices_a and matrices_b:
                    nbs_result = nbs_permutation_test(
                        matrices_a, matrices_b,
                        nbs_threshold=self._nbs_threshold,
                        n_permutations=self._nbs_permutations,
                        seed=42,
                    )
                    self._nbs_results[f"{contrast.name}_{band_name}"] = nbs_result

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df.to_csv(
                tbl_dir / "vertex_network_stats.csv", index=False,
            )
            logger.info(
                "Exported vertex_network_stats.csv (%d rows)", len(stats_df),
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
            nbs_df.to_csv(tbl_dir / "vertex_nbs_results.csv", index=False)
            logger.info("Exported vertex_nbs_results.csv")

    def figures(self) -> None:
        if self._source_coords is None:
            return

        coords = self._source_coords
        fig_dir = self.fig_dir

        # Nodal metric glass brains
        for key, info in self._cluster_results.items():
            result = info["result"]
            band = info["band"]
            metric = info["metric"]
            safe_name = f"{band}_{metric}".lower().replace(" ", "_")

            plot_band_comparison(
                coords=coords,
                mean_a=info["mean_a"],
                mean_b=info["mean_b"],
                t_map=result.t_map,
                cluster_labels=result.cluster_labels,
                cluster_pvalues=result.cluster_pvalues,
                band_name=f"{metric} — {band}",
                group_labels=info["group_labels"],
                output_path=fig_dir / f"vertex_network_{safe_name}.png",
            )

        # NBS edge glass brains
        for key, nbs in self._nbs_results.items():
            if nbs.n_significant_components == 0:
                continue

            # Extract significant edges from t_matrix
            sig_edges_mask = np.abs(nbs.t_matrix) > self._nbs_threshold
            edge_pairs = np.argwhere(np.triu(sig_edges_mask, k=1))

            if len(edge_pairs) > 0:
                edge_t_values = np.array([
                    nbs.t_matrix[i, j] for i, j in edge_pairs
                ])

                plot_glass_brain_edges(
                    coords=coords,
                    edges=edge_pairs,
                    output_path=fig_dir / f"vertex_nbs_edges_{key}.png",
                    edge_values=np.abs(edge_t_values),
                    title=f"NBS Edges — {key}",
                )

    def summary(self) -> None:
        data_dir = self.output_dir / "data"

        config_path = data_dir / "study_config.yaml"
        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        try:
            r_dir = find_r_script_dir()
            r_script = r_dir / "vertex_network_analysis.R"
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
            "# Vertex Network Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Graph-theoretic metrics on vertex connectivity (shell source space)",
            f"**Connectivity metric**: {self._metric}",
            f"**Threshold method**: {self._threshold_method} ({self._threshold_value})",
            f"**NBS threshold**: t = {self._nbs_threshold}",
            f"**NBS permutations**: {self._nbs_permutations}",
            "",
            "## Methods",
            "",
            "Graph-theoretic metrics (degree, clustering coefficient, betweenness "
            "centrality, global efficiency, modularity, small-worldness) were computed "
            "from thresholded vertex connectivity matrices using the shell source space "
            "(dorsal vertices only). Group differences in nodal metrics were tested with "
            "cluster-based permutation testing. The Network-Based Statistic "
            "(Zalesky et al., 2010) was used to identify subnetworks with significant "
            "group differences.",
            "",
        ]

        if self._epoch_config is not None:
            lines.append(
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )
            lines.append("")

        # Global metrics
        global_csv = self.output_dir / "data" / "vertex_network_global_metrics.csv"
        if global_csv.exists():
            global_df = pd.read_csv(global_csv)
            lines.append("## Global Metrics")
            lines.append("")
            lines.append(
                "| Band | Group | Efficiency | Modularity | Small-World | Edges |"
            )
            lines.append(
                "|------|-------|------------|------------|-------------|-------|"
            )
            for _, row in (
                global_df.groupby(["band", "group"])
                .mean(numeric_only=True)
                .reset_index()
                .iterrows()
            ):
                lines.append(
                    f"| {row['band']} | {row['group']} | "
                    f"{row['global_efficiency']:.3f} | {row['modularity']:.3f} | "
                    f"{row['small_worldness']:.2f} | {row['n_edges']:.0f} |"
                )
            lines.append("")

        # NBS results
        nbs_csv = tbl_dir / "vertex_nbs_results.csv"
        if nbs_csv.exists():
            nbs_df = pd.read_csv(nbs_csv)
            lines.append("## NBS Results")
            lines.append("")
            sig_nbs = nbs_df[nbs_df["p_corrected"] < 0.05]
            if len(sig_nbs) > 0:
                lines.append(
                    f"**{len(sig_nbs)} significant subnetworks** (p<0.05)"
                )
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
            "- `data/vertex_network_nodal_metrics.csv` — per-vertex graph metrics",
            "- `data/vertex_network_global_metrics.csv` — global graph metrics per subject",
            "- `tables/vertex_network_stats.csv` — cluster permutation on nodal metrics",
            "- `tables/vertex_nbs_results.csv` — NBS subnetwork results",
            "- `figures/vertex_network_*.png` — nodal metric glass brains",
            "- `figures/vertex_nbs_edges_*.png` — NBS edge visualization",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
