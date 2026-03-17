"""Vertex Network Analysis: multi-density AUC graph metrics + NBS.

Loads shell-based source timecourses (154 vertices, filtered to ~66 dorsal),
computes all-to-all vertex connectivity, then runs:
  1. Multi-density AUC: global graph metrics across a density sweep (5-40%),
     integrated via trapezoidal rule for threshold-independent inference.
  2. NBS: Network-Based Statistic for identifying subnetworks with
     significant group differences in edge connectivity.

Group differences in AUC values are tested with permutation testing.
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
from ..stats.graph_metrics import (
    GLOBAL_METRIC_NAMES,
    compute_auc,
    auc_permutation_test,
    nbs_permutation_test,
)
from ..viz.glass_brain import plot_glass_brain_edges
from .base import BaseAnalysis, find_r_script_dir

logger = logging.getLogger(__name__)


def _generate_vertex_labels(
    coords: np.ndarray,
    atlas_labels: list[str] | None = None,
) -> list[str]:
    """Generate descriptive labels for vertices.

    For vertices with atlas ROI labels, use those. For unassigned/Exterior
    vertices, generate spatial labels from coordinates.
    """
    n = len(coords)
    labels = []
    centroid = coords.mean(axis=0)

    for i in range(n):
        if atlas_labels and atlas_labels[i] not in ("Exterior", "Unknown_0"):
            labels.append(atlas_labels[i])
            continue

        x, y, z = coords[i]
        parts = []
        if y > centroid[1] + 1.0:
            parts.append("Anterior")
        elif y < centroid[1] - 1.0:
            parts.append("Posterior")
        else:
            parts.append("Central_AP")

        if x < centroid[0] - 0.5:
            parts.append("Left")
        elif x > centroid[0] + 0.5:
            parts.append("Right")
        else:
            parts.append("Midline")

        if z > centroid[2] + 0.5:
            parts.append("Dorsal")
        elif z < centroid[2] - 0.5:
            parts.append("Ventral")
        else:
            parts.append("Central_DV")

        labels.append("_".join(parts))

    return labels


class VertexNetworkAnalysis(BaseAnalysis):
    """Multi-density AUC graph metrics + NBS on vertex connectivity."""

    name = "vertex_network"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._auc_rows: list[dict] = []
        self._density_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._vertex_indices: np.ndarray | None = None
        self._vertex_labels: list[str] = []
        self._sfreq: float | None = None
        self._subject_aucs: dict[str, dict[str, dict[str, float]]] = {}
        self._subject_groups: dict[str, str] = {}
        self._conn_matrices: dict[str, dict[str, np.ndarray]] = {}

        # Config
        net_cfg = config.raw.get("vertex_network", {})
        self._nbs_threshold = float(net_cfg.get("nbs_threshold", 3.0))
        self._nbs_permutations = int(net_cfg.get("nbs_permutations", 5000))
        self._auc_permutations = int(net_cfg.get("auc_permutations", 5000))
        self._metric = net_cfg.get("metric", "imag_coherence")

        # Density sweep parameters
        self._density_min = float(net_cfg.get("density_min", 0.05))
        self._density_max = float(net_cfg.get("density_max", 0.40))
        self._density_step = float(net_cfg.get("density_step", 0.01))

        wb_cfg = config.vertex
        self._epoch_config = get_epoch_config(wb_cfg)
        self._nbs_results: dict = {}

    def setup(self) -> None:
        self._auc_rows.clear()
        self._density_rows.clear()
        self._subject_aucs.clear()
        self._subject_groups.clear()
        self._conn_matrices.clear()
        self._source_coords = None
        self._vertex_indices = None
        self._vertex_labels.clear()
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
        subject_auc_by_band = {}
        subject_conn = {}

        for band_name, (fmin, fmax) in self.config.bands.items():
            logger.info("  %s: connectivity + multi-density AUC...", band_name)

            # Check for pre-computed connectivity matrices
            conn_mat = self._load_precomputed_conn(uid, band_name)

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

            # Multi-density AUC
            auc_result = compute_auc(
                conn_mat,
                density_min=self._density_min,
                density_max=self._density_max,
                density_step=self._density_step,
            )

            subject_auc_by_band[band_name] = auc_result.auc

            # Store AUC row
            row = {
                "subject": uid,
                "group": subject.group,
                "band": band_name,
            }
            row.update(auc_result.auc)
            self._auc_rows.append(row)

            # Store per-density metrics for curve plots
            for gm in auc_result.metrics_by_density:
                drow = {
                    "subject": uid,
                    "group": subject.group,
                    "band": band_name,
                    "density": gm.density,
                }
                for mn in GLOBAL_METRIC_NAMES:
                    drow[mn] = getattr(gm, mn)
                self._density_rows.append(drow)

        self._subject_aucs[uid] = subject_auc_by_band
        self._conn_matrices[uid] = subject_conn

    def _load_precomputed_conn(
        self, uid: str, band_name: str,
    ) -> np.ndarray | None:
        """Try loading pre-computed connectivity from vertex_connectivity output."""
        vc_pkl = (
            self.config.output_dir / "vertex_connectivity" / "data"
            / "vertex_connectivity_matrices.pkl"
        )
        if not vc_pkl.exists():
            return None
        try:
            with open(vc_pkl, "rb") as f:
                all_conn = pickle.load(f)
            subj_conn = all_conn.get(uid, {})
            if band_name not in subj_conn:
                return None
            val = subj_conn[band_name]
            if isinstance(val, dict):
                conn_mat = val.get(self._metric)
            else:
                conn_mat = val if self._metric == "imag_coherence" else None
            if conn_mat is not None:
                logger.info("    Loaded pre-computed connectivity for %s", band_name)
            return conn_mat
        except Exception:
            return None

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        # AUC values per subject
        auc_df = pd.DataFrame(self._auc_rows)
        if not auc_df.empty:
            auc_df.to_csv(
                data_dir / "vertex_network_auc.csv", index=False,
            )
            logger.info(
                "Exported vertex_network_auc.csv (%d rows)", len(auc_df),
            )

        # Per-density metrics for visualization
        density_df = pd.DataFrame(self._density_rows)
        if not density_df.empty:
            density_df.to_csv(
                data_dir / "vertex_network_density_curves.csv", index=False,
            )
            logger.info(
                "Exported vertex_network_density_curves.csv (%d rows)",
                len(density_df),
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
        all_auc_stats = []

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
                # --- AUC permutation test ---
                auc_a = [
                    self._subject_aucs[uid][band_name]
                    for uid in group_a_uids
                    if band_name in self._subject_aucs.get(uid, {})
                ]
                auc_b = [
                    self._subject_aucs[uid][band_name]
                    for uid in group_b_uids
                    if band_name in self._subject_aucs.get(uid, {})
                ]

                if auc_a and auc_b:
                    logger.info(
                        "  AUC permutation: %s %s (%d vs %d)...",
                        contrast.name, band_name, len(auc_a), len(auc_b),
                    )
                    perm_results = auc_permutation_test(
                        auc_a, auc_b,
                        n_permutations=self._auc_permutations,
                        seed=42,
                    )
                    for metric_name, res in perm_results.items():
                        all_auc_stats.append({
                            "contrast": contrast.name,
                            "group_a": label_a,
                            "group_b": label_b,
                            "band": band_name,
                            "metric": metric_name,
                            "mean_a": res["mean_a"],
                            "mean_b": res["mean_b"],
                            "observed_diff": res["observed_diff"],
                            "p_value": res["p_value"],
                            "hedges_g": res["hedges_g"],
                            "significant": res["p_value"] < 0.05,
                        })

                # --- NBS on connectivity matrices ---
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

        # Export AUC stats
        if all_auc_stats:
            stats_df = pd.DataFrame(all_auc_stats)
            stats_df.to_csv(
                tbl_dir / "vertex_network_stats.csv", index=False,
            )
            logger.info(
                "Exported vertex_network_stats.csv (%d rows)", len(stats_df),
            )

        # Export NBS summary
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

        # NBS edge glass brains
        for key, nbs in self._nbs_results.items():
            if nbs.n_significant_components == 0:
                continue

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
                    cmd, capture_output=True, text=True, timeout=3600,
                )
                if result.returncode == 0:
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        self._write_python_summary()

    def _write_python_summary(self) -> None:
        tbl_dir = self.tbl_dir
        n_densities = len(np.arange(
            self._density_min,
            self._density_max + self._density_step / 2,
            self._density_step,
        ))

        lines = [
            "# Vertex Network Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Multi-density AUC graph metrics + NBS (shell source space)",
            f"**Connectivity metric**: {self._metric}",
            f"**Density range**: {self._density_min:.0%} to {self._density_max:.0%} "
            f"in {self._density_step:.0%} steps ({n_densities} densities)",
            f"**AUC permutations**: {self._auc_permutations}",
            f"**NBS threshold**: t = {self._nbs_threshold}",
            f"**NBS permutations**: {self._nbs_permutations}",
            "",
            "## Methods",
            "",
            "Graph-theoretic metrics were computed from thresholded vertex connectivity "
            "matrices across a range of proportional density thresholds "
            f"({self._density_min:.0%}–{self._density_max:.0%}). "
            "For each subject, the area under the curve (AUC) was computed for each "
            "metric using trapezoidal integration, yielding threshold-independent "
            "scalar values. Group differences in AUC were tested using permutation "
            f"testing ({self._auc_permutations} permutations). "
            "The Network-Based Statistic (Zalesky et al., 2010) was used to identify "
            "subnetworks with significant group differences in edge connectivity.",
            "",
            "**Global metrics**: global efficiency, characteristic path length, "
            "mean clustering coefficient, transitivity, modularity, assortativity, "
            "mean local efficiency, small-worldness.",
            "",
        ]

        if self._epoch_config is not None:
            lines.append(
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )
            lines.append("")

        # AUC stats
        auc_csv = tbl_dir / "vertex_network_stats.csv"
        if auc_csv.exists():
            auc_df = pd.read_csv(auc_csv)
            sig = auc_df[auc_df["significant"] == True]
            lines.append("## AUC Permutation Results")
            lines.append("")
            if len(sig) > 0:
                lines.append(
                    f"**{len(sig)} significant AUC differences** (p < 0.05):"
                )
                lines.append("")
                lines.append("| Contrast | Band | Metric | Diff | p | Hedges' g |")
                lines.append("|----------|------|--------|------|---|-----------|")
                for _, row in sig.iterrows():
                    lines.append(
                        f"| {row['contrast']} | {row['band']} | {row['metric']} | "
                        f"{row['observed_diff']:.4f} | {row['p_value']:.4f} | "
                        f"{row['hedges_g']:.3f} |"
                    )
                lines.append("")
            else:
                lines.append("No significant AUC differences at p < 0.05.")
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
                    f"**{len(sig_nbs)} significant subnetworks** (p < 0.05)"
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
            "- `data/vertex_network_auc.csv` — AUC values per subject/band/metric",
            "- `data/vertex_network_density_curves.csv` — metrics at each density",
            "- `tables/vertex_network_stats.csv` — AUC permutation test results",
            "- `tables/vertex_nbs_results.csv` — NBS subnetwork results",
            "- `figures/vertex_nbs_edges_*.png` — NBS edge visualization",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
