"""Vertex network layer: multi-density AUC graph metrics + NBS.

Loads shell-based source timecourses (filtered to dorsal vertices), builds
all-to-all vertex connectivity per band × connectivity metric (loading the
pre-computed matrices from vertex_connectivity when available), and runs two
independent analyses that share those matrices:

* :class:`VertexGraphAnalysis` (``vertex_graph``) — multi-density AUC graph
  metrics with group permutation tests.
* :class:`VertexNBSAnalysis` (``vertex_nbs``) — the Network-Based Statistic.

:class:`VertexNetworkAnalysis` (``vertex_network``) is the back-compat combined
alias that runs both, with the original output filenames.
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
)
from ..viz.glass_brain import plot_glass_brain_edges
from ._network_base import NetworkAnalysisBase
from .base import find_r_script_dir

logger = logging.getLogger(__name__)


def _generate_vertex_labels(
    coords: np.ndarray,
    atlas_labels: list[str] | None = None,
) -> list[str]:
    """Descriptive vertex labels: atlas ROI where available, else spatial."""
    n = len(coords)
    labels = []
    centroid = coords.mean(axis=0)
    for i in range(n):
        if atlas_labels and atlas_labels[i] not in ("Exterior", "Unknown_0"):
            labels.append(atlas_labels[i])
            continue
        x, y, z = coords[i]
        parts = []
        parts.append("Anterior" if y > centroid[1] + 1.0 else
                     "Posterior" if y < centroid[1] - 1.0 else "Central_AP")
        parts.append("Left" if x < centroid[0] - 0.5 else
                     "Right" if x > centroid[0] + 0.5 else "Midline")
        parts.append("Dorsal" if z > centroid[2] + 0.5 else
                     "Ventral" if z < centroid[2] - 0.5 else "Central_DV")
        labels.append("_".join(parts))
    return labels


class _VertexNetworkBase(NetworkAnalysisBase):
    """Shared vertex machinery: matrix build/load, AUC graph metrics, NBS."""

    _default_nbs_threshold = 3.0
    _nbs_results_filename = "vertex_nbs_results.csv"
    _fallback_config_key = "vertex_network"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._auc_rows: list[dict] = []
        self._density_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._vertex_indices: np.ndarray | None = None
        self._vertex_labels: list[str] = []
        self._sfreq: float | None = None
        # uid -> band -> conn_metric -> {graph_metric: auc}
        self._subject_aucs: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

        self._init_network_config()
        cfg = self._net_cfg
        self._auc_permutations = int(cfg.get("auc_permutations", 5000))
        self._density_min = float(cfg.get("density_min", 0.05))
        self._density_max = float(cfg.get("density_max", 0.40))
        self._density_step = float(cfg.get("density_step", 0.01))
        self._epoch_config = get_epoch_config(config.vertex)

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

    # ----------------------------------------------------- process steps --- #
    def _process_matrices(self, subject: SubjectInfo) -> None:
        """Build/load per band × metric vertex connectivity (graph + NBS need it)."""
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"
        stc_data = loader.load_source_timecourses()
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()
        if self._sfreq is None:
            self._sfreq = sfreq

        if self._vertex_indices is None:
            mask = self.config.get_vertex_mask(coords)
            self._vertex_indices = np.where(mask)[0]
            self._source_coords = coords[mask]
            if self.config.has_vertex_filter:
                logger.info("Vertex filter: %d/%d vertices retained",
                            len(self._vertex_indices), len(coords))
            try:
                from ..atlas import find_atlas_dir, load_vertex_roi_labels
                atlas_labels = load_vertex_roi_labels(self._source_coords, find_atlas_dir())
            except Exception:
                atlas_labels = None
            self._vertex_labels = _generate_vertex_labels(self._source_coords, atlas_labels)

        stc_data = stc_data[self._vertex_indices]
        self._subject_groups[uid] = subject.group

        subject_conn: dict[str, dict[str, np.ndarray]] = {}
        for band_name, (fmin, fmax) in self._selected_bands().items():
            band_conn: dict[str, np.ndarray] = {}
            for metric in self._connectivity_metrics:
                conn_mat = self._load_precomputed_conn(uid, band_name, metric)
                if conn_mat is None:
                    logger.info("  %s / %s: computing connectivity...", band_name, metric)
                    if self._epoch_config is not None:
                        epochs = sample_epochs(
                            stc_data, sfreq,
                            epoch_duration_sec=self._epoch_config.get("epoch_duration_sec", 2.0),
                            n_epochs=self._epoch_config.get("n_epochs", 80),
                            seed=self._epoch_config.get("seed", 42),
                            n_bootstrap=self._epoch_config.get("n_bootstrap", 1),
                        )
                        conn_mat = compute_vertex_connectivity_matrix_epochs(
                            epochs, sfreq, (fmin, fmax), metric=metric)
                    else:
                        conn_mat = compute_vertex_connectivity_matrix(
                            stc_data, sfreq, (fmin, fmax), metric=metric)
                band_conn[metric] = conn_mat
            subject_conn[band_name] = band_conn
        self._conn_matrices[uid] = subject_conn

    def _process_graph(self, subject: SubjectInfo) -> None:
        """Multi-density AUC graph metrics from the built matrices (graph only)."""
        uid = f"{subject.group}_{subject.subject_id}"
        subject_auc_by_band: dict[str, dict[str, dict[str, float]]] = {}
        for band_name, band_conn in self._conn_matrices.get(uid, {}).items():
            band_auc: dict[str, dict[str, float]] = {}
            for metric, conn_mat in band_conn.items():
                auc_result = compute_auc(
                    conn_mat, density_min=self._density_min,
                    density_max=self._density_max, density_step=self._density_step)
                band_auc[metric] = auc_result.auc
                row = {"subject": uid, "group": subject.group,
                       "band": band_name, "conn_metric": metric}
                row.update(auc_result.auc)
                self._auc_rows.append(row)
                for gm in auc_result.metrics_by_density:
                    drow = {"subject": uid, "group": subject.group, "band": band_name,
                            "conn_metric": metric, "density": gm.density}
                    for mn in GLOBAL_METRIC_NAMES:
                        drow[mn] = getattr(gm, mn)
                    self._density_rows.append(drow)
            subject_auc_by_band[band_name] = band_auc
        self._subject_aucs[uid] = subject_auc_by_band

    def _load_precomputed_conn(self, uid: str, band_name: str, metric: str) -> np.ndarray | None:
        vc_pkl = (self.config.output_dir / "vertex_connectivity" / "data"
                  / "vertex_connectivity_matrices.pkl")
        if not vc_pkl.exists():
            return None
        try:
            with open(vc_pkl, "rb") as f:
                all_conn = pickle.load(f)
            val = all_conn.get(uid, {}).get(band_name)
            if val is None:
                return None
            conn_mat = val.get(metric) if isinstance(val, dict) else (
                val if metric == "imag_coherence" else None)
            if conn_mat is not None:
                logger.info("    Loaded pre-computed connectivity for %s / %s", band_name, metric)
            return conn_mat
        except Exception:
            return None

    # --------------------------------------------------- graph aggregate --- #
    def _graph_aggregate(self) -> None:
        data_dir = self.output_dir / "data"
        if self._auc_rows:
            pd.DataFrame(self._auc_rows).to_csv(data_dir / f"{self.name}_auc.csv", index=False)
        if self._density_rows:
            pd.DataFrame(self._density_rows).to_csv(
                data_dir / f"{self.name}_density_curves.csv", index=False)
        self._write_source_coords(data_dir)

    def _write_source_coords(self, data_dir: Path) -> None:
        if self._source_coords is None:
            return
        coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
        if self._vertex_labels:
            coords_df["label"] = self._vertex_labels
        coords_df.index.name = "vertex_idx"
        coords_df.to_csv(data_dir / "source_coords.csv")

    # --------------------------------------------------- graph statistics -- #
    def _graph_statistics(self) -> None:
        all_auc_stats = []
        for contrast in self._pairwise_contrasts():
            group_a = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            group_b = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not group_a or not group_b:
                continue
            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)
            for band_name in self._selected_bands():
                for metric in self._connectivity_metrics:
                    auc_a = [self._subject_aucs[u][band_name][metric] for u in group_a
                             if metric in self._subject_aucs.get(u, {}).get(band_name, {})]
                    auc_b = [self._subject_aucs[u][band_name][metric] for u in group_b
                             if metric in self._subject_aucs.get(u, {}).get(band_name, {})]
                    if not auc_a or not auc_b:
                        continue
                    perm_results = auc_permutation_test(
                        auc_a, auc_b, n_permutations=self._auc_permutations, seed=42)
                    for metric_name, res in perm_results.items():
                        all_auc_stats.append({
                            "contrast": contrast.name, "group_a": label_a, "group_b": label_b,
                            "band": band_name, "conn_metric": metric, "metric": metric_name,
                            "mean_a": res["mean_a"], "mean_b": res["mean_b"],
                            "observed_diff": res["observed_diff"], "p_value": res["p_value"],
                            "hedges_g": res["hedges_g"], "significant": res["p_value"] < 0.05,
                        })
        if all_auc_stats:
            pd.DataFrame(all_auc_stats).to_csv(
                self.tbl_dir / f"{self.name}_stats.csv", index=False)
            logger.info("Exported %s_stats.csv (%d rows)", self.name, len(all_auc_stats))

    # -------------------------------------------------------- nbs figures -- #
    def _nbs_figures(self) -> None:
        if self._source_coords is None:
            return
        for key, nbs in self._nbs_results.items():
            if nbs.n_significant_components == 0:
                continue
            sig_mask = np.abs(nbs.t_matrix) > self._nbs_threshold
            edge_pairs = np.argwhere(np.triu(sig_mask, k=1))
            if len(edge_pairs) > 0:
                edge_t = np.array([nbs.t_matrix[i, j] for i, j in edge_pairs])
                plot_glass_brain_edges(
                    coords=self._source_coords, edges=edge_pairs,
                    output_path=self.fig_dir / f"vertex_nbs_edges_{key}.png",
                    edge_values=np.abs(edge_t), title=f"NBS Edges — {key}")

    # --------------------------------------------------------- summaries --- #
    def _write_summary(self, graph: bool = True, nbs: bool = True) -> None:
        lines = [f"# {self.name} Analysis Summary", "",
                 f"**Study**: {self.config.name}",
                 f"**Connectivity metrics**: {', '.join(self._connectivity_metrics)}"]
        if graph:
            lines += [f"**AUC permutations**: {self._auc_permutations}",
                      f"**Density range**: {self._density_min:.0%}–{self._density_max:.0%}"]
            auc_csv = self.tbl_dir / f"{self.name}_stats.csv"
            if auc_csv.exists():
                df = pd.read_csv(auc_csv)
                sig = df[df["significant"] == True]
                lines += ["", "## AUC Permutation Results", "",
                          (f"**{len(sig)} significant AUC differences** (p<0.05)"
                           if len(sig) else "No significant AUC differences at p < 0.05.")]
        if nbs:
            lines += ["", f"**NBS threshold**: t = {self._nbs_threshold}",
                      f"**NBS permutations**: {self._nbs_permutations}"]
            nbs_csv = self.tbl_dir / self._nbs_results_filename
            if nbs_csv.exists():
                ndf = pd.read_csv(nbs_csv)
                sig = ndf[ndf["p_corrected"] < 0.05]
                lines += ["", "## NBS Results", ""]
                if len(sig):
                    lines.append(f"**{len(sig)} significant subnetworks** (p<0.05)")
                    for _, row in sig.iterrows():
                        lines.append(f"- {row['key']}: {row['n_edges']} edges, p={row['p_corrected']:.4f}")
                else:
                    lines.append("No significant NBS subnetworks at p < 0.05.")
        lines.append("")
        (self.output_dir / "ANALYSIS_SUMMARY.md").write_text("\n".join(lines))


class VertexGraphAnalysis(_VertexNetworkBase):
    """Vertex multi-density AUC graph metrics, no NBS."""

    name = "vertex_graph"

    def process_subject(self, subject: SubjectInfo) -> None:
        self._process_matrices(subject)
        self._process_graph(subject)

    def aggregate(self) -> None:
        self._graph_aggregate()

    def statistics(self) -> None:
        self._graph_statistics()

    def figures(self) -> None:
        pass

    def summary(self) -> None:
        self._write_summary(graph=True, nbs=False)


class VertexNBSAnalysis(_VertexNetworkBase):
    """Vertex Network-Based Statistic, no graph metrics."""

    name = "vertex_nbs"

    def process_subject(self, subject: SubjectInfo) -> None:
        self._process_matrices(subject)

    def aggregate(self) -> None:
        self._write_source_coords(self.output_dir / "data")

    def statistics(self) -> None:
        self._run_nbs()
        self._run_nbs_hypotheses()

    def figures(self) -> None:
        self._nbs_figures()

    def summary(self) -> None:
        self._write_summary(graph=False, nbs=True)


class VertexNetworkAnalysis(_VertexNetworkBase):
    """Combined alias: AUC graph metrics + NBS (back-compat filenames)."""

    name = "vertex_network"
    _fallback_config_key = None

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
        self._nbs_figures()

    def summary(self) -> None:
        data_dir = self.output_dir / "data"
        config_path = data_dir / "study_config.yaml"
        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        try:
            r_script = find_r_script_dir() / "vertex_network_analysis.R"
            if r_script.exists():
                cmd = ["Rscript", str(r_script), "--data-dir", str(data_dir),
                       "--config", str(config_path), "--output-dir", str(self.output_dir),
                       "--fig-dir", str(self.fig_dir), "--tbl-dir", str(self.tbl_dir)]
                cmd.extend(self._r_no_figures_flags())
                if subprocess.run(cmd, capture_output=True, text=True, timeout=3600).returncode == 0:
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        self._write_summary(graph=True, nbs=True)
