"""Vertex-level connectivity analysis: multiple metrics + FCD maps.

Computes all-to-all connectivity between source vertices using one or more
metrics, derives Functional Connectivity Density (FCD) maps, and tests for
group differences using cluster-based permutation testing.
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
    compute_vertex_connectivity_matrix_multi,
    compute_vertex_connectivity_matrix_epochs,
    compute_vertex_connectivity_matrix_epochs_multi,
    compute_fcd,
    FCD_CENTER,
)
from ..spectral.epoch_sampler import sample_epochs, get_epoch_config
from ..stats.cluster_permutation import cluster_permutation_test, hedges_g
from ..viz.glass_brain import plot_glass_brain, plot_band_comparison
from .base import BaseAnalysis, find_r_script_dir

logger = logging.getLogger(__name__)


class VertexConnectivityAnalysis(BaseAnalysis):
    """All-to-all vertex connectivity with FCD mapping (multi-metric)."""

    name = "vertex_connectivity"
    SELECTABLE = {"metric": "connectivity metric", "band": "frequency band",
                  "hypothesis": "declared hypothesis"}

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._fcd_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._vertex_indices: np.ndarray | None = None
        self._sfreq: float | None = None
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}
        # conn_matrices: uid -> band -> metric -> matrix
        self._conn_matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}

        # Config
        vc_cfg = config.raw.get("vertex_connectivity", {})
        # Support both single metric (legacy) and multi-metric list
        metrics_cfg = vc_cfg.get("metrics")
        if metrics_cfg is not None:
            self._metrics = list(metrics_cfg)
        else:
            self._metrics = [vc_cfg.get("metric", "imag_coherence")]

        self._fcd_threshold = float(vc_cfg.get("fcd_threshold", 0.05))
        self._n_permutations = int(vc_cfg.get("n_permutations", 1000))

        wb_cfg = config.vertex
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))

        self._epoch_config = get_epoch_config(wb_cfg)

        self._cluster_results: dict = {}

    def setup(self) -> None:
        # Restrict to --metric / --select metric=... if requested (shared STFT
        # pass is preserved — fewer metrics are emitted from the same pass).
        self._metrics = self._select("metric", self._metrics)
        self._fcd_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._conn_matrices.clear()
        self._source_coords = None
        self._vertex_indices = None
        self._cluster_results.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        # Load signed data for phase-preserving connectivity
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

        stc_data = stc_data[self._vertex_indices]

        self._subject_groups[uid] = subject.group
        subject_fcd: dict[str, dict[str, np.ndarray]] = {}
        subject_conn: dict[str, dict[str, np.ndarray]] = {}

        use_multi = len(self._metrics) > 1

        for band_name, (fmin, fmax) in self._selected_bands().items():
            logger.info(
                "  Computing %s connectivity (%s)...",
                band_name, ", ".join(self._metrics),
            )

            if use_multi:
                # Compute all metrics in a single pass
                if self._epoch_config is not None:
                    epochs = sample_epochs(
                        stc_data, sfreq,
                        epoch_duration_sec=self._epoch_config.get(
                            "epoch_duration_sec", 2.0,
                        ),
                        n_epochs=self._epoch_config.get("n_epochs", 80),
                        seed=self._epoch_config.get("seed", 42),
                n_bootstrap=self._epoch_config.get("n_bootstrap", 1),
                    )
                    conn_results = compute_vertex_connectivity_matrix_epochs_multi(
                        epochs, sfreq, (fmin, fmax), metrics=self._metrics,
                    )
                else:
                    conn_results = compute_vertex_connectivity_matrix_multi(
                        stc_data, sfreq, (fmin, fmax), metrics=self._metrics,
                    )
            else:
                # Single metric — use original function
                metric = self._metrics[0]
                if self._epoch_config is not None:
                    epochs = sample_epochs(
                        stc_data, sfreq,
                        epoch_duration_sec=self._epoch_config.get(
                            "epoch_duration_sec", 2.0,
                        ),
                        n_epochs=self._epoch_config.get("n_epochs", 80),
                        seed=self._epoch_config.get("seed", 42),
                n_bootstrap=self._epoch_config.get("n_bootstrap", 1),
                    )
                    conn_mat = compute_vertex_connectivity_matrix_epochs(
                        epochs, sfreq, (fmin, fmax), metric=metric,
                    )
                else:
                    conn_mat = compute_vertex_connectivity_matrix(
                        stc_data, sfreq, (fmin, fmax), metric=metric,
                    )
                conn_results = {metric: conn_mat}

            band_fcd = {}
            band_conn = {}
            for metric, conn_mat in conn_results.items():
                fcd = compute_fcd(
                    conn_mat, threshold=self._fcd_threshold,
                    center=FCD_CENTER.get(metric),
                )
                band_fcd[metric] = fcd
                band_conn[metric] = conn_mat

                n_vertices = len(fcd)
                for vi in range(n_vertices):
                    self._fcd_rows.append({
                        "subject": uid,
                        "group": subject.group,
                        "vertex_idx": int(self._vertex_indices[vi]),
                        "band": band_name,
                        "metric": metric,
                        "fcd": float(fcd[vi]),
                    })

            subject_fcd[band_name] = band_fcd
            subject_conn[band_name] = band_conn

        self._subject_data[uid] = {"fcd": subject_fcd}
        self._conn_matrices[uid] = subject_conn

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        fcd_df = pd.DataFrame(self._fcd_rows)
        if fcd_df.empty:
            logger.warning("No vertex connectivity data collected")
            return
        fcd_df.to_csv(data_dir / "vertex_fcd.csv", index=False)
        logger.info("Exported vertex_fcd.csv (%d rows)", len(fcd_df))

        if self._source_coords is not None:
            coords_df = pd.DataFrame(
                self._source_coords, columns=["x", "y", "z"],
            )
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

        # Save connectivity matrices for downstream use (vertex_network)
        if self._conn_matrices:
            pkl_path = data_dir / "vertex_connectivity_matrices.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(self._conn_matrices, f)
            logger.info("Saved connectivity matrices to %s", pkl_path)

    def statistics(self) -> None:
        if self._source_coords is None:
            logger.error("No source coordinates — cannot run statistics")
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

            for band_name in self._selected_bands():
                for metric in self._metrics:
                    data_a = np.array([
                        self._subject_data[uid]["fcd"][band_name][metric]
                        for uid in group_a_uids
                        if band_name in self._subject_data.get(uid, {}).get("fcd", {})
                        and metric in self._subject_data[uid]["fcd"].get(band_name, {})
                    ])
                    data_b = np.array([
                        self._subject_data[uid]["fcd"][band_name][metric]
                        for uid in group_b_uids
                        if band_name in self._subject_data.get(uid, {}).get("fcd", {})
                        and metric in self._subject_data[uid]["fcd"].get(band_name, {})
                    ])

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

                    key = f"{contrast.name}_{band_name}_{metric}"
                    self._cluster_results[key] = {
                        "result": result,
                        "mean_a": data_a.mean(axis=0),
                        "mean_b": data_b.mean(axis=0),
                        "group_labels": (label_a, label_b),
                        "band": band_name,
                        "metric": metric,
                    }

                    for vi in range(len(result.t_map)):
                        all_stats.append({
                            "contrast": contrast.name,
                            "band": band_name,
                            "metric": metric,
                            "vertex_idx": vi,
                            "fcd_a": float(data_a.mean(axis=0)[vi]),
                            "fcd_b": float(data_b.mean(axis=0)[vi]),
                            "t": float(result.t_map[vi]),
                            "p": float(result.p_map[vi]),
                            "hedges_g": float(g_map[vi]),
                            "cluster_id": int(result.cluster_labels[vi]),
                        })

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df.to_csv(
                tbl_dir / "vertex_connectivity_stats.csv", index=False,
            )
            logger.info(
                "Exported vertex_connectivity_stats.csv (%d rows)",
                len(stats_df),
            )

        # --- Declarative hypotheses (hypothesis layer; additive, map+cluster) ---
        from ..hypothesis import write_module_hypotheses_perm

        if self._source_coords is not None and self._subject_groups:
            maps_by_cell = {
                (band_name, metric): {
                    uid: self._subject_data[uid]["fcd"][band_name][metric]
                    for uid in self._subject_groups
                }
                for band_name in self._selected_bands()
                for metric in self._metrics
            }
            wanted_hyp = self._selection.get("hypothesis")
            write_module_hypotheses_perm(
                maps_by_cell, self._subject_groups, self._source_coords, self.config,
                tbl_dir, prefix="vertex_connectivity",
                n_perms=self._n_permutations, threshold=self._cluster_threshold,
                distance_mm=self._adjacency_distance,
                hypothesis=",".join(sorted(wanted_hyp)) if wanted_hyp else None,
            )

    def figures(self) -> None:
        if self._source_coords is None:
            return

        coords = self._source_coords
        fig_dir = self.fig_dir

        for key, info in self._cluster_results.items():
            result = info["result"]
            band = info["band"]
            metric = info.get("metric", "imag_coherence")
            safe_name = f"{band}_{metric}".lower().replace(" ", "_")
            group_labels = info["group_labels"]

            plot_band_comparison(
                coords=coords,
                mean_a=info["mean_a"],
                mean_b=info["mean_b"],
                t_map=result.t_map,
                cluster_labels=result.cluster_labels,
                cluster_pvalues=result.cluster_pvalues,
                band_name=f"FCD ({metric}) — {band}",
                group_labels=group_labels,
                output_path=fig_dir / f"fcd_{safe_name}.png",
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
            r_script = r_dir / "vertex_connectivity_analysis.R"
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

        lines = [
            "# Vertex Connectivity Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            f"**Analysis**: All-to-all vertex connectivity + FCD",
            f"**Metrics**: {', '.join(self._metrics)}",
            f"**FCD threshold**: {self._fcd_threshold}",
            f"**Permutations**: {self._n_permutations}",
            "",
            "## Methods",
            "",
            f"Connectivity was computed between all pairs of source vertices "
            f"using {', '.join(self._metrics)}. Functional Connectivity "
            "Density (FCD) was derived by counting the fraction of connections "
            f"exceeding {self._fcd_threshold} per vertex. Group differences in FCD "
            "were tested using cluster-based permutation testing.",
            "",
        ]

        if self._epoch_config is not None:
            lines.append(
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )
            lines.append("")

        stats_csv = tbl_dir / "vertex_connectivity_stats.csv"
        if stats_csv.exists():
            stats_df = pd.read_csv(stats_csv)
            lines.append("## Results")
            lines.append("")
            group_cols = ["band"]
            if "metric" in stats_df.columns:
                group_cols.append("metric")
            for keys, sub in stats_df.groupby(group_cols):
                if isinstance(keys, str):
                    label = keys
                else:
                    label = " / ".join(str(k) for k in keys)
                n_sig = len(sub[sub["p"] < 0.05])
                lines.append(
                    f"- **{label}**: {n_sig}/{len(sub)} vertices nominally significant"
                )
            lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "- `data/vertex_fcd.csv` — per-subject per-vertex FCD values",
            "- `data/vertex_connectivity_matrices.pkl` — full connectivity matrices",
            "- `data/source_coords.csv` — vertex coordinates (mm)",
            "- `tables/vertex_connectivity_stats.csv` — FCD statistics",
            "- `figures/fcd_*.png` — FCD glass brain maps",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
