"""Vertex-level connectivity analysis: imaginary coherence + FCD maps.

Computes all-to-all imaginary coherence between source vertices, derives
Functional Connectivity Density (FCD) maps, and tests for group differences
using cluster-based permutation testing.
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
    compute_fcd,
)
from ..spectral.epoch_sampler import sample_epochs, get_epoch_config
from ..stats.cluster_permutation import cluster_permutation_test, hedges_g
from ..viz.glass_brain import plot_glass_brain, plot_band_comparison
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Cannot find R/ scripts directory")


class VertexConnectivityAnalysis(BaseAnalysis):
    """All-to-all vertex connectivity with FCD mapping."""

    name = "vertex_connectivity"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._fcd_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._sfreq: float | None = None
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}
        self._conn_matrices: dict[str, dict[str, np.ndarray]] = {}

        # Config
        vc_cfg = config.raw.get("vertex_connectivity", {})
        self._metric = vc_cfg.get("metric", "imag_coherence")
        self._fcd_threshold = float(vc_cfg.get("fcd_threshold", 0.05))
        self._n_permutations = int(vc_cfg.get("n_permutations", 1000))

        wb_cfg = config.wholebrain
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))

        self._epoch_config = get_epoch_config(wb_cfg)

        self._cluster_results: dict = {}

    def setup(self) -> None:
        self._fcd_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._conn_matrices.clear()
        self._source_coords = None
        self._cluster_results.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        # Load signed data for phase-preserving connectivity
        stc_data = loader.load_source_timecourses(magnitude=False)
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()

        if self._sfreq is None:
            self._sfreq = sfreq
        if self._source_coords is None:
            self._source_coords = coords

        self._subject_groups[uid] = subject.group
        subject_fcd = {}
        subject_conn = {}

        for band_name, (fmin, fmax) in self.config.bands.items():
            logger.info("  Computing %s connectivity (%s)...", band_name, self._metric)

            if self._epoch_config is not None:
                epochs = sample_epochs(
                    stc_data, sfreq,
                    epoch_duration_sec=self._epoch_config.get("epoch_duration_sec", 2.0),
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

            fcd = compute_fcd(conn_mat, threshold=self._fcd_threshold)

            subject_fcd[band_name] = fcd
            subject_conn[band_name] = conn_mat

            n_vertices = len(fcd)
            for vi in range(n_vertices):
                self._fcd_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "vertex_idx": vi,
                    "band": band_name,
                    "fcd": float(fcd[vi]),
                })

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
            coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

        # Save connectivity matrices for downstream use (network analysis)
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
        tbl_dir = self.output_dir / "tables"
        all_stats = []

        for contrast in self.config.contrasts:
            group_a_uids = [
                uid for uid, g in self._subject_groups.items() if g == contrast.group_a
            ]
            group_b_uids = [
                uid for uid, g in self._subject_groups.items() if g == contrast.group_b
            ]

            if not group_a_uids or not group_b_uids:
                continue

            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)

            for band_name in self.config.bands:
                data_a = np.array([
                    self._subject_data[uid]["fcd"][band_name] for uid in group_a_uids
                ])
                data_b = np.array([
                    self._subject_data[uid]["fcd"][band_name] for uid in group_b_uids
                ])

                result = cluster_permutation_test(
                    data_a, data_b, coords,
                    n_perms=self._n_permutations,
                    threshold=self._cluster_threshold,
                    distance_mm=self._adjacency_distance,
                    seed=42,
                )

                g_map = hedges_g(data_a, data_b)

                key = f"{contrast.name}_{band_name}"
                self._cluster_results[key] = {
                    "result": result,
                    "mean_a": data_a.mean(axis=0),
                    "mean_b": data_b.mean(axis=0),
                    "group_labels": (label_a, label_b),
                    "band": band_name,
                }

                for vi in range(len(result.t_map)):
                    all_stats.append({
                        "contrast": contrast.name,
                        "band": band_name,
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
            stats_df.to_csv(tbl_dir / "vertex_connectivity_stats.csv", index=False)
            logger.info("Exported vertex_connectivity_stats.csv (%d rows)", len(stats_df))

    def figures(self) -> None:
        if self._source_coords is None:
            return

        coords = self._source_coords
        fig_dir = self.output_dir / "figures"

        for key, info in self._cluster_results.items():
            result = info["result"]
            band = info["band"]
            safe_name = band.lower().replace(" ", "_")
            group_labels = info["group_labels"]

            plot_band_comparison(
                coords=coords,
                mean_a=info["mean_a"],
                mean_b=info["mean_b"],
                t_map=result.t_map,
                cluster_labels=result.cluster_labels,
                cluster_pvalues=result.cluster_pvalues,
                band_name=f"FCD — {band}",
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
            r_dir = _find_r_script_dir()
            r_script = r_dir / "vertex_connectivity_analysis.R"
            if r_script.exists():
                cmd = [
                    "Rscript", str(r_script),
                    "--data-dir", str(data_dir),
                    "--config", str(config_path),
                    "--output-dir", str(self.output_dir),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        self._write_python_summary()

    def _write_python_summary(self) -> None:
        tbl_dir = self.output_dir / "tables"

        lines = [
            "# Vertex Connectivity Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            f"**Analysis**: All-to-all vertex connectivity (imaginary coherence) + FCD",
            f"**Metric**: {self._metric}",
            f"**FCD threshold**: {self._fcd_threshold}",
            f"**Permutations**: {self._n_permutations}",
            "",
            "## Methods",
            "",
            "Imaginary coherence was computed between all pairs of source vertices "
            "using cross-spectral density (Welch's method). Functional Connectivity "
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
            for band in stats_df["band"].unique():
                sub = stats_df[stats_df["band"] == band]
                n_sig = len(sub[sub["p"] < 0.05])
                lines.append(f"- **{band}**: {n_sig}/{len(sub)} vertices nominally significant")
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
