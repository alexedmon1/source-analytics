"""Whole-brain vertex-level spectral analysis with cluster-based permutation testing.

Computes PSD once per subject for all 154 shell vertices, then extracts:
- Relative and absolute band power per vertex per band
- fALFF (high-gamma / total power ratio)
- Spectral slope (1/f exponent)
- Peak alpha frequency

Statistics use voxel-wise t-tests with spatial cluster-based permutation
correction (Maris & Oostenveld, 2007). Visualization via glass brain plots.
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
from ..spectral.vertex import (
    compute_psd_vertices,
    extract_band_power_vertices,
    compute_falff,
    compute_spectral_slope,
    compute_peak_frequency,
)
from ..stats.cluster_permutation import (
    cluster_permutation_test,
    hedges_g,
    voxelwise_ttest,
)
from ..viz.glass_brain import (
    plot_band_comparison,
    plot_glass_brain,
    plot_wholebrain_summary,
)
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    """Locate the R/ directory relative to this package."""
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find R/ scripts directory. Expected at: " + str(pkg_root / "R")
    )


class WholebrainAnalysis(BaseAnalysis):
    """Whole-brain vertex-level spectral analysis with cluster permutation testing.

    Processes shell_ellipsoid source data (154 vertices), computes spectral
    metrics, runs voxel-wise statistics with cluster correction, and generates
    glass brain visualizations.
    """

    name = "wholebrain"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._band_power_rows: list[dict] = []
        self._feature_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._sfreq: float | None = None
        # Per-subject arrays for statistics: {subject_uid: {band: {metric: array}}}
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}

        # Wholebrain-specific config
        wb_cfg = config.wholebrain
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))
        self._n_permutations = int(wb_cfg.get("n_permutations", 1000))
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))
        self._noise_exclude = wb_cfg.get("noise_exclude_hz")
        if self._noise_exclude is not None:
            self._noise_exclude = tuple(self._noise_exclude)

    def setup(self) -> None:
        self._band_power_rows.clear()
        self._feature_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._source_coords = None

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        # Load source time courses: (n_vertices, n_times)
        stc_data = loader.load_source_timecourses(magnitude=True)
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        if self._source_coords is None:
            self._source_coords = coords
        n_vertices = stc_data.shape[0]

        # Compute PSD for all vertices: (n_vertices, n_freqs)
        fmax = max(hi for _, hi in self.config.bands.values()) + 10
        freqs, psd = compute_psd_vertices(stc_data, sfreq, fmax=fmax)

        # Extract band power metrics
        band_power = extract_band_power_vertices(
            freqs, psd, self.config.bands,
            noise_exclude=self._noise_exclude,
        )

        # Compute additional features
        falff = compute_falff(freqs, psd)
        slope = compute_spectral_slope(freqs, psd)
        peak_alpha = compute_peak_frequency(freqs, psd, search_range=(6, 13))

        # Store per-subject data for statistics
        self._subject_groups[uid] = subject.group
        self._subject_data[uid] = {
            "band_power": band_power,
            "falff": falff,
            "slope": slope,
            "peak_alpha": peak_alpha,
        }

        # Accumulate rows for CSV export
        for band_name, bp in band_power.items():
            for vi in range(n_vertices):
                self._band_power_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "vertex_idx": vi,
                    "band": band_name,
                    "absolute": float(bp["absolute"][vi]),
                    "relative": float(bp["relative"][vi]),
                    "dB": float(bp["dB"][vi]),
                })

        for vi in range(n_vertices):
            self._feature_rows.append({
                "subject": uid,
                "group": subject.group,
                "vertex_idx": vi,
                "falff": float(falff[vi]),
                "spectral_slope": float(slope[vi]),
                "peak_alpha_freq": float(peak_alpha[vi]),
            })

    def aggregate(self) -> None:
        """Export CSVs."""
        data_dir = self.output_dir / "data"

        # Band power CSV
        band_df = pd.DataFrame(self._band_power_rows)
        if band_df.empty:
            logger.warning("No wholebrain band power data collected")
            return
        band_df.to_csv(data_dir / "wholebrain_values.csv", index=False)
        logger.info("Exported wholebrain_values.csv (%d rows)", len(band_df))

        # Features CSV
        feat_df = pd.DataFrame(self._feature_rows)
        if not feat_df.empty:
            feat_df.to_csv(data_dir / "wholebrain_features.csv", index=False)
            logger.info("Exported wholebrain_features.csv (%d rows)", len(feat_df))

        # Source coordinates CSV
        if self._source_coords is not None:
            coords_df = pd.DataFrame(
                self._source_coords,
                columns=["x", "y", "z"],
            )
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")
            logger.info("Exported source_coords.csv (%d rows)", len(coords_df))

    def statistics(self) -> None:
        """Run voxel-wise t-tests + cluster permutation for each contrast and metric."""
        if self._source_coords is None:
            logger.error("No source coordinates — cannot run statistics")
            return

        coords = self._source_coords
        tbl_dir = self.output_dir / "tables"
        data_dir = self.output_dir / "data"

        all_voxelwise = []
        all_cluster = []

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
                logger.warning(
                    "Contrast %s: missing subjects (a=%d, b=%d)",
                    contrast.name, len(group_a_uids), len(group_b_uids),
                )
                continue

            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)
            logger.info(
                "Contrast '%s': %s (n=%d) vs %s (n=%d)",
                contrast.name, label_a, len(group_a_uids),
                label_b, len(group_b_uids),
            )

            # --- Band power metrics ---
            band_cluster_results = {}
            for band_name in self.config.bands:
                for metric in ["relative", "dB"]:
                    data_a = np.array([
                        self._subject_data[uid]["band_power"][band_name][metric]
                        for uid in group_a_uids
                    ])
                    data_b = np.array([
                        self._subject_data[uid]["band_power"][band_name][metric]
                        for uid in group_b_uids
                    ])

                    # Cluster permutation test
                    result = cluster_permutation_test(
                        data_a, data_b, coords,
                        n_perms=self._n_permutations,
                        threshold=self._cluster_threshold,
                        distance_mm=self._adjacency_distance,
                        seed=42,
                    )

                    # Effect size
                    g_map = hedges_g(data_a, data_b)

                    # Store for plotting (use relative power for band plots)
                    if metric == "relative":
                        band_cluster_results[band_name] = {
                            "t_map": result.t_map,
                            "cluster_labels": result.cluster_labels,
                            "cluster_pvalues": result.cluster_pvalues,
                            "mean_a": data_a.mean(axis=0),
                            "mean_b": data_b.mean(axis=0),
                        }

                    # Voxelwise stats rows
                    for vi in range(len(result.t_map)):
                        all_voxelwise.append({
                            "contrast": contrast.name,
                            "band": band_name,
                            "metric": metric,
                            "vertex_idx": vi,
                            "t": float(result.t_map[vi]),
                            "p": float(result.p_map[vi]),
                            "hedges_g": float(g_map[vi]),
                            "cluster_id": int(result.cluster_labels[vi]),
                        })

                    # Cluster summary rows
                    for ci, (cs, cp) in enumerate(
                        zip(result.cluster_stats, result.cluster_pvalues), start=1
                    ):
                        n_verts = int(np.sum(result.cluster_labels == ci))
                        peak_t = float(
                            result.t_map[result.cluster_labels == ci][
                                np.argmax(np.abs(result.t_map[result.cluster_labels == ci]))
                            ]
                        )
                        all_cluster.append({
                            "contrast": contrast.name,
                            "band": band_name,
                            "metric": metric,
                            "cluster_id": ci,
                            "n_vertices": n_verts,
                            "cluster_stat": float(cs),
                            "peak_t": peak_t,
                            "p_corrected": float(cp),
                        })

            # --- Feature metrics (falff, slope, peak_alpha) ---
            feature_cluster_results = {}
            for feat_name, feat_key in [
                ("falff", "falff"),
                ("spectral_slope", "slope"),
                ("peak_alpha", "peak_alpha"),
            ]:
                data_a = np.array([
                    self._subject_data[uid][feat_key] for uid in group_a_uids
                ])
                data_b = np.array([
                    self._subject_data[uid][feat_key] for uid in group_b_uids
                ])

                result = cluster_permutation_test(
                    data_a, data_b, coords,
                    n_perms=self._n_permutations,
                    threshold=self._cluster_threshold,
                    distance_mm=self._adjacency_distance,
                    seed=42,
                )

                g_map = hedges_g(data_a, data_b)

                feature_cluster_results[feat_name] = {
                    "t_map": result.t_map,
                    "cluster_labels": result.cluster_labels,
                    "cluster_pvalues": result.cluster_pvalues,
                    "mean_a": data_a.mean(axis=0),
                    "mean_b": data_b.mean(axis=0),
                }

                for vi in range(len(result.t_map)):
                    all_voxelwise.append({
                        "contrast": contrast.name,
                        "band": feat_name,
                        "metric": feat_name,
                        "vertex_idx": vi,
                        "t": float(result.t_map[vi]),
                        "p": float(result.p_map[vi]),
                        "hedges_g": float(g_map[vi]),
                        "cluster_id": int(result.cluster_labels[vi]),
                    })

                for ci, (cs, cp) in enumerate(
                    zip(result.cluster_stats, result.cluster_pvalues), start=1
                ):
                    n_verts = int(np.sum(result.cluster_labels == ci))
                    peak_t = float(
                        result.t_map[result.cluster_labels == ci][
                            np.argmax(np.abs(result.t_map[result.cluster_labels == ci]))
                        ]
                    )
                    all_cluster.append({
                        "contrast": contrast.name,
                        "band": feat_name,
                        "metric": feat_name,
                        "cluster_id": ci,
                        "n_vertices": n_verts,
                        "cluster_stat": float(cs),
                        "peak_t": peak_t,
                        "p_corrected": float(cp),
                    })

            # Store results for figures phase
            self._band_cluster_results = band_cluster_results
            self._feature_cluster_results = feature_cluster_results
            self._contrast_labels = (label_a, label_b)

        # Export CSVs
        if all_voxelwise:
            vox_df = pd.DataFrame(all_voxelwise)
            vox_df.to_csv(tbl_dir / "voxelwise_stats.csv", index=False)
            logger.info("Exported voxelwise_stats.csv (%d rows)", len(vox_df))

        if all_cluster:
            clust_df = pd.DataFrame(all_cluster)
            clust_df.to_csv(tbl_dir / "cluster_results.csv", index=False)
            logger.info("Exported cluster_results.csv (%d rows)", len(clust_df))

            sig_clusters = clust_df[clust_df["p_corrected"] < 0.05]
            if len(sig_clusters) > 0:
                logger.info(
                    "Significant clusters (p<0.05): %d", len(sig_clusters),
                )
                for _, row in sig_clusters.iterrows():
                    logger.info(
                        "  %s / %s (%s): %d vertices, peak t=%.2f, p=%.4f",
                        row["band"], row["metric"], row["contrast"],
                        row["n_vertices"], row["peak_t"], row["p_corrected"],
                    )
            else:
                logger.info("No significant clusters at p<0.05")

        # Save full results dict for reuse
        results_pkl = {
            "band_cluster_results": getattr(self, "_band_cluster_results", {}),
            "feature_cluster_results": getattr(self, "_feature_cluster_results", {}),
            "source_coords": self._source_coords,
            "subject_groups": self._subject_groups,
            "config": {
                "cluster_threshold": self._cluster_threshold,
                "n_permutations": self._n_permutations,
                "adjacency_distance_mm": self._adjacency_distance,
            },
        }
        with open(data_dir / "wholebrain_results.pkl", "wb") as f:
            pickle.dump(results_pkl, f)
        logger.info("Saved wholebrain_results.pkl")

    def figures(self) -> None:
        """Generate glass brain figures."""
        if self._source_coords is None:
            logger.warning("No source coordinates — skipping figures")
            return

        coords = self._source_coords
        fig_dir = self.output_dir / "figures"
        group_labels = getattr(self, "_contrast_labels", ("Group A", "Group B"))

        # Band power figures
        band_results = getattr(self, "_band_cluster_results", {})
        for band_name, res in band_results.items():
            safe_name = band_name.lower().replace(" ", "_")
            plot_band_comparison(
                coords=coords,
                mean_a=res["mean_a"],
                mean_b=res["mean_b"],
                t_map=res["t_map"],
                cluster_labels=res["cluster_labels"],
                cluster_pvalues=res["cluster_pvalues"],
                band_name=band_name,
                group_labels=group_labels,
                output_path=fig_dir / f"wholebrain_{safe_name}.png",
            )

        # Feature figures
        feature_results = getattr(self, "_feature_cluster_results", {})
        for feat_name, res in feature_results.items():
            plot_band_comparison(
                coords=coords,
                mean_a=res["mean_a"],
                mean_b=res["mean_b"],
                t_map=res["t_map"],
                cluster_labels=res["cluster_labels"],
                cluster_pvalues=res["cluster_pvalues"],
                band_name=feat_name,
                group_labels=group_labels,
                output_path=fig_dir / f"wholebrain_{feat_name}.png",
            )

        # Summary figure
        all_results = {}
        all_results.update(band_results)
        all_results.update(feature_results)
        if all_results:
            plot_wholebrain_summary(
                band_results=all_results,
                coords=coords,
                output_path=fig_dir / "wholebrain_summary.png",
                group_labels=group_labels,
            )

    def summary(self) -> None:
        """Call R script for formatted tables and ANALYSIS_SUMMARY.md."""
        data_dir = self.output_dir / "data"

        # Write study config for R
        config_path = data_dir / "study_config.yaml"
        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Try R script
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.warning(str(e))
            self._write_python_summary()
            return

        r_script = r_dir / "wholebrain_analysis.R"
        if not r_script.exists():
            logger.warning("R script not found: %s — writing Python summary", r_script)
            self._write_python_summary()
            return

        cmd = [
            "Rscript", str(r_script),
            "--data-dir", str(data_dir),
            "--config", str(config_path),
            "--output-dir", str(self.output_dir),
        ]

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
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
                self._write_python_summary()
        except FileNotFoundError:
            logger.warning("Rscript not found — writing Python summary")
            self._write_python_summary()
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 600 seconds")
            self._write_python_summary()

    def _write_python_summary(self) -> None:
        """Fallback summary when R is not available."""
        tbl_dir = self.output_dir / "tables"
        cluster_csv = tbl_dir / "cluster_results.csv"

        lines = [
            "# Whole-Brain Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            f"**Analysis**: Vertex-level spectral analysis with cluster permutation testing",
            f"**Permutations**: {self._n_permutations}",
            f"**Cluster threshold**: t = {self._cluster_threshold}",
            f"**Adjacency distance**: {self._adjacency_distance} mm",
            "",
            "## Methods",
            "",
            "Power spectral density was computed for each source vertex using Welch's method "
            "(2-second windows, 50% overlap). Band power (absolute and relative), fALFF, "
            "spectral slope, and peak alpha frequency were extracted per vertex. "
            "Group differences were tested using independent-samples t-tests at each vertex, "
            "with cluster-based permutation correction for multiple comparisons "
            "(Maris & Oostenveld, 2007).",
            "",
        ]

        if cluster_csv.exists():
            clust_df = pd.read_csv(cluster_csv)
            sig = clust_df[clust_df["p_corrected"] < 0.05]

            lines.append("## Results")
            lines.append("")

            if len(sig) > 0:
                lines.append(f"**{len(sig)} significant clusters** (p < 0.05):")
                lines.append("")
                lines.append("| Band/Metric | Metric | Vertices | Peak t | p_corrected |")
                lines.append("|-------------|--------|----------|--------|-------------|")
                for _, row in sig.iterrows():
                    lines.append(
                        f"| {row['band']} | {row['metric']} | {row['n_vertices']} | "
                        f"{row['peak_t']:.2f} | {row['p_corrected']:.4f} |"
                    )
                lines.append("")
            else:
                lines.append("No significant clusters at p < 0.05.")
                lines.append("")

            lines.append(f"Total clusters tested: {len(clust_df)}")
            lines.append("")

        lines.append("## Output Files")
        lines.append("")
        lines.append("- `data/wholebrain_values.csv` — per-subject per-vertex band power")
        lines.append("- `data/wholebrain_features.csv` — per-subject per-vertex fALFF, slope, peak alpha")
        lines.append("- `data/source_coords.csv` — vertex coordinates (mm)")
        lines.append("- `tables/voxelwise_stats.csv` — per-vertex t, p, Hedges' g")
        lines.append("- `tables/cluster_results.csv` — cluster summaries with corrected p-values")
        lines.append("- `figures/wholebrain_*.png` — glass brain visualizations")
        lines.append("")

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
