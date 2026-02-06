"""Vertex-level spectral parameterization analysis.

Fits aperiodic (1/f) models at each vertex to determine whether gamma
elevation is a true oscillatory peak versus a broadband spectral shift.
Tests group differences in aperiodic exponent, offset, and gamma peak
presence using cluster permutation and chi-squared tests.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.vertex import compute_psd_vertices
from ..spectral.vertex_aperiodic import fit_aperiodic_vertices
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


class SpecparamVertexAnalysis(BaseAnalysis):
    """Vertex-level spectral parameterization analysis."""

    name = "specparam_vertex"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._param_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._sfreq: float | None = None
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}

        # Config
        sp_cfg = config.raw.get("specparam_vertex", {})
        self._freq_range = tuple(sp_cfg.get("freq_range", [1, 100]))
        self._peak_width_limits = tuple(sp_cfg.get("peak_width_limits", [1.0, 12.0]))
        self._max_n_peaks = int(sp_cfg.get("max_n_peaks", 6))

        wb_cfg = config.wholebrain
        self._n_permutations = int(wb_cfg.get("n_permutations", 1000))
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))
        self._noise_exclude = wb_cfg.get("noise_exclude_hz")
        if self._noise_exclude is not None:
            self._noise_exclude = tuple(self._noise_exclude)

        self._epoch_config = get_epoch_config(wb_cfg)
        self._cluster_results: dict = {}

    def setup(self) -> None:
        self._param_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._source_coords = None
        self._cluster_results.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        stc_data = loader.load_source_timecourses(magnitude=True)
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()

        if self._sfreq is None:
            self._sfreq = sfreq
        if self._source_coords is None:
            self._source_coords = coords

        # Compute PSD
        fmax = self._freq_range[1] + 10
        if self._epoch_config is not None:
            epochs = sample_epochs(
                stc_data, sfreq,
                epoch_duration_sec=self._epoch_config.get("epoch_duration_sec", 2.0),
                n_epochs=self._epoch_config.get("n_epochs", 80),
                seed=self._epoch_config.get("seed", 42),
            )
            all_psd = []
            for ep in epochs:
                f, p = compute_psd_vertices(ep, sfreq, fmax=fmax)
                all_psd.append(p)
            freqs = f
            psd = np.mean(all_psd, axis=0)
        else:
            freqs, psd = compute_psd_vertices(stc_data, sfreq, fmax=fmax)

        # Fit specparam at each vertex
        params = fit_aperiodic_vertices(
            freqs, psd,
            freq_range=self._freq_range,
            max_n_peaks=self._max_n_peaks,
            peak_width_limits=self._peak_width_limits,
        )

        self._subject_groups[uid] = subject.group
        self._subject_data[uid] = params

        n_vertices = psd.shape[0]
        for vi in range(n_vertices):
            self._param_rows.append({
                "subject": uid,
                "group": subject.group,
                "vertex_idx": vi,
                "exponent": float(params["exponent"][vi]),
                "offset": float(params["offset"][vi]),
                "r_squared": float(params["r_squared"][vi]),
                "n_peaks": int(params["n_peaks"][vi]),
                "has_gamma_peak": bool(params["has_gamma_peak"][vi]),
                "gamma_peak_freq": float(params["gamma_peak_freq"][vi]),
                "gamma_peak_power": float(params["gamma_peak_power"][vi]),
                "method": params["method"][vi],
            })

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        param_df = pd.DataFrame(self._param_rows)
        if param_df.empty:
            logger.warning("No specparam data collected")
            return
        param_df.to_csv(data_dir / "specparam_vertex.csv", index=False)
        logger.info("Exported specparam_vertex.csv (%d rows)", len(param_df))

        if self._source_coords is not None:
            coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

    def statistics(self) -> None:
        if self._source_coords is None:
            logger.error("No source coordinates")
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

            # Cluster permutation on exponent and offset maps
            for param_name in ["exponent", "offset"]:
                data_a = np.array([
                    self._subject_data[uid][param_name] for uid in group_a_uids
                ])
                data_b = np.array([
                    self._subject_data[uid][param_name] for uid in group_b_uids
                ])

                result = cluster_permutation_test(
                    data_a, data_b, coords,
                    n_perms=self._n_permutations,
                    threshold=self._cluster_threshold,
                    distance_mm=self._adjacency_distance,
                    seed=42,
                )

                g_map = hedges_g(data_a, data_b)

                self._cluster_results[f"{contrast.name}_{param_name}"] = {
                    "result": result,
                    "mean_a": data_a.mean(axis=0),
                    "mean_b": data_b.mean(axis=0),
                    "group_labels": (label_a, label_b),
                    "param": param_name,
                }

                for vi in range(len(result.t_map)):
                    all_stats.append({
                        "contrast": contrast.name,
                        "parameter": param_name,
                        "vertex_idx": vi,
                        "t": float(result.t_map[vi]),
                        "p": float(result.p_map[vi]),
                        "hedges_g": float(g_map[vi]),
                        "cluster_id": int(result.cluster_labels[vi]),
                    })

            # Chi-squared test on gamma peak presence
            gamma_a = np.array([
                self._subject_data[uid]["has_gamma_peak"] for uid in group_a_uids
            ])  # (n_a, n_vertices) bool
            gamma_b = np.array([
                self._subject_data[uid]["has_gamma_peak"] for uid in group_b_uids
            ])

            rate_a = gamma_a.mean(axis=0)  # fraction with gamma peak per vertex
            rate_b = gamma_b.mean(axis=0)

            # Per-vertex chi-squared test
            n_a, n_b = len(group_a_uids), len(group_b_uids)
            chi2_stats = []
            for vi in range(len(rate_a)):
                # 2x2 contingency table
                a_yes = int(gamma_a[:, vi].sum())
                a_no = n_a - a_yes
                b_yes = int(gamma_b[:, vi].sum())
                b_no = n_b - b_yes

                table = np.array([[a_yes, a_no], [b_yes, b_no]])
                if table.sum() > 0 and np.all(table.sum(axis=0) > 0):
                    chi2, p_val, _, _ = sp_stats.chi2_contingency(table, correction=True)
                else:
                    chi2, p_val = 0.0, 1.0

                chi2_stats.append({
                    "contrast": contrast.name,
                    "vertex_idx": vi,
                    "gamma_rate_a": float(rate_a[vi]),
                    "gamma_rate_b": float(rate_b[vi]),
                    "chi2": float(chi2),
                    "p": float(p_val),
                })

            if chi2_stats:
                chi2_df = pd.DataFrame(chi2_stats)
                chi2_df.to_csv(tbl_dir / "gamma_peak_chi2.csv", index=False)

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df.to_csv(tbl_dir / "specparam_vertex_stats.csv", index=False)
            logger.info("Exported specparam_vertex_stats.csv (%d rows)", len(stats_df))

    def figures(self) -> None:
        if self._source_coords is None:
            return

        coords = self._source_coords
        fig_dir = self.output_dir / "figures"

        for key, info in self._cluster_results.items():
            result = info["result"]
            param = info["param"]
            group_labels = info["group_labels"]

            plot_band_comparison(
                coords=coords,
                mean_a=info["mean_a"],
                mean_b=info["mean_b"],
                t_map=result.t_map,
                cluster_labels=result.cluster_labels,
                cluster_pvalues=result.cluster_pvalues,
                band_name=param,
                group_labels=group_labels,
                output_path=fig_dir / f"specparam_{param}.png",
            )

        # Gamma peak presence map (average across all subjects)
        if self._subject_data:
            all_gamma = np.array([
                d["has_gamma_peak"].astype(float) for d in self._subject_data.values()
            ])
            mean_gamma_rate = all_gamma.mean(axis=0)
            plot_glass_brain(
                coords=coords,
                values=mean_gamma_rate,
                title="Gamma Peak Presence Rate",
                output_path=fig_dir / "gamma_peak_presence.png",
                cmap="YlOrRd",
                vlim=(0, 1),
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
            r_script = r_dir / "specparam_vertex_analysis.R"
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
            "# Spectral Parameterization (Vertex-Level) Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Vertex-level spectral parameterization (aperiodic + peaks)",
            f"**Frequency range**: {self._freq_range[0]}-{self._freq_range[1]} Hz",
            f"**Max peaks**: {self._max_n_peaks}",
            f"**Peak width limits**: {self._peak_width_limits[0]}-{self._peak_width_limits[1]} Hz",
            "",
            "## Methods",
            "",
            "Spectral parameterization (specparam/FOOOF) was applied to the PSD at "
            "each source vertex to decompose the spectrum into aperiodic (1/f) and "
            "oscillatory components. This determines whether elevated gamma power "
            "reflects a true oscillatory peak or a broadband spectral shift. "
            "Group differences in aperiodic exponent and offset were tested using "
            "cluster-based permutation testing. Gamma peak presence rates were "
            "compared using per-vertex chi-squared tests.",
            "",
        ]

        if self._epoch_config is not None:
            lines.append(
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )
            lines.append("")

        # Specparam stats
        stats_csv = tbl_dir / "specparam_vertex_stats.csv"
        if stats_csv.exists():
            stats_df = pd.read_csv(stats_csv)
            lines.append("## Aperiodic Parameter Results")
            lines.append("")
            for param in stats_df["parameter"].unique():
                sub = stats_df[stats_df["parameter"] == param]
                n_clust = len(set(sub["cluster_id"]) - {0})
                lines.append(f"- **{param}**: {n_clust} clusters found")
            lines.append("")

        # Chi-squared results
        chi2_csv = tbl_dir / "gamma_peak_chi2.csv"
        if chi2_csv.exists():
            chi2_df = pd.read_csv(chi2_csv)
            n_sig = len(chi2_df[chi2_df["p"] < 0.05])
            lines.append("## Gamma Peak Presence")
            lines.append("")
            lines.append(
                f"- {n_sig}/{len(chi2_df)} vertices with significant "
                "group differences in gamma peak presence (uncorrected p<0.05)"
            )
            lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "- `data/specparam_vertex.csv` — per-subject per-vertex specparam parameters",
            "- `tables/specparam_vertex_stats.csv` — cluster permutation results",
            "- `tables/gamma_peak_chi2.csv` — gamma peak presence chi-squared tests",
            "- `figures/specparam_*.png` — aperiodic parameter glass brains",
            "- `figures/gamma_peak_presence.png` — gamma peak prevalence map",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
