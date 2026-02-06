"""TFCE (Threshold-Free Cluster Enhancement) whole-brain analysis.

Uses TFCE instead of a fixed cluster-forming threshold for more sensitive
detection of diffuse effects. Computes vertex-level PSD, extracts band power,
then runs TFCE permutation testing per band and metric.
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
from ..io.loader import SubjectLoader
from ..spectral.vertex import compute_psd_vertices, extract_band_power_vertices
from ..spectral.epoch_sampler import sample_epochs, get_epoch_config
from ..stats.tfce import tfce_permutation_test
from ..viz.glass_brain import plot_glass_brain
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


class TFCEAnalysis(BaseAnalysis):
    """Whole-brain TFCE analysis with permutation testing."""

    name = "tfce"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._band_power_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._sfreq: float | None = None
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}

        # TFCE config
        tfce_cfg = config.raw.get("tfce", {})
        self._n_permutations = int(tfce_cfg.get("n_permutations", 1000))
        self._E = float(tfce_cfg.get("E", 0.5))
        self._H = float(tfce_cfg.get("H", 2.0))
        self._dh = float(tfce_cfg.get("dh", 0.1))
        self._adjacency_distance = float(tfce_cfg.get("adjacency_distance_mm", 5.0))

        # Noise exclusion from wholebrain config
        wb_cfg = config.wholebrain
        self._noise_exclude = wb_cfg.get("noise_exclude_hz")
        if self._noise_exclude is not None:
            self._noise_exclude = tuple(self._noise_exclude)

        # Epoch sampling
        self._epoch_config = get_epoch_config(wb_cfg)

        # Store results for figures
        self._tfce_results: dict[str, dict] = {}

    def setup(self) -> None:
        self._band_power_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._source_coords = None
        self._tfce_results.clear()

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

        # Epoch sampling if enabled
        if self._epoch_config is not None:
            epochs = sample_epochs(
                stc_data, sfreq,
                epoch_duration_sec=self._epoch_config.get("epoch_duration_sec", 2.0),
                n_epochs=self._epoch_config.get("n_epochs", 80),
                seed=self._epoch_config.get("seed", 42),
            )
            # Average PSD across epochs
            fmax = max(hi for _, hi in self.config.bands.values()) + 10
            all_psd = []
            for ep in epochs:
                f, p = compute_psd_vertices(ep, sfreq, fmax=fmax)
                all_psd.append(p)
            freqs = f
            psd = np.mean(all_psd, axis=0)
        else:
            fmax = max(hi for _, hi in self.config.bands.values()) + 10
            freqs, psd = compute_psd_vertices(stc_data, sfreq, fmax=fmax)

        band_power = extract_band_power_vertices(
            freqs, psd, self.config.bands, noise_exclude=self._noise_exclude,
        )

        self._subject_groups[uid] = subject.group
        self._subject_data[uid] = {"band_power": band_power}

        n_vertices = stc_data.shape[0]
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

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        band_df = pd.DataFrame(self._band_power_rows)
        if band_df.empty:
            logger.warning("No TFCE data collected")
            return
        band_df.to_csv(data_dir / "tfce_band_power.csv", index=False)
        logger.info("Exported tfce_band_power.csv (%d rows)", len(band_df))

        if self._source_coords is not None:
            coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

    def statistics(self) -> None:
        if self._source_coords is None:
            logger.error("No source coordinates — cannot run TFCE")
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
                logger.warning("Contrast %s: missing subjects", contrast.name)
                continue

            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)
            logger.info(
                "TFCE contrast '%s': %s (n=%d) vs %s (n=%d)",
                contrast.name, label_a, len(group_a_uids), label_b, len(group_b_uids),
            )

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

                    result = tfce_permutation_test(
                        data_a, data_b, coords,
                        n_perms=self._n_permutations,
                        E=self._E, H=self._H, dh=self._dh,
                        distance_mm=self._adjacency_distance,
                        seed=42,
                    )

                    key = f"{contrast.name}_{band_name}_{metric}"
                    self._tfce_results[key] = {
                        "band": band_name,
                        "metric": metric,
                        "contrast": contrast.name,
                        "result": result,
                        "group_labels": (label_a, label_b),
                        "mean_a": data_a.mean(axis=0),
                        "mean_b": data_b.mean(axis=0),
                    }

                    for vi in range(len(result.t_map)):
                        all_stats.append({
                            "contrast": contrast.name,
                            "band": band_name,
                            "metric": metric,
                            "vertex_idx": vi,
                            "t": float(result.t_map[vi]),
                            "tfce_score": float(result.tfce_scores[vi]),
                            "p_corrected": float(result.p_corrected[vi]),
                            "hedges_g": float(result.hedges_g_map[vi]),
                        })

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df.to_csv(tbl_dir / "tfce_stats.csv", index=False)
            logger.info("Exported tfce_stats.csv (%d rows)", len(stats_df))

    def figures(self) -> None:
        if self._source_coords is None:
            return

        coords = self._source_coords
        fig_dir = self.output_dir / "figures"

        for key, info in self._tfce_results.items():
            result = info["result"]
            band = info["band"]
            metric = info["metric"]
            safe_name = f"{band}_{metric}".lower().replace(" ", "_")

            # TFCE scores glass brain
            plot_glass_brain(
                coords=coords,
                values=result.tfce_scores,
                title=f"TFCE Scores — {band} ({metric})",
                output_path=fig_dir / f"tfce_scores_{safe_name}.png",
                cmap="RdBu_r",
            )

            # Significant vertices
            sig_mask = result.p_corrected < 0.05
            if np.any(sig_mask):
                sig_values = np.where(sig_mask, result.hedges_g_map, 0.0)
                plot_glass_brain(
                    coords=coords,
                    values=sig_values,
                    title=f"TFCE Significant (p<0.05) — {band} ({metric})",
                    output_path=fig_dir / f"tfce_significant_{safe_name}.png",
                    cmap="RdBu_r",
                )

    def summary(self) -> None:
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
            r_script = r_dir / "tfce_analysis.R"
            if r_script.exists():
                cmd = [
                    "Rscript", str(r_script),
                    "--data-dir", str(data_dir),
                    "--config", str(config_path),
                    "--output-dir", str(self.output_dir),
                ]
                logger.info("Calling R: %s", " ".join(cmd))
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.stdout:
                    for line in result.stdout.strip().split("\n"):
                        logger.info("[R] %s", line)
                if result.stderr:
                    for line in result.stderr.strip().split("\n"):
                        if line.strip():
                            logger.info("[R] %s", line)
                if result.returncode == 0:
                    return
                logger.error("R script failed (exit %d), falling back", result.returncode)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        self._write_python_summary()

    def _write_python_summary(self) -> None:
        tbl_dir = self.output_dir / "tables"
        stats_csv = tbl_dir / "tfce_stats.csv"

        lines = [
            "# TFCE Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            f"**Analysis**: Threshold-Free Cluster Enhancement (TFCE)",
            f"**Permutations**: {self._n_permutations}",
            f"**Parameters**: E={self._E}, H={self._H}, dh={self._dh}",
            f"**Adjacency distance**: {self._adjacency_distance} mm",
            "",
            "## Methods",
            "",
            "TFCE (Smith & Nichols, 2009) was applied to vertex-level band power maps. "
            "TFCE integrates cluster extent and height across all possible thresholds, "
            "eliminating the need for an arbitrary cluster-forming threshold. "
            "Statistical significance was assessed via permutation testing, "
            "comparing each vertex's |TFCE| score against a null distribution "
            "of max|TFCE| across the brain.",
            "",
        ]

        if self._epoch_config is not None:
            lines.append(
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )
            lines.append("")

        if stats_csv.exists():
            stats_df = pd.read_csv(stats_csv)
            sig = stats_df[stats_df["p_corrected"] < 0.05]

            lines.append("## Results")
            lines.append("")

            for band in stats_df["band"].unique():
                for metric in stats_df["metric"].unique():
                    subset = sig[(sig["band"] == band) & (sig["metric"] == metric)]
                    n_sig = len(subset)
                    total = len(stats_df[
                        (stats_df["band"] == band) & (stats_df["metric"] == metric)
                    ])
                    lines.append(f"- **{band}** ({metric}): {n_sig}/{total} vertices significant")

            lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "- `data/tfce_band_power.csv` — per-subject per-vertex band power",
            "- `data/source_coords.csv` — vertex coordinates (mm)",
            "- `tables/tfce_stats.csv` — per-vertex t, TFCE score, corrected p, Hedges' g",
            "- `figures/tfce_scores_*.png` — TFCE score glass brains",
            "- `figures/tfce_significant_*.png` — significant vertex glass brains",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
