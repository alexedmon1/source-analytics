"""MVPA (Multivariate Pattern Analysis) whole-brain analysis.

Uses linear SVM + LOOCV to classify groups based on whole-brain spatial
patterns of band power. Provides a single omnibus test per band: can the
spatial pattern of activity distinguish KO from WT?
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.vertex import compute_psd_vertices, extract_band_power_vertices
from ..spectral.epoch_sampler import sample_epochs, get_epoch_config
from ..stats.mvpa import run_mvpa
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


class MVPAAnalysis(BaseAnalysis):
    """Whole-brain MVPA classification analysis."""

    name = "mvpa"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._feature_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._sfreq: float | None = None
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}
        self._subject_order: list[str] = []

        # Config
        mvpa_cfg = config.raw.get("mvpa", {})
        self._classifier = mvpa_cfg.get("classifier", "svm_linear")
        self._cv_method = mvpa_cfg.get("cv_method", "loocv")
        self._n_permutations = int(mvpa_cfg.get("n_permutations", 1000))

        wb_cfg = config.wholebrain
        self._noise_exclude = wb_cfg.get("noise_exclude_hz")
        if self._noise_exclude is not None:
            self._noise_exclude = tuple(self._noise_exclude)

        self._epoch_config = get_epoch_config(wb_cfg)
        self._mvpa_results: dict[str, object] = {}

    def setup(self) -> None:
        self._feature_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._subject_order.clear()
        self._source_coords = None
        self._mvpa_results.clear()

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
        fmax = max(hi for _, hi in self.config.bands.values()) + 10
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

        band_power = extract_band_power_vertices(
            freqs, psd, self.config.bands, noise_exclude=self._noise_exclude,
        )

        self._subject_groups[uid] = subject.group
        self._subject_order.append(uid)
        self._subject_data[uid] = {"band_power": band_power}

        n_vertices = stc_data.shape[0]
        for band_name, bp in band_power.items():
            for vi in range(n_vertices):
                self._feature_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "vertex_idx": vi,
                    "band": band_name,
                    "relative": float(bp["relative"][vi]),
                })

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        feat_df = pd.DataFrame(self._feature_rows)
        if feat_df.empty:
            logger.warning("No MVPA feature data collected")
            return
        feat_df.to_csv(data_dir / "mvpa_features.csv", index=False)
        logger.info("Exported mvpa_features.csv (%d rows)", len(feat_df))

        if self._source_coords is not None:
            coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

    def statistics(self) -> None:
        if not self._subject_data:
            logger.error("No subject data for MVPA")
            return

        tbl_dir = self.output_dir / "tables"
        all_results = []

        for contrast in self.config.contrasts:
            group_a_uids = [
                uid for uid in self._subject_order
                if self._subject_groups[uid] == contrast.group_a
            ]
            group_b_uids = [
                uid for uid in self._subject_order
                if self._subject_groups[uid] == contrast.group_b
            ]

            if not group_a_uids or not group_b_uids:
                continue

            ordered_uids = group_a_uids + group_b_uids
            labels = np.array(
                [0] * len(group_a_uids) + [1] * len(group_b_uids)
            )

            for band_name in self.config.bands:
                # Build feature matrix: (n_subjects, n_vertices)
                features = np.array([
                    self._subject_data[uid]["band_power"][band_name]["relative"]
                    for uid in ordered_uids
                ])

                result = run_mvpa(
                    features, labels,
                    classifier=self._classifier,
                    cv_method=self._cv_method,
                    n_permutations=self._n_permutations,
                    seed=42,
                )

                self._mvpa_results[f"{contrast.name}_{band_name}"] = result

                all_results.append({
                    "contrast": contrast.name,
                    "band": band_name,
                    "accuracy": result.accuracy,
                    "p_value": result.p_value,
                    "sensitivity": result.sensitivity,
                    "specificity": result.specificity,
                    "auc": result.auc,
                    "ci_lower": result.accuracy_ci[0],
                    "ci_upper": result.accuracy_ci[1],
                    "n_permutations": result.n_permutations,
                })

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(tbl_dir / "mvpa_results.csv", index=False)
            logger.info("Exported mvpa_results.csv")

    def figures(self) -> None:
        if self._source_coords is None:
            return

        coords = self._source_coords
        fig_dir = self.output_dir / "figures"

        for key, result in self._mvpa_results.items():
            safe_name = key.lower().replace(" ", "_")

            # Feature importance glass brain
            plot_glass_brain(
                coords=coords,
                values=result.feature_weights,
                title=f"Feature Importance — {key}",
                output_path=fig_dir / f"mvpa_importance_{safe_name}.png",
                cmap="YlOrRd",
            )

            # Null distribution histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(result.null_distribution, bins=30, color="#3498DB",
                    alpha=0.7, edgecolor="white", label="Null distribution")
            ax.axvline(result.accuracy, color="#E74C3C", linewidth=2,
                       linestyle="--", label=f"Observed: {result.accuracy:.1%}")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Count")
            ax.set_title(f"MVPA Permutation Test — {key}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"mvpa_null_{safe_name}.png", dpi=150)
            plt.close(fig)

            # Confusion matrix
            fig, ax = plt.subplots(figsize=(5, 4))
            preds = result.predictions
            true = result.true_labels
            cm = np.array([
                [(true == 0) & (preds == 0), (true == 0) & (preds == 1)],
                [(true == 1) & (preds == 0), (true == 1) & (preds == 1)],
            ])
            cm_counts = np.array([[s.sum() for s in row] for row in cm])
            ax.imshow(cm_counts, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm_counts[i, j]),
                            ha="center", va="center", fontsize=16)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(f"Confusion Matrix — {key}")
            fig.tight_layout()
            fig.savefig(fig_dir / f"mvpa_confusion_{safe_name}.png", dpi=150)
            plt.close(fig)

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
            r_script = r_dir / "mvpa_analysis.R"
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
            "# MVPA Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Multivariate Pattern Analysis (MVPA)",
            f"**Classifier**: {self._classifier}",
            f"**CV method**: {self._cv_method}",
            f"**Permutations**: {self._n_permutations}",
            "",
            "## Methods",
            "",
            "Linear SVM with Leave-One-Out Cross-Validation (LOOCV) was used to classify "
            "groups based on the spatial pattern of vertex-level relative band power. "
            "Statistical significance was assessed via permutation testing: group labels "
            "were shuffled and LOOCV accuracy recomputed to build a null distribution. "
            "Feature importance was derived from SVM coefficients averaged across folds.",
            "",
        ]

        if self._epoch_config is not None:
            lines.append(
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )
            lines.append("")

        results_csv = tbl_dir / "mvpa_results.csv"
        if results_csv.exists():
            results_df = pd.read_csv(results_csv)
            lines.append("## Results")
            lines.append("")
            lines.append(
                "| Band | Accuracy | p-value | Sensitivity | Specificity | AUC | 95% CI |"
            )
            lines.append(
                "|------|----------|---------|-------------|-------------|-----|--------|"
            )
            for _, row in results_df.iterrows():
                lines.append(
                    f"| {row['band']} | {row['accuracy']:.1%} | {row['p_value']:.4f} | "
                    f"{row['sensitivity']:.1%} | {row['specificity']:.1%} | "
                    f"{row['auc']:.3f} | [{row['ci_lower']:.1%}, {row['ci_upper']:.1%}] |"
                )
            lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "- `data/mvpa_features.csv` — feature matrix (per-subject per-vertex band power)",
            "- `tables/mvpa_results.csv` — classification results per band",
            "- `figures/mvpa_importance_*.png` — feature importance glass brains",
            "- `figures/mvpa_null_*.png` — permutation null distribution histograms",
            "- `figures/mvpa_confusion_*.png` — confusion matrices",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
