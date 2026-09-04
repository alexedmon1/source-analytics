"""Vertex neural-signature (whole-brain classification) analysis.

Classifies groups from whole-brain spatial patterns of band power with LOOCV +
permutation testing, across one or more classifiers (the interpretable linear
trio svm_linear/logistic/lda give feature-importance maps; svm_rbf is
accuracy-only). Provides an omnibus test per band × classifier: can the spatial
pattern of activity distinguish KO from WT?
"""

from __future__ import annotations

import logging
import pickle
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
from ..spectral.epoch_sampler import sample_epochs
from ..stats.signature import (
    SignatureResult,
    classifier_label,
    normalize_classifier,
    run_signature,
)
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


class VertexSignatureAnalysis(BaseAnalysis):
    """Whole-brain vertex-level neural-signature (classification) analysis."""

    name = "vertex_signature"
    SELECTABLE = {"band": "frequency band"}

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._feature_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._sfreq: float | None = None
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}
        self._subject_order: list[str] = []

        # Config. `classifiers:` (list) drives the multi-model run; `classifier:`
        # (scalar) is the single-model fallback. Names are normalised/deduped in
        # config order. (vertex_mvpa/mvpa keys kept as back-compat aliases.)
        sig_cfg = config.raw.get(
            "vertex_signature",
            config.raw.get("vertex_mvpa", config.raw.get("mvpa", {})))
        raw_clfs = sig_cfg.get("classifiers") or [sig_cfg.get("classifier", "svm_linear")]
        seen: set[str] = set()
        self._classifiers: list[str] = []
        for c in raw_clfs:
            key = normalize_classifier(c)
            if key not in seen:
                seen.add(key)
                self._classifiers.append(key)
        self._cv_method = sig_cfg.get("cv_method", "loocv")
        self._n_permutations = int(sig_cfg.get("n_permutations", 1000))

        wb_cfg = config.vertex
        self._noise_exclude = wb_cfg.get("noise_exclude_hz")
        if self._noise_exclude is not None:
            self._noise_exclude = tuple(self._noise_exclude)

        # Global epoch_sampling → vertex: block → per-analysis block (see base).
        self._epoch_config = self._vertex_epoch_config()
        self._signature_results: dict[str, object] = {}

    def setup(self) -> None:
        self._feature_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._subject_order.clear()
        self._source_coords = None
        self._signature_results.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        stc_data = loader.load_source_timecourses()
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
                n_bootstrap=self._epoch_config.get("n_bootstrap", 1),
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
            freqs, psd, self._selected_bands(), noise_exclude=self._noise_exclude,
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
            logger.warning("No signature feature data collected")
            return
        feat_df.to_csv(data_dir / "vertex_signature_features.csv", index=False)
        logger.info("Exported vertex_signature_features.csv (%d rows)", len(feat_df))

        if self._source_coords is not None:
            coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

    def statistics(self) -> None:
        if not self._subject_data:
            logger.error("No subject data for signature analysis")
            return

        tbl_dir = self.tbl_dir
        all_results = []

        for contrast in self._pairwise_contrasts():
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

            for band_name in self._selected_bands():
                # Build feature matrix: (n_subjects, n_vertices)
                features = np.array([
                    self._subject_data[uid]["band_power"][band_name]["relative"]
                    for uid in ordered_uids
                ])

                for clf in self._classifiers:
                    result = run_signature(
                        features, labels,
                        classifier=clf,
                        cv_method=self._cv_method,
                        n_permutations=self._n_permutations,
                        seed=42,
                    )

                    key = f"{contrast.name}_{band_name}_{clf}"
                    self._signature_results[key] = result

                    all_results.append({
                        "contrast": contrast.name,
                        "band": band_name,
                        "classifier": clf,
                        "model": classifier_label(clf),
                        "accuracy": result.accuracy,
                        "p_value": result.p_value,
                        "balanced_accuracy": result.balanced_accuracy,
                        "balanced_p_value": result.balanced_p_value,
                        "sensitivity": result.sensitivity,
                        "specificity": result.specificity,
                        "auc": result.auc,
                        "ci_lower": result.accuracy_ci[0],
                        "ci_upper": result.accuracy_ci[1],
                        "balanced_ci_lower": result.balanced_accuracy_ci[0],
                        "balanced_ci_upper": result.balanced_accuracy_ci[1],
                        "n_permutations": result.n_permutations,
                    })

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(tbl_dir / "vertex_signature_results.csv", index=False)
            logger.info("Exported vertex_signature_results.csv")

        # Save full results for --steps figures support
        if self._signature_results:
            data_dir = self.output_dir / "data"
            pkl_data = {}
            for key, result in self._signature_results.items():
                pkl_data[key] = {
                    "feature_weights": result.feature_weights,
                    "null_distribution": result.null_distribution,
                    "predictions": result.predictions,
                    "true_labels": result.true_labels,
                    "accuracy": result.accuracy,
                    "p_value": result.p_value,
                    "sensitivity": result.sensitivity,
                    "specificity": result.specificity,
                    "auc": result.auc,
                    "accuracy_ci": result.accuracy_ci,
                    "balanced_accuracy": result.balanced_accuracy,
                    "balanced_p_value": result.balanced_p_value,
                    "balanced_accuracy_ci": result.balanced_accuracy_ci,
                    "n_permutations": result.n_permutations,
                    "classifier": result.classifier,
                    "has_weights": result.has_weights,
                }
            with open(data_dir / "vertex_signature_results.pkl", "wb") as f:
                pickle.dump(pkl_data, f)
            logger.info("Saved vertex_signature_results.pkl")

    def _load_state_from_disk(self) -> bool:
        """Load saved signature state from pickle for --steps figures support."""
        data_dir = self.output_dir / "data"
        pkl_path = data_dir / "vertex_signature_results.pkl"
        if not pkl_path.exists():
            logger.warning("No saved signature state at %s; skipping figures", pkl_path)
            return False

        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)

        for key, d in saved.items():
            self._signature_results[key] = SignatureResult(
                accuracy=d["accuracy"],
                p_value=d["p_value"],
                sensitivity=d["sensitivity"],
                specificity=d["specificity"],
                auc=d["auc"],
                accuracy_ci=tuple(d["accuracy_ci"]),
                feature_weights=d["feature_weights"],
                null_distribution=d["null_distribution"],
                predictions=d["predictions"],
                true_labels=d["true_labels"],
                n_permutations=d["n_permutations"],
                classifier=d.get("classifier", "svm_linear"),
                has_weights=d.get("has_weights", True),
                # .get() so a pkl written before the balanced-metric change still loads.
                balanced_accuracy=d.get("balanced_accuracy", float("nan")),
                balanced_p_value=d.get("balanced_p_value", float("nan")),
                balanced_accuracy_ci=tuple(
                    d.get("balanced_accuracy_ci", (float("nan"), float("nan")))),
            )

        # Load source coords
        coords_csv = data_dir / "source_coords.csv"
        if coords_csv.exists():
            coords_df = pd.read_csv(coords_csv)
            self._source_coords = coords_df[["x", "y", "z"]].values

        logger.info("Loaded signature state from %s", pkl_path)
        return True

    def figures(self) -> None:
        # Load from disk if in-memory state is missing (--steps support)
        if not self._signature_results or self._source_coords is None:
            if not self._load_state_from_disk():
                return

        if self._source_coords is None:
            return

        coords = self._source_coords
        fig_dir = self.fig_dir

        for key, result in self._signature_results.items():
            safe_name = key.lower().replace(" ", "_")

            # Feature importance glass brain — only for linear models that expose
            # coef_ (non-linear e.g. svm_rbf has no per-vertex weight map).
            if getattr(result, "has_weights", True) and not np.all(np.isnan(result.feature_weights)):
                plot_glass_brain(
                    coords=coords,
                    values=result.feature_weights,
                    title=f"Feature Importance — {key}",
                    output_path=fig_dir / f"vertex_signature_importance_{safe_name}.png",
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
            ax.set_title(f"Signature Permutation Test — {key}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"vertex_signature_null_{safe_name}.png", dpi=150)
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
            fig.savefig(fig_dir / f"vertex_signature_confusion_{safe_name}.png", dpi=150)
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
            r_script = r_dir / "vertex_signature_analysis.R"
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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode == 0:
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        self._write_python_summary()

    def _write_python_summary(self) -> None:
        tbl_dir = self.tbl_dir

        models = ", ".join(classifier_label(c) for c in self._classifiers)
        lines = [
            "# Neural Signature Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Whole-brain vertex-level neural signature (classification)",
            f"**Classifiers**: {models}",
            f"**CV method**: {self._cv_method}",
            f"**Permutations**: {self._n_permutations}",
            "",
            "## Methods",
            "",
            "Each classifier, with Leave-One-Out Cross-Validation (LOOCV), was trained to "
            "distinguish groups from the spatial pattern of vertex-level relative band "
            "power. Significance was assessed by permutation testing: group labels were "
            "shuffled and LOOCV accuracy recomputed to build a null distribution. For "
            "linear models (SVM/logistic/LDA), feature importance is the mean |coefficient| "
            "across folds; non-linear models (RBF SVM) report accuracy only.",
            "",
        ]

        if self._epoch_config is not None:
            lines.append(
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )
            lines.append("")

        results_csv = tbl_dir / "vertex_signature_results.csv"
        if results_csv.exists():
            results_df = pd.read_csv(results_csv)
            has_model = "model" in results_df.columns
            lines.append("## Results")
            lines.append("")
            lines.append(
                "| Model | Band | Accuracy | p-value | Sensitivity | Specificity | AUC | 95% CI |"
            )
            lines.append(
                "|-------|------|----------|---------|-------------|-------------|-----|--------|"
            )
            for _, row in results_df.iterrows():
                model = row["model"] if has_model else "—"
                lines.append(
                    f"| {model} | {row['band']} | {row['accuracy']:.1%} | {row['p_value']:.4f} | "
                    f"{row['sensitivity']:.1%} | {row['specificity']:.1%} | "
                    f"{row['auc']:.3f} | [{row['ci_lower']:.1%}, {row['ci_upper']:.1%}] |"
                )
            lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "- `data/vertex_signature_features.csv` — feature matrix (per-subject per-vertex band power)",
            "- `tables/vertex_signature_results.csv` — classification results per band",
            "- `figures/vertex_signature_importance_*.png` — feature importance glass brains",
            "- `figures/vertex_signature_null_*.png` — permutation null distribution histograms",
            "- `figures/vertex_signature_confusion_*.png` — confusion matrices",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)
