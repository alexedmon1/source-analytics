"""Electrode (sensor-level) neural-signature classification analysis.

The sensor-space counterpart of ``vertex_signature``: it runs the SAME
classifiers (LOOCV + permutation testing) on per-electrode band power instead of
per-vertex source power, so the two can be compared directly — does source
localization buy us predictability over the raw sensor montage? When the sibling
``vertex_signature`` results are present, a source-vs-sensor accuracy comparison
(per contrast × band × classifier) is emitted.

Reuses the level-agnostic ``stats/signature.py`` machinery unchanged.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.electrode_loader import load_eeglab_set
from ..spectral.epoch_sampler import sample_epochs
from ..spectral.psd import compute_psd
from ..spectral.band_power import extract_band_power, relative_power_kwargs
from ..stats.signature import (
    SignatureResult,
    classifier_label,
    normalize_classifier,
    run_signature,
)
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


class ElectrodeSignatureAnalysis(BaseAnalysis):
    """Sensor-level whole-montage neural-signature (classification) analysis."""

    name = "electrode_signature"
    SELECTABLE = {"band": "frequency band"}

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._sfreq: float | None = None
        self._roster: pd.DataFrame | None = None
        self._ch_names: list[str] | None = None
        self._ch_coords: np.ndarray | None = None
        self._feature_rows: list[dict] = []
        # subject uid -> {band -> {channel -> relative power}}
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}
        self._subject_order: list[str] = []

        # Classifiers: `classifiers:` (list) or `classifier:` (scalar), normalised
        # and deduped in config order — same contract as vertex_signature.
        sig_cfg = config.raw.get("electrode_signature", {})
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

        self._signature_results: dict[str, object] = {}

    def setup(self) -> None:
        self._feature_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._subject_order.clear()
        self._ch_names = None
        self._ch_coords = None
        self._signature_results.clear()

        roster_path = self.config.electrode.get("subject_roster")
        if not roster_path:
            raise ValueError(
                "electrode.subject_roster not set in study config. "
                "Add 'electrode: {subject_roster: /path/to/subject_roster.csv}'."
            )
        roster_path = Path(roster_path)
        if not roster_path.exists():
            raise FileNotFoundError(f"Subject roster not found: {roster_path}")
        self._roster = pd.read_csv(roster_path)
        required_cols = {"subject_id", "eeg_filename", "eeg_dir"}
        missing = required_cols - set(self._roster.columns)
        if missing:
            raise ValueError(f"Subject roster missing columns: {missing}")

    def _find_eeg_path(self, subject: SubjectInfo) -> Path | None:
        roster_group = subject.pipeline_dir.parent.name
        matches = self._roster[
            (self._roster["subject_id"] == subject.subject_id)
            & (self._roster["group"] == roster_group)
        ]
        if matches.empty:
            matches = self._roster[self._roster["subject_id"] == subject.subject_id]
        if matches.empty:
            logger.warning("Subject %s not in roster; skipping", subject.subject_id)
            return None
        row = matches.iloc[0]
        eeg_path = Path(row["eeg_dir"]) / row["eeg_filename"]
        if not eeg_path.exists():
            logger.warning("Raw EEG file not found: %s", eeg_path)
            return None
        return eeg_path

    def _get_draws(self, data: np.ndarray, sfreq: float) -> list[np.ndarray]:
        """Epoch-sample raw electrode data → list of (n_channels, n_samples)."""
        if not self._epoch_equalize or self._epoch_n_bootstrap <= 0:
            return [data]
        n_boot = max(1, self._epoch_n_bootstrap)
        rng = np.random.default_rng(self._epoch_seed)
        seeds = ([self._epoch_seed] if n_boot == 1
                 else list(rng.integers(0, 2**31, size=n_boot)))
        draws: list[np.ndarray] = []
        for s in seeds:
            epochs = sample_epochs(
                data, sfreq,
                epoch_duration_sec=self._epoch_duration_sec,
                n_epochs=self._epoch_n_epochs,
                seed=(int(s) if s is not None else None),
            )
            n_ep, n_ch, ep_len = epochs.shape
            draws.append(epochs.transpose(1, 0, 2).reshape(n_ch, n_ep * ep_len))
        return draws

    def process_subject(self, subject: SubjectInfo) -> None:
        eeg_path = self._find_eeg_path(subject)
        if eeg_path is None:
            return

        target_sfreq = (
            self.config.electrode.get("target_sfreq")
            or self.config.raw.get(self.name, {}).get("target_sfreq")
        )
        data, sfreq, ch_names, ch_coords = load_eeglab_set(
            eeg_path, target_sfreq=target_sfreq)

        if self._sfreq is None:
            self._sfreq = sfreq
        if self._ch_names is None:
            self._ch_names = list(ch_names)
        if self._ch_coords is None and ch_coords is not None:
            self._ch_coords = ch_coords

        uid = f"{subject.group}_{subject.subject_id}"
        fmax = max(hi for _, hi in self.config.bands.values()) + 10

        draws = self._get_draws(data, sfreq)

        # Per (channel, band): mean relative power across draws.
        ch_bp: dict[tuple[str, str], list[float]] = {}
        for draw_data in draws:
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = draw_data[ch_idx, :]
                if np.all(ch_data == 0) or np.any(np.isnan(ch_data)):
                    continue
                freqs, psd = compute_psd(ch_data, sfreq, fmax=fmax)
                bp = extract_band_power(
                    freqs, psd, self._selected_bands(),
                    **relative_power_kwargs(self.config.raw.get("relative_power")))
                for band_name, vals in bp.items():
                    ch_bp.setdefault((ch_name, band_name), []).append(vals["relative"])

        # Collapse to a band -> {channel -> relative} map for the feature matrix.
        band_power: dict[str, dict[str, float]] = {}
        for (ch_name, band_name), vals in ch_bp.items():
            band_power.setdefault(band_name, {})[ch_name] = float(np.mean(vals))
            self._feature_rows.append({
                "subject": uid,
                "group": subject.group,
                "channel": ch_name,
                "band": band_name,
                "relative": float(np.mean(vals)),
            })

        self._subject_groups[uid] = subject.group
        self._subject_order.append(uid)
        self._subject_data[uid] = {"band_power": band_power}

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        feat_df = pd.DataFrame(self._feature_rows)
        if feat_df.empty:
            logger.warning("No electrode signature feature data collected")
            return
        feat_df.to_csv(data_dir / "electrode_signature_features.csv", index=False)
        logger.info("Exported electrode_signature_features.csv (%d rows)", len(feat_df))

        # Persist the montage layout (channel, x, y, z) for topomap rendering —
        # the sensor analog of source_coords.csv.
        if self._ch_names is not None and self._ch_coords is not None:
            layout = pd.DataFrame(self._ch_coords, columns=["x", "y", "z"])
            layout.insert(0, "channel", self._ch_names)
            layout.to_csv(data_dir / "electrode_layout.csv", index=False)

    def statistics(self) -> None:
        if not self._subject_data:
            logger.error("No subject data for electrode signature analysis")
            return

        tbl_dir = self.tbl_dir
        # Stable channel order (from the first subject's collected features).
        channels = self._ch_names or sorted(
            {ch for d in self._subject_data.values()
             for bm in d["band_power"].values() for ch in bm})

        all_results = []
        for contrast in self._pairwise_contrasts():
            a_uids = [u for u in self._subject_order
                      if self._subject_groups[u] == contrast.group_a]
            b_uids = [u for u in self._subject_order
                      if self._subject_groups[u] == contrast.group_b]
            if not a_uids or not b_uids:
                continue
            ordered = a_uids + b_uids
            labels = np.array([0] * len(a_uids) + [1] * len(b_uids))

            for band_name in self._selected_bands():
                # Feature matrix (n_subjects, n_channels); NaN for missing channels.
                features = np.array([
                    [self._subject_data[u]["band_power"].get(band_name, {}).get(ch, np.nan)
                     for ch in channels]
                    for u in ordered
                ])
                # Drop channels missing for any subject (classifier needs complete cols).
                good = ~np.any(np.isnan(features), axis=0)
                feats = features[:, good]
                if feats.shape[1] == 0:
                    continue

                for clf in self._classifiers:
                    result = run_signature(
                        feats, labels, classifier=clf,
                        cv_method=self._cv_method,
                        n_permutations=self._n_permutations, seed=42)
                    # Re-expand weights back to the full channel vector (NaN for dropped).
                    full_w = np.full(len(channels), np.nan)
                    if result.has_weights:
                        full_w[good] = result.feature_weights
                    result.feature_weights = full_w

                    key = f"{contrast.name}_{band_name}_{clf}"
                    self._signature_results[key] = result
                    all_results.append({
                        "contrast": contrast.name, "band": band_name,
                        "classifier": clf, "model": classifier_label(clf),
                        "accuracy": result.accuracy, "p_value": result.p_value,
                        "balanced_accuracy": result.balanced_accuracy,
                        "balanced_p_value": result.balanced_p_value,
                        "sensitivity": result.sensitivity,
                        "specificity": result.specificity, "auc": result.auc,
                        "ci_lower": result.accuracy_ci[0],
                        "ci_upper": result.accuracy_ci[1],
                        "balanced_ci_lower": result.balanced_accuracy_ci[0],
                        "balanced_ci_upper": result.balanced_accuracy_ci[1],
                        "n_permutations": result.n_permutations,
                    })

        if all_results:
            pd.DataFrame(all_results).to_csv(
                tbl_dir / "electrode_signature_results.csv", index=False)
            logger.info("Exported electrode_signature_results.csv (%d rows)", len(all_results))

        if self._signature_results:
            data_dir = self.output_dir / "data"
            pkl_data = {}
            for key, r in self._signature_results.items():
                pkl_data[key] = {
                    "feature_weights": r.feature_weights,
                    "null_distribution": r.null_distribution,
                    "predictions": r.predictions, "true_labels": r.true_labels,
                    "accuracy": r.accuracy, "p_value": r.p_value,
                    "sensitivity": r.sensitivity, "specificity": r.specificity,
                    "auc": r.auc, "accuracy_ci": r.accuracy_ci,
                    "balanced_accuracy": r.balanced_accuracy,
                    "balanced_p_value": r.balanced_p_value,
                    "balanced_accuracy_ci": r.balanced_accuracy_ci,
                    "n_permutations": r.n_permutations,
                    "classifier": r.classifier, "has_weights": r.has_weights,
                }
            with open(data_dir / "electrode_signature_results.pkl", "wb") as f:
                pickle.dump(pkl_data, f)

    def _load_state_from_disk(self) -> bool:
        data_dir = self.output_dir / "data"
        pkl_path = data_dir / "electrode_signature_results.pkl"
        if not pkl_path.exists():
            logger.warning("No saved electrode signature state at %s; skipping figures", pkl_path)
            return False
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        for key, d in saved.items():
            self._signature_results[key] = SignatureResult(
                accuracy=d["accuracy"], p_value=d["p_value"],
                sensitivity=d["sensitivity"], specificity=d["specificity"],
                auc=d["auc"], accuracy_ci=tuple(d["accuracy_ci"]),
                feature_weights=d["feature_weights"],
                null_distribution=d["null_distribution"],
                predictions=d["predictions"], true_labels=d["true_labels"],
                n_permutations=d["n_permutations"],
                classifier=d.get("classifier", "svm_linear"),
                has_weights=d.get("has_weights", True),
                # .get() so a pkl written before the balanced-metric change still loads.
                balanced_accuracy=d.get("balanced_accuracy", float("nan")),
                balanced_p_value=d.get("balanced_p_value", float("nan")),
                balanced_accuracy_ci=tuple(
                    d.get("balanced_accuracy_ci", (float("nan"), float("nan")))))
        layout_csv = data_dir / "electrode_layout.csv"
        if layout_csv.exists():
            lay = pd.read_csv(layout_csv)
            self._ch_names = lay["channel"].tolist()
            self._ch_coords = lay[["x", "y", "z"]].values
        return True

    def _plot_importance_topomap(self, values: np.ndarray, title: str, out_path: Path) -> None:
        """Lightweight sensor importance map: channels at their (x, y) montage
        positions, colored by |weight|. Portable (no montage library); the
        FORGE mouse-silhouette topomap is an optional downstream polish."""
        if self._ch_coords is None:
            return
        xy = self._ch_coords[:, :2]
        ok = ~np.any(np.isnan(xy), axis=1) & ~np.isnan(values)
        if not np.any(ok):
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=values[ok], s=260,
                        cmap="YlOrRd", edgecolors="black", linewidths=0.6)
        if self._ch_names is not None:
            for i in np.where(ok)[0]:
                ax.annotate(self._ch_names[i], (xy[i, 0], xy[i, 1]),
                            ha="center", va="center", fontsize=6)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, label="|weight|", shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def figures(self) -> None:
        if not self._signature_results:
            if not self._load_state_from_disk():
                return
        fig_dir = self.fig_dir

        for key, result in self._signature_results.items():
            safe = key.lower().replace(" ", "_")

            if getattr(result, "has_weights", True) and not np.all(np.isnan(result.feature_weights)):
                self._plot_importance_topomap(
                    result.feature_weights, f"Feature Importance — {key}",
                    fig_dir / f"electrode_signature_importance_{safe}.png")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(result.null_distribution, bins=30, color="#3498DB",
                    alpha=0.7, edgecolor="white", label="Null distribution")
            ax.axvline(result.accuracy, color="#E74C3C", linewidth=2,
                       linestyle="--", label=f"Observed: {result.accuracy:.1%}")
            ax.set_xlabel("Accuracy"); ax.set_ylabel("Count")
            ax.set_title(f"Signature Permutation Test — {key}")
            ax.legend(); fig.tight_layout()
            fig.savefig(fig_dir / f"electrode_signature_null_{safe}.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 4))
            preds, true = result.predictions, result.true_labels
            cm = np.array([[((true == 0) & (preds == 0)).sum(), ((true == 0) & (preds == 1)).sum()],
                           [((true == 1) & (preds == 0)).sum(), ((true == 1) & (preds == 1)).sum()]])
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(f"Confusion Matrix — {key}")
            fig.tight_layout()
            fig.savefig(fig_dir / f"electrode_signature_confusion_{safe}.png", dpi=150)
            plt.close(fig)

        self._render_source_vs_sensor(fig_dir)

    def _find_vertex_results(self) -> Path | None:
        """Locate the sibling vertex_signature results table (best-effort)."""
        # tbl_dir = .../results/tables/<paradigm>/electrode_signature
        tables_root = self.tbl_dir.parent.parent
        for cand in tables_root.glob("**/vertex_signature_results.csv"):
            return cand
        return None

    def _render_source_vs_sensor(self, fig_dir: Path) -> None:
        """Source (vertex) vs sensor (electrode) decoding accuracy per
        contrast × band × classifier — the headline comparison."""
        sensor_csv = self.tbl_dir / "electrode_signature_results.csv"
        vertex_csv = self._find_vertex_results()
        if not sensor_csv.exists() or vertex_csv is None:
            logger.info("Source-vs-sensor comparison skipped (need both "
                        "vertex_signature and electrode_signature results).")
            return
        sensor = pd.read_csv(sensor_csv)
        source = pd.read_csv(vertex_csv)
        keys = ["contrast", "band", "classifier"]
        if not all(k in source.columns for k in keys):
            return
        merged = source.merge(sensor, on=keys, suffixes=("_source", "_sensor"))
        if merged.empty:
            return
        merged["accuracy_gain"] = merged["accuracy_source"] - merged["accuracy_sensor"]
        for m in ("balanced_accuracy", "auc"):
            cols = (f"{m}_source", f"{m}_sensor")
            if all(c in merged.columns for c in cols):
                merged[f"{m}_gain"] = merged[cols[0]] - merged[cols[1]]
        # A cell is only INFORMATIVE about the two modalities if at least one of
        # them actually decodes; contrasts where both sit at chance contribute
        # noise, and averaging them into a headline gain hides real differences.
        if {"p_value_source", "p_value_sensor"} <= set(merged.columns):
            merged["either_significant"] = (
                (merged["p_value_source"] < 0.05) | (merged["p_value_sensor"] < 0.05))
        merged.to_csv(self.tbl_dir / "signature_source_vs_sensor.csv", index=False)

        # Per-contrast breakdown — the global mean is dominated by underpowered
        # treated-vs-treated contrasts, so report the split explicitly.
        gain_cols = [c for c in merged.columns if c.endswith("_gain")]
        by_contrast = merged.groupby("contrast")[gain_cols].mean().round(4)
        by_contrast.to_csv(self.tbl_dir / "signature_source_vs_sensor_by_contrast.csv")

        # One panel per classifier. Plot BALANCED accuracy when available (the
        # unequal-n contrasts inflate raw accuracy), and mark the cells where at
        # least one modality reached significance.
        metric = ("balanced_accuracy"
                  if "balanced_accuracy_source" in merged.columns else "accuracy")
        mlabel = "Balanced accuracy" if metric == "balanced_accuracy" else "Accuracy"
        for clf in sorted(merged["classifier"].unique()):
            sub = merged[merged["classifier"] == clf]
            fig, ax = plt.subplots(figsize=(6, 6))
            sig = sub.get("either_significant")
            if sig is not None:
                ax.scatter(sub.loc[~sig, f"{metric}_sensor"], sub.loc[~sig, f"{metric}_source"],
                           c="#BDC3C7", s=45, alpha=0.7, edgecolors="white",
                           label="neither significant")
                ax.scatter(sub.loc[sig, f"{metric}_sensor"], sub.loc[sig, f"{metric}_source"],
                           c="#8E44AD", s=70, alpha=0.9, edgecolors="white",
                           label="≥1 significant")
                ax.legend(fontsize=8, loc="lower right", frameon=False)
            else:
                ax.scatter(sub[f"{metric}_sensor"], sub[f"{metric}_source"],
                           c="#8E44AD", s=60, alpha=0.8, edgecolors="white")
            lo, hi = 0.3, 1.0
            ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            ax.set_xlabel(f"Sensor {mlabel.lower()} (electrode)")
            ax.set_ylabel(f"Source {mlabel.lower()} (vertex)")
            ax.set_title(f"Source vs Sensor decoding — {classifier_label(clf)}\n"
                         "(above line = source localization gains predictability)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"signature_source_vs_sensor_{clf}.png", dpi=150)
            plt.close(fig)
        logger.info("Rendered source-vs-sensor comparison (%d matched cells)", len(merged))

    def summary(self) -> None:
        tbl_dir = self.tbl_dir
        models = ", ".join(classifier_label(c) for c in self._classifiers)
        lines = [
            "# Electrode Neural Signature Analysis Summary", "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Sensor-level (electrode) neural signature (classification)",
            f"**Classifiers**: {models}",
            f"**CV method**: {self._cv_method}",
            f"**Permutations**: {self._n_permutations}", "",
            "## Methods", "",
            "Each classifier, with LOOCV, was trained to distinguish groups from the "
            "spatial pattern of per-electrode relative band power. Significance was "
            "assessed by permutation testing. This is the sensor-space counterpart of "
            "the vertex (source-localized) neural signature; the "
            "`signature_source_vs_sensor` table/figure compare the two.", "",
        ]
        results_csv = tbl_dir / "electrode_signature_results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            has_model = "model" in df.columns
            lines += ["## Results", "",
                      "| Model | Band | Accuracy | p-value | Sensitivity | Specificity | AUC |",
                      "|-------|------|----------|---------|-------------|-------------|-----|"]
            for _, r in df.iterrows():
                model = r["model"] if has_model else "—"
                lines.append(
                    f"| {model} | {r['band']} | {r['accuracy']:.1%} | {r['p_value']:.4f} | "
                    f"{r['sensitivity']:.1%} | {r['specificity']:.1%} | {r['auc']:.3f} |")
            lines.append("")
        (self.output_dir / "ANALYSIS_SUMMARY.md").write_text("\n".join(lines))
        logger.info("Wrote %s/ANALYSIS_SUMMARY.md", self.output_dir)
