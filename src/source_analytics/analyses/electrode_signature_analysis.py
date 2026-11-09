"""Electrode (sensor-level) neural-signature classification analysis.

The sensor-space counterpart of the source signatures (``roi_signature``,
``vertex_signature``): it runs the SAME classifiers (LOOCV + permutation testing)
on per-electrode band power instead of source power, so the two can be compared
directly — does source localization buy us predictability over the raw sensor
montage? When a source signature ran in the SAME paradigm, a source-vs-sensor
accuracy comparison (per contrast × band × classifier) is emitted. Only that
paradigm is searched: a search of the whole results tree once matched a stale
vertex table from an earlier study version and would have reported it as this
run's source side.

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
from ..figure_record import record_figure
from ..io.discovery import SubjectInfo
from ..io.electrode_loader import load_eeglab_set
from ..spectral.epoch_sampler import sample_epochs
from ..spectral.psd import compute_psd
from ..spectral.band_power import extract_band_power, relative_power_kwargs
from ..stats.signature import (
    SignatureResult,
    balanced_accuracy_interval,
    bh_qvalues,
    classifier_label,
    normalize_classifier,
    run_signature,
)
from ..hypothesis.power import mde_not_computed
from ..results_spec import ResultsSpec, SpecTable
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def signature_spec(features: str, unit: str) -> ResultsSpec:
    """The results specification of a signature module (electrode_signature, roi_signature):
    one row per contrast x band x classifier, balanced accuracy as the effect."""
    chance = ("the permutation null's {p} percentile of {what}: the band chance falls in, not an interval "
              "for the observed value")
    return ResultsSpec(
        domain="decoding",
        measures={"balanced_accuracy": "decoding_accuracy"},
        units={"balanced_accuracy": "proportion correct, the mean of the two groups' hit rates (0.5 = chance)"},
        input_file=None,
        method=(f"per declared two-group contrast, band and classifier, a classifier is trained on {features} "
                "with leave-one-subject-out cross-validation; its balanced accuracy is tested against "
                "label permutations (the add-one permutation p, Phipson & Smyth 2010), and the p-values are "
                "BH-FDR corrected over each contrast's bands x classifiers"),
        tables=(
            SpecTable("_results.csv", "tests",
                      "one declared two-group contrast decoded in one band by one classifier",
                      f"cross-validated classification of each contrast's two groups from {unit} band power",
                      element=False, headline=True),
        ),
        extra_columns=(
            ("contrast", {"Description": "the declared hypothesis whose two groups are classified",
                          "Standard": "contrast"}),
            ("classifier", {"Description": "the classifier", "Standard": "facet", "FacetName": "classifier"}),
            ("model", {"Description": "the classifier in words"}),
            ("balanced_accuracy", {"Description": "cross-validated balanced accuracy: the mean of the two "
                                                  "groups' hit rates (0.5 = chance)",
                                   "Standard": "effect_size", "EffectMeasure": "balanced_accuracy"}),
            ("balanced_accuracy_lcl", {"Description": "an approximate lower 95% bound of the balanced accuracy: "
                                                      "the mean of the two groups' Wilson bounds",
                                       "Standard": "effect_ci_low", "CILevel": 0.95}),
            ("balanced_accuracy_ucl", {"Description": "the approximate upper 95% bound, likewise",
                                       "Standard": "effect_ci_high", "CILevel": 0.95}),
            ("balanced_p_value", {"Description": "the permutation p of the balanced accuracy, before correction",
                                  "Standard": "p_uncorrected"}),
            ("accuracy", {"Description": "cross-validated accuracy (proportion of subjects classified right)"}),
            ("p_value", {"Description": "the permutation p of the accuracy (not corrected; the balanced "
                                        "accuracy's is the one tested)"}),
            ("sensitivity", {"Description": "group_b's hit rate"}),
            ("specificity", {"Description": "group_a's hit rate"}),
            ("auc", {"Description": "area under the ROC curve of the cross-validated decision scores"}),
            ("ci_lower", {"Description": chance.format(p="2.5th", what="accuracy")}),
            ("ci_upper", {"Description": chance.format(p="97.5th", what="accuracy")}),
            ("balanced_ci_lower", {"Description": chance.format(p="2.5th", what="balanced accuracy")}),
            ("balanced_ci_upper", {"Description": chance.format(p="97.5th", what="balanced accuracy")}),
            ("n_permutations", {"Description": "label permutations in the null distribution"}),
        ),
        effect={"measure": "balanced_accuracy", "ci_level": 0.95,
                "definition": "cross-validated balanced accuracy (0.5 = chance); its interval is approximate: "
                              "the mean of the two groups' Wilson 95% intervals"},
        caveats=(("The interval of the balanced accuracy is approximate: cross-validated predictions are not "
                  "independent, and with a few subjects per group it is wide."),
                 ("ci_lower/ci_upper and balanced_ci_lower/balanced_ci_upper are the permutation null's "
                  "2.5th-97.5th percentiles, the band chance falls in, not intervals for the observed values."),
                 ("The FDR correction is within each contrast (its bands x classifiers), not across contrasts."),
                 ("For a normalization contrast, failing to classify the two groups is not evidence that they "
                  "are equivalent.")),
    )


def _add_fdr(rows: list[dict]) -> None:
    """BH-FDR of the balanced-accuracy permutation p over each contrast's bands x classifiers:
    ``q_value``, ``significant`` (q < 0.05) and ``fdr_family``, labelled as the hypothesis
    layer labels its families, so the results specification reads them the same way."""
    import hashlib

    by_contrast: dict[str, list[dict]] = {}
    for r in rows:
        by_contrast.setdefault(r["contrast"], []).append(r)
    for contrast, members in by_contrast.items():
        q = bh_qvalues([m["balanced_p_value"] for m in members])
        keys = sorted(f"{m['band']}|{m['classifier']}" for m in members)
        digest = hashlib.md5("|".join(keys).encode()).hexdigest()[:8]
        label = (f"scope=contrast method=BH key={contrast} "
                 f"members=band_x_classifier[{len(members)}] hash={digest}")
        for m, qv in zip(members, q):
            m["q_value"] = float(qv)
            m["significant"] = bool(qv < 0.05) if not np.isnan(qv) else False
            m["fdr_family"] = label


def render_signature_comparison(
    source_csv: Path, sensor_csv: Path, tbl_dir: Path, fig_dir: Path, *,
    source_module: str,
) -> None:
    """Source vs sensor decoding accuracy per contrast × band × classifier.

    Writes ``signature_source_vs_sensor.csv`` (+ ``_by_contrast``) into ``tbl_dir``
    and one scatter per classifier into ``fig_dir``, recording which source module
    was compared. Called by the sensor module when a source signature ran in its
    paradigm, and by ``roi_signature`` when the sensor signature ran in another.
    """
    sensor = pd.read_csv(sensor_csv)
    source = pd.read_csv(source_csv)
    keys = ["contrast", "band", "classifier"]
    if not all(k in source.columns for k in keys):
        return
    merged = source.merge(sensor, on=keys, suffixes=("_source", "_sensor"))
    if merged.empty:
        return
    merged.insert(0, "source_module", source_module)
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
    merged.to_csv(tbl_dir / "signature_source_vs_sensor.csv", index=False)

    # Per-contrast breakdown — the global mean is dominated by underpowered
    # treated-vs-treated contrasts, so report the split explicitly.
    gain_cols = [c for c in merged.columns if c.endswith("_gain")]
    by_contrast = merged.groupby("contrast")[gain_cols].mean().round(4)
    by_contrast.to_csv(tbl_dir / "signature_source_vs_sensor_by_contrast.csv")

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
        ax.set_ylabel(f"Source {mlabel.lower()} ({source_module})")
        ax.set_title(f"Source vs Sensor decoding — {classifier_label(clf)}\n"
                     "(above line = source localization gains predictability)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"signature_source_vs_sensor_{clf}.png", dpi=150)
        plt.close(fig)
        record_figure(fig_dir, f"signature_source_vs_sensor_{clf}.png", measure=metric,
                      facets={"classifier": clf}, kind="other", shows="signature_source_vs_sensor.csv")
    logger.info("Rendered source-vs-sensor comparison (%d matched cells)", len(merged))


class ElectrodeSignatureAnalysis(BaseAnalysis):
    """Sensor-level whole-montage neural-signature (classification) analysis."""

    name = "electrode_signature"
    RESULTS_SPEC = signature_spec("each electrode's relative band power", "per-electrode")
    SELECTABLE = {"band": "frequency band"}

    # Source-signature modules this sensor signature pairs with, in preference
    # order. Looked for in THIS paradigm only (see _find_source_results).
    _SOURCE_SIGNATURES = ("roi_signature", "vertex_signature")

    _SUMMARY_TITLE = "Electrode Neural Signature Analysis Summary"
    _SUMMARY_ANALYSIS = "Sensor-level (electrode) neural signature (classification)"
    _SUMMARY_METHODS = (
        "Each classifier, with LOOCV, was trained to distinguish groups from the "
        "spatial pattern of per-electrode relative band power. Significance was "
        "assessed by permutation testing. This is the sensor-space counterpart of "
        "the source-localized neural signatures (roi_signature, vertex_signature); "
        "the `signature_source_vs_sensor` table/figure compare them when one ran in "
        "the same paradigm.")

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
        sig_cfg = config.raw.get(self.name, {})
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
        feat_df.to_csv(data_dir / f"{self.name}_features.csv", index=False)
        logger.info(f"Exported {self.name}_features.csv (%d rows)", len(feat_df))

        # Persist the montage layout (channel, x, y, z) for topomap rendering —
        # the sensor analog of source_coords.csv.
        if self._ch_names is not None and self._ch_coords is not None:
            layout = pd.DataFrame(self._ch_coords, columns=["x", "y", "z"])
            layout.insert(0, "channel", self._ch_names)
            layout.to_csv(data_dir / "electrode_layout.csv", index=False)

    def _load_features_from_disk(self) -> bool:
        """The per-subject features an earlier run's aggregate step saved
        (``data/<name>_features.csv``: subject, group, channel or roi, band, relative), so
        ``--steps statistics`` classifies without reprocessing the recordings."""
        path = self.output_dir / "data" / f"{self.name}_features.csv"
        if not path.exists():
            return False
        df = pd.read_csv(path)
        unit = "channel" if "channel" in df.columns else "roi"
        bands = set(self._selected_bands())
        df = df[df["band"].isin(bands)]
        for r in df.itertuples(index=False):
            uid = r.subject
            if uid not in self._subject_data:
                self._subject_data[uid] = {"band_power": {}}
                self._subject_groups[uid] = r.group
                self._subject_order.append(uid)
            self._subject_data[uid]["band_power"].setdefault(r.band, {})[getattr(r, unit)] = float(r.relative)
        order = self.output_dir / "data" / "roi_order.csv"
        if not self._ch_names:
            self._ch_names = (pd.read_csv(order)["roi"].tolist() if unit == "roi" and order.exists()
                              else list(dict.fromkeys(df[unit])))
        logger.info("Signature features from %s: %d subjects", path.name, len(self._subject_data))
        return bool(self._subject_data)

    def statistics(self) -> None:
        if not self._subject_data and not self._load_features_from_disk():
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
                    # labels: group_a 0, group_b 1, so sensitivity is group_b's hit rate
                    bal_lo, bal_hi = balanced_accuracy_interval(
                        result.sensitivity, result.specificity, len(b_uids), len(a_uids))
                    all_results.append({
                        "contrast": contrast.name, "kind": "contrast", "dv": "balanced_accuracy",
                        "group_a": contrast.group_a, "group_b": contrast.group_b,
                        "n_a": len(a_uids), "n_b": len(b_uids), "band": band_name,
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
                        "balanced_accuracy_lcl": bal_lo, "balanced_accuracy_ucl": bal_hi,
                        "n_permutations": result.n_permutations,
                        **mde_not_computed("permutation test of a cross-validated classifier"),
                    })

        if all_results:
            _add_fdr(all_results)
            pd.DataFrame(all_results).to_csv(
                tbl_dir / f"{self.name}_results.csv", index=False)
            logger.info(f"Exported {self.name}_results.csv (%d rows)", len(all_results))

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
            with open(data_dir / f"{self.name}_results.pkl", "wb") as f:
                pickle.dump(pkl_data, f)

    def _load_state_from_disk(self) -> bool:
        data_dir = self.output_dir / "data"
        pkl_path = data_dir / f"{self.name}_results.pkl"
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
        # result keys join contrast, band and classifier with "_", which the names themselves use
        parts = {f"{c.name}_{b}_{clf}": (c.name, b, clf) for c in self._pairwise_contrasts()
                 for b in self._selected_bands() for clf in self._classifiers}

        def record(out: Path, key: str, kind: str) -> None:
            if not out.exists():
                return
            contrast, band, clf = parts.get(key, (None, None, None))
            record_figure(fig_dir, out, measure="balanced_accuracy", contrast=contrast,
                          facets={"band": band, "classifier": clf} if band else None, kind=kind,
                          shows=f"{self.name}_results.csv")

        for key, result in self._signature_results.items():
            safe = key.lower().replace(" ", "_")

            if getattr(result, "has_weights", True) and not np.all(np.isnan(result.feature_weights)):
                out = fig_dir / f"{self.name}_importance_{safe}.png"
                self._plot_importance_topomap(result.feature_weights, f"Feature Importance — {key}", out)
                record(out, key, "brain")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(result.null_distribution, bins=30, color="#3498DB",
                    alpha=0.7, edgecolor="white", label="Null distribution")
            ax.axvline(result.accuracy, color="#E74C3C", linewidth=2,
                       linestyle="--", label=f"Observed: {result.accuracy:.1%}")
            ax.set_xlabel("Accuracy"); ax.set_ylabel("Count")
            ax.set_title(f"Signature Permutation Test — {key}")
            ax.legend(); fig.tight_layout()
            fig.savefig(fig_dir / f"{self.name}_null_{safe}.png", dpi=150)
            plt.close(fig)
            record(fig_dir / f"{self.name}_null_{safe}.png", key, "diagnostic")

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
            fig.savefig(fig_dir / f"{self.name}_confusion_{safe}.png", dpi=150)
            plt.close(fig)
            record(fig_dir / f"{self.name}_confusion_{safe}.png", key, "diagnostic")

        self._render_source_vs_sensor(fig_dir)

    def _find_source_results(self) -> tuple[str, Path] | None:
        """The source-signature results table in THIS paradigm, if one ran.

        ``tbl_dir`` is ``.../tables/<paradigm>/<module>``, so its siblings are the
        other modules of the same paradigm. Nothing outside it is searched: the
        previous results-tree-wide glob returned the first
        ``vertex_signature_results.csv`` anywhere, which was a stale table from an
        earlier study version.
        """
        for mod in self._SOURCE_SIGNATURES:
            cand = self.tbl_dir.parent / mod / f"{mod}_results.csv"
            if cand.exists():
                return mod, cand
        return None

    def _render_source_vs_sensor(self, fig_dir: Path) -> None:
        """Compare against the source signature of this paradigm, if one ran."""
        sensor_csv = self.tbl_dir / f"{self.name}_results.csv"
        found = self._find_source_results()
        if not sensor_csv.exists() or found is None:
            logger.info("Source-vs-sensor comparison skipped (no source signature in "
                        "this paradigm; looked for %s).", ", ".join(self._SOURCE_SIGNATURES))
            return
        render_signature_comparison(found[1], sensor_csv, self.tbl_dir, fig_dir,
                                    source_module=found[0])

    def summary(self) -> None:
        tbl_dir = self.tbl_dir
        models = ", ".join(classifier_label(c) for c in self._classifiers)
        lines = [
            f"# {self._SUMMARY_TITLE}", "",
            f"**Study**: {self.config.name}",
            f"**Analysis**: {self._SUMMARY_ANALYSIS}",
            f"**Classifiers**: {models}",
            f"**CV method**: {self._cv_method}",
            f"**Permutations**: {self._n_permutations}", "",
            "## Methods", "",
            self._SUMMARY_METHODS, "",
        ]
        results_csv = tbl_dir / f"{self.name}_results.csv"
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
