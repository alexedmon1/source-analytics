"""PSD Analysis module: computes PSD, band power, and group statistics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.psd import compute_psd_multiroi
from ..spectral.band_power import extract_band_power_multiroi
from ..stats.parametric import ttest_between_groups, lmm_group_effect, TestResult
from ..stats.effect_size import hedges_g
from ..stats.correction import fdr_correction
from ..viz.psd_plots import (
    plot_psd_by_region,
    plot_band_power_boxplots,
    plot_regional_power_heatmap,
)
from ..viz.report import ReportWriter
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


class PSDAnalysis(BaseAnalysis):
    """Power spectral density analysis with group comparisons.

    Produces per-subject PSD, band power extraction, t-tests, LMMs,
    FDR correction, and publication-quality figures.
    """

    name = "psd"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        # Per-subject storage — keyed by (group, subject_id) to handle
        # subjects with the same name in different groups
        self._subject_psds: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._subject_band_power: list[dict] = []
        self._subject_groups: dict[str, str] = {}
        self._sfreq: float | None = None
        # Aggregated
        self._band_df: pd.DataFrame | None = None
        self._group_psd_means: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        # Statistics
        self._stats_df: pd.DataFrame | None = None

    def setup(self) -> None:
        self._subject_psds.clear()
        self._subject_band_power.clear()
        self._subject_groups.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)

        # Load ROI timeseries (magnitude for power analysis)
        roi_ts = loader.load_roi_timeseries(signed=False)
        sfreq = loader.load_sfreq()

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        # Use composite key to handle duplicate subject IDs across groups
        uid = f"{subject.group}_{subject.subject_id}"

        # Compute PSD for all ROIs
        fmax = max(hi for _, hi in self.config.bands.values()) + 10
        roi_psds = compute_psd_multiroi(roi_ts, sfreq, fmax=fmax)
        self._subject_psds[uid] = roi_psds
        self._subject_groups[uid] = subject.group

        # Extract band power
        band_power = extract_band_power_multiroi(roi_psds, self.config.bands)

        for roi_name, bp_dict in band_power.items():
            for band_name, power_vals in bp_dict.items():
                self._subject_band_power.append({
                    "subject": uid,
                    "group": subject.group,
                    "roi": roi_name,
                    "band": band_name,
                    "absolute": power_vals["absolute"],
                    "relative": power_vals["relative"],
                    "dB": power_vals["dB"],
                })

    def aggregate(self) -> None:
        # Build band power DataFrame
        self._band_df = pd.DataFrame(self._subject_band_power)

        if self._band_df.empty:
            logger.warning("No band power data collected")
            return

        # Save raw data
        self._band_df.to_csv(self.output_dir / "data" / "band_power_all.csv", index=False)

        # Compute group-mean PSDs for plotting
        for group_id in self.config.group_order:
            group_subjects = [
                sid for sid, g in self._subject_groups.items() if g == group_id
            ]
            if not group_subjects:
                continue

            roi_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            # Collect all ROI names from the first subject
            first_subj = group_subjects[0]
            roi_names = list(self._subject_psds[first_subj].keys())

            for roi_name in roi_names:
                all_psds = []
                freqs = None
                for sid in group_subjects:
                    if roi_name in self._subject_psds.get(sid, {}):
                        f, p = self._subject_psds[sid][roi_name]
                        all_psds.append(p)
                        freqs = f

                if all_psds and freqs is not None:
                    roi_means[roi_name] = (freqs, np.mean(all_psds, axis=0))

            self._group_psd_means[group_id] = roi_means

    def statistics(self) -> None:
        if self._band_df is None or self._band_df.empty:
            self._stats_df = pd.DataFrame()
            return

        rows = []

        for contrast in self.config.contrasts:
            ga = contrast.group_a
            gb = contrast.group_b

            for band_name in self.config.bands:
                band_data = self._band_df[self._band_df["band"] == band_name]
                if band_data.empty:
                    continue

                # Subject-level means (average across ROIs to avoid pseudoreplication)
                subj_means = band_data.groupby(["subject", "group"])["relative"].mean().reset_index()

                vals_a = subj_means[subj_means["group"] == ga]["relative"].values
                vals_b = subj_means[subj_means["group"] == gb]["relative"].values

                # T-test
                tresult = ttest_between_groups(vals_a, vals_b)
                # Effect size
                g = hedges_g(vals_a, vals_b)

                # LMM on ROI-level data with random intercept per subject
                lmm_data = band_data[band_data["group"].isin([ga, gb])].copy()
                lmm_result = lmm_group_effect(
                    lmm_data,
                    value_col="relative",
                    group_col="group",
                    subject_col="subject",
                    roi_col="roi",
                )

                rows.append({
                    "contrast": contrast.name,
                    "band": band_name,
                    "group_a": ga,
                    "group_b": gb,
                    "n_a": tresult.n_a,
                    "n_b": tresult.n_b,
                    "group_a_mean": tresult.group_a_mean,
                    "group_b_mean": tresult.group_b_mean,
                    "group_a_std": tresult.group_a_std,
                    "group_b_std": tresult.group_b_std,
                    "t_stat": tresult.statistic,
                    "p_value": tresult.p_value,
                    "hedges_g": g,
                    "lmm_z": lmm_result.statistic,
                    "lmm_p": lmm_result.p_value,
                    "lmm_converged": lmm_result.converged,
                })

        self._stats_df = pd.DataFrame(rows)

        if self._stats_df.empty:
            return

        # FDR correction per contrast (t-test p-values)
        for contrast_name in self._stats_df["contrast"].unique():
            mask = self._stats_df["contrast"] == contrast_name
            pvals = self._stats_df.loc[mask, "p_value"].values
            rejected, qvals = fdr_correction(pvals)
            self._stats_df.loc[mask, "q_value"] = qvals
            self._stats_df.loc[mask, "significant"] = rejected

            # FDR for LMM p-values
            lmm_pvals = self._stats_df.loc[mask, "lmm_p"].values
            lmm_rejected, lmm_qvals = fdr_correction(lmm_pvals)
            self._stats_df.loc[mask, "lmm_q"] = lmm_qvals
            self._stats_df.loc[mask, "lmm_significant"] = lmm_rejected

        # Save statistics
        self._stats_df.to_csv(self.output_dir / "tables" / "psd_statistics.csv", index=False)
        logger.info("Statistics saved to tables/psd_statistics.csv")

    def figures(self) -> None:
        fig_dir = self.output_dir / "figures"

        if self._group_psd_means and self.config.roi_categories:
            plot_psd_by_region(
                psd_data=self._group_psd_means,
                roi_categories=self.config.roi_categories,
                group_labels=self.config.groups,
                group_colors=self.config.group_colors,
                group_order=self.config.group_order,
                output_dir=fig_dir,
            )

        if self._band_df is not None and not self._band_df.empty:
            for ptype in ["relative", "absolute", "dB"]:
                plot_band_power_boxplots(
                    band_df=self._band_df,
                    group_colors=self.config.group_colors,
                    group_order=self.config.group_order,
                    output_dir=fig_dir,
                    power_type=ptype,
                )

            if self.config.roi_categories:
                plot_regional_power_heatmap(
                    band_df=self._band_df,
                    roi_categories=self.config.roi_categories,
                    group_order=self.config.group_order,
                    group_labels=self.config.groups,
                    output_dir=fig_dir,
                )

    def summary(self) -> None:
        report = ReportWriter(f"PSD Analysis — {self.config.name}")

        # Methods
        n_subjects = {}
        for group_id in self.config.group_order:
            n = sum(1 for g in self._subject_groups.values() if g == group_id)
            if n > 0:
                n_subjects[self.config.get_group_label(group_id)] = n

        report.add_methods(
            n_subjects=n_subjects,
            bands=self.config.bands,
            sfreq=self._sfreq or 0,
            analysis_name="Power Spectral Density",
        )

        # Statistics table
        if self._stats_df is not None and not self._stats_df.empty:
            report.add_statistics_table(self._stats_df, heading="Band Power Statistics")

            # Key findings
            findings = []
            sig = self._stats_df[self._stats_df.get("significant", pd.Series(dtype=bool)) == True]
            for _, row in sig.iterrows():
                direction = ">" if row.get("hedges_g", 0) > 0 else "<"
                findings.append(
                    f"**{row['band']}**: {row['group_a']} {direction} {row['group_b']} "
                    f"(t={row['t_stat']:.2f}, q={row['q_value']:.4f}, g={row['hedges_g']:.2f})"
                )

            if not findings:
                findings.append("No bands reached significance after FDR correction.")

            report.add_key_findings(findings)

        # Figure references
        fig_dir = self.output_dir / "figures"
        for fig_file in sorted(fig_dir.glob("*.png")):
            report.add_figure_reference(fig_file, fig_file.stem.replace("_", " ").title())

        report.write(self.output_dir / "ANALYSIS_SUMMARY.md")
