"""Electrode vs Source comparison analysis.

Reads outputs from both ``electrode`` and ``psd`` analyses, computes
correlations, effect sizes, and regional specificity metrics.  Generates
matplotlib figures for the manuscript.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..viz.constants import BAND_ORDER, CC_ROIS
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Hedges' g (bias-corrected Cohen's d)."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return np.nan
    mean_diff = np.mean(group_a) - np.mean(group_b)
    pooled_var = ((na - 1) * np.var(group_a, ddof=1) + (nb - 1) * np.var(group_b, ddof=1)) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return np.nan
    d = mean_diff / pooled_sd
    # Hedges' correction factor
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return d * correction


def _hedges_g_ci(group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Compute Hedges' g with confidence interval."""
    from scipy import stats as sp_stats

    g = _hedges_g(group_a, group_b)
    na, nb = len(group_a), len(group_b)
    if np.isnan(g) or na < 2 or nb < 2:
        return g, np.nan, np.nan
    # Approximate SE of Hedges' g
    se = np.sqrt((na + nb) / (na * nb) + g**2 / (2 * (na + nb)))
    z = sp_stats.norm.ppf(1 - alpha / 2)
    return g, g - z * se, g + z * se


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


class ElectrodeComparisonAnalysis(BaseAnalysis):
    """Compare electrode-level and source-level analysis results.

    Reads pre-computed CSVs from the ``electrode`` and ``psd`` analysis
    output directories.  Computes correlations, effect sizes, and regional
    specificity.  Generates publication-quality matplotlib figures.
    """

    name = "electrode_comparison"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._electrode_df: pd.DataFrame | None = None
        self._source_df: pd.DataFrame | None = None
        self._comparison_df: pd.DataFrame | None = None

    def setup(self) -> None:
        """Load CSVs from both electrode and psd output directories."""
        base_dir = self.config.output_dir

        # Load electrode data
        electrode_csv = base_dir / "electrode" / "data" / "electrode_band_power.csv"
        if not electrode_csv.exists():
            raise FileNotFoundError(
                f"Electrode band power CSV not found: {electrode_csv}. "
                "Run the 'electrode' analysis first."
            )
        self._electrode_df = pd.read_csv(electrode_csv)
        logger.info("Loaded electrode data: %d rows", len(self._electrode_df))

        # Load source (PSD) data
        source_csv = base_dir / "psd" / "data" / "band_power.csv"
        if not source_csv.exists():
            raise FileNotFoundError(
                f"Source band power CSV not found: {source_csv}. "
                "Run the 'psd' analysis first."
            )
        self._source_df = pd.read_csv(source_csv)
        # Exclude Corpus Callosum (white matter) ROIs
        n_before = len(self._source_df)
        self._source_df = self._source_df[~self._source_df["roi"].isin(CC_ROIS)]
        logger.info("Loaded source data: %d rows (%d CC rows excluded)",
                     len(self._source_df), n_before - len(self._source_df))

    def process_subject(self, subject: SubjectInfo) -> None:
        """No-op — data already loaded from CSVs."""
        pass

    def aggregate(self) -> None:
        """Merge electrode and source data by subject and export."""
        data_dir = self.output_dir / "data"

        # Per-subject mean power across channels (electrode level)
        elec_subj = (
            self._electrode_df
            .groupby(["subject", "group", "band"])
            .agg(
                elec_absolute=("absolute", "mean"),
                elec_relative=("relative", "mean"),
                elec_dB=("dB", "mean"),
            )
            .reset_index()
        )

        # Per-subject mean power across ROIs (source level)
        src_subj = (
            self._source_df
            .groupby(["subject", "group", "band"])
            .agg(
                source_absolute=("absolute", "mean"),
                source_relative=("relative", "mean"),
                source_dB=("dB", "mean"),
            )
            .reset_index()
        )

        # Merge on subject + band
        self._comparison_df = pd.merge(
            elec_subj, src_subj,
            on=["subject", "group", "band"],
            how="inner",
        )

        if self._comparison_df.empty:
            logger.warning("No matching subjects between electrode and source data")
            return

        self._comparison_df.to_csv(data_dir / "comparison_data.csv", index=False)
        logger.info("Exported comparison_data.csv (%d rows)", len(self._comparison_df))

        # Also compute per-subject regional source power for regional comparison
        if self.config.roi_categories:
            roi_to_region = {}
            for region, rois in self.config.roi_categories.items():
                for roi in rois:
                    roi_to_region[roi] = region

            src_with_region = self._source_df.copy()
            src_with_region["region"] = src_with_region["roi"].map(roi_to_region)
            src_with_region = src_with_region.dropna(subset=["region"])

            regional_src = (
                src_with_region
                .groupby(["subject", "group", "band", "region"])
                .agg(
                    source_absolute=("absolute", "mean"),
                    source_relative=("relative", "mean"),
                    source_dB=("dB", "mean"),
                )
                .reset_index()
            )

            regional_src.to_csv(data_dir / "regional_source_power.csv", index=False)
            logger.info("Exported regional_source_power.csv (%d rows)", len(regional_src))

    def statistics(self) -> None:
        """Compute correlations, effect sizes, and regional specificity."""
        from scipy import stats as sp_stats

        if self._comparison_df is None or self._comparison_df.empty:
            logger.warning("No comparison data — skipping statistics")
            return

        data_dir = self.output_dir / "data"
        tbl_dir = self.output_dir / "tables"

        # --- Correlation and effect sizes per band ---
        stats_rows = []
        for band in self._comparison_df["band"].unique():
            bdata = self._comparison_df[self._comparison_df["band"] == band]

            for power_type in ["relative", "dB"]:
                elec_col = f"elec_{power_type}"
                src_col = f"source_{power_type}"

                # Pearson correlation
                valid = bdata[[elec_col, src_col]].dropna()
                if len(valid) > 2:
                    r, p = sp_stats.pearsonr(valid[elec_col], valid[src_col])
                else:
                    r, p = np.nan, np.nan

                # Effect sizes per group contrast
                for contrast in self.config.contrasts:
                    ga_data = bdata[bdata["group"] == contrast.group_a]
                    gb_data = bdata[bdata["group"] == contrast.group_b]

                    elec_g, elec_ci_lo, elec_ci_hi = _hedges_g_ci(
                        ga_data[elec_col].values, gb_data[elec_col].values
                    )
                    src_g, src_ci_lo, src_ci_hi = _hedges_g_ci(
                        ga_data[src_col].values, gb_data[src_col].values
                    )

                    stats_rows.append({
                        "band": band,
                        "power_type": power_type,
                        "contrast": contrast.name,
                        "correlation_r": r,
                        "correlation_p": p,
                        "n_subjects": len(valid),
                        "electrode_hedges_g": elec_g,
                        "electrode_ci_lo": elec_ci_lo,
                        "electrode_ci_hi": elec_ci_hi,
                        "source_hedges_g": src_g,
                        "source_ci_lo": src_ci_lo,
                        "source_ci_hi": src_ci_hi,
                    })

        stats_df = pd.DataFrame(stats_rows)
        if not stats_df.empty:
            stats_df.to_csv(tbl_dir / "comparison_stats.csv", index=False)
            logger.info("Exported comparison_stats.csv (%d rows)", len(stats_df))

        # --- Regional effect sizes ---
        regional_csv = data_dir / "regional_source_power.csv"
        if regional_csv.exists() and self.config.roi_categories:
            regional_src = pd.read_csv(regional_csv)
            regional_rows = []

            for band in regional_src["band"].unique():
                for power_type in ["relative", "dB"]:
                    src_col = f"source_{power_type}"

                    for contrast in self.config.contrasts:
                        # Global electrode effect (for reference)
                        comp_band = self._comparison_df[self._comparison_df["band"] == band]
                        elec_col = f"elec_{power_type}"
                        ga_elec = comp_band[comp_band["group"] == contrast.group_a][elec_col].values
                        gb_elec = comp_band[comp_band["group"] == contrast.group_b][elec_col].values
                        elec_g, elec_ci_lo, elec_ci_hi = _hedges_g_ci(ga_elec, gb_elec)

                        # Per-region source effect
                        for region in regional_src["region"].unique():
                            rdata = regional_src[
                                (regional_src["band"] == band) &
                                (regional_src["region"] == region)
                            ]
                            ga_src = rdata[rdata["group"] == contrast.group_a][src_col].values
                            gb_src = rdata[rdata["group"] == contrast.group_b][src_col].values
                            reg_g, reg_ci_lo, reg_ci_hi = _hedges_g_ci(ga_src, gb_src)

                            regional_rows.append({
                                "band": band,
                                "power_type": power_type,
                                "contrast": contrast.name,
                                "region": region,
                                "electrode_hedges_g": elec_g,
                                "electrode_ci_lo": elec_ci_lo,
                                "electrode_ci_hi": elec_ci_hi,
                                "region_hedges_g": reg_g,
                                "region_ci_lo": reg_ci_lo,
                                "region_ci_hi": reg_ci_hi,
                                "exceeds_electrode": (
                                    abs(reg_g) > abs(elec_g)
                                    if not (np.isnan(reg_g) or np.isnan(elec_g))
                                    else False
                                ),
                            })

            regional_df = pd.DataFrame(regional_rows)
            if not regional_df.empty:
                regional_df.to_csv(tbl_dir / "regional_effect_sizes.csv", index=False)
                logger.info("Exported regional_effect_sizes.csv (%d rows)", len(regional_df))

    def figures(self) -> None:
        """Generate matplotlib comparison figures."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tbl_dir = self.output_dir / "tables"
        fig_dir = self.output_dir / "figures"

        stats_csv = tbl_dir / "comparison_stats.csv"
        if not stats_csv.exists():
            logger.warning("comparison_stats.csv not found — skipping figures")
            return

        stats_df = pd.read_csv(stats_csv)
        comp_df = pd.read_csv(self.output_dir / "data" / "comparison_data.csv")

        group_colors = self.config.group_colors
        group_labels = self.config.groups

        # --- Fig 1: Correlation scatter (one panel per band) ---
        self._plot_correlation_scatter(comp_df, stats_df, fig_dir, group_colors, group_labels)

        # --- Fig 2: Effect size comparison ---
        self._plot_effect_size_comparison(stats_df, fig_dir)

        # --- Fig 3 & 4: Regional specificity ---
        regional_csv = tbl_dir / "regional_effect_sizes.csv"
        if regional_csv.exists():
            regional_df = pd.read_csv(regional_csv)
            self._plot_regional_forest(regional_df, fig_dir)
            self._plot_spatial_advantage(regional_df, fig_dir)

        # --- Fig 5: ROI-level forest plot (from PSD posthoc) ---
        self._plot_roi_forest(fig_dir)

    def _plot_correlation_scatter(
        self, comp_df: pd.DataFrame, stats_df: pd.DataFrame,
        fig_dir: Path, group_colors: dict, group_labels: dict,
    ) -> None:
        import matplotlib.pyplot as plt

        for power_type in ["relative", "dB"]:
            pt_stats = stats_df[stats_df["power_type"] == power_type]
            # Sort bands by ascending Hz order
            bands = [b for b in BAND_ORDER if b in pt_stats["band"].values]
            n_bands = len(bands)
            if n_bands == 0:
                continue

            ncols = min(3, n_bands)
            nrows = (n_bands + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

            elec_col = f"elec_{power_type}"
            src_col = f"source_{power_type}"

            for idx, band in enumerate(bands):
                ax = axes[idx // ncols, idx % ncols]
                bdata = comp_df[comp_df["band"] == band]

                for grp in self.config.group_order:
                    gd = bdata[bdata["group"] == grp]
                    color = group_colors.get(grp, "#333333")
                    label = group_labels.get(grp, grp)
                    ax.scatter(gd[elec_col], gd[src_col], c=color, label=label,
                               alpha=0.7, s=40, edgecolors="white", linewidth=0.5)

                # Annotation
                row = pt_stats[pt_stats["band"] == band]
                if not row.empty:
                    r = row["correlation_r"].values[0]
                    p = row["correlation_p"].values[0]
                    ax.annotate(
                        f"r={r:.2f}, p={p:.3f}",
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=9, va="top",
                    )

                ax.set_xlabel(f"Electrode {power_type}")
                ax.set_ylabel(f"Source {power_type}")
                ax.set_title(band)

            # Hide empty axes
            for idx in range(n_bands, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"Electrode vs Source Power Correlation ({power_type})", fontsize=14, y=1.02)
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig1_correlation_{power_type}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved fig1_correlation_%s.png", power_type)

    def _plot_effect_size_comparison(self, stats_df: pd.DataFrame, fig_dir: Path) -> None:
        import matplotlib.pyplot as plt

        for power_type in ["relative", "dB"]:
            pt_df = stats_df[stats_df["power_type"] == power_type].copy()
            if pt_df.empty:
                continue

            # Sort by standard ascending Hz order
            pt_df["band"] = pd.Categorical(pt_df["band"], categories=BAND_ORDER, ordered=True)
            pt_df = pt_df.sort_values("band").dropna(subset=["band"])

            bands = pt_df["band"].values
            n = len(bands)
            x = np.arange(n)
            width = 0.35

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6 + n * 1.2, 5))

            # Bar chart with error bars (CIs)
            elec_g = pt_df["electrode_hedges_g"].values
            src_g = pt_df["source_hedges_g"].values
            elec_ci_lo = pt_df["electrode_ci_lo"].values
            elec_ci_hi = pt_df["electrode_ci_hi"].values
            src_ci_lo = pt_df["source_ci_lo"].values
            src_ci_hi = pt_df["source_ci_hi"].values

            elec_err = np.array([elec_g - elec_ci_lo, elec_ci_hi - elec_g])
            src_err = np.array([src_g - src_ci_lo, src_ci_hi - src_g])

            ax1.bar(x - width / 2, elec_g, width, label="Electrode", color="#7FB3D8",
                    edgecolor="white", yerr=elec_err, capsize=3, error_kw={"linewidth": 1})
            ax1.bar(x + width / 2, src_g, width, label="Source", color="#E8A0A0",
                    edgecolor="white", yerr=src_err, capsize=3, error_kw={"linewidth": 1})

            # Significance markers: * where CI excludes zero
            for i in range(n):
                # Electrode significance
                if elec_ci_lo[i] > 0 or elec_ci_hi[i] < 0:
                    y_pos_star = max(elec_g[i] + elec_err[1, i], 0) + 0.05
                    ax1.annotate("*", xy=(x[i] - width / 2, y_pos_star),
                                 fontsize=14, fontweight="bold", color="#4A90D9",
                                 ha="center", va="bottom")
                # Source significance
                if src_ci_lo[i] > 0 or src_ci_hi[i] < 0:
                    y_pos_star = max(src_g[i] + src_err[1, i], 0) + 0.05
                    ax1.annotate("*", xy=(x[i] + width / 2, y_pos_star),
                                 fontsize=14, fontweight="bold", color="#C0392B",
                                 ha="center", va="bottom")

            ax1.set_xticks(x)
            ax1.set_xticklabels(bands, rotation=45, ha="right")
            ax1.set_ylabel("Hedges' g")
            ax1.set_title("Effect Sizes by Band")
            ax1.axhline(y=0, color="grey", linewidth=0.5)
            ax1.legend(fontsize=9)

            # Scatter: electrode g vs source g
            ax2.scatter(elec_g, src_g, c="#444444", s=50, zorder=3)
            for i, band in enumerate(bands):
                ax2.annotate(str(band), (elec_g[i], src_g[i]), fontsize=8,
                             xytext=(5, 5), textcoords="offset points")
            lims = [
                min(min(elec_g), min(src_g)) - 0.2,
                max(max(elec_g), max(src_g)) + 0.2,
            ]
            ax2.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
            ax2.set_xlabel("Electrode Hedges' g")
            ax2.set_ylabel("Source Hedges' g")
            ax2.set_title("Electrode vs Source Effect Size")

            fig.suptitle(f"Effect Size Comparison ({power_type})", fontsize=13)
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig2_effect_sizes_{power_type}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved fig2_effect_sizes_%s.png", power_type)

    def _plot_regional_forest(self, regional_df: pd.DataFrame, fig_dir: Path) -> None:
        """Forest plot: electrode global effect as reference line, regional source effects as dots.

        Manuscript-quality 2×3 grid with significance markers (*) for CIs
        excluding zero, color-coded by whether region exceeds electrode effect.
        """
        import matplotlib.pyplot as plt

        for power_type in ["relative", "dB"]:
            pt_df = regional_df[regional_df["power_type"] == power_type].copy()
            if pt_df.empty:
                continue

            # Sort bands by ascending Hz order
            bands = [b for b in BAND_ORDER if b in pt_df["band"].values]
            if not bands:
                continue

            # Determine significance (CI excludes zero)
            pt_df["significant"] = (pt_df["region_ci_lo"] > 0) | (pt_df["region_ci_hi"] < 0)

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes_flat = axes.flatten()

            for idx, band in enumerate(bands):
                ax = axes_flat[idx]
                bdata = pt_df[pt_df["band"] == band].copy()
                bdata = bdata.sort_values("region_hedges_g", ascending=True)

                electrode_g = bdata["electrode_hedges_g"].iloc[0]
                regions = bdata["region"].values
                g_vals = bdata["region_hedges_g"].values
                ci_lo = bdata["region_ci_lo"].values
                ci_hi = bdata["region_ci_hi"].values
                sig = bdata["significant"].values
                y_pos = np.arange(len(regions))

                # Reference lines
                ax.axvline(x=electrode_g, color="gray", linestyle="--", linewidth=1.5,
                           alpha=0.7, label=f"Electrode g={electrode_g:.2f}")
                ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

                for i, (y, g, lo, hi, s) in enumerate(zip(y_pos, g_vals, ci_lo, ci_hi, sig)):
                    color = "#E74C3C" if g > electrode_g else "#3498DB"
                    marker_size = 9 if s else 7

                    # CI line
                    ax.plot([lo, hi], [y, y], color=color,
                            linewidth=2 if s else 1.5, alpha=0.9 if s else 0.6)
                    # Point
                    ax.plot(g, y, "o", color=color, markersize=marker_size, zorder=5)

                    # Significance star
                    if s:
                        ax.annotate("*", xy=(hi + 0.03, y), fontsize=16, fontweight="bold",
                                    color=color, va="center", ha="left")

                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(regions, fontsize=11)
                ax.set_xlabel("Hedges' g", fontsize=12)
                ax.set_title(band, fontsize=14, fontweight="bold")
                ax.legend(fontsize=9, loc="lower right")

            # Hide unused axes
            for idx in range(len(bands), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle(
                f"Regional Source Effect Sizes vs Electrode Reference ({power_type})",
                fontsize=16, fontweight="bold", y=0.98,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(fig_dir / f"fig3_regional_forest_{power_type}.png",
                        dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            logger.info("Saved fig3_regional_forest_%s.png", power_type)

    def _plot_spatial_advantage(self, regional_df: pd.DataFrame, fig_dir: Path) -> None:
        """Heatmap: which regions exceed electrode-level sensitivity."""
        import matplotlib.pyplot as plt

        for power_type in ["relative", "dB"]:
            pt_df = regional_df[regional_df["power_type"] == power_type]
            if pt_df.empty:
                continue

            bands = sorted(pt_df["band"].unique(), key=lambda b: list(self.config.bands.keys()).index(b)
                           if b in self.config.bands else 999)
            regions = sorted(pt_df["region"].unique())

            # Build matrix: region_g - electrode_g
            advantage = np.full((len(regions), len(bands)), np.nan)
            for i, region in enumerate(regions):
                for j, band in enumerate(bands):
                    row = pt_df[(pt_df["region"] == region) & (pt_df["band"] == band)]
                    if not row.empty:
                        advantage[i, j] = (
                            abs(row["region_hedges_g"].values[0]) -
                            abs(row["electrode_hedges_g"].values[0])
                        )

            fig, ax = plt.subplots(figsize=(max(6, len(bands) * 1.5), max(4, len(regions) * 0.5)))

            vmax = np.nanmax(np.abs(advantage))
            im = ax.imshow(advantage, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels(bands, rotation=45, ha="right")
            ax.set_yticks(range(len(regions)))
            ax.set_yticklabels(regions, fontsize=9)

            # Annotate cells
            for i in range(len(regions)):
                for j in range(len(bands)):
                    val = advantage[i, j]
                    if not np.isnan(val):
                        color = "white" if abs(val) > vmax * 0.6 else "black"
                        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                                fontsize=8, color=color)

            fig.colorbar(im, ax=ax, label="|Regional g| - |Electrode g|", shrink=0.8)
            ax.set_title(f"Spatial Specificity: Source vs Electrode ({power_type})")
            ax.set_xlabel("Frequency Band")
            ax.set_ylabel("Brain Region")
            fig.tight_layout()
            fig.savefig(fig_dir / f"fig4_spatial_advantage_{power_type}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved fig4_spatial_advantage_%s.png", power_type)

    def _plot_roi_forest(self, fig_dir: Path) -> None:
        """ROI-level forest plot from PSD posthoc results.

        Two-panel plot (Low Gamma, High Gamma) showing per-ROI disease
        effect sizes with three significance tiers: Holm-corrected (*),
        uncorrected p<0.05 (dagger), and non-significant (gray).
        """
        import matplotlib.pyplot as plt

        # Locate posthoc CSV from sibling PSD analysis
        base_dir = self.config.output_dir
        posthoc_csv = base_dir / "psd" / "tables" / "psd_posthoc_roi.csv"
        if not posthoc_csv.exists():
            logger.info("psd_posthoc_roi.csv not found — skipping ROI forest plot")
            return

        df = pd.read_csv(posthoc_csv)

        # Filter to dB power, Low Gamma and High Gamma
        df = df[df["power_type"] == "dB"].copy()
        bands = ["Low Gamma", "High Gamma"]
        df = df[df["band"].isin(bands)].copy()
        if df.empty:
            logger.info("No dB Low/High Gamma rows in posthoc — skipping ROI forest")
            return

        # Exclude Corpus Callosum ROIs
        df = df[~df["roi"].isin(CC_ROIS)].copy()

        # Compute approximate 95% CI for Hedges' g
        # Infer group sizes from config contrasts
        contrast = self.config.contrasts[0] if self.config.contrasts else None
        if contrast and self._comparison_df is not None:
            n1 = len(self._comparison_df[self._comparison_df["group"] == contrast.group_a]["subject"].unique())
            n2 = len(self._comparison_df[self._comparison_df["group"] == contrast.group_b]["subject"].unique())
        else:
            # Fallback: estimate from unique subjects in posthoc
            n1 = n2 = 14

        df["se_g"] = np.sqrt(1 / n1 + 1 / n2 + df["hedges_g"] ** 2 / (2 * (n1 + n2)))
        df["g_ci_lo"] = df["hedges_g"] - 1.96 * df["se_g"]
        df["g_ci_hi"] = df["hedges_g"] + 1.96 * df["se_g"]

        # Significance markers
        df["uncorrected_sig"] = df["p_value"] < 0.05
        df["holm_sig"] = df["significant"]  # already Holm-corrected

        # Clean ROI names for display
        df["roi_label"] = df["roi"].str.replace("_", " ").str.replace("Cortex ", "")

        fig, axes = plt.subplots(1, 2, figsize=(16, 14))

        for idx, band in enumerate(bands):
            ax = axes[idx]
            bdf = df[df["band"] == band].copy()
            bdf = bdf.sort_values("hedges_g", ascending=True)

            rois = bdf["roi_label"].values
            g_vals = bdf["hedges_g"].values
            ci_lo = bdf["g_ci_lo"].values
            ci_hi = bdf["g_ci_hi"].values
            uncorr = bdf["uncorrected_sig"].values
            holm = bdf["holm_sig"].values
            y_pos = np.arange(len(rois))

            # Zero reference
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)

            for i, (y, g, lo, hi, u, h) in enumerate(
                zip(y_pos, g_vals, ci_lo, ci_hi, uncorr, holm)
            ):
                if h:
                    color, lw, ms, alpha = "#C0392B", 2.5, 10, 1.0
                elif u:
                    color, lw, ms, alpha = "#E67E22", 2.0, 8, 0.9
                else:
                    color, lw, ms, alpha = "#95A5A6", 1.2, 6, 0.5

                ax.plot([lo, hi], [y, y], color=color, linewidth=lw, alpha=alpha)
                ax.plot(g, y, "o", color=color, markersize=ms, zorder=5, alpha=alpha)

                if h:
                    ax.annotate("*", xy=(hi + 0.05, y), fontsize=18, fontweight="bold",
                                color=color, va="center", ha="left")
                elif u:
                    ax.annotate("\u2020", xy=(hi + 0.05, y), fontsize=14,
                                color=color, va="center", ha="left")

            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(rois, fontsize=9)
            ax.set_xlabel("Hedges' g (KO \u2212 WT)", fontsize=12)
            ax.set_title(band, fontsize=14, fontweight="bold")

            n_uncorr = int(uncorr.sum())
            n_holm = int(holm.sum())
            ax.text(
                0.02, 0.98,
                f"{n_uncorr} uncorrected p<0.05 (\u2020)\n{n_holm} Holm-corrected (*)",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8),
            )

        fig.suptitle(
            "ROI-Level Disease Effect Sizes (dB Power)",
            fontsize=16, fontweight="bold", y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_dir / "fig5_roi_forest_dB.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved fig5_roi_forest_dB.png")

    def summary(self) -> None:
        """Call Rscript for formatted comparison report."""
        data_dir = self.output_dir / "data"
        tbl_dir = self.output_dir / "tables"

        # Verify prerequisite files exist
        if not (tbl_dir / "comparison_stats.csv").exists():
            logger.error("comparison_stats.csv not found — skipping R report")
            return

        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "electrode_comparison_analysis.R"
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return

        # Write study config YAML for R
        config_path = data_dir / "study_config.yaml"
        import yaml

        config_data = dict(self.config.raw)
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        cmd = [
            "Rscript", str(r_script),
            "--data-dir", str(data_dir),
            "--tables-dir", str(tbl_dir),
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
        except FileNotFoundError:
            logger.error("Rscript not found. Install R to enable comparison report.")
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 600 seconds")
