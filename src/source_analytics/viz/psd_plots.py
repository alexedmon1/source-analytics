"""PSD and band power visualizations."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .style import apply_style, get_group_color

logger = logging.getLogger(__name__)


def plot_psd_by_region(
    psd_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    roi_categories: dict[str, list[str]],
    group_labels: dict[str, str],
    group_colors: dict[str, str],
    group_order: list[str],
    output_dir: Path,
    fmax: float = 120.0,
):
    """Plot mean PSD curves per region, one subplot per region category.

    Parameters
    ----------
    psd_data : dict[group_id -> dict[roi_name -> (freqs, psd_mean)]]
        Group-averaged PSDs per ROI.
    roi_categories : dict[category_name -> list[roi_name]]
    """
    apply_style()
    categories = list(roi_categories.keys())
    if not categories:
        return

    n_cats = len(categories)
    n_cols = min(3, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, cat_name in enumerate(categories):
        ax = axes[idx // n_cols, idx % n_cols]
        roi_list = roi_categories[cat_name]

        for group_id in group_order:
            if group_id not in psd_data:
                continue
            group_psds = psd_data[group_id]
            color = get_group_color(group_id, group_colors)
            label = group_labels.get(group_id, group_id)

            # Average PSD across ROIs in this category
            cat_psds = []
            freqs = None
            for roi in roi_list:
                if roi in group_psds:
                    f, p = group_psds[roi]
                    cat_psds.append(p)
                    freqs = f

            if not cat_psds or freqs is None:
                continue

            mean_psd = np.mean(cat_psds, axis=0)
            sem_psd = np.std(cat_psds, axis=0) / np.sqrt(len(cat_psds)) if len(cat_psds) > 1 else np.zeros_like(mean_psd)

            freq_mask = freqs <= fmax
            ax.semilogy(freqs[freq_mask], mean_psd[freq_mask], color=color, label=label)
            if len(cat_psds) > 1:
                ax.fill_between(
                    freqs[freq_mask],
                    (mean_psd - sem_psd)[freq_mask],
                    (mean_psd + sem_psd)[freq_mask],
                    color=color,
                    alpha=0.2,
                )

        ax.set_title(cat_name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_cats, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Power Spectral Density by Region", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = output_dir / "psd_by_region.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved PSD plot: %s", out_path)


def plot_band_power_boxplots(
    band_df: pd.DataFrame,
    group_colors: dict[str, str],
    group_order: list[str],
    output_dir: Path,
    power_type: str = "relative",
):
    """Box plots of band power by group, one panel per band.

    Parameters
    ----------
    band_df : DataFrame
        Columns: subject, group, band, roi, absolute, relative, dB
    """
    apply_style()
    bands = band_df["band"].unique()
    n_bands = len(bands)

    fig, axes = plt.subplots(1, n_bands, figsize=(3.5 * n_bands, 5), squeeze=False)

    palette = {g: get_group_color(g, group_colors) for g in group_order}

    for i, band in enumerate(bands):
        ax = axes[0, i]
        bdata = band_df[band_df["band"] == band]

        # Average across ROIs per subject to avoid pseudoreplication
        subj_means = bdata.groupby(["subject", "group"])[power_type].mean().reset_index()

        sns.boxplot(
            data=subj_means,
            x="group",
            y=power_type,
            hue="group",
            order=group_order,
            hue_order=group_order,
            palette=palette,
            ax=ax,
            width=0.5,
            fliersize=3,
            legend=False,
        )
        sns.stripplot(
            data=subj_means,
            x="group",
            y=power_type,
            hue="group",
            order=group_order,
            hue_order=group_order,
            palette=palette,
            ax=ax,
            size=4,
            alpha=0.6,
            jitter=True,
            legend=False,
        )
        ax.set_title(band)
        ax.set_xlabel("")
        ax.set_ylabel(f"{power_type.capitalize()} Power" if i == 0 else "")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"Band Power ({power_type.capitalize()}) by Group", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path = output_dir / f"band_power_{power_type}.png"
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved band power plot: %s", out_path)


def plot_regional_power_heatmap(
    band_df: pd.DataFrame,
    roi_categories: dict[str, list[str]],
    group_order: list[str],
    group_labels: dict[str, str],
    output_dir: Path,
    power_type: str = "relative",
):
    """Heatmap of mean band power by region category x band, per group.

    Parameters
    ----------
    band_df : DataFrame
        Columns: subject, group, band, roi, absolute, relative, dB
    """
    apply_style()

    for group_id in group_order:
        gdata = band_df[band_df["group"] == group_id]
        if gdata.empty:
            continue

        # Map ROIs to categories
        roi_to_cat = {}
        for cat, rois in roi_categories.items():
            for roi in rois:
                roi_to_cat[roi] = cat

        gdata = gdata.copy()
        gdata["category"] = gdata["roi"].map(roi_to_cat)
        gdata = gdata.dropna(subset=["category"])

        if gdata.empty:
            continue

        pivot = gdata.groupby(["category", "band"])[power_type].mean().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.5 + 1)))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title(f"{group_labels.get(group_id, group_id)} — {power_type.capitalize()} Power")
        ax.set_ylabel("Region")
        ax.set_xlabel("Band")
        fig.tight_layout()
        out_path = output_dir / f"heatmap_{power_type}_{group_id}.png"
        fig.savefig(out_path)
        plt.close(fig)
        logger.info("Saved heatmap: %s", out_path)
