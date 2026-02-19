"""Radar (spider) chart for regional profiles across treatment groups.

Plots one polar subplot per frequency band showing treatment group values
relative to a reference group (typically Vehicle).  Each spoke represents
a brain region.

Requires: matplotlib, numpy, pandas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sensible defaults for the Autifony study
DEFAULT_GROUP_COLORS = {
    "6mgkg": "#3182bd",
    "30mgkg": "#e6550d",
}
DEFAULT_GROUP_LABELS = {
    "6mgkg": "AUT00201 (6 mg/kg)",
    "30mgkg": "AUT00206 (30 mg/kg)",
}


def plot_radar(
    region_df: pd.DataFrame,
    output_path: str | Path,
    *,
    value_col: str = "dB",
    reference_group: str = "Vehicle",
    treatment_groups: list[str] | None = None,
    bands: list[str] | None = None,
    group_colors: dict[str, str] | None = None,
    group_labels: dict[str, str] | None = None,
    title: str = "Regional Profile vs Vehicle",
    figsize: tuple[float, float] = (20, 7),
    dpi: int = 200,
) -> Path:
    """Render a radar chart of regional profiles by treatment group.

    Parameters
    ----------
    region_df : DataFrame
        Must contain columns: ``subject``, ``group``, ``region``, ``band``,
        and ``value_col``.  One row per subject × region × band.
    output_path : str or Path
        Output PNG path.
    value_col : str
        Column name for the measure to plot (default ``"dB"``).
    reference_group : str
        Group to normalize against (dashed zero line).
    treatment_groups : list[str], optional
        Groups to plot.  Auto-detected if *None* (all non-reference groups).
    bands : list[str], optional
        Frequency bands to show (one subplot each).
        Auto-detected if *None*.
    group_colors : dict, optional
        Group name → hex color.
    group_labels : dict, optional
        Group name → display label.
    title : str
        Figure suptitle.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if group_colors is None:
        group_colors = DEFAULT_GROUP_COLORS
    if group_labels is None:
        group_labels = DEFAULT_GROUP_LABELS

    # Auto-detect treatment groups
    all_groups = region_df["group"].unique()
    if treatment_groups is None:
        treatment_groups = [g for g in all_groups if g != reference_group]

    # Auto-detect bands
    if bands is None:
        bands = sorted(region_df["band"].unique())

    # Compute reference group means per region × band
    ref_means = (
        region_df[region_df["group"] == reference_group]
        .groupby(["region", "band"])[value_col]
        .mean()
        .rename("ref_mean")
    )
    plot_df = region_df.merge(ref_means, on=["region", "band"])
    plot_df["delta"] = plot_df[value_col] - plot_df["ref_mean"]

    # Group stats
    stats = (
        plot_df.groupby(["region", "group", "band"])["delta"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    regions = sorted(stats["region"].unique())
    n_regions = len(regions)
    n_bands = len(bands)
    angles = np.linspace(0, 2 * np.pi, n_regions, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(
        1, n_bands, figsize=figsize,
        subplot_kw=dict(polar=True), facecolor="white",
    )
    if n_bands == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)

    for ax, band in zip(axes, bands):
        bstats = stats[stats["band"] == band]

        # Zero reference (Vehicle baseline)
        ax.plot(
            angles, [0] * len(angles),
            color="gray", linewidth=1.2, linestyle="--", alpha=0.7,
            label=reference_group, zorder=1,
        )

        for group in treatment_groups:
            gstats = bstats[bstats["group"] == group].set_index("region")
            values = [
                gstats.loc[r, "mean"] if r in gstats.index else 0.0
                for r in regions
            ]
            sems = [
                gstats.loc[r, "sem"] if r in gstats.index else 0.0
                for r in regions
            ]
            values += values[:1]
            sems += sems[:1]
            values, sems = np.array(values), np.array(sems)

            color = group_colors.get(group, "#333333")
            label = group_labels.get(group, group)

            ax.plot(
                angles, values, color=color, linewidth=2.5,
                label=label, zorder=3,
            )
            ax.fill_between(
                angles, values - sems, values + sems,
                color=color, alpha=0.12, zorder=2,
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [r.replace("_", "\n") for r in regions], fontsize=7,
        )
        ax.set_title(
            band.replace("_", " ").title(), fontsize=12, pad=20,
        )
        ax.grid(True, alpha=0.25)

    axes[0].legend(
        loc="upper left", bbox_to_anchor=(-0.3, 1.15),
        fontsize=10, frameon=False,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Saved radar chart: %s", output_path)
    return output_path
