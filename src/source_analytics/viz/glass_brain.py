"""Glass brain visualizations for whole-brain vertex-level results.

Three-view (axial, coronal, sagittal) scatter plots of source-space data,
suitable for mouse EEG source localization results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _setup_axes(fig, title: str | None = None):
    """Create 3-view glass brain axes (axial, coronal, sagittal)."""
    axes = []
    # Axial (top-down): x vs y
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title("Axial")
    ax1.set_aspect("equal")
    axes.append(ax1)

    # Coronal (front): x vs z
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Z (mm)")
    ax2.set_title("Coronal")
    ax2.set_aspect("equal")
    axes.append(ax2)

    # Sagittal (side): y vs z
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_xlabel("Y (mm)")
    ax3.set_ylabel("Z (mm)")
    ax3.set_title("Sagittal")
    ax3.set_aspect("equal")
    axes.append(ax3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    return axes


def plot_glass_brain(
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    output_path: str | Path,
    *,
    cmap: str = "RdBu_r",
    vlim: tuple[float, float] | None = None,
    marker_size: float = 40,
    alpha: float = 0.8,
) -> None:
    """Plot vertex-level values on a 3-view glass brain.

    Parameters
    ----------
    coords : ndarray, shape (n_vertices, 3)
        Source coordinates (x, y, z) in mm.
    values : ndarray, shape (n_vertices,)
        Values to plot (e.g., t-statistics, power).
    title : str
        Figure title.
    output_path : Path
        Where to save the figure.
    cmap : str
        Colormap name.
    vlim : tuple, optional
        (vmin, vmax) for colorbar. If None, symmetric around 0.
    marker_size : float
        Scatter point size.
    alpha : float
        Scatter point transparency.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vlim is None:
        vmax = np.nanmax(np.abs(values))
        vlim = (-vmax, vmax)

    fig = plt.figure(figsize=(15, 5))
    axes = _setup_axes(fig, title)

    # Projections: (x,y), (x,z), (y,z)
    proj_pairs = [(0, 1), (0, 2), (1, 2)]
    for ax, (i, j) in zip(axes, proj_pairs):
        sc = ax.scatter(
            coords[:, i], coords[:, j],
            c=values, cmap=cmap,
            vmin=vlim[0], vmax=vlim[1],
            s=marker_size, alpha=alpha,
            edgecolors="0.3", linewidths=0.3,
        )

    fig.colorbar(sc, ax=axes, shrink=0.8, label="Value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved glass brain: %s", output_path)


def plot_band_comparison(
    coords: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    t_map: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_pvalues: list[float],
    band_name: str,
    group_labels: tuple[str, str],
    output_path: str | Path,
    *,
    alpha_threshold: float = 0.05,
) -> None:
    """Generate 6-panel comparison figure for a single band/metric.

    Panels:
    1. Group A mean
    2. Group B mean
    3. Difference (A - B)
    4. T-statistic map
    5. Significant clusters
    6. Histogram of vertex-level differences

    Parameters
    ----------
    coords : ndarray, shape (n_vertices, 3)
    mean_a, mean_b : ndarray, shape (n_vertices,)
        Group-level mean values.
    t_map : ndarray, shape (n_vertices,)
        T-statistics per vertex.
    cluster_labels : ndarray, shape (n_vertices,)
        Cluster label per vertex (0 = no cluster).
    cluster_pvalues : list[float]
        Corrected p-value per cluster.
    band_name : str
    group_labels : tuple of str
        (label_a, label_b) for display.
    output_path : Path
    alpha_threshold : float
        Significance threshold for clusters.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Whole-Brain: {band_name}", fontsize=16, fontweight="bold")

    diff = mean_a - mean_b

    # Determine consistent color limits for group means
    vmax_groups = max(np.nanmax(np.abs(mean_a)), np.nanmax(np.abs(mean_b)))

    # Panel 1: Group A mean (axial view)
    ax = axes[0, 0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=mean_a,
                    cmap="viridis", s=30, alpha=0.8, edgecolors="0.3", linewidths=0.3)
    ax.set_title(f"{group_labels[0]} Mean")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)

    # Panel 2: Group B mean
    ax = axes[0, 1]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=mean_b,
                    cmap="viridis", s=30, alpha=0.8, edgecolors="0.3", linewidths=0.3,
                    vmin=sc.get_clim()[0], vmax=sc.get_clim()[1])
    ax.set_title(f"{group_labels[1]} Mean")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)

    # Panel 3: Difference
    ax = axes[0, 2]
    vmax_diff = np.nanmax(np.abs(diff))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=diff,
                    cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff,
                    s=30, alpha=0.8, edgecolors="0.3", linewidths=0.3)
    ax.set_title(f"Difference ({group_labels[0]} - {group_labels[1]})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)

    # Panel 4: T-map
    ax = axes[1, 0]
    vmax_t = np.nanmax(np.abs(t_map))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=t_map,
                    cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t,
                    s=30, alpha=0.8, edgecolors="0.3", linewidths=0.3)
    ax.set_title("T-statistic")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)

    # Panel 5: Significant clusters
    ax = axes[1, 1]
    # Build significance mask
    sig_mask = np.zeros(len(t_map), dtype=bool)
    for ci, p_val in enumerate(cluster_pvalues, start=1):
        if p_val < alpha_threshold:
            sig_mask |= (cluster_labels == ci)

    colors = np.where(sig_mask, t_map, 0.0)
    vmax_sig = np.nanmax(np.abs(colors)) if sig_mask.any() else 1.0
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                    cmap="RdBu_r", vmin=-vmax_sig, vmax=vmax_sig,
                    s=30, alpha=0.8, edgecolors="0.3", linewidths=0.3)
    n_sig = sum(1 for p in cluster_pvalues if p < alpha_threshold)
    ax.set_title(f"Significant Clusters (n={n_sig}, p<{alpha_threshold})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)

    # Panel 6: Histogram
    ax = axes[1, 2]
    ax.hist(diff, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Vertex-Level Differences")
    ax.set_xlabel(f"Difference ({group_labels[0]} - {group_labels[1]})")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved band comparison: %s", output_path)


def plot_wholebrain_summary(
    band_results: dict[str, dict],
    coords: np.ndarray,
    output_path: str | Path,
    group_labels: tuple[str, str] = ("Group A", "Group B"),
) -> None:
    """Summary figure showing t-maps for all bands and features.

    Parameters
    ----------
    band_results : dict
        band_name -> {"t_map": ndarray, "cluster_labels": ndarray,
        "cluster_pvalues": list[float]}
    coords : ndarray, shape (n_vertices, 3)
    output_path : Path
    group_labels : tuple
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bands = len(band_results)
    if n_bands == 0:
        return

    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4))
    if n_bands == 1:
        axes = [axes]

    fig.suptitle(
        f"Whole-Brain T-maps: {group_labels[0]} vs {group_labels[1]}",
        fontsize=14, fontweight="bold",
    )

    for ax, (band_name, res) in zip(axes, band_results.items()):
        t_map = res["t_map"]
        vmax = np.nanmax(np.abs(t_map))
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], c=t_map,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            s=20, alpha=0.8, edgecolors="0.3", linewidths=0.2,
        )
        # Mark significant clusters
        sig_vertices = np.zeros(len(t_map), dtype=bool)
        for ci, p_val in enumerate(res.get("cluster_pvalues", []), start=1):
            if p_val < 0.05:
                sig_vertices |= (res["cluster_labels"] == ci)
        if sig_vertices.any():
            ax.scatter(
                coords[sig_vertices, 0], coords[sig_vertices, 1],
                facecolors="none", edgecolors="black", linewidths=1.5,
                s=40, zorder=5,
            )

        ax.set_title(band_name, fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        fig.colorbar(sc, ax=ax, shrink=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved summary figure: %s", output_path)
