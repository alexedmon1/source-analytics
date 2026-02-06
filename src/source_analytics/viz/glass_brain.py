"""Glass brain visualizations for whole-brain vertex-level results.

Three-view (axial, coronal, sagittal) scatter plots of source-space data,
suitable for mouse EEG source localization results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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


def plot_pvalue_glass_brain(
    source_coords: np.ndarray,
    p_values: np.ndarray,
    electrode_coords: np.ndarray,
    title: str,
    output_path: str | Path,
    *,
    p_threshold: float = 0.05,
    effect_direction: Optional[np.ndarray] = None,
    min_radius: float = 20,
    max_radius: float = 200,
    alpha: float = 0.6,
    show_nonsig: bool = True,
    nonsig_alpha: float = 0.15,
) -> None:
    """Plot source p-values with electrode-distance-based sizing.

    Creates a 3-view glass brain where:
    - Circle SIZE reflects distance from nearest electrode (larger = farther = less certain)
    - Circle COLOR reflects p-value (using hot colormap, darker = more significant)
    - Significant sources (p < threshold) are highlighted

    Parameters
    ----------
    source_coords : ndarray, shape (n_sources, 3)
        Source coordinates (x, y, z) in mm.
    p_values : ndarray, shape (n_sources,)
        P-values for each source.
    electrode_coords : ndarray, shape (n_electrodes, 3)
        Electrode coordinates (x, y, z) in mm.
    title : str
        Figure title.
    output_path : Path
        Where to save the figure.
    p_threshold : float
        Significance threshold (default 0.05).
    effect_direction : ndarray, optional
        Sign of effect (+1 or -1) per source. If provided, uses diverging colormap.
    min_radius : float
        Minimum marker size (for closest to electrodes).
    max_radius : float
        Maximum marker size (for farthest from electrodes).
    alpha : float
        Transparency for significant sources.
    show_nonsig : bool
        Whether to show non-significant sources.
    nonsig_alpha : float
        Transparency for non-significant sources.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable

    n_sources = len(source_coords)

    # Compute distance from each source to nearest electrode
    distances = np.zeros(n_sources)
    for i, src in enumerate(source_coords):
        dists = np.linalg.norm(electrode_coords - src, axis=1)
        distances[i] = np.min(dists)

    # Normalize distances to marker sizes (farther = larger)
    dist_min, dist_max = distances.min(), distances.max()
    if dist_max > dist_min:
        norm_dist = (distances - dist_min) / (dist_max - dist_min)
    else:
        norm_dist = np.zeros(n_sources)

    marker_sizes = min_radius + norm_dist * (max_radius - min_radius)

    # Convert p-values to color intensity (-log10 scale)
    # Clamp to avoid log(0)
    p_clamp = np.clip(p_values, 1e-10, 1.0)
    neg_log_p = -np.log10(p_clamp)

    # Identify significant sources
    sig_mask = p_values < p_threshold

    # Create figure
    fig = plt.figure(figsize=(18, 6))

    # Three views
    view_configs = [
        ("Axial (top)", 0, 1, "X (mm)", "Y (mm)"),
        ("Coronal (front)", 0, 2, "X (mm)", "Z (mm)"),
        ("Sagittal (side)", 1, 2, "Y (mm)", "Z (mm)"),
    ]

    # Colormap setup
    if effect_direction is not None:
        # Diverging colormap for bidirectional effects
        cmap = plt.cm.RdBu_r
        # Color = -log10(p) * sign(effect)
        color_values = neg_log_p * np.sign(effect_direction)
        vmax = np.max(np.abs(color_values[sig_mask])) if sig_mask.any() else 3
        norm = Normalize(vmin=-vmax, vmax=vmax)
        cbar_label = "-log₁₀(p) × sign(effect)"
    else:
        # Sequential colormap (hot) for unsigned p-values
        cmap = plt.cm.hot_r
        color_values = neg_log_p
        norm = Normalize(vmin=0, vmax=max(3, neg_log_p.max()))
        cbar_label = "-log₁₀(p)"

    axes = []
    for idx, (view_title, xi, yi, xlabel, ylabel) in enumerate(view_configs):
        ax = fig.add_subplot(1, 3, idx + 1)
        axes.append(ax)

        # Plot non-significant sources first (background)
        if show_nonsig:
            nonsig_mask = ~sig_mask
            if nonsig_mask.any():
                ax.scatter(
                    source_coords[nonsig_mask, xi],
                    source_coords[nonsig_mask, yi],
                    s=marker_sizes[nonsig_mask],
                    c="lightgray",
                    alpha=nonsig_alpha,
                    edgecolors="gray",
                    linewidths=0.3,
                    zorder=1,
                )

        # Plot significant sources
        if sig_mask.any():
            sc = ax.scatter(
                source_coords[sig_mask, xi],
                source_coords[sig_mask, yi],
                s=marker_sizes[sig_mask],
                c=color_values[sig_mask],
                cmap=cmap,
                norm=norm,
                alpha=alpha,
                edgecolors="black",
                linewidths=0.8,
                zorder=2,
            )

        # Plot electrodes as small markers
        ax.scatter(
            electrode_coords[:, xi],
            electrode_coords[:, yi],
            s=15,
            c="green",
            marker="^",
            alpha=0.7,
            zorder=3,
            label="Electrodes",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(view_title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Add legend for electrode markers
    axes[0].legend(loc="upper left", fontsize=8)

    # Colorbar
    if sig_mask.any():
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(cbar_label, fontsize=10)

    # Title with stats
    n_sig = sig_mask.sum()
    pct_sig = 100 * n_sig / n_sources
    fig.suptitle(
        f"{title}\n{n_sig}/{n_sources} vertices p<{p_threshold} ({pct_sig:.1f}%)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Add size legend
    # Create a separate axis for size legend
    size_legend_ax = fig.add_axes([0.02, 0.02, 0.15, 0.08])
    size_legend_ax.axis("off")

    # Show example sizes
    example_sizes = [min_radius, (min_radius + max_radius) / 2, max_radius]
    example_dists = [dist_min, (dist_min + dist_max) / 2, dist_max]
    for i, (sz, d) in enumerate(zip(example_sizes, example_dists)):
        size_legend_ax.scatter([i * 0.4], [0.5], s=sz, c="gray", alpha=0.5, edgecolors="black")
        size_legend_ax.text(i * 0.4, -0.3, f"{d:.1f}mm", ha="center", fontsize=7)
    size_legend_ax.text(0.4, 1.2, "Distance to electrode", ha="center", fontsize=8)
    size_legend_ax.set_xlim(-0.2, 1.0)
    size_legend_ax.set_ylim(-0.8, 1.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved p-value glass brain: %s", output_path)

    return {
        "n_significant": int(n_sig),
        "pct_significant": float(pct_sig),
        "distance_range": (float(dist_min), float(dist_max)),
    }


def plot_band_comparison(
    coords: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    t_map: np.ndarray,
    cluster_labels: np.ndarray | None,
    cluster_pvalues: list[float] | None,
    band_name: str,
    group_labels: tuple[str, str],
    output_path: str | Path,
    *,
    alpha_threshold: float = 0.05,
    p_corrected: np.ndarray | None = None,
) -> None:
    """Generate 6-panel comparison figure for a single band/metric.

    Panels:
    1. Group A mean
    2. Group B mean
    3. Difference (A - B)
    4. T-statistic map
    5. Significant clusters/vertices
    6. Histogram of vertex-level differences

    Parameters
    ----------
    coords : ndarray, shape (n_vertices, 3)
    mean_a, mean_b : ndarray, shape (n_vertices,)
        Group-level mean values.
    t_map : ndarray, shape (n_vertices,)
        T-statistics per vertex.
    cluster_labels : ndarray or None, shape (n_vertices,)
        Cluster label per vertex (0 = no cluster). Used with cluster correction.
    cluster_pvalues : list[float] or None
        Corrected p-value per cluster. Used with cluster correction.
    band_name : str
    group_labels : tuple of str
        (label_a, label_b) for display.
    output_path : Path
    alpha_threshold : float
        Significance threshold.
    p_corrected : ndarray or None, shape (n_vertices,)
        Per-vertex corrected p-values (e.g., from TFCE). If provided, used
        instead of cluster_labels/cluster_pvalues for significance.
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

    # Panel 5: Significant clusters/vertices
    ax = axes[1, 1]
    # Build significance mask
    sig_mask = np.zeros(len(t_map), dtype=bool)
    if p_corrected is not None:
        sig_mask = p_corrected < alpha_threshold
        correction_label = "TFCE Vertices"
    elif cluster_pvalues is not None and cluster_labels is not None:
        for ci, p_val in enumerate(cluster_pvalues, start=1):
            if p_val < alpha_threshold:
                sig_mask |= (cluster_labels == ci)
        correction_label = "Clusters"
    else:
        correction_label = "Vertices"

    colors = np.where(sig_mask, t_map, 0.0)
    vmax_sig = np.nanmax(np.abs(colors)) if sig_mask.any() else 1.0
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                    cmap="RdBu_r", vmin=-vmax_sig, vmax=vmax_sig,
                    s=30, alpha=0.8, edgecolors="0.3", linewidths=0.3)
    n_sig = int(sig_mask.sum())
    ax.set_title(f"Significant {correction_label} (n={n_sig}, p<{alpha_threshold})")
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
        # Mark significant vertices/clusters
        sig_vertices = np.zeros(len(t_map), dtype=bool)
        if "p_corrected" in res:
            sig_vertices = res["p_corrected"] < 0.05
        else:
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
