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

# Default atlas intensity template filename (continuous-valued MRI volume)
_ATLAS_INTENSITY_NIFTI = "Atlas_3DRois.nii"
_ATLAS_VOXEL_SCALE = 0.1


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


def _load_atlas_slices(
    atlas_dir: str | Path,
    slice_coords: dict[str, float] | None = None,
) -> dict:
    """Load atlas intensity template and extract center slices.

    Parameters
    ----------
    atlas_dir : Path
        Directory containing Atlas_3DRois.nii.
    slice_coords : dict, optional
        Override slice positions: {"axial_z": 1.5, "coronal_y": 0.0,
        "sagittal_x": 0.0}. Defaults to midline/dorsal center.

    Returns
    -------
    dict with keys: axial_slice, coronal_slice, sagittal_slice,
        axial_extent, coronal_extent, sagittal_extent, vmin_img, vmax_img,
        x_coords, y_coords, z_coords.
    """
    import nibabel as nib

    atlas_dir = Path(atlas_dir)
    nii_path = atlas_dir / _ATLAS_INTENSITY_NIFTI
    if not nii_path.exists():
        raise FileNotFoundError(f"Atlas intensity template not found: {nii_path}")

    nii = nib.load(str(nii_path))
    vol = nii.get_fdata()
    affine = nii.affine.copy()
    affine[:3, :3] *= _ATLAS_VOXEL_SCALE
    affine[:3, 3] *= _ATLAS_VOXEL_SCALE
    nx, ny, nz = vol.shape

    x_coords = affine[0, 3] + np.arange(nx) * affine[0, 0]
    y_coords = affine[1, 3] + np.arange(ny) * affine[1, 1]
    z_coords = affine[2, 3] + np.arange(nz) * affine[2, 2]

    sc = slice_coords or {}
    iz = int(np.argmin(np.abs(z_coords - sc.get("axial_z", 1.5))))
    iy = int(np.argmin(np.abs(y_coords - sc.get("coronal_y", 0.0))))
    ix = int(np.argmin(np.abs(x_coords - sc.get("sagittal_x", 0.0))))

    vmin_img, vmax_img = np.percentile(vol[vol > 0.1], [2, 98])

    return {
        "axial_slice": vol[:, :, iz].T,       # (ny, nx)
        "coronal_slice": vol[:, iy, :].T,      # (nz, nx)
        "sagittal_slice": vol[ix, :, :].T,     # (nz, ny)
        "axial_extent": [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
        "coronal_extent": [x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
        "sagittal_extent": [y_coords[0], y_coords[-1], z_coords[0], z_coords[-1]],
        "vmin_img": float(vmin_img),
        "vmax_img": float(vmax_img),
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "slice_z": float(z_coords[iz]),
        "slice_y": float(y_coords[iy]),
        "slice_x": float(x_coords[ix]),
    }


def plot_anatomical_glass_brain(
    coords: np.ndarray,
    band_data: dict[str, dict],
    output_path: str | Path,
    *,
    atlas_dir: str | Path | None = None,
    title: str = "",
    subtitle: str = "",
    cmap: str = "RdYlBu_r",
    vlim: tuple[float, float] | None = None,
    sig_threshold: float | None = 2.0,
    slice_coords: dict[str, float] | None = None,
    atlas_alpha: float = 0.55,
    colorbar_label: str = "Value",
    sig_marker_size: float = 130,
    nonsig_marker_size: float = 50,
    dpi: int = 200,
) -> None:
    """Plot vertex-level values on anatomical atlas center slices.

    Creates a 3-view (axial, coronal, sagittal) figure with MRI center
    slices as background and scatter points colored by value. Supports
    multiple bands displayed as side-by-side columns.

    Parameters
    ----------
    coords : ndarray, shape (n_vertices, 3)
        Source coordinates (x, y, z) in mm.
    band_data : dict
        Mapping of band_name -> dict with keys:
        - "values" : ndarray, shape (n_vertices,) — values to plot
        Optional keys for labeling:
        - "n_sig" : int — number of significant vertices (for subtitle)
        - "n_total" : int — total vertices
    output_path : Path
        Where to save the figure.
    atlas_dir : Path, optional
        Directory containing Atlas_3DRois.nii. If None, auto-detected via
        ``source_analytics.atlas.find_atlas_dir()``.
    title : str
        Main figure title.
    subtitle : str
        Second line of title.
    cmap : str
        Colormap name for scatter points.
    vlim : tuple, optional
        (vmin, vmax) for color scale. If None, uses [0, max(|values|)].
    sig_threshold : float, optional
        Value threshold for highlighting significant vertices with larger
        markers and black edges. Set to None to disable.
    slice_coords : dict, optional
        Override slice positions: {"axial_z", "coronal_y", "sagittal_x"}.
    atlas_alpha : float
        Transparency of anatomical background (0=invisible, 1=opaque).
    colorbar_label : str
        Label for the colorbar.
    sig_marker_size : float
        Marker size for significant vertices.
    nonsig_marker_size : float
        Marker size for non-significant vertices.
    dpi : int
        Output resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Auto-detect atlas directory
    if atlas_dir is None:
        from ..atlas import find_atlas_dir
        atlas_dir = find_atlas_dir()

    # Load atlas slices
    atlas = _load_atlas_slices(atlas_dir, slice_coords)

    # View definitions
    views = [
        {
            "label": f"Axial (z = {atlas['slice_z']:.1f} mm)",
            "xi": 0, "yi": 1,
            "xlabel": "X (L-R, mm)", "ylabel": "Y (A-P, mm)",
            "slice": atlas["axial_slice"], "extent": atlas["axial_extent"],
            "xlim": [-7.0, 7.0], "ylim": [-10.5, 10.5],
            "annotations": True,
        },
        {
            "label": f"Coronal (y = {atlas['slice_y']:.1f} mm)",
            "xi": 0, "yi": 2,
            "xlabel": "X (L-R, mm)", "ylabel": "Z (D-V, mm)",
            "slice": atlas["coronal_slice"], "extent": atlas["coronal_extent"],
            "xlim": [-7.0, 7.0], "ylim": [-2.0, 4.5],
            "annotations": False,
        },
        {
            "label": f"Sagittal (x = {atlas['slice_x']:.1f} mm)",
            "xi": 1, "yi": 2,
            "xlabel": "Y (A-P, mm)", "ylabel": "Z (D-V, mm)",
            "slice": atlas["sagittal_slice"], "extent": atlas["sagittal_extent"],
            "xlim": [-10.5, 10.5], "ylim": [-2.0, 4.5],
            "annotations": False,
        },
    ]

    # Compute height ratios from data aspect
    h_ratios = []
    for v in views:
        y_range = v["ylim"][1] - v["ylim"][0]
        x_range = v["xlim"][1] - v["xlim"][0]
        h_ratios.append(y_range / x_range)

    n_bands = len(band_data)
    band_names = list(band_data.keys())

    # Compute color limits
    if vlim is None:
        all_vals = np.concatenate([bd["values"] for bd in band_data.values()])
        vmax_val = float(np.ceil(np.nanmax(np.abs(all_vals)) * 10) / 10)
        vlim = (0, vmax_val)

    # Figure size: scale by number of bands
    fig_w = 6.5 * n_bands + 1.0  # extra for colorbar
    fig_h = 16
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        3, n_bands + 1, figure=fig,
        height_ratios=h_ratios,
        width_ratios=[1] * n_bands + [0.05],
        wspace=0.15, hspace=0.10,
    )

    text_fx = [pe.withStroke(linewidth=2, foreground="black")]

    for col_j, band_name in enumerate(band_names):
        bd = band_data[band_name]
        values = bd["values"]
        n_sig = bd.get("n_sig")
        n_total = bd.get("n_total", len(values))
        if n_sig is None and sig_threshold is not None:
            n_sig = int((np.abs(values) > sig_threshold).sum())

        for row_i, view in enumerate(views):
            ax = fig.add_subplot(gs[row_i, col_j])
            xi, yi = view["xi"], view["yi"]

            # Anatomical background
            ax.imshow(
                view["slice"], extent=view["extent"], origin="lower",
                cmap="gray", vmin=atlas["vmin_img"], vmax=atlas["vmax_img"],
                alpha=atlas_alpha, aspect="equal", interpolation="bilinear",
            )

            # Scatter: non-significant
            if sig_threshold is not None:
                nonsig = np.abs(values) <= sig_threshold
                sig = np.abs(values) > sig_threshold
            else:
                nonsig = np.ones(len(values), dtype=bool)
                sig = np.zeros(len(values), dtype=bool)

            if nonsig.any():
                ax.scatter(
                    coords[nonsig, xi], coords[nonsig, yi],
                    c=values[nonsig], cmap=cmap,
                    vmin=vlim[0], vmax=vlim[1],
                    s=nonsig_marker_size, alpha=0.65,
                    edgecolors="white", linewidths=0.5, zorder=2,
                )

            if sig.any():
                ax.scatter(
                    coords[sig, xi], coords[sig, yi],
                    c=values[sig], cmap=cmap,
                    vmin=vlim[0], vmax=vlim[1],
                    s=sig_marker_size, alpha=0.95,
                    edgecolors="black", linewidths=1.3, zorder=3,
                )

            ax.set_xlim(view["xlim"])
            ax.set_ylim(view["ylim"])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)

            # Axis labels
            if row_i == len(views) - 1:
                ax.set_xlabel(view["xlabel"], fontsize=10)
            else:
                ax.set_xticklabels([])
            if col_j == 0:
                ax.set_ylabel(view["ylabel"], fontsize=10)
            else:
                ax.set_yticklabels([])

            # View label on left column
            if col_j == 0:
                ax.text(
                    -0.18, 0.5, view["label"], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", ha="center", va="center",
                    rotation=90, color="#333333",
                )

            # Band title on top row
            if row_i == 0:
                sig_label = f"\n({n_sig}/{n_total} sig.)" if n_sig is not None else ""
                ax.set_title(
                    f"{band_name}{sig_label}",
                    fontsize=12, fontweight="bold", pad=10,
                )

            # L/R, A/P annotations on axial views
            if view["annotations"]:
                xlim, ylim = view["xlim"], view["ylim"]
                for txt, pos, ha, va in [
                    ("L", (xlim[0] + 0.4, 0), "left", "center"),
                    ("R", (xlim[1] - 0.4, 0), "right", "center"),
                    ("A", (0, ylim[1] - 0.4), "center", "top"),
                    ("P", (0, ylim[0] + 0.4), "center", "bottom"),
                ]:
                    ax.text(
                        *pos, txt, fontsize=9, fontweight="bold",
                        ha=ha, va=va, color="white", path_effects=text_fx,
                    )

    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, n_bands])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vlim[0], vmax=vlim[1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(colorbar_label, fontsize=11)
    if sig_threshold is not None:
        cbar.ax.axhline(y=sig_threshold, color="black", linestyle="--", linewidth=1.0)
        cbar.ax.text(
            1.5, sig_threshold, f"t = {sig_threshold}", fontsize=8,
            va="center", transform=cbar.ax.get_yaxis_transform(),
        )

    # Title
    title_text = title
    if subtitle:
        title_text += f"\n{subtitle}"
    if title_text:
        fig.suptitle(title_text, fontsize=13, fontweight="bold", y=0.98)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved anatomical glass brain: %s", output_path)
