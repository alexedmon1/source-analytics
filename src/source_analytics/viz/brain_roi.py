"""Mouse brain visualization with ROIs colored by effect size.

Three entry points:
- ``plot_brain_roi_mosaic``: 2D atlas slice mosaic (coronal, axial, sagittal)
  with anatomy background.  Publication-ready; no 3D dependencies.
- ``render_posthoc_mosaics``: batch helper — reads a posthoc CSV, groups by
  facet columns, and calls ``plot_brain_roi_mosaic`` for every group.
- ``plot_brain_roi``: 3D rendered views (dorsal, lateral, posterior) using PyVista.

Requires: nibabel, matplotlib.  PyVista needed only for ``plot_brain_roi``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from ..atlas import find_atlas_dir, load_atlas, load_roi_mapping
from .palettes import get_diverging_cmap_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_name_to_id(roi_mapping: dict) -> dict[str, int]:
    """Map ROI abbreviation -> atlas label ID."""
    name_to_id = {}
    for lid, info in roi_mapping["rois"].items():
        name_to_id[info["abbreviation"]] = int(lid)
    return name_to_id


def _build_label_to_value(
    region_values: dict[str, float],
    roi_categories: dict[str, list[str]],
    roi_mapping: dict,
) -> dict[int, float]:
    """Map atlas label ID -> scalar value via region_values + roi_categories."""
    name_to_id = _build_name_to_id(roi_mapping)
    label_to_value: dict[int, float] = {}
    for region, roi_names in roi_categories.items():
        for rname in roi_names:
            if rname in name_to_id:
                label_to_value[name_to_id[rname]] = region_values.get(region, 0.0)
    return label_to_value


def _region_to_label_ids(
    roi_categories: dict[str, list[str]],
    roi_mapping: dict,
) -> dict[str, list[int]]:
    """Map region names from roi_categories -> list of atlas label IDs."""
    name_to_id = _build_name_to_id(roi_mapping)
    result = {}
    for region, roi_names in roi_categories.items():
        ids = [name_to_id[n] for n in roi_names if n in name_to_id]
        if ids:
            result[region] = ids
    return result


# ---------------------------------------------------------------------------
# 2-D mosaic (slice-based) — preferred for publication
# ---------------------------------------------------------------------------

# Default slice positions (voxel indices into the Antwerp atlas 64×256×50)
DEFAULT_CORONAL_SLICES = [178, 145, 112]   # anterior → posterior
DEFAULT_AXIAL_SLICES = [35, 28, 20]        # dorsal → ventral
DEFAULT_SAGITTAL_SLICES = [18, 25, 42]     # left → midline → right


def _make_slice_rgb(
    label_slice: np.ndarray,
    anat_slice: np.ndarray,
    label_to_value: dict[int, float],
    cmap,
    norm,
    vmin: float,
    vmax: float,
    roi_opacity: float = 0.85,
) -> np.ndarray:
    """Render a 2-D label slice as RGB with anatomy background."""
    gray = anat_slice / max(anat_slice.max(), 1e-6)
    bg = np.stack([gray, gray, gray], axis=-1)
    bg[gray < 0.01] = 1.0

    rgb = bg.copy()
    for lid, val in label_to_value.items():
        mask = label_slice == lid
        if mask.any():
            roi_color = np.array(cmap(norm(np.clip(val, vmin, vmax)))[:3])
            rgb[mask] = roi_opacity * roi_color + (1 - roi_opacity) * bg[mask]

    return np.clip(rgb, 0, 1)


def _add_direction_labels(ax, left, right, bottom, top, fontsize=9):
    """Add L/R/A/P/D/V direction indicators at panel edges."""
    kw = dict(fontsize=fontsize, color="0.4", fontweight="bold",
              ha="center", va="center", transform=ax.transAxes)
    ax.text(0.0, 0.5, left, **{**kw, "ha": "left"})
    ax.text(1.0, 0.5, right, **{**kw, "ha": "right"})
    ax.text(0.5, 0.01, bottom, **{**kw, "va": "bottom"})
    ax.text(0.5, 0.99, top, **{**kw, "va": "top"})


def plot_brain_roi_mosaic(
    region_values: dict[str, float],
    roi_categories: dict[str, list[str]],
    output_path: str | Path,
    *,
    title: str = "",
    cmap_name: str = "RdYlBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float = 0.0,
    colorbar_label: str = "Hedges' g",
    atlas_dir: str | Path | None = None,
    roi_opacity: float = 0.85,
    gamma: float = 0.5,
    coronal_slices: list[int] | None = None,
    axial_slices: list[int] | None = None,
    sagittal_slices: list[int] | None = None,
    figsize: tuple[float, float] = (11, 12),
    dpi: int = 250,
) -> Path:
    """Render a 3×3 slice mosaic of the mouse brain with ROIs colored by a scalar.

    Rows: coronal, axial, sagittal.  Each row shows three slices at the
    positions given (voxel indices).  The skull-stripped anatomy atlas is
    shown as a grayscale background.

    Parameters
    ----------
    region_values : dict
        Region name → scalar value (e.g. Hedges' g).
    roi_categories : dict
        Region name → list of ROI abbreviations.
    output_path : str or Path
        Output PNG path.
    title : str
        Figure suptitle.
    cmap_name : str
        Matplotlib colormap name (diverging recommended).
    vmin, vmax : float, optional
        Symmetric color limits.  Auto-computed if *None*.
    vcenter : float
        Center of the diverging colormap (default 0).
    colorbar_label : str
        Label for the colorbar.
    atlas_dir : str or Path, optional
        Atlas directory.  Auto-detected if *None*.
    roi_opacity : float
        Blending opacity for ROI colors over anatomy (0–1).
    gamma : float
        Gamma correction for anatomy background (< 1 brightens).
    coronal_slices, axial_slices, sagittal_slices : list[int], optional
        Voxel indices for each row.  Three per row.
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

    if coronal_slices is None:
        coronal_slices = DEFAULT_CORONAL_SLICES
    if axial_slices is None:
        axial_slices = DEFAULT_AXIAL_SLICES
    if sagittal_slices is None:
        sagittal_slices = DEFAULT_SAGITTAL_SLICES

    # Load atlas volumes
    atlas_dir_path = find_atlas_dir(atlas_dir)
    label_data, affine = load_atlas(atlas_dir_path)
    roi_mapping = load_roi_mapping(atlas_dir_path)

    # Anatomy background (skull-stripped)
    anat_path = Path(atlas_dir_path) / "Atlas_3DRois_brain.nii.gz"
    anat_data = nib.load(str(anat_path)).get_fdata()
    anat_norm = (anat_data / anat_data.max()) ** gamma

    # Label → value lookup
    label_to_value = _build_label_to_value(region_values, roi_categories, roi_mapping)

    # Color scale
    vals = [v for v in region_values.values() if np.isfinite(v)]
    if not vals:
        logger.warning("No finite values to plot")
        return output_path
    if vmin is None:
        vmin = min(-0.1, min(vals))
    if vmax is None:
        vmax = max(0.1, max(vals))
    abs_max = max(abs(vmin - vcenter), abs(vmax - vcenter))
    vmin, vmax = vcenter - abs_max, vcenter + abs_max

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.colormaps.get_cmap(cmap_name)

    def _vox_to_mm(axis, idx):
        return affine[axis, axis] * idx + affine[axis, 3]

    # --- Figure layout ---
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        4, 3, height_ratios=[0.9, 1.4, 0.9, 0.06],
        hspace=0.08, wspace=0.08,
        left=0.02, right=0.98, top=0.94, bottom=0.05,
    )
    fig.suptitle(title, fontsize=17, fontweight="bold")

    slice_kw = dict(label_to_value=label_to_value, cmap=cmap, norm=norm,
                    vmin=vmin, vmax=vmax, roi_opacity=roi_opacity)

    # Row 0: Coronal (Y slices) — axes: X (L-R) horizontal, Z (D-V) vertical
    for col, yi in enumerate(coronal_slices):
        ax = fig.add_subplot(gs[0, col])
        rgb = _make_slice_rgb(
            label_data[:, yi, :].T, anat_norm[:, yi, :].T, **slice_kw)
        ext = [_vox_to_mm(0, 0), _vox_to_mm(0, label_data.shape[0]),
               _vox_to_mm(2, 0), _vox_to_mm(2, label_data.shape[2])]
        ax.imshow(rgb, extent=ext, aspect="equal", interpolation="nearest",
                  origin="lower")
        ax.set_title(f"Coronal  Y = {_vox_to_mm(1, yi):.1f} mm",
                     fontsize=12, pad=3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-4.5, 4.5)
        ax.axis("off")
        _add_direction_labels(ax, "L", "R", "V", "D")

    # Row 1: Axial (Z slices) — axes: X (L-R) horizontal, Y (A-P) vertical
    for col, zi in enumerate(axial_slices):
        ax = fig.add_subplot(gs[1, col])
        rgb = _make_slice_rgb(
            label_data[:, :, zi].T, anat_norm[:, :, zi].T, **slice_kw)
        ext = [_vox_to_mm(0, 0), _vox_to_mm(0, label_data.shape[0]),
               _vox_to_mm(1, 0), _vox_to_mm(1, label_data.shape[1])]
        ax.imshow(rgb, extent=ext, aspect="equal", interpolation="nearest",
                  origin="lower")
        ax.set_title(f"Axial  Z = {_vox_to_mm(2, zi):.1f} mm",
                     fontsize=12, pad=3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-9, 9)
        ax.axis("off")
        _add_direction_labels(ax, "L", "R", "P", "A")

    # Row 2: Sagittal (X slices) — axes: Y (A-P) horizontal, Z (D-V) vertical
    for col, xi in enumerate(sagittal_slices):
        ax = fig.add_subplot(gs[2, col])
        rgb = _make_slice_rgb(
            label_data[xi, :, :].T, anat_norm[xi, :, :].T, **slice_kw)
        ext = [_vox_to_mm(1, 0), _vox_to_mm(1, label_data.shape[1]),
               _vox_to_mm(2, 0), _vox_to_mm(2, label_data.shape[2])]
        ax.imshow(rgb, extent=ext, aspect="equal", interpolation="nearest",
                  origin="lower")
        ax.set_title(f"Sagittal  X = {_vox_to_mm(0, xi):.1f} mm",
                     fontsize=12, pad=3)
        ax.set_xlim(-9, 9)
        ax.set_ylim(-4.5, 4.5)
        ax.axis("off")
        _add_direction_labels(ax, "P", "A", "V", "D")

    # Colorbar
    cbar_ax = fig.add_subplot(gs[3, :])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Saved brain ROI mosaic: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Batch helper — read posthoc CSV → one mosaic per facet group
# ---------------------------------------------------------------------------

def render_posthoc_mosaics(
    posthoc_csv: Path,
    roi_categories: dict[str, list[str]],
    output_dir: Path,
    *,
    analysis_name: str = "psd",
    effect_col: str = "hedges_g",
    roi_col: str = "roi",
    facet_cols: list[str] | None = None,
    colorbar_label: str = "Hedges' g",
) -> list[Path]:
    """Render brain ROI mosaics from a posthoc effect-size CSV.

    For each unique combination of *facet_cols* the function builds a
    ``{roi: effect_size}`` dict and delegates to :func:`plot_brain_roi_mosaic`
    using the analysis-specific diverging colormap.

    Parameters
    ----------
    posthoc_csv : Path
        CSV with at least *roi_col*, *effect_col*, and any *facet_cols*.
    roi_categories : dict
        Region name → list of ROI abbreviations (same as the study config).
    output_dir : Path
        Directory for output PNGs.
    analysis_name : str
        Key into ``ANALYSIS_CMAPS`` (e.g. ``"psd"``, ``"aperiodic"``, ``"pac"``).
    effect_col : str
        Column containing the scalar to color-map (default ``"hedges_g"``).
    roi_col : str
        Column identifying individual ROIs or regions.
    facet_cols : list[str] | None
        Columns whose unique combinations define separate mosaics.
        If *None*, a single mosaic is rendered for the whole CSV.
    colorbar_label : str
        Label for the mosaic colorbar.

    Returns
    -------
    list[Path]
        Paths to all saved PNGs.
    """
    posthoc_csv = Path(posthoc_csv)
    if not posthoc_csv.exists():
        logger.warning("Posthoc CSV not found: %s — skipping mosaics", posthoc_csv)
        return []

    df = pd.read_csv(posthoc_csv)
    if df.empty or effect_col not in df.columns or roi_col not in df.columns:
        logger.warning("Posthoc CSV empty or missing columns — skipping mosaics")
        return []

    if facet_cols is None:
        facet_cols = []

    # Validate facet cols exist
    facet_cols = [c for c in facet_cols if c in df.columns]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap_name = get_diverging_cmap_name(analysis_name)
    saved: list[Path] = []

    if facet_cols:
        groups = df.groupby(facet_cols)
    else:
        groups = [(("all",), df)]

    for group_key, group_df in groups:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        # Build {roi: effect} dict — average if duplicates
        roi_effects = (
            group_df.groupby(roi_col)[effect_col]
            .mean()
            .to_dict()
        )
        if not roi_effects:
            continue

        # Map ROIs to regions: roi_categories maps region → [roi_abbrevs],
        # but posthoc CSV may use region names directly (for region-level data)
        # or ROI names. Try both approaches.
        region_values: dict[str, float] = {}
        for region, roi_names in roi_categories.items():
            # Check if ROI names from the CSV match region names (region-level data)
            if region in roi_effects:
                region_values[region] = roi_effects[region]
            else:
                # Check for individual ROI matches
                vals = [roi_effects[r] for r in roi_names if r in roi_effects]
                if vals:
                    region_values[region] = float(np.mean(vals))

        if not region_values:
            continue

        # Build filename from facet values
        tag = "_".join(str(v).replace(" ", "_").replace("/", "-") for v in group_key)
        fname = f"brain_roi_{tag}.png"
        out_path = output_dir / fname

        # Title from facet values
        if facet_cols:
            title_parts = [f"{c}: {v}" for c, v in zip(facet_cols, group_key)]
            title = " | ".join(title_parts)
        else:
            title = analysis_name.upper()

        plot_brain_roi_mosaic(
            region_values,
            roi_categories,
            out_path,
            title=title,
            cmap_name=cmap_name,
            colorbar_label=colorbar_label,
        )
        saved.append(out_path)

    logger.info("Rendered %d brain mosaics in %s", len(saved), output_dir)
    return saved


# ---------------------------------------------------------------------------
# 3-D rendered views (PyVista) — kept for optional use
# ---------------------------------------------------------------------------

VIEWS = {
    "Dorsal": dict(
        position=(0, 0, 15), focal_point=(0, 0, 0), viewup=(0, 1, 0),
    ),
    "Left lateral": dict(
        position=(-15, 0, 2), focal_point=(0, 0, 0), viewup=(0, 0, 1),
    ),
    "Posterior": dict(
        position=(0, -15, 2), focal_point=(0, 0, 0), viewup=(0, 0, 1),
    ),
}


def plot_brain_roi(
    region_values: dict[str, float],
    roi_categories: dict[str, list[str]],
    output_path: str | Path,
    *,
    title: str = "",
    cmap_name: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float = 0.0,
    colorbar_label: str = "Hedges' g",
    atlas_dir: str | Path | None = None,
    views: dict[str, dict] | None = None,
    brain_opacity: float = 0.15,
    roi_opacity: float = 0.85,
    roi_scale: float = 1.0,
    window_size: tuple[int, int] = (800, 800),
    figsize: tuple[float, float] = (18, 6),
    dpi: int = 200,
) -> Path:
    """Render 3D mouse brain with ROIs colored by a scalar value.

    Parameters
    ----------
    region_values : dict
        Mapping of region name -> scalar value (e.g., Hedges' g).
    roi_categories : dict
        Mapping of region name -> list of ROI abbreviations (atlas names).
    output_path : str or Path
        Output PNG path.
    title : str
        Figure title.
    cmap_name : str
        Matplotlib colormap name (diverging recommended).
    vmin, vmax : float, optional
        Color scale limits. Auto-computed if None.
    vcenter : float
        Center of diverging colormap (default 0).
    colorbar_label : str
        Label for the colorbar.
    atlas_dir : str or Path, optional
        Atlas directory. Auto-detected if None.
    views : dict, optional
        Camera views. Uses default dorsal/lateral/posterior if None.
    brain_opacity : float
        Opacity of the whole-brain outline (0-1).
    roi_opacity : float
        Opacity of ROI surfaces (0-1).
    roi_scale : float
        Scale factor for ROI meshes (0-1). Shrinks each ROI toward
        its centroid to reduce overlap. Default 1.0.
    window_size : tuple
        Pixel size for each pyvista render window.
    figsize : tuple
        Matplotlib figure size (inches).
    dpi : int
        Output resolution.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    import pyvista as pv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if views is None:
        views = VIEWS

    atlas_dir_resolved = find_atlas_dir(atlas_dir)
    label_data, affine = load_atlas(atlas_dir_resolved)
    roi_mapping = load_roi_mapping(atlas_dir_resolved)
    region_to_labels = _region_to_label_ids(roi_categories, roi_mapping)

    vals = [v for v in region_values.values() if np.isfinite(v)]
    if not vals:
        logger.warning("No finite values to plot")
        return output_path
    if vmin is None:
        vmin = min(-0.1, min(vals))
    if vmax is None:
        vmax = max(0.1, max(vals))
    abs_max = max(abs(vmin - vcenter), abs(vmax - vcenter))
    vmin, vmax = vcenter - abs_max, vcenter + abs_max

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.colormaps.get_cmap(cmap_name)

    brain_surface = _extract_brain_surface(label_data, affine)

    roi_meshes = {}
    for region, label_ids in region_to_labels.items():
        mesh = _extract_roi_mesh(label_data, affine, label_ids, scale=roi_scale)
        if mesh is not None:
            roi_meshes[region] = mesh

    brain_center = np.array(brain_surface.center)

    pv.OFF_SCREEN = True
    n_views = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=figsize)
    if n_views == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    for ax_idx, (view_name, cam) in enumerate(views.items()):
        p = pv.Plotter(off_screen=True, window_size=list(window_size))
        p.set_background("white")
        p.add_mesh(brain_surface, color="lightgray", opacity=brain_opacity,
                    silhouette=dict(color="gray", line_width=1.5))
        for region, mesh in roi_meshes.items():
            val = region_values.get(region, 0.0)
            rgba = cmap(norm(np.clip(val, vmin, vmax)))
            p.add_mesh(mesh, color=rgba[:3], opacity=roi_opacity,
                        smooth_shading=True)
        pos = np.array(cam["position"]) + brain_center
        p.camera_position = [pos.tolist(), brain_center.tolist(), cam["viewup"]]
        p.enable_parallel_projection()
        p.reset_camera(bounds=brain_surface.bounds)
        p.camera.zoom(1.3)
        img = p.screenshot(transparent_background=False, return_img=True)
        p.close()
        axes[ax_idx].imshow(img)
        axes[ax_idx].set_title(view_name, fontsize=13)
        axes[ax_idx].axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04,
                        pad=0.08, aspect=40, shrink=0.6)
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.90, wspace=0.05)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Saved brain ROI figure: %s", output_path)
    return output_path
