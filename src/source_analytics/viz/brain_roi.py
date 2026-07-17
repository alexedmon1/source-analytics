"""Mouse brain visualization with ROIs colored by effect size.

Entry points:
- ``plot_effect_size_mosaic``: compact 1-or-2 row mosaic (all ROIs; plus an
  FDR-thresholded row when any survive).  Preferred for publication.
- ``render_posthoc_mosaics``: batch helper — reads a posthoc CSV, groups by
  facet columns, and calls ``plot_effect_size_mosaic`` for every group.
- ``plot_significance_mosaic``: compact mosaic colored by 1 − p.
- ``plot_brain_roi_mosaic``: 3×3 slice mosaic with anatomy background (legacy;
  used when an already-aggregated region→value dict is all that's available).
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

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : array-like
        Raw p-values.

    Returns
    -------
    np.ndarray
        FDR-corrected q-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    q = pvals * n / rank
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    return np.minimum(q, 1.0)


def _build_name_to_id(roi_mapping: dict) -> dict[str, int]:
    """Map ROI name and abbreviation -> atlas label ID."""
    name_to_id = {}
    for lid, info in roi_mapping["rois"].items():
        name_to_id[info["name"]] = int(lid)
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


def _build_roi_label_to_value(
    roi_values: dict[str, float],
    roi_mapping: dict,
) -> dict[int, float]:
    """Map ROI names directly to atlas label IDs (no region aggregation)."""
    name_to_id = _build_name_to_id(roi_mapping)
    label_to_value: dict[int, float] = {}
    for roi_name, val in roi_values.items():
        if roi_name in name_to_id:
            label_to_value[name_to_id[roi_name]] = val
    return label_to_value


def _roi_boundaries(label_slice: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True on ROI boundary pixels.

    Checks two directions (right and down) so each border is drawn once,
    producing a 1-pixel-wide line. Used by the legacy pixel-boundary path.
    """
    h, w = label_slice.shape
    boundary = np.zeros((h, w), dtype=bool)
    diff_r = label_slice[:, :-1] != label_slice[:, 1:]
    either_r = (label_slice[:, :-1] > 0) | (label_slice[:, 1:] > 0)
    boundary[:, :-1] |= (diff_r & either_r)
    diff_d = label_slice[:-1, :] != label_slice[1:, :]
    either_d = (label_slice[:-1, :] > 0) | (label_slice[1:, :] > 0)
    boundary[:-1, :] |= (diff_d & either_d)
    return boundary


def _draw_roi_contours(
    ax,
    label_slice: np.ndarray,
    extent: list[float],
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    alpha: float = 0.4,
    linewidth: float = 0.6,
) -> None:
    """Draw smooth, anti-aliased ROI boundaries on `ax` using sub-pixel contours.

    Uses skimage.measure.find_contours to extract polylines at the 0.5
    level set of each ROI's boolean mask, then draws each polyline with
    matplotlib's default anti-aliasing. Produces clean curves that follow
    the underlying anatomy rather than the jagged voxel-grid edges of
    the legacy pixel-boundary path.

    Silently no-ops if scikit-image is not installed.
    """
    try:
        from skimage import measure
    except ImportError:
        return
    h, w = label_slice.shape
    extent_w = extent[1] - extent[0]
    extent_h = extent[3] - extent[2]
    # find_contours returns (row, col) coordinates in image-array space.
    # Convert to data coordinates using the imshow `extent`.
    unique_labels = np.unique(label_slice)
    unique_labels = unique_labels[unique_labels > 0]
    for lid in unique_labels:
        mask = (label_slice == lid).astype(float)
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ys = extent[2] + (contour[:, 0] / max(h - 1, 1)) * extent_h
            xs = extent[0] + (contour[:, 1] / max(w - 1, 1)) * extent_w
            ax.plot(
                xs, ys,
                color=color, alpha=alpha,
                linewidth=linewidth,
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )


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
    boundary_opacity: float = 0.3,
) -> np.ndarray:
    """Render a 2-D label slice as RGB with anatomy background and ROI outlines."""
    gray = anat_slice / max(anat_slice.max(), 1e-6)
    bg = np.stack([gray, gray, gray], axis=-1)
    bg[gray < 0.01] = 1.0

    rgb = bg.copy()
    for lid, val in label_to_value.items():
        mask = label_slice == lid
        if mask.any():
            roi_color = np.array(cmap(norm(np.clip(val, vmin, vmax)))[:3])
            rgb[mask] = roi_opacity * roi_color + (1 - roi_opacity) * bg[mask]

    boundaries = _roi_boundaries(label_slice)
    rgb[boundaries] = (
        boundary_opacity * np.array([0.0, 0.0, 0.0])
        + (1 - boundary_opacity) * rgb[boundaries]
    )

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
# Compact mosaic: significance (p-value) and effect-size maps
# ---------------------------------------------------------------------------

DEFAULT_COMPACT_CORONAL = 145
DEFAULT_COMPACT_AXIAL = 28
DEFAULT_COMPACT_SAGITTAL = 25

# Axis order used throughout the compact mosaic: 0 = X (sagittal),
# 1 = Y (coronal), 2 = Z (axial).
_AXIS_SAGITTAL, _AXIS_CORONAL, _AXIAL = 0, 1, 2


def _slice_label_areas(
    label_data: np.ndarray,
    axis: int,
    index_of: dict[int, int],
) -> np.ndarray:
    """Cross-sectional area of each target label in every slice along *axis*.

    Returns an ``(n_slices, n_targets)`` array of voxel counts, so plane choice
    can favour planes that show an ROI substantially rather than clipping its
    edge into a one-voxel sliver.
    """
    n = label_data.shape[axis]
    areas = np.zeros((n, len(index_of)), dtype=np.int64)
    for i in range(n):
        sl = np.take(label_data, i, axis=axis)
        labs, counts = np.unique(sl, return_counts=True)
        for lab, cnt in zip(labs, counts):
            k = index_of.get(int(lab))
            if k is not None:
                areas[i, k] = int(cnt)
    return areas


def _pick_informative_slices(
    label_data: np.ndarray,
    target_labels: set[int],
    *,
    defaults: tuple[int, int, int] = (
        DEFAULT_COMPACT_CORONAL,
        DEFAULT_COMPACT_AXIAL,
        DEFAULT_COMPACT_SAGITTAL,
    ),
    shortlist: int = 32,
) -> tuple[int, int, int]:
    """Choose (coronal, axial, sagittal) slices that show *target_labels*.

    The historical fixed planes were chosen once and never adapted to the data,
    so an ROI could be significant yet appear in none of the three panels.

    Planes are scored by, in order: how many target ROIs the three panels cover
    between them; then the *largest* cross-section the worst-served ROI gets in
    any panel (maximin — this is what stops a deep ROI from being technically
    present but clipped to a sliver); then total shown area; then closeness to
    the historical planes, so figures whose ROIs are already well served stay
    visually stable.

    Falls back to *defaults* when *target_labels* is empty or absent from the
    atlas volume.
    """
    targets = {int(t) for t in target_labels if int(t) != 0}
    if not targets:
        return defaults

    present = targets & {int(v) for v in np.unique(label_data)}
    if not present:
        logger.warning(
            "None of the requested ROIs exist in the atlas volume; "
            "falling back to default mosaic slices."
        )
        return defaults

    index_of = {lab: k for k, lab in enumerate(sorted(present))}
    n_targets = len(index_of)

    axis_defaults = {
        _AXIS_CORONAL: defaults[0],
        _AXIAL: defaults[1],
        _AXIS_SAGITTAL: defaults[2],
    }
    areas: dict[int, np.ndarray] = {}
    candidates: dict[int, list[int]] = {}
    for axis in (_AXIS_CORONAL, _AXIAL, _AXIS_SAGITTAL):
        axis_areas = _slice_label_areas(label_data, axis, index_of)
        areas[axis] = axis_areas
        ranked = sorted(
            range(label_data.shape[axis]),
            key=lambda i: (
                -int(np.count_nonzero(axis_areas[i])),
                -int(axis_areas[i].sum()),
                abs(i - axis_defaults[axis]),
            ),
        )
        candidates[axis] = ranked[:shortlist]

    best: tuple[int, int, int] | None = None
    best_key: tuple[int, int, int, int] | None = None
    for yi in candidates[_AXIS_CORONAL]:
        for zi in candidates[_AXIAL]:
            for xi in candidates[_AXIS_SAGITTAL]:
                # Per ROI, the best cross-section any of the three panels gives it.
                shown = np.maximum.reduce([
                    areas[_AXIS_CORONAL][yi],
                    areas[_AXIAL][zi],
                    areas[_AXIS_SAGITTAL][xi],
                ])
                key = (
                    int(np.count_nonzero(shown)),
                    int(shown.min()),
                    int(shown.sum()),
                    -(abs(yi - defaults[0])
                      + abs(zi - defaults[1])
                      + abs(xi - defaults[2])),
                )
                if best_key is None or key > best_key:
                    best_key, best = key, (yi, zi, xi)

    assert best is not None and best_key is not None
    covered = best_key[0]
    if covered < n_targets:
        shown = np.maximum.reduce([
            areas[_AXIS_CORONAL][best[0]],
            areas[_AXIAL][best[1]],
            areas[_AXIS_SAGITTAL][best[2]],
        ])
        missing = sorted(lab for lab, k in index_of.items() if shown[k] == 0)
        logger.warning(
            "Mosaic slices cover %d/%d requested ROIs; no three planes show "
            "label(s) %s simultaneously.",
            covered, n_targets, missing,
        )
    else:
        logger.info(
            "Mosaic slices cover all %d requested ROIs "
            "(coronal=%d, axial=%d, sagittal=%d); smallest cross-section "
            "shown = %d voxels.",
            covered, best[0], best[1], best[2], best_key[1],
        )
    return best


def _load_atlas_and_anat(
    atlas_name: str = "allen",
    gamma: float = 0.5,
):
    """Load atlas label volume, affine, ROI mapping, and anatomy background."""
    atlas_dir = find_atlas_dir(atlas_name=atlas_name)
    label_data, affine = load_atlas(atlas_dir)
    roi_mapping = load_roi_mapping(atlas_dir)

    anat_path = atlas_dir.parent / "Atlas_3DRois_brain.nii.gz"
    anat_data = nib.load(str(anat_path)).get_fdata()
    anat_norm = (anat_data / anat_data.max()) ** gamma

    return label_data, affine, roi_mapping, anat_norm


def _compact_mosaic(
    row_configs: list[tuple[dict[int, float], str]],
    title: str,
    label_data: np.ndarray,
    anat_norm: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    cmap,
    norm,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    colorbar_ticks: list[float] | None = None,
    colorbar_ticklabels: list[str] | None = None,
    roi_opacity: float = 0.85,
    boundary_opacity: float = 0.3,
    coronal_slice: int = DEFAULT_COMPACT_CORONAL,
    axial_slice: int = DEFAULT_COMPACT_AXIAL,
    sagittal_slice: int = DEFAULT_COMPACT_SAGITTAL,
    dpi: int = 300,
) -> Path:
    """Core rendering for compact 1-or-2 row mosaic (coronal, axial, sagittal).

    Parameters
    ----------
    row_configs : list of (label_to_value_dict, row_label_string)
        Each entry becomes one row of the mosaic.  Typically 1 row (all ROIs)
        or 2 rows (all + FDR-thresholded).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(row_configs)
    height_ratios = [1] * n_rows + [0.05]
    # Tightened from (5.0 + 3.5*(n-1)) and width 13 -> 11 to reduce whitespace.
    fig_height = 4.0 + 2.8 * (n_rows - 1)
    fig_width = 11.0

    def _vox_to_mm(axis, idx):
        return affine[axis, axis] * idx + affine[axis, 3]

    def _axis_range(axis):
        lo = _vox_to_mm(axis, 0)
        hi = _vox_to_mm(axis, label_data.shape[axis])
        margin = (hi - lo) * 0.02
        return (lo - margin, hi + margin)

    xr, yr, zr = _axis_range(0), _axis_range(1), _axis_range(2)

    slice_configs = [
        ("Coronal", coronal_slice,
         lambda d, yi: d[:, yi, :].T,
         lambda: [_vox_to_mm(0, 0), _vox_to_mm(0, label_data.shape[0]),
                  _vox_to_mm(2, 0), _vox_to_mm(2, label_data.shape[2])],
         xr, zr, ("L", "R", "V", "D"),
         lambda yi: f"Y = {_vox_to_mm(1, yi):.1f} mm"),
        ("Axial", axial_slice,
         lambda d, zi: d[:, :, zi].T,
         lambda: [_vox_to_mm(0, 0), _vox_to_mm(0, label_data.shape[0]),
                  _vox_to_mm(1, 0), _vox_to_mm(1, label_data.shape[1])],
         xr, yr, ("L", "R", "P", "A"),
         lambda zi: f"Z = {_vox_to_mm(2, zi):.1f} mm"),
        ("Sagittal", sagittal_slice,
         lambda d, xi: d[xi, :, :].T,
         lambda: [_vox_to_mm(1, 0), _vox_to_mm(1, label_data.shape[1]),
                  _vox_to_mm(2, 0), _vox_to_mm(2, label_data.shape[2])],
         yr, zr, ("P", "A", "V", "D"),
         lambda xi: f"X = {_vox_to_mm(0, xi):.1f} mm"),
    ]

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    # Tighter spacing: wspace/hspace minimized; top/bottom margins reduced
    # to claw back vertical whitespace around the panels and colorbar.
    gs = gridspec.GridSpec(
        n_rows + 1, 3, height_ratios=height_ratios,
        hspace=0.04, wspace=0.02,
        left=0.07, right=0.99,
        top=0.92 if n_rows > 1 else 0.88,
        bottom=0.05 if n_rows > 1 else 0.08,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Suppress baked-in pixel boundaries; we'll draw smooth contour
    # boundaries on each axis after imshow.
    slice_kw = dict(cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                    roi_opacity=roi_opacity, boundary_opacity=0.0)
    axes_grid = []

    for row_idx, (label_vals, row_label) in enumerate(row_configs):
        row_axes = []
        for col_idx, (view_name, slice_idx, slice_fn, extent_fn,
                      xlim, ylim, dirs, coord_label) in enumerate(slice_configs):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            row_axes.append(ax)
            lbl_slice = slice_fn(label_data, slice_idx)
            anat_slice = slice_fn(anat_norm, slice_idx)
            rgb = _make_slice_rgb(lbl_slice, anat_slice, label_vals, **slice_kw)
            ext = extent_fn()
            ax.imshow(rgb, extent=ext, aspect="equal", interpolation="nearest",
                      origin="lower")
            # Smooth contour-based ROI outlines (sub-pixel, anti-aliased).
            _draw_roi_contours(
                ax, lbl_slice, ext,
                color=(0.0, 0.0, 0.0),
                alpha=boundary_opacity if boundary_opacity > 0 else 0.4,
                linewidth=0.55,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.axis("off")
            _add_direction_labels(ax, *dirs)
            if row_idx == 0:
                ax.set_title(f"{view_name}  {coord_label(slice_idx)}",
                             fontsize=10, pad=2)
        axes_grid.append(row_axes)

    for row_idx, (_, row_label) in enumerate(row_configs):
        ax_left = axes_grid[row_idx][0]
        bbox = ax_left.get_position()
        fig.text(0.02, (bbox.y0 + bbox.y1) / 2, row_label,
                 fontsize=11, fontweight="bold", rotation=90,
                 ha="center", va="center")

    cbar_ax = fig.add_subplot(gs[n_rows, :])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    if colorbar_ticks is not None:
        cbar.set_ticks(colorbar_ticks)
    if colorbar_ticklabels is not None:
        cbar.set_ticklabels(colorbar_ticklabels)
    cbar.set_label(colorbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Saved: %s", output_path)
    return output_path


def plot_significance_mosaic(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "",
    roi_col: str = "roi",
    p_col: str = "p_value",
    alpha: float = 0.05,
    cmap_name: str = "YlOrRd",
    atlas_name: str = "allen",
    roi_opacity: float = 0.85,
    dpi: int = 300,
) -> Path:
    """Plot a significance mosaic colored by p-value (1-p internally).

    Top row: all ROIs colored by 1-p (continuous).
    Bottom row (if any survive): FDR-corrected ROIs only.

    Parameters
    ----------
    df : DataFrame
        ROI-level posthoc results.  Must contain *roi_col* and *p_col*.
    output_path : str or Path
        Output PNG path.
    title : str
        Figure title.
    roi_col : str
        Column with ROI names matching the atlas.
    p_col : str
        Column with uncorrected p-values.
    alpha : float
        Significance threshold for the FDR-corrected row.
    cmap_name : str
        Sequential colormap name.
    atlas_name : str
        Atlas to load (default ``"allen"``).
    """
    label_data, affine, roi_mapping, anat_norm = _load_atlas_and_anat(atlas_name)

    df = df.copy()
    df["_q_fdr"] = fdr_bh(df[p_col].values)

    uncorr_values = dict(zip(df[roi_col], 1.0 - df[p_col]))
    label_vals_uncorr = _build_roi_label_to_value(uncorr_values, roi_mapping)

    sig = df[df["_q_fdr"] < alpha]
    corr_values = dict(zip(sig[roi_col], 1.0 - sig["_q_fdr"]))
    label_vals_corr = _build_roi_label_to_value(corr_values, roi_mapping)
    n_corr = len(corr_values)

    vmin, vmax = 0.5, 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap(cmap_name)

    row_configs = [(label_vals_uncorr, "p (uncorrected, all ROIs)")]
    if n_corr > 0:
        row_configs.append(
            (label_vals_corr, f"q (FDR-corrected, q < {alpha}, n={n_corr})")
        )

    tick_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    return _compact_mosaic(
        row_configs, title, label_data, anat_norm, affine,
        Path(output_path), cmap, norm, vmin, vmax,
        colorbar_label="p-value",
        colorbar_ticks=tick_vals,
        colorbar_ticklabels=[f"{1 - t:.2f}" for t in tick_vals],
        roi_opacity=roi_opacity, dpi=dpi,
    )


def plot_effect_size_mosaic(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "",
    roi_col: str = "roi",
    effect_col: str = "hedges_g",
    p_col: str = "p_value",
    q_col: str | None = None,
    correction_label: str = "FDR",
    alpha: float = 0.05,
    cmap_name: str = "RdBu_r",
    colorbar_label: str | None = None,
    atlas_name: str = "allen",
    roi_opacity: float = 0.85,
    auto_slices: bool = True,
    dpi: int = 300,
) -> Path:
    """Plot an effect-size mosaic with diverging colormap.

    Top row: all ROIs colored by Hedges' g.
    Bottom row (if any survive): post-hoc-corrected ROIs only, thresholded
    at ``q < alpha``.

    By default the bottom row uses BH-FDR computed from *p_col* internally
    (legacy behavior). Pass ``q_col`` to use an already-corrected q-value
    column from the CSV directly — useful when the study uses a different
    post-hoc correction (e.g., Holm step-down within a (band × power-type)
    cell, per the analysis pipeline) and that column is already populated.

    Parameters
    ----------
    df : DataFrame
        ROI-level posthoc results. Must contain *roi_col*, *effect_col*,
        *p_col*, and optionally *q_col*.
    output_path : str or Path
        Output PNG path.
    title : str
        Figure title.
    roi_col : str
        Column with ROI names matching the atlas.
    effect_col : str
        Column with effect sizes (default ``"hedges_g"``).
    p_col : str
        Column with uncorrected p-values. Used only when *q_col* is None
        (legacy BH-FDR fallback).
    q_col : str, optional
        Column with pre-computed q-values (e.g., Holm-corrected). When
        provided, the bottom-row threshold is applied to this column
        directly and ``p_col`` is ignored for thresholding.
    correction_label : str
        Label inserted into the bottom-row label, e.g., "FDR" or "Holm".
    alpha : float
        Significance threshold for the bottom-row column.
    cmap_name : str
        Diverging colormap name.
    colorbar_label : str, optional
        Custom colorbar label. If *None*, defaults to
        ``"Hedges' g  (blue: KO < WT    red: KO > WT)"``.
    atlas_name : str
        Atlas to load (default ``"allen"``).
    auto_slices : bool
        When True (default), pick the three mosaic planes from the data so the
        post-hoc survivors are actually visible, rather than using the fixed
        historical planes (which could omit a significant ROI entirely — e.g.
        a deep thalamic locus missing every panel). Set False to restore the
        legacy fixed planes.
    """
    label_data, affine, roi_mapping, anat_norm = _load_atlas_and_anat(atlas_name)

    df = df.copy()
    if q_col is not None:
        if q_col not in df.columns:
            raise ValueError(f"q_col={q_col!r} not in DataFrame columns: {list(df.columns)}")
        df["_q_thresh"] = df[q_col]
    else:
        df["_q_thresh"] = fdr_bh(df[p_col].values)

    g_abs_max = max(abs(df[effect_col].min()), abs(df[effect_col].max()), 0.5)
    vmin, vmax = -g_abs_max, g_abs_max
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.colormaps.get_cmap(cmap_name)

    all_values = dict(zip(df[roi_col], df[effect_col]))
    label_vals_all = _build_roi_label_to_value(all_values, roi_mapping)

    sig = df[df["_q_thresh"] < alpha]
    sig_values = dict(zip(sig[roi_col], sig[effect_col]))
    label_vals_sig = _build_roi_label_to_value(sig_values, roi_mapping)
    n_sig = len(sig_values)

    row_configs = [(label_vals_all, "Hedges' g (all ROIs)")]
    if n_sig > 0:
        row_configs.append(
            (label_vals_sig, f"Hedges' g ({correction_label} q < {alpha}, n={n_sig})")
        )

    slice_kwargs: dict[str, int] = {}
    if auto_slices:
        # Drive plane choice off the survivors when there are any: the bottom
        # row is the panel a reader interrogates ROI-by-ROI, so it is the row
        # that must not silently drop one. With no survivors there is only the
        # all-ROI row, so cover that instead.
        targets = set(label_vals_sig) if n_sig > 0 else set(label_vals_all)
        coronal, axial, sagittal = _pick_informative_slices(label_data, targets)
        slice_kwargs = dict(
            coronal_slice=coronal, axial_slice=axial, sagittal_slice=sagittal,
        )

    return _compact_mosaic(
        row_configs, title, label_data, anat_norm, affine,
        Path(output_path), cmap, norm, vmin, vmax,
        colorbar_label=colorbar_label or "Hedges' g  (blue: KO < WT    red: KO > WT)",
        roi_opacity=roi_opacity, dpi=dpi, **slice_kwargs,
    )


# ---------------------------------------------------------------------------
# Batch helper — read posthoc CSV → one mosaic per facet group
# ---------------------------------------------------------------------------

def _expand_regions_to_rois(
    df: pd.DataFrame,
    region_col: str,
    roi_categories: dict[str, list[str]],
) -> pd.DataFrame:
    """Duplicate region-level rows into one row per member ROI.

    Region-level posthoc CSVs (e.g. PAC) have one row per anatomical group.
    ``plot_effect_size_mosaic`` needs atlas ROI names to color individual
    labels, so each region row is replicated across its member ROIs with
    the effect size and p-value carried through unchanged.
    """
    rows = []
    for _, row in df.iterrows():
        member_rois = roi_categories.get(row[region_col], [])
        for roi in member_rois:
            new_row = row.copy()
            new_row["roi"] = roi
            rows.append(new_row)
    if not rows:
        return df.assign(roi=df[region_col])
    return pd.DataFrame(rows).reset_index(drop=True)


def render_posthoc_mosaics(
    posthoc_csv: Path,
    roi_categories: dict[str, list[str]],
    output_dir: Path,
    *,
    analysis_name: str = "psd",
    effect_col: str = "hedges_g",
    roi_col: str = "roi",
    p_col: str = "p_value",
    q_col: str | None = None,
    correction_label: str = "FDR",
    facet_cols: list[str] | None = None,
    colorbar_label: str = "Hedges' g",
    alpha: float = 0.05,
    auto_slices: bool = True,
) -> list[Path]:
    """Render effect-size brain mosaics from a posthoc CSV.

    For each unique combination of *facet_cols* the function delegates to
    :func:`plot_effect_size_mosaic`, producing a compact 1-or-2 row mosaic
    (all ROIs, plus an FDR-thresholded row when any survive *alpha*).

    Parameters
    ----------
    posthoc_csv : Path
        CSV with at least *roi_col*, *effect_col*, *p_col*, and any *facet_cols*.
    roi_categories : dict
        Region name → list of ROI abbreviations. Used only when *roi_col*
        holds region names rather than atlas ROI names (e.g. PAC); rows are
        expanded so each member ROI is colored with the region's effect.
    output_dir : Path
        Directory for output PNGs.
    analysis_name : str
        Key into ``ANALYSIS_CMAPS`` (e.g. ``"psd"``, ``"aperiodic"``, ``"pac"``).
        A leading ``"roi_"`` prefix is stripped for lookup.
    effect_col, roi_col, p_col : str
        Columns for effect size, ROI identifier, and uncorrected p-values.
    q_col : str | None
        Column of pre-corrected q-values (e.g. the study's ``q_value``). When
        given, the thresholded second row uses ``q_col < alpha`` with
        *correction_label*, so the mosaic's ``n=`` matches the study's survivor
        count — instead of the viz layer's internal BH-FDR fallback on *p_col*.
    correction_label : str
        Legend/label for the correction behind *q_col* (e.g. "FDR", "Holm").
    facet_cols : list[str] | None
        Columns whose unique combinations define separate mosaics.
    colorbar_label : str
        Label for the mosaic colorbar.
    alpha : float
        FDR threshold used to build the second row.
    auto_slices : bool
        Pick the three mosaic planes from the data (coverage + maximin over the
        survivors) so no significant ROI is clipped out (MS1 revision). When
        False, fixed default planes are used.

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
    required = {roi_col, effect_col, p_col}
    if df.empty or not required.issubset(df.columns):
        logger.warning(
            "Posthoc CSV empty or missing required columns %s — skipping mosaics",
            required - set(df.columns),
        )
        return []

    if facet_cols is None:
        facet_cols = []
    facet_cols = [c for c in facet_cols if c in df.columns]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap_key = analysis_name[4:] if analysis_name.startswith("roi_") else analysis_name
    cmap_name = get_diverging_cmap_name(cmap_key)

    # Region-level CSV? The roi_col values are region names (keys of
    # roi_categories) rather than atlas ROI names — expand to ROI-level
    # so each atlas label can be colored.
    is_region_level = bool(
        set(df[roi_col].astype(str).unique()) & set(roi_categories.keys())
    )
    if is_region_level:
        df = _expand_regions_to_rois(df, roi_col, roi_categories)
        plot_roi_col = "roi"
    else:
        plot_roi_col = roi_col

    saved: list[Path] = []
    if facet_cols:
        groups = df.groupby(facet_cols)
    else:
        groups = [(("all",), df)]

    for group_key, group_df in groups:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        if group_df.empty:
            continue

        tag = "_".join(str(v).replace(" ", "_").replace("/", "-") for v in group_key)
        out_path = output_dir / f"effect_size_{tag}.png"

        if facet_cols:
            title = " | ".join(str(v) for v in group_key)
        else:
            title = analysis_name.upper()

        try:
            plot_effect_size_mosaic(
                group_df,
                out_path,
                title=title,
                roi_col=plot_roi_col,
                effect_col=effect_col,
                p_col=p_col,
                q_col=q_col if (q_col and q_col in group_df.columns) else None,
                correction_label=correction_label,
                alpha=alpha,
                cmap_name=cmap_name,
                colorbar_label=colorbar_label,
                auto_slices=auto_slices,
            )
            saved.append(out_path)
        except Exception as exc:
            logger.warning("Failed to render mosaic for %s: %s", tag, exc)

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
