"""Attribution maps: ROI effects spread by each parcel's measured localization error.

This is the display the published MS1 paper used for its brain figures (NeuroImage
NIMG-26-1224, Figures 4-6; ported from that revision's ``figures/brainmap.py``), made
general: the atlas and the error table are parameters, so any atlas and any inverse
operator can use it.

What the colour is
------------------
Each parcel's effect (Hedges' g) comes from the analysis **unchanged**. The only thing
added is where it is drawn: the value is spread by a Gaussian whose sigma is that
parcel's own **peak displacement** for the operator that produced the data, and
overlapping contributions combine as a weighted average

    displayed(v) = SUM_i g_i w_i(v) / SUM_i w_i(v)

with each kernel w_i normalised to unit **integral**: a parcel's signal is spread by the
reconstruction, not duplicated. Combining in value space keeps every displayed value
inside the range of the tested g's, and opposite signs cancel numerically instead of
blending toward the neutral midpoint. Opacity over the anatomy is constant and encodes
nothing. Parcels that passed the test get a bold outline: the test was per parcel, so the
parcel boundary carries the inferential claim, not the smoothed field.

Peak displacement
-----------------
Per true parcel i, from the operator's confusion matrix (simulated sources in i, the
parcel j the reconstruction puts the peak in):

    displacement_i = SUM_j P(estimated = j | true = i) * || centroid_j - centroid_i ||

It belongs to one operator on one atlas. A table measured for one pipeline does not apply
to another, so pass the table measured for the data being drawn.

Two things the caption must carry, because the display cannot (``attribution_caption``
returns the standard wording):

* the value at a point is **not** the tested per-ROI effect size;
* the map shows attribution uncertainty, not a second forward pass.

Entry points: ``plot_attribution_mosaic`` (draws the figure), ``attribution_volume`` (the
field alone), ``attribution_field_range`` + ``shared_limit`` (one colour scale for a set of
maps, set from what the maps reach), ``attribution_caption``.

Requires: nibabel, scipy, matplotlib.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ..atlas import load_atlas, load_roi_mapping, resolve_atlas

logger = logging.getLogger(__name__)

# The published figures' look (MS1 brainmap.py), unchanged.
SLICE_MM = {"sagittal": 0.2, "coronal": 1.4, "axial": 0.9}
ATTRIBUTION_CMAP = LinearSegmentedColormap.from_list("kowt", [
    (0.00, "#0a2f5e"), (0.13, "#1a5fb4"), (0.27, "#3987e5"), (0.40, "#7fb0ea"), (0.44, "#c9dcf5"),
    (0.50, "#f4f3f0"), (0.56, "#f7d3c4"), (0.60, "#f09a7c"), (0.73, "#e05a30"), (0.87, "#b02d14"),
    (1.00, "#701a08")])
ROI_ALPHA = 0.92
ANAT_GAMMA = 0.65
SIGMA_FLOOR_MM = 0.20
_SIG_INK = "#0b0b0b"
_INK, _INK_2, _INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"


def _load(atlas):
    """Labels, their affine (10x convention corrected from the header), anatomy, id -> name.

    The anatomy volume is used only as a backdrop raster on the labels' grid; its own
    affine is never read, so a shape check is what ties the two together."""
    spec = resolve_atlas(atlas_name=atlas) if isinstance(atlas, str) else resolve_atlas(atlas)
    labels, affine = load_atlas(spec)
    labels = np.asarray(labels).astype(int)
    anat = np.asarray(nib.load(str(spec.brain_mask)).get_fdata())
    if anat.shape != labels.shape:
        raise ValueError(f"anatomy {anat.shape} != labels {labels.shape}")
    anat = (anat / anat.max()) ** ANAT_GAMMA
    rois = load_roi_mapping(spec)["rois"]
    id2name = {int(k): v["name"] for k, v in rois.items() if int(k) != 0 and v.get("name")}
    return labels, affine, anat, id2name


def attribution_volume(labels: np.ndarray, affine: np.ndarray, id2name: dict[int, str],
                       effects: dict[str, float], sigma_mm: dict[str, float]) -> np.ndarray:
    """Per-parcel effects spread by their own sigma (mm), combined as a weighted average.

    Voxels no kernel reaches are NaN. Parcels named in *effects* but absent from the atlas
    raise: a silently skipped parcel would vanish from the figure."""
    from scipy.ndimage import gaussian_filter

    missing = sorted(set(effects) - set(id2name.values()))
    if missing:
        raise ValueError(f"parcels not in the atlas: {missing}")
    no_sigma = sorted(set(effects) - set(sigma_mm))
    if no_sigma:
        raise ValueError(f"no displacement for: {no_sigma}")
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    num = np.zeros(labels.shape, dtype=np.float64)
    den = np.zeros(labels.shape, dtype=np.float64)
    for lid, name in id2name.items():
        if name not in effects:
            continue
        ind = (labels == lid).astype(np.float64)
        n = ind.sum()
        if n == 0:
            continue
        w = gaussian_filter(ind, sigma=max(float(sigma_mm[name]), SIGMA_FLOOR_MM) / zooms, mode="constant")
        w /= n                                   # unit integral: spread, not duplicated
        num += effects[name] * w
        den += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-12, num / np.maximum(den, 1e-300), np.nan)


def attribution_field_range(rows: pd.DataFrame | list[tuple[str, pd.DataFrame]], displacement: pd.DataFrame,
                            *, atlas="allen", roi_col: str = "roi",
                            effect_col: str = "hedges_g") -> list[tuple[float, float]]:
    """(min, max) of the displayed field per row, without drawing anything.

    Use it to set a shared ``g_limit`` the way the published figures did: from the largest
    |value| the maps of a set actually reach, not the largest raw effect. A limit set from a
    value the field never attains leaves part of the colour ramp unused and the maps pale.
    """
    if isinstance(rows, pd.DataFrame):
        rows = [("", rows)]
    labels, affine, _, id2name = _load(atlas)
    name2id = {v: k for k, v in id2name.items()}
    sigma = dict(zip(displacement["roi"], displacement["displacement_mm"]))
    out = []
    for _, eff in rows:
        effects = dict(zip(eff[roi_col], eff[effect_col].astype(float)))
        vol = attribution_volume(labels, affine, id2name, effects, sigma)
        ok = np.isfinite(vol) & np.isin(labels, [name2id[n] for n in effects])
        out.append((float(vol[ok].min()), float(vol[ok].max())))
    return out


def shared_limit(ranges: list[tuple[float, float]], step: float = 0.05) -> float:
    """Smallest multiple of *step* covering every range: one colour scale for a set of maps."""
    peak = max(max(abs(lo), abs(hi)) for lo, hi in ranges)
    return float(np.ceil(peak / step - 1e-9) * step)


def _mm_to_vox(affine, axis, mm):
    return int(round((mm - affine[axis, 3]) / affine[axis, axis]))


def _cut(vol, axis, idx):
    sl = [slice(None)] * 3
    sl[axis] = idx
    return vol[tuple(sl)].T


def _views(affine, slice_mm):
    z = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return [(0, _mm_to_vox(affine, 0, slice_mm["sagittal"]), f"Sagittal  X = {slice_mm['sagittal']} mm",
             ("P", "A", "V", "D"), z[2] / z[1]),
            (1, _mm_to_vox(affine, 1, slice_mm["coronal"]), f"Coronal  Y = {slice_mm['coronal']} mm",
             ("L", "R", "V", "D"), z[2] / z[0]),
            (2, _mm_to_vox(affine, 2, slice_mm["axial"]), f"Axial  Z = {slice_mm['axial']} mm",
             ("L", "R", "P", "A"), z[1] / z[0])]


def _panel(ax, field2d, anat2d, labels2d, sig_ids, drawn_ids, norm, aspect):
    ax.imshow(np.ma.masked_where(anat2d <= 0.02, anat2d), origin="lower", cmap="gray", vmin=0, vmax=1,
              interpolation="bilinear", zorder=1)
    inside = np.isin(labels2d, drawn_ids)
    ax.imshow(np.ma.masked_invalid(np.where(inside, field2d, np.nan)), origin="lower", cmap=ATTRIBUTION_CMAP,
              norm=norm, alpha=ROI_ALPHA, interpolation="nearest", zorder=2)
    b = np.zeros_like(labels2d, dtype=bool)
    b[:, :-1] |= (labels2d[:, :-1] != labels2d[:, 1:]) & ((labels2d[:, :-1] > 0) | (labels2d[:, 1:] > 0))
    b[:-1, :] |= (labels2d[:-1, :] != labels2d[1:, :]) & ((labels2d[:-1, :] > 0) | (labels2d[1:, :] > 0))
    ax.imshow(np.ma.masked_where(~b, np.ones_like(b, dtype=float)), origin="lower", cmap="gray_r", vmin=0,
              vmax=1.9, zorder=3, interpolation="nearest")
    for lid in sig_ids:
        m = (labels2d == lid).astype(float)
        if m.sum():
            ax.contour(m, levels=[0.5], colors=[_SIG_INK], linewidths=1.5, zorder=4)
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_attribution_mosaic(
    rows: pd.DataFrame | list[tuple[str, pd.DataFrame]],
    output_path: str | Path,
    displacement: pd.DataFrame,
    *,
    atlas="allen",
    g_limit: float,
    roi_col: str = "roi",
    effect_col: str = "hedges_g",
    significant_col: str | None = "significant",
    significance_label: str = "q < 0.05",
    title: str = "",
    colorbar_label: str = "Hedges' g  (blue: KO < WT    red: KO > WT)",
    slice_mm: dict[str, float] | None = None,
    width_in: float = 7.2,
    dpi: int = 300,
    formats: tuple[str, ...] = ("png",),
) -> list[tuple[float, float]]:
    """Draw attribution maps: one row per (label, effects table), three orthogonal views.

    Parameters
    ----------
    rows : DataFrame, or list of (row label, DataFrame)
        Per-ROI effects: *roi_col*, *effect_col* and (optional) boolean *significant_col*.
        Only parcels listed are coloured, so a cortex-only source space leaves the rest grey.
    displacement : DataFrame
        ``roi``, ``displacement_mm`` for the operator and atlas the effects came from.
    atlas : str or AtlasSpec
        The data's own atlas (e.g. ``"allen26"``, ``"allen"``).
    g_limit : float
        Symmetric colour limit, shared by every map of a set. A map whose field exceeds it
        raises instead of clipping silently; raise the limit.
    formats : tuple of str
        File types written beside *output_path* (its suffix is replaced).

    Returns
    -------
    list of (min, max) of the displayed field, per row.
    """
    if isinstance(rows, pd.DataFrame):
        rows = [("", rows)]
    slice_mm = slice_mm or SLICE_MM
    labels, affine, anat, id2name = _load(atlas)
    name2id = {v: k for k, v in id2name.items()}
    sigma = dict(zip(displacement["roi"], displacement["displacement_mm"]))
    norm = Normalize(-g_limit, g_limit)
    vw = _views(affine, slice_mm)

    fig, axes = plt.subplots(len(rows), 3, figsize=(width_in, 1.75 * len(rows) + 1.35), squeeze=False)
    ranges, n_sig = [], 0
    for r, (row_label, eff) in enumerate(rows):
        effects = dict(zip(eff[roi_col], eff[effect_col].astype(float)))
        vol = attribution_volume(labels, affine, id2name, effects, sigma)
        drawn = [name2id[n] for n in effects]
        sig_names = list(eff.loc[eff[significant_col].astype(bool), roi_col]) if significant_col else []
        n_sig += len(sig_names)
        sig_ids = [name2id[n] for n in sig_names]
        ok = np.isfinite(vol) & np.isin(labels, drawn)
        lo, hi = float(vol[ok].min()), float(vol[ok].max())
        if max(abs(lo), abs(hi)) > g_limit:
            raise ValueError(f"{row_label or 'map'} reaches {max(abs(lo), abs(hi)):.3f}, beyond g_limit "
                             f"{g_limit}; raise it rather than clip silently")
        ranges.append((lo, hi))
        for c, (axis, idx, _, dirs, aspect) in enumerate(vw):
            ax = axes[r][c]
            _panel(ax, _cut(vol, axis, idx), _cut(anat, axis, idx), _cut(labels, axis, idx), sig_ids, drawn,
                   norm, aspect)
            for txt, (x, y, ha, va) in zip(dirs, [(0.012, 0.5, "left", "center"), (0.988, 0.5, "right", "center"),
                                                  (0.5, 0.015, "center", "bottom"), (0.5, 0.985, "center", "top")]):
                ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=5.8, color=_INK_MUTED)
        if row_label:
            axes[r][0].set_ylabel(row_label, fontsize=8, weight="bold", color=_INK, labelpad=12)

    top = 0.86 if len(rows) > 1 else 0.80
    fig.subplots_adjust(left=0.085, right=0.985, top=top, bottom=0.20, wspace=0.10, hspace=0.06)
    for c, (_, _, vtitle, _, _) in enumerate(vw):
        pos = axes[0][c].get_position()
        fig.text(pos.x0 + pos.width / 2, top + 0.02, vtitle, ha="center", va="bottom", fontsize=7.5, color=_INK_2)
    if title:
        fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=8.5, color=_INK, weight="bold")
    if significant_col:
        fig.text(0.5, 0.165, f"bold outline: {significance_label}" if n_sig else
                 f"no parcel reached {significance_label}", ha="center", va="top", fontsize=6.6, color=_INK_2)
    cax = fig.add_axes((0.20, 0.085, 0.60, 0.026))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ATTRIBUTION_CMAP), cax=cax, orientation="horizontal")
    cb.set_label(colorbar_label, color=_INK_2, fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, color=_INK_MUTED, labelsize=6.5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved: %s", output_path)
    return ranges


def attribution_caption(displacement: pd.DataFrame, table_ref: str,
                        significance_text: str = "Bold outlines mark q < 0.05.") -> str:
    """The standard caption (MS1 Figure 4 wording), with this table's displacement range."""
    lo, hi = displacement["displacement_mm"].min(), displacement["displacement_mm"].max()
    return (f"Each parcel's Hedges' g is spread by a Gaussian whose standard deviation is that parcel's own "
            f"measured peak displacement ({lo:.2f}–{hi:.2f} mm; {table_ref}). The map shows where an effect can "
            f"be attributed, not a new estimate; a value at any point is not that parcel's tested effect size. "
            f"{significance_text}")
