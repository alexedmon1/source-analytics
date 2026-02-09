"""Connectivity circos diagrams and heatmaps for ROI-level results.

Produces publication-quality 3-panel comparison figures (Group A | Group B |
Difference) in both circos/chord and annotated-heatmap styles.  All 46 ROIs
are shown individually, ordered and colored by their 10 anatomical region
groups.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Short label helpers
# ---------------------------------------------------------------------------

def _short_roi_label(roi: str) -> str:
    """Shorten ROI names for plot labels.

    ``Cortex_Frontal_Association_L`` → ``Frontal Assoc L``
    ``Olfactory_Bulb_R`` → ``Olfactory Bulb R``
    """
    s = roi
    # Strip common prefixes
    for prefix in ("Cortex_", "Corpus_Callosum_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    # Hemisphere suffix
    side = ""
    if s.endswith("_L"):
        s, side = s[:-2], " L"
    elif s.endswith("_R"):
        s, side = s[:-2], " R"

    # Underscore → space, abbreviate long words
    s = s.replace("_", " ")
    s = s.replace("Somatosensory 1", "S1")
    s = s.replace("Association", "Assoc")
    s = s.replace("Retrosplenial", "RSC")

    return s + side


# ---------------------------------------------------------------------------
# ROI matrix construction (full 46×46, ordered by region)
# ---------------------------------------------------------------------------

def build_roi_matrix(
    edges_df: pd.DataFrame,
    roi_categories: dict[str, list[str]],
    metric: str,
    group: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """Build a full ROI-pair matrix, with ROIs ordered by region group.

    Parameters
    ----------
    edges_df : DataFrame
        Edge-level connectivity (one band), with columns
        ``roi1, roi2, group, <metric>``.  Should already be filtered
        to a single band.
    roi_categories : dict
        ``{region_name: [roi_name, ...]}`` mapping.
    metric : str
        Column name to average (``"coherence"`` or ``"imag_coherence"``).
    group : str, optional
        If given, filter ``edges_df`` to this group first.

    Returns
    -------
    matrix : ndarray, shape (n_rois, n_rois)
        Symmetric mean-connectivity matrix.
    roi_labels : list[str]
        ROI names in matrix order (ordered by region).
    region_names : list[str]
        Unique region names in order.
    region_sizes : list[int]
        Number of ROIs per region (same order as *region_names*).
    """
    if group is not None:
        edges_df = edges_df[edges_df["group"] == group]

    # Build ordered ROI list: sort regions alphabetically, ROIs within each
    region_names = sorted(roi_categories.keys())
    roi_labels: list[str] = []
    region_sizes: list[int] = []
    for region in region_names:
        rois = sorted(roi_categories[region])
        roi_labels.extend(rois)
        region_sizes.append(len(rois))

    roi_idx = {name: i for i, name in enumerate(roi_labels)}
    n = len(roi_labels)

    sums = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)

    for _, row in edges_df.iterrows():
        r1, r2 = row["roi1"], row["roi2"]
        i = roi_idx.get(r1)
        j = roi_idx.get(r2)
        if i is None or j is None:
            continue
        val = row[metric]
        sums[i, j] += val
        sums[j, i] += val
        counts[i, j] += 1
        counts[j, i] += 1

    with np.errstate(invalid="ignore"):
        matrix = np.where(counts > 0, sums / counts, 0.0)

    return matrix, roi_labels, region_names, region_sizes


# Keep backward-compatible alias
build_region_matrix = build_roi_matrix


def build_significance_matrix(
    posthoc_df: pd.DataFrame,
    roi_labels: list[str],
    region_names: list[str],
    region_sizes: list[int],
    band: str,
    metric: str,
    *,
    p_col: str = "p_value",
    alpha: float = 0.05,
) -> np.ndarray | None:
    """Map region-pair p-values to an ROI-level boolean significance mask.

    Parameters
    ----------
    posthoc_df : DataFrame
        Region-pair posthoc results with columns
        ``band, metric, region_pair, <p_col>``.
    roi_labels, region_names, region_sizes
        From :func:`build_roi_matrix`.
    band, metric : str
        Filter the posthoc table to this band/metric.
    p_col : str
        Column to threshold (``"p_value"`` for uncorrected,
        ``"q_value"`` for corrected).
    alpha : float
        Significance threshold.

    Returns
    -------
    sig_mask : ndarray (n, n) of bool, or None if no matching rows.
    """
    sub = posthoc_df[
        (posthoc_df["band"] == band) & (posthoc_df["metric"] == metric)
    ]
    if sub.empty:
        return None

    n = len(roi_labels)

    # Build ROI index → region name lookup
    roi_to_region: dict[str, str] = {}
    offset = 0
    for ri, sz in enumerate(region_sizes):
        for j in range(sz):
            roi_to_region[roi_labels[offset + j]] = region_names[ri]
        offset += sz

    # Collect significant region pairs as frozensets for fast lookup
    sig_pairs: set[frozenset[str]] = set()
    for _, row in sub.iterrows():
        if row[p_col] < alpha:
            parts = [p.strip() for p in row["region_pair"].split(" - ")]
            sig_pairs.add(frozenset(parts))

    if not sig_pairs:
        return np.zeros((n, n), dtype=bool)

    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            rp = frozenset([roi_to_region[roi_labels[i]],
                            roi_to_region[roi_labels[j]]])
            if rp in sig_pairs:
                mask[i, j] = True
                mask[j, i] = True

    return mask


def plot_significance_circos(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    roi_labels: list[str],
    region_names: list[str],
    region_sizes: list[int],
    sig_mask: np.ndarray,
    output_path: str | Path,
    *,
    group_labels: tuple[str, str] = ("Group A", "Group B"),
    title: str = "",
    dpi: int = 200,
) -> None:
    """Single-panel circos highlighting significant connections.

    Significant ROI pairs (from *sig_mask*) are drawn opaque; all others
    are ghosted so the significant ones stand out.

    Parameters
    ----------
    mat_a, mat_b : ndarray (n, n)
    roi_labels, region_names, region_sizes : from build_roi_matrix
    sig_mask : ndarray (n, n) of bool
    output_path : Path
    group_labels : tuple[str, str]
    title : str
    dpi : int
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    diff = mat_a - mat_b
    diff_vmax = np.nanmax(np.abs(diff))
    if diff_vmax == 0:
        diff_vmax = 1.0

    n_sig = int(sig_mask[np.triu_indices_from(sig_mask, k=1)].sum())

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_circos(
        diff, roi_labels, region_names, region_sizes, ax,
        cmap="RdBu_r",
        threshold=0.0,
        vmin=-diff_vmax, vmax=diff_vmax,
        sig_mask=sig_mask,
    )

    ax.set_title(
        f"Difference ({group_labels[0]} \u2013 {group_labels[1]})"
        f"\n{n_sig} region pairs p < 0.05 uncorrected",
        fontsize=11, fontweight="bold", pad=12,
    )

    sm = ScalarMappable(
        cmap="RdBu_r", norm=Normalize(vmin=-diff_vmax, vmax=diff_vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal",
        fraction=0.04, pad=0.04, shrink=0.5,
    )
    cbar.set_label("Difference", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved significance circos: %s", output_path)


# ---------------------------------------------------------------------------
# Circos / chord diagram
# ---------------------------------------------------------------------------

def plot_circos(
    matrix: np.ndarray,
    roi_labels: list[str],
    region_names: list[str],
    region_sizes: list[int],
    ax,
    *,
    cmap: str = "YlOrRd",
    threshold: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    linewidth_range: tuple[float, float] = (0.3, 4.0),
    sig_mask: np.ndarray | None = None,
):
    """Draw a circos / chord diagram on *ax*.

    Each of the *n* ROIs gets an individual arc segment, colored by region.
    Region groups are separated by wider gaps.  Chords connect ROI pairs
    above *threshold*.

    Parameters
    ----------
    matrix : ndarray (n, n)
        Symmetric connectivity matrix (full ROI-level).
    roi_labels : list[str]
        ROI names in matrix order.
    region_names : list[str]
        Region group names (determines colour).
    region_sizes : list[int]
        Number of ROIs per region.
    ax : matplotlib Axes
    cmap : str
        Colormap for chords.
    threshold : float
        Omit chords with ``|value| < threshold``.
    vmin, vmax : float, optional
        Colormap range.
    linewidth_range : tuple
        ``(min_lw, max_lw)`` for chord width scaling.
    sig_mask : ndarray (n, n) of bool, optional
        If provided, significant pairs (True) are drawn opaque and thick;
        non-significant pairs are drawn as faint background chords.
        Overrides the magnitude-based alpha scaling.

    Returns
    -------
    ax
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path as MplPath
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    n = len(roi_labels)
    n_regions = len(region_names)

    # Region colours (tab10)
    tab10 = plt.cm.tab10
    region_colors = [tab10(i % 10) for i in range(n_regions)]

    # Map each ROI index → region index and colour
    roi_region_idx = []
    for ri, sz in enumerate(region_sizes):
        roi_region_idx.extend([ri] * sz)
    node_colors = [region_colors[ri] for ri in roi_region_idx]

    # Angular layout: small gap within regions, larger gap between regions
    inner_gap_deg = 0.6
    region_gap_deg = 3.5
    total_gaps = inner_gap_deg * (n - n_regions) + region_gap_deg * n_regions
    arc_deg = (360.0 - total_gaps) / n

    starts: list[float] = []
    angle = 0.0
    roi_i = 0
    for ri, sz in enumerate(region_sizes):
        for j in range(sz):
            starts.append(angle)
            angle += arc_deg
            if j < sz - 1:
                angle += inner_gap_deg
            roi_i += 1
        angle += region_gap_deg

    r_inner, r_outer = 0.88, 0.96
    r_region_outer = 1.0  # outer region band

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw outer region band (single arc per region)
    roi_i = 0
    for ri, sz in enumerate(region_sizes):
        theta1 = starts[roi_i]
        theta2 = starts[roi_i + sz - 1] + arc_deg
        wedge = mpatches.Wedge(
            (0, 0), r_region_outer, theta1, theta2,
            width=r_region_outer - r_outer,
            facecolor=region_colors[ri],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )
        ax.add_patch(wedge)

        # Region label at midpoint of region arc
        mid_angle = np.radians((theta1 + theta2) / 2)
        lx = 1.12 * np.cos(mid_angle)
        ly = 1.12 * np.sin(mid_angle)
        rotation = np.degrees(mid_angle)
        ha = "left"
        if 90 < rotation % 360 < 270:
            rotation += 180
            ha = "right"
        ax.text(
            lx, ly, region_names[ri],
            ha=ha, va="center",
            fontsize=9, fontweight="bold",
            rotation=rotation,
            rotation_mode="anchor",
            color=region_colors[ri],
        )
        roi_i += sz

    # Draw individual ROI arcs
    short_labels = [_short_roi_label(r) for r in roi_labels]
    for i in range(n):
        theta1 = starts[i]
        theta2 = starts[i] + arc_deg
        wedge = mpatches.Wedge(
            (0, 0), r_outer, theta1, theta2,
            width=r_outer - r_inner,
            facecolor=node_colors[i],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.7,
        )
        ax.add_patch(wedge)

        # ROI label
        mid_angle = np.radians((theta1 + theta2) / 2)
        lx = 1.35 * np.cos(mid_angle)
        ly = 1.35 * np.sin(mid_angle)
        rotation = np.degrees(mid_angle)
        ha = "left"
        if 90 < rotation % 360 < 270:
            rotation += 180
            ha = "right"
        ax.text(
            lx, ly, short_labels[i],
            ha=ha, va="center",
            fontsize=6,
            rotation=rotation,
            rotation_mode="anchor",
            color="0.2",
        )

    # Collect upper-triangle values for colour scaling
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            v = matrix[i, j]
            if abs(v) >= threshold:
                vals.append(v)
    if not vals:
        return ax

    vals_arr = np.array(vals)
    if vmin is None:
        vmin = float(vals_arr.min())
    if vmax is None:
        vmax = float(vals_arr.max())

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    abs_max = max(abs(vals_arr.min()), abs(vals_arr.max()))
    lw_min, lw_max = linewidth_range

    # Draw chords (quadratic Bezier through origin)
    # When sig_mask is provided: draw non-sig first (background), then sig on top
    if sig_mask is not None:
        draw_order = [(False, True)]  # (is_sig, draw_now) — non-sig first
        draw_order.append((True, True))
    else:
        draw_order = [(None, True)]  # no mask, single pass

    for is_sig_pass, _ in draw_order:
        for i in range(n):
            for j in range(i + 1, n):
                v = matrix[i, j]
                if abs(v) < threshold:
                    continue

                # Skip chords not in this pass
                if sig_mask is not None:
                    pair_sig = bool(sig_mask[i, j] or sig_mask[j, i])
                    if is_sig_pass and not pair_sig:
                        continue
                    if not is_sig_pass and pair_sig:
                        continue

                a1 = np.radians(starts[i] + arc_deg / 2)
                a2 = np.radians(starts[j] + arc_deg / 2)
                x1, y1 = r_inner * np.cos(a1), r_inner * np.sin(a1)
                x2, y2 = r_inner * np.cos(a2), r_inner * np.sin(a2)

                verts = [(x1, y1), (0, 0), (x2, y2)]
                codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
                path = MplPath(verts, codes)

                if abs_max > 0:
                    frac = abs(v) / abs_max
                else:
                    frac = 0.5

                if sig_mask is not None:
                    if is_sig_pass:
                        lw = lw_min + frac * (lw_max - lw_min) * 1.5
                        alpha = 0.85
                    else:
                        lw = lw_min + frac * (lw_max - lw_min) * 0.5
                        alpha = 0.08
                else:
                    lw = lw_min + frac * (lw_max - lw_min)
                    alpha = 0.15 + frac * 0.60

                color = sm.to_rgba(v)
                patch = mpatches.FancyArrowPatch(
                    path=path,
                    arrowstyle="-",
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                )
                ax.add_patch(patch)

    return ax


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_connectivity_heatmap(
    matrix: np.ndarray,
    roi_labels: list[str],
    region_names: list[str],
    region_sizes: list[int],
    ax,
    *,
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    mask_diagonal: bool = True,
):
    """Annotated symmetric heatmap on *ax* with region separators.

    Parameters
    ----------
    matrix : ndarray (n, n)
        Symmetric connectivity matrix.
    roi_labels : list[str]
        ROI names in matrix order.
    region_names : list[str]
        Region group names.
    region_sizes : list[int]
        Number of ROIs per region.
    ax : matplotlib Axes
    cmap : str
        Colormap.
    vmin, vmax : float, optional
        Colormap range.
    mask_diagonal : bool
        Grey out diagonal cells.

    Returns
    -------
    (ax, im)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(roi_labels)
    display = matrix.copy()

    mask = np.zeros_like(matrix, dtype=bool)
    if mask_diagonal:
        np.fill_diagonal(mask, True)

    display_masked = np.ma.array(display, mask=mask)

    im = ax.imshow(
        display_masked, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest",
    )

    # Short tick labels
    short_labels = [_short_roi_label(r) for r in roi_labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=90, ha="center", fontsize=4)
    ax.set_yticklabels(short_labels, fontsize=4)

    # Region colours for tick labels
    tab10 = plt.cm.tab10
    offset = 0
    for ri, sz in enumerate(region_sizes):
        color = tab10(ri % 10)
        for j in range(sz):
            idx = offset + j
            ax.get_xticklabels()[idx].set_color(color)
            ax.get_yticklabels()[idx].set_color(color)
        offset += sz

    # Region separator lines
    offset = 0
    for ri, sz in enumerate(region_sizes):
        if ri > 0:
            pos = offset - 0.5
            ax.axhline(pos, color="black", linewidth=0.8, alpha=0.6)
            ax.axvline(pos, color="black", linewidth=0.8, alpha=0.6)
        offset += sz

    # Region name annotations along the top
    offset = 0
    for ri, sz in enumerate(region_sizes):
        mid = offset + sz / 2 - 0.5
        color = tab10(ri % 10)
        ax.text(
            mid, -1.8, region_names[ri],
            ha="center", va="bottom", fontsize=4.5,
            fontweight="bold", color=color, rotation=45,
        )
        offset += sz

    # Grey out masked cells
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        facecolor="#e0e0e0", edgecolor="white", linewidth=0.3,
                    )
                )

    return ax, im


# ---------------------------------------------------------------------------
# 3-panel comparison figure
# ---------------------------------------------------------------------------

def plot_connectivity_comparison(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    roi_labels: list[str],
    region_names: list[str],
    region_sizes: list[int],
    output_path: str | Path,
    *,
    plot_type: str = "circos",
    group_labels: tuple[str, str] = ("Group A", "Group B"),
    title: str = "",
    threshold: float = 0.0,
    dpi: int = 300,
) -> None:
    """Three-panel figure: Group A | Group B | Difference.

    Parameters
    ----------
    mat_a, mat_b : ndarray (n, n)
        ROI-level connectivity matrices for two groups.
    roi_labels : list[str]
        ROI names in matrix order.
    region_names : list[str]
        Region group names.
    region_sizes : list[int]
        Number of ROIs per region.
    output_path : Path
        Where to save.
    plot_type : {"circos", "heatmap"}
        Visualization style.
    group_labels : tuple[str, str]
        Display labels for the two groups.
    title : str
        Figure suptitle.
    threshold : float
        For circos: omit chords below this value.
    dpi : int
        Output resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    diff = mat_a - mat_b

    # Shared colour range for group panels
    group_vmax = max(np.nanmax(np.abs(mat_a)), np.nanmax(np.abs(mat_b)))
    group_vmin = 0.0  # coherence is non-negative

    # Diff colour range (symmetric around 0)
    diff_vmax = np.nanmax(np.abs(diff))
    if diff_vmax == 0:
        diff_vmax = 1.0

    if plot_type == "circos":
        # Difference panel: threshold at 1 SD of differences so only
        # the notable connections show, independent of group threshold
        diff_ut = diff[np.triu_indices_from(diff, k=1)]
        diff_thresh = float(np.std(diff_ut))

        fig, axes = plt.subplots(1, 3, figsize=(30, 12))
        for ax_i, (mat, label, cm, lo, hi, thresh) in enumerate([
            (mat_a, group_labels[0], "YlOrRd", group_vmin, group_vmax, threshold),
            (mat_b, group_labels[1], "YlOrRd", group_vmin, group_vmax, threshold),
            (diff, "Difference", "RdBu_r", -diff_vmax, diff_vmax, diff_thresh),
        ]):
            plot_circos(
                mat, roi_labels, region_names, region_sizes, axes[ax_i],
                cmap=cm,
                threshold=thresh,
                vmin=lo, vmax=hi,
            )
            axes[ax_i].set_title(label, fontsize=14, fontweight="bold", pad=12)

        # Colorbars below
        sm_grp = ScalarMappable(
            cmap="YlOrRd", norm=Normalize(vmin=group_vmin, vmax=group_vmax),
        )
        sm_grp.set_array([])
        cbar_grp = fig.colorbar(
            sm_grp, ax=[axes[0], axes[1]], orientation="horizontal",
            fraction=0.04, pad=0.06, shrink=0.6,
        )
        cbar_grp.set_label("Mean connectivity", fontsize=11)

        sm_diff = ScalarMappable(
            cmap="RdBu_r", norm=Normalize(vmin=-diff_vmax, vmax=diff_vmax),
        )
        sm_diff.set_array([])
        cbar_diff = fig.colorbar(
            sm_diff, ax=axes[2], orientation="horizontal",
            fraction=0.04, pad=0.06, shrink=0.6,
        )
        cbar_diff.set_label("Difference", fontsize=11)

    elif plot_type == "heatmap":
        fig, axes = plt.subplots(1, 3, figsize=(30, 9))
        for ax_i, (mat, label, cm, lo, hi) in enumerate([
            (mat_a, group_labels[0], "YlOrRd", group_vmin, group_vmax),
            (mat_b, group_labels[1], "YlOrRd", group_vmin, group_vmax),
            (diff, "Difference", "RdBu_r", -diff_vmax, diff_vmax),
        ]):
            _, im = plot_connectivity_heatmap(
                mat, roi_labels, region_names, region_sizes, axes[ax_i],
                cmap=cm, vmin=lo, vmax=hi,
                mask_diagonal=True,
            )
            axes[ax_i].set_title(label, fontsize=12, fontweight="bold")
            fig.colorbar(im, ax=axes[ax_i], shrink=0.75)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type!r}")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    if plot_type == "heatmap":
        fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved connectivity %s: %s", plot_type, output_path)
