"""On-demand summary figure generators for the ``figure`` CLI command.

Each function reads pre-computed stats CSVs from ``tbl_dir`` and writes
publication-quality figures to ``fig_dir``.  All functions share the
signature ``(tbl_dir, fig_dir, **kwargs) -> list[Path]``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from .theme import (
    apply_theme,
    savefig,
    COLOR_NONSIG,
    COLOR_SIG,
    COLOR_TREND,
    FIGSIZE_WIDE,
    FIGSIZE_CIRCOS,
    FIGSIZE_GLASS_BRAIN,
    FONT_TITLE,
    FONT_SUBTITLE,
    FONT_ANNOTATION,
)
from .palettes import get_diverging_cmap_name

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────

def _infer_analysis(tbl_dir: Path) -> str:
    """Guess analysis name from the table directory path."""
    return tbl_dir.name


def _load_and_normalise(
    tbl_dir: Path,
    analysis: str | None = None,
    **filter_kwargs,
) -> tuple[pd.DataFrame, "TableSchema"] | tuple[None, None]:
    """Load posthoc CSV and apply optional contrast/band filters."""
    from .figure_registry import TABLE_SCHEMAS, load_posthoc

    analysis = analysis or _infer_analysis(tbl_dir)
    schema = TABLE_SCHEMAS.get(analysis)
    if schema is None:
        logger.warning("No table schema for '%s'", analysis)
        return None, None

    df = load_posthoc(tbl_dir, analysis)
    if df is None:
        return None, None

    contrast = filter_kwargs.get("contrast")
    band = filter_kwargs.get("band")
    if contrast and schema.contrast_col in df.columns:
        df = df[df[schema.contrast_col] == contrast]
    if band and schema.band_col and schema.band_col in df.columns:
        df = df[df[schema.band_col] == band]

    if df.empty:
        logger.warning("No rows after filtering (contrast=%s, band=%s)", contrast, band)
        return None, None

    return df, schema


def _build_row_label(row: pd.Series, schema) -> str:
    """Build a human-readable label for a row, combining label + band."""
    label = str(row.get(schema.label_col, ""))
    if schema.band_col and schema.band_col in row.index:
        band_val = str(row[schema.band_col])
        if band_val and band_val.upper() != "NA":
            label = f"{label} | {band_val}"
    return label


# ── Effect heatmap ───────────────────────────────────────────────────

def plot_effect_heatmap(
    tbl_dir: Path,
    fig_dir: Path,
    **kwargs,
) -> list[Path]:
    """Heatmap of estimated differences across measures/bands and contrasts.

    Cell color = estimated difference (estimate, coefficient, or mean diff).
    Non-significant cells are greyed out.  Cells annotated with the estimate
    value plus Hedges' g in parentheses.
    """
    analysis = kwargs.pop("analysis", None) or _infer_analysis(tbl_dir)
    df, schema = _load_and_normalise(tbl_dir, analysis, **kwargs)
    if df is None:
        return []

    apply_theme()

    # Need either _estimate or effect_col
    has_estimate = "_estimate" in df.columns and df["_estimate"].notna().any()
    has_effect = schema.effect_col in df.columns
    if not has_estimate and not has_effect:
        logger.warning("No estimate or effect column found for %s", analysis)
        return []

    # Build pivot: rows = label (dv+band), cols = contrast
    df["_label"] = df.apply(lambda r: _build_row_label(r, schema), axis=1)
    contrasts = df[schema.contrast_col].unique().tolist()
    labels = df["_label"].unique().tolist()

    # Create matrices for estimate (color), effect size (annotation), and significance
    est_matrix = np.full((len(labels), len(contrasts)), np.nan)
    g_matrix = np.full((len(labels), len(contrasts)), np.nan)
    sig_matrix = np.zeros((len(labels), len(contrasts)), dtype=bool)

    label_idx = {l: i for i, l in enumerate(labels)}
    contrast_idx = {c: i for i, c in enumerate(contrasts)}

    for _, row in df.iterrows():
        li = label_idx[row["_label"]]
        ci = contrast_idx[row[schema.contrast_col]]
        if has_estimate:
            est_matrix[li, ci] = row["_estimate"]
        if has_effect:
            g_matrix[li, ci] = row[schema.effect_col]
        sig_matrix[li, ci] = row.get("_significant", False)

    # Use estimate for color if available, otherwise fall back to effect col
    color_matrix = est_matrix if has_estimate else g_matrix

    # Determine symmetric color limits
    vmax = np.nanmax(np.abs(color_matrix))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    cmap_name = get_diverging_cmap_name(analysis)
    cmap = plt.colormaps.get_cmap(cmap_name).copy()
    cmap.set_bad(COLOR_NONSIG)

    # Figure sizing
    n_rows, n_cols = color_matrix.shape
    fig_w = max(6, 2 + n_cols * 2.5)
    fig_h = max(4, 1.5 + n_rows * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Plot full heatmap (all cells)
    im = ax.imshow(
        color_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest",
    )

    # Overlay grey on non-significant cells and annotate
    for i in range(n_rows):
        for j in range(n_cols):
            if not sig_matrix[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=COLOR_NONSIG, alpha=0.55, edgecolor="white", linewidth=0.5,
                ))
            # Build annotation text: estimate value + (g=X.XX)
            est_val = est_matrix[i, j]
            g_val = g_matrix[i, j]
            color_val = color_matrix[i, j]
            if np.isnan(color_val):
                continue
            sig = sig_matrix[i, j]

            # Primary: estimate value
            if has_estimate and not np.isnan(est_val):
                txt = f"{est_val:.2f}"
                if has_effect and not np.isnan(g_val):
                    txt += f"\n(g={g_val:.2f})"
            elif has_effect and not np.isnan(g_val):
                txt = f"g={g_val:.2f}"
            else:
                continue

            if sig:
                txt += " *"
            text_color = "white" if abs(color_val) > vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=FONT_ANNOTATION - 1, color=text_color,
                    fontweight="bold" if sig else "normal")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c.replace("_vs_", "\nvs\n") for c in contrasts], ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(schema.estimate_label)

    paradigm = tbl_dir.parent.name if tbl_dir.parent.name != "tables" else ""
    title = f"{analysis.replace('_', ' ').title()} — {schema.estimate_label}s"
    if paradigm:
        title = f"{paradigm.replace('_', ' ').title()} — {title}"
    ax.set_title(title, fontsize=FONT_TITLE, pad=12)
    ax.grid(False)

    out = fig_dir / f"{analysis}_effect_heatmap.png"
    savefig(fig, out)
    logger.info("Saved effect heatmap: %s", out)
    return [out]


# ── Volcano plot ─────────────────────────────────────────────────────

def plot_volcano(
    tbl_dir: Path,
    fig_dir: Path,
    **kwargs,
) -> list[Path]:
    """Volcano plot: estimated difference (x) vs -log10(p) (y), colored by significance.

    Uses the actual measured difference (estimate/coefficient/mean diff) on the
    x-axis.  Significant points are labeled with name and Hedges' g.
    """
    analysis = kwargs.pop("analysis", None) or _infer_analysis(tbl_dir)
    df, schema = _load_and_normalise(tbl_dir, analysis, **kwargs)
    if df is None:
        return []

    apply_theme()

    if schema.p_col not in df.columns:
        logger.warning("P-value column '%s' not found", schema.p_col)
        return []

    # Prefer _estimate for x-axis, fall back to effect_col
    has_estimate = "_estimate" in df.columns and df["_estimate"].notna().any()
    has_effect = schema.effect_col in df.columns
    x_col = "_estimate" if has_estimate else schema.effect_col
    if x_col not in df.columns:
        logger.warning("No x-axis column found for volcano")
        return []

    df = df.dropna(subset=[x_col, schema.p_col])
    if df.empty:
        return []

    x_vals = df[x_col].values
    pvals = df[schema.p_col].values.clip(min=1e-300)
    neg_log_p = -np.log10(pvals)
    is_sig = df["_significant"].values

    # Trend: p < 0.10 but not significant
    is_trend = (pvals < 0.10) & ~is_sig

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Non-significant
    mask_ns = ~is_sig & ~is_trend
    ax.scatter(x_vals[mask_ns], neg_log_p[mask_ns],
               c=COLOR_NONSIG, s=30, alpha=0.6, edgecolors="none", label="NS")
    # Trend
    if is_trend.any():
        ax.scatter(x_vals[is_trend], neg_log_p[is_trend],
                   c=COLOR_TREND, s=45, alpha=0.8, edgecolors="none", label="Trend")
    # Significant
    if is_sig.any():
        ax.scatter(x_vals[is_sig], neg_log_p[is_sig],
                   c=COLOR_SIG, s=60, alpha=0.9, edgecolors="black", linewidths=0.5,
                   label="Significant")
        # Label significant points with name + g
        df["_label"] = df.apply(lambda r: _build_row_label(r, schema), axis=1)
        for idx in df.index[is_sig]:
            row = df.loc[idx]
            lbl = row["_label"]
            if has_effect and schema.effect_col in row.index:
                g = row[schema.effect_col]
                if not pd.isna(g):
                    lbl += f" (g={g:.2f})"
            ax.annotate(
                lbl,
                (row[x_col], -np.log10(max(row[schema.p_col], 1e-300))),
                fontsize=FONT_ANNOTATION,
                xytext=(5, 5), textcoords="offset points",
            )

    # Reference lines
    ax.axhline(-np.log10(0.05), color=COLOR_SIG, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(-np.log10(0.10), color=COLOR_TREND, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xlabel(schema.estimate_label)
    ax.set_ylabel("-log10(p)")
    ax.legend(loc="upper right", framealpha=0.9)

    paradigm = tbl_dir.parent.name if tbl_dir.parent.name != "tables" else ""
    title = f"{analysis.replace('_', ' ').title()} — Volcano"
    if paradigm:
        title = f"{paradigm.replace('_', ' ').title()} — {title}"
    ax.set_title(title, fontsize=FONT_TITLE, pad=12)

    out = fig_dir / f"{analysis}_volcano.png"
    savefig(fig, out)
    logger.info("Saved volcano plot: %s", out)
    return [out]


# ── Summary circos (connectivity) ────────────────────────────────────

def plot_summary_circos(
    tbl_dir: Path,
    fig_dir: Path,
    **kwargs,
) -> list[Path]:
    """Circos showing only significant region-pair connections.

    Reads ``roi_connectivity_posthoc_region_pair.csv`` and filters to
    significant pairs. One figure per contrast x band with significant results.
    """
    analysis = kwargs.pop("analysis", None) or "roi_connectivity"
    config = kwargs.get("config")

    apply_theme()

    pair_file = tbl_dir / "roi_connectivity_posthoc_region_pair.csv"
    if not pair_file.exists():
        logger.warning("Region-pair posthoc not found: %s", pair_file)
        return []

    df = pd.read_csv(pair_file)

    # Normalise significance
    if "significant" in df.columns:
        if df["significant"].dtype == object:
            df["_significant"] = df["significant"].str.upper().eq("TRUE")
        else:
            df["_significant"] = df["significant"].astype(bool)
    else:
        df["_significant"] = False

    # Apply filters
    contrast_filter = kwargs.get("contrast")
    band_filter = kwargs.get("band")
    if contrast_filter:
        df = df[df["contrast"] == contrast_filter]
    if band_filter:
        df = df[df["band"] == band_filter]

    sig_df = df[df["_significant"]]
    if sig_df.empty:
        logger.info("No significant region pairs found for circos plot")
        return []

    # Get ROI categories from config or build from region_pair column
    roi_categories = None
    if config and hasattr(config, "roi_categories"):
        roi_categories = config.roi_categories

    outputs = []
    for (cname, metric, band), grp in sig_df.groupby(["contrast", "metric", "band"]):
        if grp.empty:
            continue

        # Parse region pairs and build a connectivity matrix
        regions = set()
        for rp in grp["region_pair"]:
            parts = [p.strip() for p in rp.split(" - ")]
            regions.update(parts)
        regions = sorted(regions)
        n = len(regions)
        reg_idx = {r: i for i, r in enumerate(regions)}

        # Effect size matrix
        matrix = np.zeros((n, n))
        for _, row in grp.iterrows():
            parts = [p.strip() for p in row["region_pair"].split(" - ")]
            if len(parts) == 2 and parts[0] in reg_idx and parts[1] in reg_idx:
                i, j = reg_idx[parts[0]], reg_idx[parts[1]]
                val = row.get("hedges_g", row.get("estimate", 0))
                matrix[i, j] = val
                matrix[j, i] = val

        fig, ax = plt.subplots(figsize=FIGSIZE_CIRCOS, subplot_kw={"polar": True})

        # Draw circos manually: arcs for regions, chords for connections
        n_regions = len(regions)
        angles = np.linspace(0, 2 * np.pi, n_regions, endpoint=False)
        width = 2 * np.pi / n_regions * 0.8

        # Draw region arcs
        for i, (region, angle) in enumerate(zip(regions, angles)):
            ax.bar(angle, 1, width=width, bottom=0.9, alpha=0.6, color=f"C{i % 10}")
            ax.text(angle, 1.15, region, ha="center", va="center",
                    fontsize=FONT_ANNOTATION, rotation=np.degrees(angle) - 90 if angle < np.pi else np.degrees(angle) + 90,
                    rotation_mode="anchor")

        # Draw chords for significant connections
        for _, row in grp.iterrows():
            parts = [p.strip() for p in row["region_pair"].split(" - ")]
            if len(parts) != 2 or parts[0] not in reg_idx or parts[1] not in reg_idx:
                continue
            i, j = reg_idx[parts[0]], reg_idx[parts[1]]
            g = row.get("hedges_g", 0)
            color = COLOR_SIG if abs(g) > 0.5 else COLOR_TREND
            lw = min(3, max(0.5, abs(g)))
            ax.plot([angles[i], angles[j]], [0.9, 0.9],
                    color=color, linewidth=lw, alpha=0.7)

        ax.set_ylim(0, 1.4)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_title(
            f"{cname.replace('_', ' ')} | {metric} | {band}",
            fontsize=FONT_SUBTITLE, pad=20,
        )

        fname = f"circos_sig_{cname}_{metric}_{band}.png"
        out = fig_dir / fname
        savefig(fig, out)
        outputs.append(out)
        logger.info("Saved circos: %s", out)

    return outputs


# ── Summary glass brain (vertex_cluster / vertex_spatial) ────────────

def plot_summary_glass_brain(
    tbl_dir: Path,
    fig_dir: Path,
    **kwargs,
) -> list[Path]:
    """Glass-brain showing significant clusters/vertices.

    For vertex_cluster: reads cluster_results.csv + voxelwise_stats.csv
    For vertex_spatial: reads vertex_spatial_results.csv
    For vertex_specparam: reads vertex_specparam_stats.csv
    """
    from .glass_brain import plot_glass_brain

    analysis = kwargs.pop("analysis", None) or _infer_analysis(tbl_dir)
    data_dir = kwargs.get("data_dir")

    apply_theme()

    # Load vertex coordinates
    coords = _find_coords(tbl_dir, data_dir, analysis)
    if coords is None:
        logger.warning("Could not find source coordinates for glass brain")
        return []

    outputs = []

    if analysis in ("vertex_cluster", "wholebrain"):
        outputs.extend(_glass_brain_vertex_cluster(tbl_dir, fig_dir, coords, **kwargs))
    elif analysis in ("vertex_spatial", "spatial_lmm"):
        outputs.extend(_glass_brain_vertex_spatial(tbl_dir, fig_dir, coords, **kwargs))
    elif analysis in ("vertex_specparam", "specparam_vertex"):
        outputs.extend(_glass_brain_vertex_specparam(tbl_dir, fig_dir, coords, **kwargs))

    return outputs


def _find_coords(tbl_dir: Path, data_dir: Path | None, analysis: str) -> np.ndarray | None:
    """Search for source_coords.csv in data_dir or nearby directories."""
    search_paths = []
    if data_dir:
        search_paths.append(Path(data_dir) / "source_coords.csv")

    # Mirror tbl_dir's position under results/ into the parallel analytics/ tree:
    #   results/[<profile>/]tables/<paradigm>/<analysis>
    #     -> analytics/[<profile>/]<paradigm>/<analysis>/data/
    # Found by naming the `results` ancestor rather than counting levels, so an
    # optional profile segment doesn't shift the walk. (The previous fixed 4-level
    # walk reached `results` with zero slack and failed *silently* — returning None
    # here just drops the glass brain with a warning.)
    results_root = next(
        (anc for anc in tbl_dir.parents if anc.name == "results"), None,
    )
    if results_root is not None:
        analytics_root = results_root.parent / "analytics"
        # ("tables", <paradigm>, <analysis>) or (<profile>, "tables", <paradigm>, <analysis>)
        rel = [p for p in tbl_dir.relative_to(results_root).parts if p != "tables"]
        if rel:
            search_paths.append(
                analytics_root.joinpath(*rel) / "data" / "source_coords.csv"
            )
            # vertex_cluster is the canonical producer of source_coords.csv, so fall
            # back to it within the same profile+paradigm.
            search_paths.append(
                analytics_root.joinpath(*rel[:-1])
                / "vertex_cluster" / "data" / "source_coords.csv"
            )

    for p in search_paths:
        if p.exists():
            df = pd.read_csv(p)
            return df[["x", "y", "z"]].values

    return None


def _glass_brain_vertex_cluster(
    tbl_dir: Path, fig_dir: Path, coords: np.ndarray, **kwargs,
) -> list[Path]:
    """Glass brain from vertex cluster + voxelwise stats."""
    from .glass_brain import plot_glass_brain

    cluster_file = tbl_dir / "cluster_results.csv"
    voxel_file = tbl_dir / "voxelwise_stats.csv"

    if not voxel_file.exists():
        logger.warning("voxelwise_stats.csv not found")
        return []

    vox = pd.read_csv(voxel_file)
    clusters = pd.read_csv(cluster_file) if cluster_file.exists() else pd.DataFrame()

    contrast_filter = kwargs.get("contrast")
    band_filter = kwargs.get("band")

    # Find significant clusters
    if not clusters.empty:
        sig_clusters = clusters[clusters["p_corrected"] < 0.05]
        if contrast_filter:
            sig_clusters = sig_clusters[sig_clusters["contrast"] == contrast_filter]
        if band_filter:
            sig_clusters = sig_clusters[sig_clusters["band"] == band_filter]
    else:
        sig_clusters = pd.DataFrame()

    if sig_clusters.empty:
        logger.info("No significant vertex clusters; plotting top uncorrected t-map")

    # Group by contrast x band x metric and plot the t-map
    if contrast_filter:
        vox = vox[vox["contrast"] == contrast_filter]
    if band_filter:
        vox = vox[vox["band"] == band_filter]

    outputs = []
    for (cname, band, metric), grp in vox.groupby(["contrast", "band", "metric"]):
        grp = grp.sort_values("vertex_idx")
        t_vals = grp["t"].values
        if len(t_vals) != len(coords):
            # Vertex count mismatch — skip
            continue

        title = f"{cname} | {band} | {metric}"
        fname = f"glass_brain_{cname}_{band}_{metric}.png"
        out = fig_dir / fname
        plot_glass_brain(coords, t_vals, title=title, output_path=out, cmap="RdBu_r")
        outputs.append(out)
        logger.info("Saved glass brain: %s", out)

    return outputs


def _glass_brain_vertex_spatial(
    tbl_dir: Path, fig_dir: Path, coords: np.ndarray, **kwargs,
) -> list[Path]:
    """Glass brain from vertex spatial results (significant bands/metrics only)."""
    from .glass_brain import plot_glass_brain

    lmm_file = tbl_dir / "vertex_spatial_results.csv"
    if not lmm_file.exists():
        return []

    df = pd.read_csv(lmm_file)
    # Normalise significance
    if "significant" in df.columns:
        if df["significant"].dtype == object:
            df["_sig"] = df["significant"].str.upper().eq("TRUE")
        else:
            df["_sig"] = df["significant"].astype(bool)
    else:
        df["_sig"] = False

    contrast_filter = kwargs.get("contrast")
    band_filter = kwargs.get("band")
    if contrast_filter:
        df = df[df["contrast"] == contrast_filter]
    if band_filter:
        df = df[df["band"] == band_filter]

    sig_df = df[df["_sig"]]
    if sig_df.empty:
        logger.info("No significant vertex spatial results for glass brain")
        return []

    # For each significant result, try to find per-vertex residuals or use coefficient
    # Since spatial_lmm is a single coefficient per band/metric, we show an info plot
    outputs = []
    residual_file = tbl_dir / "vertex_spatial_residuals.csv"
    if residual_file.exists():
        resid = pd.read_csv(residual_file)
        for _, row in sig_df.iterrows():
            cname, band, metric = row["contrast"], row["band"], row["metric"]
            sub = resid[(resid.get("contrast", "") == cname) &
                        (resid.get("band", "") == band) &
                        (resid.get("metric", "") == metric)]
            if sub.empty or "vertex_idx" not in sub.columns:
                continue
            sub = sub.sort_values("vertex_idx")
            vals = sub.iloc[:, -1].values  # last column is residual
            if len(vals) != len(coords):
                continue
            title = f"Vertex Spatial | {cname} | {band} | {metric}"
            fname = f"glass_brain_slmm_{cname}_{band}_{metric}.png"
            out = fig_dir / fname
            plot_glass_brain(coords, vals, title=title, output_path=out, cmap="RdBu_r")
            outputs.append(out)
            logger.info("Saved vertex spatial glass brain: %s", out)

    return outputs


def _glass_brain_vertex_specparam(
    tbl_dir: Path, fig_dir: Path, coords: np.ndarray, **kwargs,
) -> list[Path]:
    """Glass brain from specparam vertex stats (t-values per vertex)."""
    from .glass_brain import plot_glass_brain

    stats_file = tbl_dir / "vertex_specparam_stats.csv"
    if not stats_file.exists():
        return []

    df = pd.read_csv(stats_file)
    contrast_filter = kwargs.get("contrast")
    if contrast_filter:
        df = df[df["contrast"] == contrast_filter]

    outputs = []
    for (cname, param), grp in df.groupby(["contrast", "parameter"]):
        grp = grp.sort_values("vertex_idx")
        t_vals = grp["t"].values
        if len(t_vals) != len(coords):
            continue
        title = f"Specparam | {cname} | {param}"
        fname = f"glass_brain_specparam_{cname}_{param}.png"
        out = fig_dir / fname
        plot_glass_brain(coords, t_vals, title=title, output_path=out, cmap="PiYG")
        outputs.append(out)
        logger.info("Saved specparam glass brain: %s", out)

    return outputs
