"""Registry mapping (analysis, figure_type) to generator functions.

Provides a central dispatch for ``source-analytics figure`` without
per-analysis branching logic in the CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

logger = logging.getLogger(__name__)


# ── Column schema per analysis ───────────────────────────────────────

class TableSchema:
    """Describes how to read a posthoc CSV for a given analysis."""

    def __init__(
        self,
        posthoc_file: str,
        estimate_col: str | None = "estimate",
        effect_col: str = "hedges_g",
        p_col: str = "p_value",
        q_col: str = "q_value",
        sig_col: str = "significant",
        label_col: str = "dv",
        band_col: str | None = "band",
        contrast_col: str = "contrast",
        estimate_label: str = "Estimate",
    ):
        self.posthoc_file = posthoc_file
        self.estimate_col = estimate_col
        self.effect_col = effect_col
        self.p_col = p_col
        self.q_col = q_col
        self.sig_col = sig_col
        self.label_col = label_col
        self.band_col = band_col
        self.contrast_col = contrast_col
        self.estimate_label = estimate_label


TABLE_SCHEMAS: dict[str, TableSchema] = {
    "roi_psd": TableSchema(
        posthoc_file="roi_psd_posthoc_global.csv",
        estimate_col="estimate",
        label_col="dv",
        band_col="band",
        estimate_label="Difference",
    ),
    "roi_aperiodic": TableSchema(
        posthoc_file="roi_aperiodic_posthoc_global.csv",
        estimate_col="estimate",
        label_col="dv",
        band_col=None,
        estimate_label="Difference",
    ),
    "roi_evoked": TableSchema(
        posthoc_file="roi_evoked_posthoc_global.csv",
        estimate_col="estimate",
        label_col="dv",
        band_col=None,
        estimate_label="Difference",
    ),
    "roi_connectivity": TableSchema(
        posthoc_file="roi_connectivity_global.csv",
        estimate_col=None,  # computed as mean_a - mean_b
        label_col="metric",
        band_col="band",
        estimate_label="Mean Difference",
    ),
    "roi_cross_freq": TableSchema(
        posthoc_file="roi_pac_global.csv",  # PAC R script output name unchanged
        estimate_col=None,  # computed as mean_a - mean_b
        label_col="freq_pair",
        band_col=None,
        estimate_label="Mean Difference",
    ),
    "roi_network": TableSchema(
        posthoc_file="roi_network_global_pairwise.csv",
        estimate_col=None,
        label_col="gm_name",
        band_col="band",
        effect_col="hedges_g",
        q_col="q_value",
        estimate_label="Difference",
    ),
    "vertex_cluster": TableSchema(
        posthoc_file="cluster_results.csv",
        estimate_col="cluster_stat",
        label_col="metric",
        band_col="band",
        effect_col="peak_t",
        p_col="p_corrected",
        q_col="p_corrected",
        sig_col="p_corrected",
        contrast_col="contrast",
        estimate_label="Cluster Stat",
    ),
    "vertex_spatial": TableSchema(
        posthoc_file="vertex_spatial_results.csv",
        estimate_col="coefficient",
        label_col="metric",
        band_col="band",
        effect_col="t_value",
        p_col="p_value",
        q_col="q_value",
        estimate_label="Coefficient",
    ),
    "vertex_specparam": TableSchema(
        posthoc_file="vertex_specparam_stats.csv",
        estimate_col=None,
        label_col="parameter",
        band_col=None,
        effect_col="hedges_g",
        p_col="p",
        q_col="p",
        estimate_label="Hedges g",
    ),
    "vertex_mvpa": TableSchema(
        posthoc_file="vertex_mvpa_results.csv",
        estimate_col="accuracy",
        label_col="band",
        band_col="band",
        effect_col="accuracy",
        p_col="p_value",
        q_col="p_value",
        estimate_label="Accuracy",
    ),
}

# Backward-compatible aliases for old analysis names
for _old, _new in [
    ("psd", "roi_psd"), ("aperiodic", "roi_aperiodic"), ("evoked", "roi_evoked"),
    ("pac", "roi_cross_freq"), ("roi_pac", "roi_cross_freq"),
    ("wholebrain", "vertex_cluster"), ("spatial_lmm", "vertex_spatial"),
    ("specparam_vertex", "vertex_specparam"), ("mvpa", "vertex_mvpa"),
]:
    if _new in TABLE_SCHEMAS:
        TABLE_SCHEMAS[_old] = TABLE_SCHEMAS[_new]


# ── Figure type registry ─────────────────────────────────────────────

FigureGenerator = Callable[..., list[Path]]

# Populated by register() calls at module import
FIGURE_REGISTRY: dict[tuple[str, str], FigureGenerator] = {}
FIGURE_TYPES: dict[str, list[str]] = {}


def register(analysis: str, fig_type: str, func: FigureGenerator) -> None:
    """Register a figure generator for (analysis, fig_type)."""
    FIGURE_REGISTRY[(analysis, fig_type)] = func
    FIGURE_TYPES.setdefault(analysis, []).append(fig_type)


def list_figure_types(analysis: str | None = None) -> dict[str, list[str]]:
    """Return available figure types, optionally filtered to one analysis."""
    if analysis:
        return {analysis: FIGURE_TYPES.get(analysis, [])}
    return dict(FIGURE_TYPES)


def generate_figure(
    analysis: str,
    fig_type: str,
    tbl_dir: Path,
    fig_dir: Path,
    **kwargs: Any,
) -> list[Path]:
    """Dispatch to the registered generator for (analysis, fig_type).

    Returns list of generated file paths.
    """
    key = (analysis, fig_type)
    if key not in FIGURE_REGISTRY:
        available = FIGURE_TYPES.get(analysis, [])
        raise ValueError(
            f"No figure type '{fig_type}' for analysis '{analysis}'. "
            f"Available: {available}"
        )
    func = FIGURE_REGISTRY[key]
    kwargs.setdefault("analysis", analysis)
    return func(tbl_dir=tbl_dir, fig_dir=fig_dir, **kwargs)


def load_posthoc(tbl_dir: Path, analysis: str) -> pd.DataFrame | None:
    """Load the posthoc CSV for *analysis* from *tbl_dir*, or None if missing."""
    schema = TABLE_SCHEMAS.get(analysis)
    if schema is None:
        logger.warning("No table schema for analysis '%s'", analysis)
        return None
    path = tbl_dir / schema.posthoc_file
    if not path.exists():
        logger.warning("Posthoc file not found: %s", path)
        return None
    df = pd.read_csv(path)
    # Normalize the 'significant' column to boolean
    if schema.sig_col in df.columns:
        col = df[schema.sig_col]
        if col.dtype == object:
            df["_significant"] = col.str.upper().eq("TRUE")
        elif col.dtype in ("float64", "int64"):
            # For cluster results, sig = p_corrected < 0.05
            df["_significant"] = col < 0.05
        else:
            df["_significant"] = col.astype(bool)
    else:
        df["_significant"] = False

    # Compute estimate for analyses that store mean_a / mean_b instead
    if schema.estimate_col is None and "mean_a" in df.columns and "mean_b" in df.columns:
        df["_estimate"] = df["mean_a"] - df["mean_b"]
    elif schema.estimate_col and schema.estimate_col in df.columns:
        df["_estimate"] = df[schema.estimate_col]
    else:
        df["_estimate"] = pd.NA

    return df


# ── Registration (imports summary_figures) ────────────────────────────

def _register_all() -> None:
    """Wire up all (analysis, fig_type) -> generator mappings."""
    from . import summary_figures as sf

    # Analyses that support the standard heatmap + volcano
    heatmap_analyses = [
        "roi_psd", "roi_aperiodic", "roi_evoked", "roi_connectivity", "roi_cross_freq",
        "vertex_spatial",
    ]
    volcano_analyses = [
        "roi_psd", "roi_aperiodic", "roi_evoked", "roi_connectivity", "roi_cross_freq",
        "vertex_spatial",
    ]

    for a in heatmap_analyses:
        register(a, "effect_heatmap", sf.plot_effect_heatmap)
    for a in volcano_analyses:
        register(a, "volcano", sf.plot_volcano)

    # Connectivity gets circos
    register("roi_connectivity", "circos", sf.plot_summary_circos)

    # Vertex-level analyses get glass_brain
    for a in ("vertex_cluster", "vertex_spatial", "vertex_specparam"):
        register(a, "glass_brain", sf.plot_summary_glass_brain)


_register_all()
