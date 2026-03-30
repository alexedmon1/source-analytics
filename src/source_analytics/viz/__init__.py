"""Visualization — R for ROI-level plots, Python for glass brain & connectivity."""

from .constants import (
    BAND_ORDER,
    BAND_FREQ_RANGES,
    BAND_COLORS,
    CC_ROIS,
    GROUP_COLORS,
    GROUP_LABELS,
    METRIC_LABELS,
)
from .glass_brain import (
    plot_glass_brain,
    plot_band_comparison,
    plot_vertex_cluster_summary,
)
from .connectivity_plots import (
    build_roi_matrix,
    build_region_matrix,
    build_significance_matrix,
    plot_circos,
    plot_connectivity_heatmap,
    plot_connectivity_comparison,
    plot_significance_circos,
)
from .brain_roi import (
    fdr_bh,
    plot_brain_roi,
    plot_brain_roi_mosaic,
    plot_effect_size_mosaic,
    plot_significance_mosaic,
    render_posthoc_mosaics,
)
from .palettes import (
    ANALYSIS_CMAPS,
    R_GRADIENT2_COLORS,
    get_diverging_cmap,
    get_sequential_cmap,
    get_diverging_cmap_name,
    get_r_gradient2,
)
from .radar import plot_radar
from .figure_registry import (
    generate_figure,
    list_figure_types,
    FIGURE_REGISTRY,
    FIGURE_TYPES,
    TABLE_SCHEMAS,
)

__all__ = [
    "BAND_ORDER",
    "BAND_FREQ_RANGES",
    "BAND_COLORS",
    "CC_ROIS",
    "GROUP_COLORS",
    "GROUP_LABELS",
    "METRIC_LABELS",
    "plot_glass_brain",
    "plot_band_comparison",
    "plot_vertex_cluster_summary",
    "build_roi_matrix",
    "build_region_matrix",
    "build_significance_matrix",
    "plot_circos",
    "plot_connectivity_heatmap",
    "plot_connectivity_comparison",
    "plot_significance_circos",
    "fdr_bh",
    "plot_brain_roi",
    "plot_brain_roi_mosaic",
    "plot_effect_size_mosaic",
    "plot_significance_mosaic",
    "render_posthoc_mosaics",
    "ANALYSIS_CMAPS",
    "R_GRADIENT2_COLORS",
    "get_diverging_cmap",
    "get_sequential_cmap",
    "get_diverging_cmap_name",
    "get_r_gradient2",
    "plot_radar",
    "generate_figure",
    "list_figure_types",
    "FIGURE_REGISTRY",
    "FIGURE_TYPES",
    "TABLE_SCHEMAS",
]
