"""Atlas integration: map vertex coordinates to anatomical ROI labels."""

from .atlas_utils import (
    AtlasSpec,
    find_atlas_dir,
    header_is_inflated,
    registered_atlases,
    resolve_atlas,
    load_atlas,
    load_roi_mapping,
    load_roi_categories,
    load_vertex_roi_labels,
    extract_roi_timeseries,
)

__all__ = [
    "AtlasSpec",
    "find_atlas_dir",
    "header_is_inflated",
    "registered_atlases",
    "resolve_atlas",
    "load_atlas",
    "load_roi_mapping",
    "load_roi_categories",
    "load_vertex_roi_labels",
    "extract_roi_timeseries",
]
