"""Per-analysis color palettes for consistent, grayscale-safe figures.

Each analysis gets a diverging colormap (for effect sizes) and a sequential
colormap.  All chosen from matplotlib's perceptually uniform families where
lightness monotonically encodes magnitude.

Matching hex colors are provided for R ``scale_fill_gradient2()`` calls so
that Python brain mosaics and R heatmaps share the same palette.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

# ── Registry ──────────────────────────────────────────────────────────

ANALYSIS_CMAPS: dict[str, dict[str, str]] = {
    "psd":          {"diverging": "RdBu_r",   "sequential": "Blues"},
    "aperiodic":    {"diverging": "PiYG",      "sequential": "Greens"},
    "connectivity": {"diverging": "PuOr",      "sequential": "Oranges"},
    "pac":          {"diverging": "PRGn",      "sequential": "Purples"},
    "network":      {"diverging": "BrBG",      "sequential": "YlGn"},
    "evoked":       {"diverging": "RdYlBu_r", "sequential": "YlOrRd"},
}

# Hex endpoints for R scale_fill_gradient2(low, mid, high).
# These are the outer-quartile colors from each diverging cmap.
R_GRADIENT2_COLORS: dict[str, dict[str, str]] = {
    "psd":       {"low": "#2166AC", "mid": "white", "high": "#B2182B"},
    "aperiodic": {"low": "#4D9221", "mid": "white", "high": "#C51B7D"},
    "pac":       {"low": "#1B7837", "mid": "white", "high": "#762A83"},
    "connectivity": {"low": "#B35806", "mid": "white", "high": "#542788"},
    "network":   {"low": "#01665E", "mid": "white", "high": "#8C510A"},
    "evoked":    {"low": "#2166AC", "mid": "white", "high": "#B2182B"},
}

_DEFAULT_DIVERGING = "RdBu_r"
_DEFAULT_SEQUENTIAL = "Blues"


# ── Helpers ───────────────────────────────────────────────────────────

def get_diverging_cmap(analysis: str) -> Colormap:
    """Return the diverging matplotlib Colormap for *analysis*."""
    name = ANALYSIS_CMAPS.get(analysis, {}).get("diverging", _DEFAULT_DIVERGING)
    return plt.colormaps.get_cmap(name)


def get_sequential_cmap(analysis: str) -> Colormap:
    """Return the sequential matplotlib Colormap for *analysis*."""
    name = ANALYSIS_CMAPS.get(analysis, {}).get("sequential", _DEFAULT_SEQUENTIAL)
    return plt.colormaps.get_cmap(name)


def get_diverging_cmap_name(analysis: str) -> str:
    """Return the diverging colormap *name* string for *analysis*."""
    return ANALYSIS_CMAPS.get(analysis, {}).get("diverging", _DEFAULT_DIVERGING)


def get_r_gradient2(analysis: str) -> dict[str, str]:
    """Return ``{low, mid, high}`` hex dict for R ``scale_fill_gradient2``."""
    return R_GRADIENT2_COLORS.get(analysis, R_GRADIENT2_COLORS["psd"])
