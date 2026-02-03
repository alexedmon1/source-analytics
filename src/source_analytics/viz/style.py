"""Publication-quality plot defaults."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


PUBLICATION_PARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "lines.linewidth": 1.5,
    "font.family": "sans-serif",
}


def apply_style():
    """Apply publication defaults to matplotlib."""
    mpl.rcParams.update(PUBLICATION_PARAMS)


def get_group_color(group_id: str, group_colors: dict[str, str]) -> str:
    """Return a group's color, with fallback to a default palette."""
    if group_id in group_colors:
        return group_colors[group_id]
    default_palette = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12"]
    idx = hash(group_id) % len(default_palette)
    return default_palette[idx]
