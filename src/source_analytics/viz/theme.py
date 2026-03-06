"""Unified visual theme for all source-analytics figures.

Single source of truth for font sizes, figure dimensions, DPI, and
matplotlib rcParams.  Import ``apply_theme()`` at the top of any plotting
function, or use the constants directly.

Also exports ``r_theme_args()`` which returns a JSON-serialisable dict
that R scripts can consume to match the Python style.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Typography ────────────────────────────────────────────────────────

FONT_FAMILY = "sans-serif"
FONT_TITLE = 14          # figure suptitle / main title
FONT_SUBTITLE = 12       # subplot titles, panel headers
FONT_AXIS_LABEL = 11     # x/y axis labels
FONT_TICK = 9            # tick labels
FONT_ANNOTATION = 8      # small annotations, direction labels
FONT_LEGEND = 10         # legend text
FONT_LEGEND_TITLE = 11   # legend title

# ── Figure sizes (inches) ────────────────────────────────────────────

# Named presets — (width, height)
FIGSIZE_SINGLE = (6, 5)        # single-panel plot
FIGSIZE_WIDE = (10, 5)         # wide single-panel (e.g. bar chart)
FIGSIZE_MULTI = (10, 6)        # multi-panel (e.g. 2×3 grid)
FIGSIZE_GLASS_BRAIN = (12, 4)  # 3-view glass brain
FIGSIZE_RADAR = (5, 5)         # per-subplot for radar (scales with n_bands)
FIGSIZE_HEATMAP = (8, 7)       # single heatmap / matrix
FIGSIZE_CIRCOS = (8, 8)        # single circos plot

DPI = 300                      # all output figures

# ── Colors ────────────────────────────────────────────────────────────

COLOR_BG = "white"
COLOR_GRID = "#E8E8E8"
COLOR_GRID_MINOR = "#F2F2F2"
COLOR_TEXT = "#333333"
COLOR_AXIS = "#666666"
COLOR_NONSIG = "#CCCCCC"       # non-significant results in summary plots

# Significance markers
COLOR_SIG = "#E74C3C"          # p < 0.05
COLOR_TREND = "#F39C12"        # 0.05 < p < 0.10

# ── Line & marker ────────────────────────────────────────────────────

LINEWIDTH_DEFAULT = 1.2
LINEWIDTH_THICK = 2.0
LINEWIDTH_THIN = 0.8
MARKER_SIZE = 5
ERRORBAR_CAPSIZE = 3

# ── Spacing ───────────────────────────────────────────────────────────

TITLE_PAD = 12          # pts above title
LABEL_PAD = 6           # pts between axis label and ticks
TIGHT_PAD = 1.5         # tight_layout pad


# ── Apply ─────────────────────────────────────────────────────────────

def apply_theme() -> None:
    """Set matplotlib rcParams to the unified theme.

    Call once at the start of any plotting function.  Safe to call
    repeatedly (idempotent).
    """
    rc = {
        # Font
        "font.family": FONT_FAMILY,
        "font.size": FONT_TICK,
        "axes.titlesize": FONT_SUBTITLE,
        "axes.labelsize": FONT_AXIS_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "legend.title_fontsize": FONT_LEGEND_TITLE,
        "figure.titlesize": FONT_TITLE,

        # Colors
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_BG,
        "savefig.facecolor": COLOR_BG,
        "text.color": COLOR_TEXT,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_AXIS,
        "ytick.color": COLOR_AXIS,

        # Grid
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": LINEWIDTH_THIN,
        "grid.alpha": 0.8,

        # Lines & markers
        "lines.linewidth": LINEWIDTH_DEFAULT,
        "lines.markersize": MARKER_SIZE,

        # Layout
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": False,

        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": LINEWIDTH_THIN,
    }
    mpl.rcParams.update(rc)


def r_theme_json() -> str:
    """Return theme constants as a JSON string for R scripts.

    R scripts can parse this to build a matching ggplot2 theme.
    """
    d = {
        "font_title": FONT_TITLE,
        "font_subtitle": FONT_SUBTITLE,
        "font_axis_label": FONT_AXIS_LABEL,
        "font_tick": FONT_TICK,
        "font_annotation": FONT_ANNOTATION,
        "font_legend": FONT_LEGEND,
        "dpi": DPI,
        "color_bg": COLOR_BG,
        "color_grid": COLOR_GRID,
        "color_text": COLOR_TEXT,
        "color_axis": COLOR_AXIS,
        "color_nonsig": COLOR_NONSIG,
        "color_sig": COLOR_SIG,
        "color_trend": COLOR_TREND,
        "linewidth_default": LINEWIDTH_DEFAULT,
        "linewidth_thin": LINEWIDTH_THIN,
    }
    return json.dumps(d)


def savefig(fig: plt.Figure, path: str | Path, **kwargs) -> Path:
    """Save a figure with standard settings.

    Applies tight bbox, white facecolor, and configured DPI.
    Returns the resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        dpi=DPI,
        bbox_inches="tight",
        facecolor=COLOR_BG,
        pad_inches=0.15,
    )
    defaults.update(kwargs)
    fig.savefig(path, **defaults)
    plt.close(fig)
    return path
