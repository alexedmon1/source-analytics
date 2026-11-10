"""Heatmaps over a module's declared tests, drawn from its hypotheses tables (results plan 5b.4).

These replace the table figures source-lightbox drew from the same CSVs, so the stars come from
each test's own ``significant`` column (the declared correction), never from tests run for the
figure. One helper lays out the grid; the modules choose rows, columns and panels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def grid(df: pd.DataFrame, rows: str, cols: str, value: str,
         row_order: list[str] | None = None, col_order: list[str] | None = None,
         ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """``value`` and ``significant`` laid out as rows x cols; NaN where no test ran."""
    r = [x for x in (row_order or []) if x in set(df[rows])]
    r += [x for x in dict.fromkeys(df[rows]) if x not in r]
    c = [x for x in (col_order or []) if x in set(df[cols])]
    c += [x for x in dict.fromkeys(df[cols]) if x not in c]
    vals = np.full((len(r), len(c)), np.nan)
    sig = np.zeros((len(r), len(c)), dtype=bool)
    flags = df["significant"].astype(str).str.lower().isin(["true", "1"])
    numbers = pd.to_numeric(df[value], errors="coerce")
    for (rv, cv, v, s) in zip(df[rows], df[cols], numbers, flags):
        i, j = r.index(rv), c.index(cv)
        vals[i, j] = v
        sig[i, j] = sig[i, j] or bool(s)
    return vals, sig, r, c


def draw(ax, vals: np.ndarray, sig: np.ndarray, rows: list[str], cols: list[str], *,
         vmax: float | None = None, cmap: str = "RdBu_r", annotate: bool = True,
         row_labels: dict[str, str] | None = None):
    """One panel: a diverging map centred on 0, the value in each cell, * where significant."""
    top = vmax if vmax is not None else (np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1)
    top = top or 1
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=-top, vmax=top, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            text = (f"{v:.1f}" if annotate else "") + ("*" if sig[i, j] else "")
            if text:
                ax.text(j, i, text, ha="center", va="center", fontsize=7 if annotate else 10,
                        fontweight="bold" if sig[i, j] else "normal",
                        color="white" if abs(v) > 0.65 * top else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    labels = row_labels or {}
    ax.set_yticklabels([labels.get(x, x).replace("_", " ") for x in rows], fontsize=8)
    return im


def save(fig, out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
