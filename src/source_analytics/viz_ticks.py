"""No two numbers on a colour bar overlap, in any figure source-analytics saves.

A colour bar's numbers can collide when a figure sets its own ticks (a threshold beside the
automatic ones) or the bar is short. ``declutter`` hides the labels that would overlap a label
already kept, keeping the ends and any tick the figure marks as important first; ``install``
runs it on every colour bar as a figure is saved, so each module needs nothing of its own.
R's figures (ggplot2) space their legends themselves.
"""

from __future__ import annotations

from typing import Any

#: Pixels kept clear between two labels.
GAP = 3.0
_installed = False


def _priority(cbar_ax: Any, value: float, lo: float, hi: float) -> int:
    keep = getattr(cbar_ax, "_keep_ticks", ())
    if any(abs(value - k) <= 1e-9 * max(1.0, abs(k)) for k in keep):
        return 0  # the figure's own marks (a cut, a threshold) come first
    if abs(value - lo) <= 1e-12 or abs(value - hi) <= 1e-12 or value == 0:
        return 1
    return 2


def declutter(fig: Any) -> int:
    """Hide colour-bar tick labels that would overlap another; the number hidden."""
    axes = [ax for ax in fig.axes if getattr(ax, "_colorbar", None) is not None]
    if not axes:
        return 0
    renderer = fig.canvas.get_renderer()
    fig.draw_without_rendering()
    hidden = 0
    for ax in axes:
        cbar = ax._colorbar
        axis = ax.xaxis if cbar.orientation == "horizontal" else ax.yaxis
        lo, hi = sorted(ax.get_xlim() if cbar.orientation == "horizontal" else ax.get_ylim())
        ticks = [t for t in axis.get_major_ticks() if t.label1.get_visible() and t.label1.get_text()]
        boxes = []
        for t in ticks:
            value = t.get_loc()
            if value < lo - 1e-9 or value > hi + 1e-9:
                t.label1.set_visible(False)
                continue
            boxes.append((_priority(ax, value, lo, hi), value, t, t.label1.get_window_extent(renderer)))
        kept: list[Any] = []
        for _, _, tick, box in sorted(boxes, key=lambda b: (b[0], b[1])):
            padded = box.expanded(1.0, 1.0).padded(GAP / 2)
            if any(padded.overlaps(k) for k in kept):
                tick.label1.set_visible(False)
                hidden += 1
            else:
                kept.append(box)
    return hidden


def keep_ticks(cbar: Any, values: list[float]) -> None:
    """Mark colour-bar ticks to keep before any other when labels collide."""
    cbar.ax._keep_ticks = tuple(float(v) for v in values)


def install() -> None:
    """Declutter every colour bar when a figure is saved (once per process)."""
    global _installed
    if _installed:
        return
    import functools

    from matplotlib.figure import Figure

    original = Figure.savefig

    # wraps keeps the name pyplot checks (Figure.savefig) when it is imported after this
    @functools.wraps(original)
    def savefig(self: Figure, *args: Any, **kwargs: Any) -> Any:
        try:
            declutter(self)
        except Exception:  # noqa: BLE001 - tidying must never stop a figure being saved
            pass
        return original(self, *args, **kwargs)

    Figure.savefig = savefig  # type: ignore[method-assign]
    _installed = True
