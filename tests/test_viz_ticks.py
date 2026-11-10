"""No two numbers on a colour bar overlap, in any figure source-analytics saves (viz_ticks)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from source_analytics import viz_ticks  # noqa: E402


def _labels(cbar):
    fig = cbar.ax.figure
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    return [(t.get_loc(), t.label1.get_window_extent(r)) for t in cbar.ax.xaxis.get_major_ticks()
            if t.label1.get_visible() and t.label1.get_text()]


def test_crowded_colour_bar_numbers_are_thinned_and_marked_ones_kept(tmp_path):
    viz_ticks.install()
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(np.linspace(-5, 5, 9).reshape(3, 3), cmap="RdBu_r", vmin=-5, vmax=5)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.4)
    cbar.set_ticks([-5, -4.6, -4, -2.5, -2.3, 0, 2.3, 2.5, 4, 4.6, 5])
    viz_ticks.keep_ticks(cbar, [-2.5, 2.5])
    fig.savefig(tmp_path / "f.png")  # the check runs as the figure is saved
    shown = _labels(cbar)
    boxes = [b for _, b in shown]
    assert all(not a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    values = [v for v, _ in shown]
    assert -2.5 in values and 2.5 in values  # the marked ticks win their collisions
    assert -2.3 not in values and 2.3 not in values
    plt.close(fig)


def test_a_figure_without_colour_bars_is_left_alone(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    assert viz_ticks.declutter(fig) == 0
    plt.close(fig)


def test_pyplot_imported_after_install_still_loads():
    """pyplot checks Figure.savefig's name when it is first imported (fresh interpreter)."""
    import subprocess
    import sys

    code = ("from source_analytics import viz_ticks; viz_ticks.install(); "
            "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; "
            "fig, ax = plt.subplots(); plt.savefig(__import__('os').devnull, format='png')")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0
