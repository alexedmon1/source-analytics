"""Attribution maps (viz/attribution_map.py): the field's guarantees, and an end-to-end render.

The field is what the published figures promise readers, so its properties are tested
directly on a synthetic label volume (no atlas needed):

* a value never exceeds the range of the tested effects (weighted average in value space);
* opposite effects of equal weight cancel to zero between them, not to a blend;
* kernels have unit integral, so a large parcel does not outweigh a small one by size alone;
* a parcel the atlas lacks, or one without a displacement, raises instead of vanishing.

The render test needs a registered atlas on disk and is skipped without one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from source_analytics.viz.attribution_map import (
    SIGMA_FLOOR_MM,
    attribution_caption,
    attribution_field_range,
    attribution_volume,
    plot_attribution_mosaic,
    shared_limit,
)

AFFINE = np.diag([0.2, 0.2, 0.2, 1.0])
ID2NAME = {1: "A", 2: "B", 3: "C"}


def _labels():
    vol = np.zeros((40, 40, 20), dtype=int)
    vol[5:15, 15:25, 5:15] = 1          # A, left
    vol[25:35, 15:25, 5:15] = 2         # B, right, mirror of A
    vol[18:22, 2:6, 8:12] = 3           # C, small
    return vol


def test_single_parcel_field_is_its_effect():
    v = attribution_volume(_labels(), AFFINE, ID2NAME, {"A": 1.3}, {"A": 0.6})
    ok = np.isfinite(v)
    assert ok.any()
    np.testing.assert_allclose(v[ok], 1.3)


def test_field_stays_within_tested_range():
    v = attribution_volume(_labels(), AFFINE, ID2NAME, {"A": 2.0, "B": -0.5, "C": 0.7},
                           {"A": 1.0, "B": 1.0, "C": 0.4})
    ok = np.isfinite(v)
    assert v[ok].max() <= 2.0 + 1e-9
    assert v[ok].min() >= -0.5 - 1e-9


def test_opposite_effects_cancel_midway():
    v = attribution_volume(_labels(), AFFINE, ID2NAME, {"A": 1.0, "B": -1.0}, {"A": 1.0, "B": 1.0})
    # x = 19.5 is the mirror plane between A (5-14) and B (25-34)
    mid = 0.5 * (v[19, 20, 10] + v[20, 20, 10])
    assert abs(mid) < 1e-6


def test_kernels_have_unit_integral():
    """A large and a small parcel at equal distance and sigma weigh the same at their midpoint."""
    vol = np.zeros((60, 20, 20), dtype=int)
    vol[0:20, 5:15, 5:15] = 1           # large: 20 voxels deep along x, nearest face at x = 19
    vol[38:40, 5:15, 5:15] = 2          # small: 2 voxels deep along x, nearest face at x = 38
    v_big = attribution_volume(vol, AFFINE, {1: "A", 2: "B"}, {"A": 1.0, "B": -1.0}, {"A": 2.0, "B": 2.0})
    # with unit-PEAK kernels the large parcel would dominate everywhere between them; with unit
    # integral its weight per voxel is spread over 10x the voxels, so the small one is not swamped
    between = v_big[29, 10, 10]
    assert -1.0 < between < 1.0
    assert v_big[37, 10, 10] < 0         # next to the small parcel, its sign holds


def test_sigma_floor():
    a = attribution_volume(_labels(), AFFINE, ID2NAME, {"A": 1.0, "B": -1.0}, {"A": 0.0, "B": 0.0})
    b = attribution_volume(_labels(), AFFINE, ID2NAME, {"A": 1.0, "B": -1.0},
                           {"A": SIGMA_FLOOR_MM, "B": SIGMA_FLOOR_MM})
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_unknown_parcel_raises():
    with pytest.raises(ValueError, match="not in the atlas"):
        attribution_volume(_labels(), AFFINE, ID2NAME, {"A": 1.0, "Z": 1.0}, {"A": 1.0, "Z": 1.0})


def test_missing_displacement_raises():
    with pytest.raises(ValueError, match="no displacement"):
        attribution_volume(_labels(), AFFINE, ID2NAME, {"A": 1.0, "B": 1.0}, {"A": 1.0})


def test_shared_limit_covers_every_map():
    assert shared_limit([(-0.2, 1.62), (0.1, 0.9)]) == pytest.approx(1.65)
    assert shared_limit([(-2.68, 0.3)]) == pytest.approx(2.70)
    assert shared_limit([(0.0, 1.60)]) == pytest.approx(1.60)


def test_caption_carries_range():
    d = pd.DataFrame({"roi": ["A", "B"], "displacement_mm": [0.33, 5.87]})
    c = attribution_caption(d, "Table S10")
    assert "0.33–5.87 mm; Table S10" in c and "not that parcel's tested effect size" in c


def _atlas_or_skip(name):
    from source_analytics.atlas import load_roi_mapping, resolve_atlas
    try:
        spec = resolve_atlas(atlas_name=name)
        rois = load_roi_mapping(spec)["rois"]
    except Exception as exc:  # atlas files not on this machine
        pytest.skip(f"atlas {name} unavailable: {exc}")
    return [v["name"] for k, v in rois.items() if int(k) != 0 and v.get("name")]


def test_render_end_to_end(tmp_path):
    names = _atlas_or_skip("allen26")
    rng = np.random.default_rng(0)
    eff = pd.DataFrame({"roi": names, "hedges_g": rng.uniform(-1.2, 1.2, len(names))})
    eff["significant"] = eff.hedges_g.abs() > 1.0
    disp = pd.DataFrame({"roi": names, "displacement_mm": rng.uniform(0.3, 4.0, len(names))})
    out = tmp_path / "map.png"
    ranges = plot_attribution_mosaic([("row", eff)], out, disp, atlas="allen26", g_limit=1.5, dpi=60)
    assert out.exists() and out.stat().st_size > 0
    lo, hi = ranges[0]
    assert -1.2 <= lo <= hi <= 1.2
    assert attribution_field_range(eff, disp, atlas="allen26")[0] == pytest.approx((lo, hi))
    with pytest.raises(ValueError, match="beyond g_limit"):
        plot_attribution_mosaic(eff, tmp_path / "clip.png", disp, atlas="allen26", g_limit=0.1, dpi=60)
