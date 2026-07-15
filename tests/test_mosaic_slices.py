"""Guard: mosaic planes must actually show the ROIs the figure claims to show.

The bug this locks out: ``_compact_mosaic`` rendered three hard-coded planes
that were never adapted to the data. For the FORGE KO-vs-WT Low Gamma figure
that silently dropped Thalamus_R -- the largest individual effect (g = 1.92),
named in the figure caption -- from all three panels, while the row label still
read "n=6". A significant ROI must never be invisible in its own figure.

These tests use a synthetic label volume, so they do not need the atlas on disk.
"""

from __future__ import annotations

import numpy as np

from source_analytics.viz.brain_roi import (
    DEFAULT_COMPACT_AXIAL,
    DEFAULT_COMPACT_CORONAL,
    DEFAULT_COMPACT_SAGITTAL,
    _pick_informative_slices,
)

DEFAULTS = (DEFAULT_COMPACT_CORONAL, DEFAULT_COMPACT_AXIAL, DEFAULT_COMPACT_SAGITTAL)


def _volume() -> np.ndarray:
    """Label volume with a blind-spot ROI that the default planes all miss.

    Label 1 sits at the defaults. Label 2 is placed off every default plane,
    reproducing the Thalamus_R geometry: near-misses on two axes and the
    opposite side on the third.
    """
    vol = np.zeros((64, 256, 50), dtype=np.int16)
    vol[20:30, 140:150, 24:32] = 1
    vol[34:44, 100:130, 12:24] = 2
    return vol


def test_default_planes_have_a_blind_spot():
    """Sanity-check the fixture: the old fixed planes really do miss label 2."""
    vol = _volume()
    y, z, x = DEFAULTS
    assert not np.any(vol[:, y, :] == 2)
    assert not np.any(vol[:, :, z] == 2)
    assert not np.any(vol[x, :, :] == 2)


def test_picker_covers_every_target_roi():
    vol = _volume()
    y, z, x = _pick_informative_slices(vol, {1, 2})
    for label in (1, 2):
        shown = (
            np.any(vol[:, y, :] == label)
            or np.any(vol[:, :, z] == label)
            or np.any(vol[x, :, :] == label)
        )
        assert shown, f"label {label} invisible at coronal={y} axial={z} sagittal={x}"


def test_picker_avoids_sliver_cross_sections():
    """Coverage alone is not enough: each ROI needs a substantial cross-section.

    Maximin scoring should do better than clipping an ROI's outermost voxel.
    """
    vol = _volume()
    y, z, x = _pick_informative_slices(vol, {1, 2})
    for label in (1, 2):
        best = max(
            int(np.count_nonzero(vol[:, y, :] == label)),
            int(np.count_nonzero(vol[:, :, z] == label)),
            int(np.count_nonzero(vol[x, :, :] == label)),
        )
        assert best >= 25, f"label {label} shown as a {best}-voxel sliver"


def test_empty_or_absent_targets_fall_back_to_defaults():
    vol = _volume()
    assert _pick_informative_slices(vol, set()) == DEFAULTS
    assert _pick_informative_slices(vol, {999}) == DEFAULTS
    # Background is not a target.
    assert _pick_informative_slices(vol, {0}) == DEFAULTS


def test_well_served_rois_stay_near_the_historical_planes():
    """Stability: when the defaults already show the ROI, do not wander far.

    Figures whose ROIs were always visible should not shift appearance for no
    reason, so distance-to-default is the final tie-break.
    """
    vol = _volume()
    y, z, x = _pick_informative_slices(vol, {1})
    assert abs(y - DEFAULT_COMPACT_CORONAL) <= 6
    assert abs(z - DEFAULT_COMPACT_AXIAL) <= 6
    assert abs(x - DEFAULT_COMPACT_SAGITTAL) <= 6
