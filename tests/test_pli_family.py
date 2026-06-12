"""Known-signal sanity tests for the PLI family kernels (pli, dwpli, wpli, dpli).

wPLI (Vinck 2011, non-debiased) and dPLI (Stam & van Straaten 2012, directed)
are the two new MS2 kernels; these tests pin down their defining behaviour and
guard the asymmetry contract of dPLI.
"""

from __future__ import annotations

import numpy as np
import pytest

from source_analytics.spectral.connectivity import compute_connectivity_matrix
from source_analytics.spectral.vertex_connectivity import (
    compute_vertex_connectivity_matrix,
)

FS = 200.0
DUR = 12.0
F0 = 10.0          # carrier
# Narrow band around the carrier: with 0.5 Hz STFT resolution a wide band
# averages the coherent carrier bin with many incoherent noise bins, diluting
# the (correctly) magnitude-insensitive PLI/wPLI. Kept tight so the carrier
# dominates and the sanity thresholds are meaningful.
BAND = (9.5, 10.5)


def _pair(phase_lag, noise=0.02, seed=0):
    """Two sinusoids at F0 where signal A leads signal B by `phase_lag` rad."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(DUR * FS)) / FS
    a = np.sin(2 * np.pi * F0 * t)
    b = np.sin(2 * np.pi * F0 * t - phase_lag)   # B lags A -> A leads B
    a = a + noise * rng.standard_normal(t.size)
    b = b + noise * rng.standard_normal(t.size)
    return a, b


def _roi(a, b):
    res, names = compute_connectivity_matrix({"A": a, "B": b}, FS, {"band": BAND})
    assert names == ["A", "B"]
    return res["band"]


def _vtx(a, b, metric):
    stc = np.vstack([a, b])
    return compute_vertex_connectivity_matrix(stc, FS, BAND, metric=metric)


# ------------------------------------------------------------ phase-lead pair
def test_roi_phase_lead():
    m = _roi(*_pair(np.pi / 3))
    # consistent phase difference -> strong phase-locking
    assert m["pli"][0, 1] > 0.8
    assert m["wpli"][0, 1] > 0.8
    assert m["dwpli"][0, 1] > 0.7
    # directed: A (row 0) leads B (row 1)
    assert m["dpli"][0, 1] > 0.6
    assert m["dpli"][1, 0] < 0.4


def test_dpli_antisymmetry_sums_to_one():
    m = _roi(*_pair(np.pi / 4))
    assert m["dpli"][0, 1] + m["dpli"][1, 0] == pytest.approx(1.0, abs=1e-9)


def test_dpli_direction_flips():
    # negative lag -> B leads A -> dpli[A,B] < 0.5
    m = _roi(*_pair(-np.pi / 3))
    assert m["dpli"][0, 1] < 0.4
    assert m["dpli"][1, 0] > 0.6


def test_wpli_symmetric_dpli_asymmetric():
    m = _roi(*_pair(np.pi / 3))
    assert m["wpli"][0, 1] == pytest.approx(m["wpli"][1, 0])
    assert m["dpli"][0, 1] != pytest.approx(m["dpli"][1, 0])


# ----------------------------------------------------------------- zero lag
def test_zero_lag_no_directionality():
    # in-phase signals -> no consistent imaginary cross-spectrum
    m = _roi(*_pair(0.0, noise=0.1, seed=3))
    assert m["dpli"][0, 1] == pytest.approx(0.5, abs=0.15)
    assert m["pli"][0, 1] < 0.4
    assert m["wpli"][0, 1] < 0.4


# ------------------------------------------------------------------- ranges
def test_ranges():
    m = _roi(*_pair(np.pi / 3))
    for metric in ("pli", "dwpli", "wpli", "dpli"):
        mat = m[metric]
        assert mat.min() >= 0.0 and mat.max() <= 1.0


# ------------------------------------------------------- vertex kernel parity
def test_vertex_matches_roi_direction():
    a, b = _pair(np.pi / 3)
    for metric in ("wpli", "dpli"):
        vtx = _vtx(a, b, metric)
        roi = _roi(a, b)[metric]
        # same qualitative answer at vertex and ROI level
        assert vtx[0, 1] == pytest.approx(roi[0, 1], abs=0.05)


def test_vertex_dpli_asymmetry():
    a, b = _pair(np.pi / 3)
    d = _vtx(a, b, "dpli")
    assert d[0, 1] > 0.6 and d[1, 0] < 0.4
    assert d[0, 1] + d[1, 0] == pytest.approx(1.0, abs=1e-9)


def test_vertex_wpli_symmetric():
    a, b = _pair(np.pi / 3)
    w = _vtx(a, b, "wpli")
    assert w[0, 1] == pytest.approx(w[1, 0])
    assert w[0, 1] > 0.8
