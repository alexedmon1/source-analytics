"""Known-signal sanity tests for the cross-frequency kernels (AAC, PPC)."""

from __future__ import annotations

import numpy as np
import pytest

from source_analytics.spectral.cross_freq import compute_aac, compute_ppc
from source_analytics.analyses.roi_cross_freq_analysis import _nm_ratio

FS = 200.0
DUR = 20.0
BAND_LOW = (8.0, 12.0)     # ~10 Hz
BAND_HIGH = (18.0, 22.0)   # ~20 Hz
BAND_GAMMA = (35.0, 45.0)  # ~40 Hz


def _t():
    return np.arange(int(DUR * FS)) / FS


# --------------------------------------------------------------------- AAC
def test_aac_shared_envelope_high_independent_low():
    t = _t()
    rng = np.random.default_rng(0)
    # slow 0.5 Hz amplitude envelope shared by a 10 Hz and a 40 Hz carrier
    env = 1.0 + 0.8 * np.sin(2 * np.pi * 0.5 * t)
    s_low = env * np.sin(2 * np.pi * 10 * t)            # sig 0: AM @10Hz
    s_gamma_shared = env * np.sin(2 * np.pi * 40 * t)   # sig 1: AM @40Hz, same env
    # genuinely independent envelope: a *different* modulation frequency
    # (a same-freq phase shift would still correlate as cos(Δφ))
    env2 = 1.0 + 0.8 * np.sin(2 * np.pi * 0.31 * t + 2.1)
    s_gamma_indep = env2 * np.sin(2 * np.pi * 40 * t)   # sig 2: AM @40Hz, diff env

    data = np.vstack([s_low, s_gamma_shared, s_gamma_indep])
    m = compute_aac(data, FS, BAND_LOW, BAND_GAMMA)

    # env_low(sig0) vs env_gamma(sig1) share `env` -> high; vs sig2 -> low
    assert m[0, 1] > 0.7
    assert abs(m[0, 2]) < 0.4


def test_aac_same_band_symmetric():
    t = _t()
    env = 1.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    a = env * np.sin(2 * np.pi * 10 * t)
    b = env * np.sin(2 * np.pi * 10 * t + 0.3)
    m = compute_aac(np.vstack([a, b]), FS, BAND_LOW, BAND_LOW)
    assert m[0, 1] == pytest.approx(m[1, 0], abs=1e-9)  # symmetric for equal bands
    assert m[0, 1] > 0.7


def test_aac_range():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, int(DUR * FS)))
    m = compute_aac(data, FS, BAND_LOW, BAND_GAMMA)
    assert m.min() >= -1.0 and m.max() <= 1.0


# --------------------------------------------------------------------- PPC
def test_ppc_nm_locked():
    t = _t()
    x = np.sin(2 * np.pi * 10 * t)   # phase φ
    y = np.sin(2 * np.pi * 20 * t)   # phase 2φ  -> locked 2:1 (n·f_x = m·f_y: n=2,m=1)
    rng = np.random.default_rng(2)
    # unlocked 20 Hz control: phase drifts via random walk
    drift = np.cumsum(rng.standard_normal(t.size)) * 0.05
    y_free = np.sin(2 * np.pi * 20 * t + drift)

    data = np.vstack([x, y, y_free])
    ppc = compute_ppc(data, FS, BAND_LOW, BAND_HIGH, n=2, m=1)

    assert ppc[0, 1] > 0.9          # x (10Hz) locked to y (20Hz) at 2:1
    assert ppc[0, 2] < 0.5          # x not locked to the drifting control


def test_ppc_range_and_self():
    t = _t()
    x = np.sin(2 * np.pi * 10 * t)
    y = np.sin(2 * np.pi * 20 * t)
    ppc = compute_ppc(np.vstack([x, y]), FS, BAND_LOW, BAND_HIGH, n=2, m=1)
    assert ppc.min() >= 0.0 and ppc.max() <= 1.0


# ------------------------------------------------------- n:m ratio helper
def test_nm_ratio():
    # theta(~7) -> low gamma(~42): ~6:1
    assert _nm_ratio((4, 10), (30, 55)) == (6, 1)
    # delta(~2.5) -> alpha(~11.5): ~5:1 (rounds)
    assert _nm_ratio((1, 4), (10, 13))[1] == 1
    # same band -> 1:1
    assert _nm_ratio((8, 12), (8, 12)) == (1, 1)


def test_ppc_unlocked_pair_low():
    t = _t()
    rng = np.random.default_rng(5)
    a = np.sin(2 * np.pi * 10 * t + np.cumsum(rng.standard_normal(t.size)) * 0.05)
    b = np.sin(2 * np.pi * 20 * t + np.cumsum(rng.standard_normal(t.size)) * 0.05)
    ppc = compute_ppc(np.vstack([a, b]), FS, BAND_LOW, BAND_HIGH, n=2, m=1)
    assert ppc[0, 1] < 0.5
