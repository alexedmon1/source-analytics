"""Known-signal test for orthogonalized AEC (Hipp et al. 2012).

The defining property: a shared amplitude envelope with a genuine phase lag is
detected, but a zero-lag (volume-conduction) mixture of the same source is
suppressed by the orthogonalization.
"""

from __future__ import annotations

import numpy as np

from source_analytics.spectral.connectivity import _band_orthogonalized_aec
from source_analytics.spectral.vertex_connectivity import _compute_aec

FS = 200.0
DUR = 30.0
F0 = 10.0
BAND = (8.0, 12.0)


def _shared_env(t, rng):
    e = rng.standard_normal(t.size)
    k = int(FS / 2)
    e = np.convolve(e, np.ones(k) / k, mode="same")
    return 1.5 + e / (np.abs(e).max() + 1e-9)


def _signals(rng):
    t = np.arange(int(DUR * FS)) / FS
    env = _shared_env(t, rng)
    carrier = np.sin(2 * np.pi * F0 * t)
    sig0 = env * carrier
    # genuine coupling: same envelope, phase-lagged by pi/3 (non-zero lag)
    sig1 = env * np.sin(2 * np.pi * F0 * t - np.pi / 3)
    # volume conduction: zero-lag scaled copy of sig0 + independent in-band noise
    noise = np.sin(2 * np.pi * F0 * t + rng.uniform(0, 2 * np.pi)) * 0.1
    sig2 = 0.8 * sig0 + noise
    return [sig0, sig1, sig2]


def test_roi_aec_suppresses_zero_lag():
    rng = np.random.default_rng(0)
    sigs = _signals(rng)
    aec = _band_orthogonalized_aec(sigs, FS, *BAND)
    # genuine lagged amplitude coupling detected; zero-lag mixture suppressed
    assert aec[0, 1] > 0.3
    assert abs(aec[0, 2]) < aec[0, 1]
    assert abs(aec[0, 2]) < 0.25
    # symmetric, diagonal 1
    assert np.allclose(aec, aec.T)
    assert np.allclose(np.diag(aec), 1.0)


def test_vertex_aec_matches_roi():
    rng = np.random.default_rng(0)
    sigs = _signals(rng)
    stc = np.vstack(sigs)
    aec_v = _compute_aec(stc, FS, BAND, 3)
    aec_r = _band_orthogonalized_aec(sigs, FS, *BAND)
    # same orthogonalized-AEC values at vertex and ROI entry points
    assert abs(aec_v[0, 1] - aec_r[0, 1]) < 1e-6
    assert aec_v[0, 1] > 0.3 and abs(aec_v[0, 2]) < 0.25
