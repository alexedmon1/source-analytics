"""Known-signal test for the vertex local-PAC kernel (within-vertex PAC map)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from source_analytics.spectral.pac import compute_local_pac_vertices

FS = 200.0
DUR = 16.0
F_PHASE = 6.0    # theta
F_AMP = 40.0     # gamma
PHASE_BAND = (4.0, 8.0)
AMP_BAND = (30.0, 50.0)


def _theta_drive(t, rng):
    """Aperiodic theta = band-limited (4-8 Hz) noise, so circular-shift
    surrogates actually decorrelate (a pure sine would not)."""
    sos = butter(2, [F_PHASE / 2 / (FS / 2), 8 / (FS / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, rng.standard_normal(t.size))


def _slow_env(t, rng):
    """A slow, random (theta-independent) amplitude envelope for uncoupled gamma."""
    e = rng.standard_normal(t.size)
    k = int(FS / 3)
    e = np.convolve(e, np.ones(k) / k, mode="same")
    return 1.0 + 0.9 * (e / (np.abs(e).max() + 1e-9))


def _pac_signal(t, coupled, rng):
    """A signal with (coupled) or without theta-gamma phase-amplitude coupling."""
    theta = _theta_drive(t, rng)
    if coupled:
        # gamma amplitude tracks the (aperiodic) theta phase
        phase = np.angle(hilbert(theta))
        amp_env = 1.0 + 0.9 * np.cos(phase)
    else:
        amp_env = _slow_env(t, rng)  # random envelope, not theta-locked
    gamma = amp_env * np.sin(2 * np.pi * F_AMP * t)
    return theta + gamma + 0.05 * rng.standard_normal(t.size)


def test_local_pac_detects_coupled_vertex():
    rng = np.random.default_rng(0)
    t = np.arange(int(DUR * FS)) / FS
    # vertex 0 coupled, vertices 1-2 uncoupled
    stc = np.vstack([
        _pac_signal(t, True, rng),
        _pac_signal(t, False, rng),
        _pac_signal(t, False, rng),
    ])
    z, mi = compute_local_pac_vertices(stc, FS, PHASE_BAND, AMP_BAND, n_surrogates=200)

    assert z.shape == (3,) and mi.shape == (3,)
    # coupled vertex stands well above surrogate noise and clearly above uncoupled
    assert z[0] > 8.0
    assert z[0] > z[1:].max() + 5.0
    assert mi[0] > mi[1] and mi[0] > mi[2]


def test_local_pac_all_uncoupled_modest():
    rng = np.random.default_rng(1)
    t = np.arange(int(DUR * FS)) / FS
    stc = np.vstack([_pac_signal(t, False, rng) for _ in range(4)])
    z, _ = compute_local_pac_vertices(stc, FS, PHASE_BAND, AMP_BAND, n_surrogates=200)
    # no theta-locked coupling -> no vertex reaches the strong-coupling regime
    assert np.all(z < 8.0)
