"""compute_connectivity_matrix(metrics=...) returns exactly the full call's values for the requested metrics;
compute_transfer_entropy's band-matched lag (the default since the MEA30 directed-connectivity validation)."""

from __future__ import annotations

import numpy as np
import pytest

from source_analytics.spectral.connectivity import ALL_METRICS, compute_connectivity_matrix
from source_analytics.spectral.transfer_entropy import band_lag, compute_transfer_entropy

FS = 500.0


def _lagged(lag_samples=3, n=30000, seed=0, band=(15.0, 25.0)):
    """Three channels: b follows a by ``lag_samples``; c is independent. Band-limited carriers plus noise."""
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.default_rng(seed)
    sos = butter(4, band, btype="bandpass", fs=FS, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n + lag_samples))
    a, b = x[lag_samples:], x[:n]
    c = sosfiltfilt(sos, rng.standard_normal(n))
    noise = 0.3 * rng.standard_normal((3, n))
    return {"a": a + noise[0], "b": b + noise[1], "c": c + noise[2]}


def test_metric_subset_equals_full_call():
    ts = _lagged()
    bands = {"beta": (15.0, 25.0), "theta": (6.0, 8.0)}
    full, names = compute_connectivity_matrix(ts, FS, bands)
    for subset in (["dpli"], ["coherence"], ["aec", "wpli"], ["partial_corr", "imag_coherence"]):
        part, pn = compute_connectivity_matrix(ts, FS, bands, metrics=subset)
        assert pn == names
        for band in bands:
            assert set(part[band]) == set(subset)
            for m in subset:
                assert np.array_equal(part[band][m], full[band][m]), (band, m)
    assert set(full["beta"]) == set(ALL_METRICS)


def test_metric_subset_rejects_unknown():
    with pytest.raises(ValueError, match="unknown connectivity metric"):
        compute_connectivity_matrix(_lagged(), FS, {"beta": (15.0, 25.0)}, metrics=["dtf"])


def test_band_lag_rule():
    # one eighth of the band-centre period at 500 Hz: theta 7 Hz -> 9, beta 20 Hz -> 3, low gamma 40 Hz -> 2
    assert band_lag(FS, (6.0, 8.0)) == 9
    assert band_lag(FS, (15.0, 25.0)) == 3
    assert band_lag(FS, (35.0, 45.0)) == 2
    assert band_lag(100.0, (40.0, 50.0)) == 1                  # never below one sample


def test_te_default_is_the_band_lag_and_lag1_reproduces_old_default():
    ts = _lagged()
    b = {"beta": (15.0, 25.0)}
    default, n0 = compute_transfer_entropy(ts, FS, b)
    explicit, n1 = compute_transfer_entropy(ts, FS, b, lag=band_lag(FS, b["beta"]))
    assert n0 == n1 and np.array_equal(default["beta"]["te"], explicit["beta"]["te"])
    old, _ = compute_transfer_entropy(ts, FS, b, lag=1)
    assert not np.array_equal(old["beta"]["te"], default["beta"]["te"])


def test_te_band_lag_direction():
    ts = _lagged(lag_samples=3)
    res, names = compute_transfer_entropy(ts, FS, {"beta": (15.0, 25.0)})
    net = res["beta"]["net_te"]
    assert net[names.index("a"), names.index("b")] > 0                       # a leads b
    assert abs(net[names.index("a"), names.index("c")]) < net[names.index("a"), names.index("b")]
