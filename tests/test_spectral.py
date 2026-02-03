"""Tests for spectral analysis modules."""

import numpy as np

from source_analytics.spectral.psd import compute_psd, compute_psd_multiroi
from source_analytics.spectral.band_power import extract_band_power


def test_compute_psd():
    rng = np.random.default_rng(42)
    sfreq = 500.0
    t = np.arange(5000) / sfreq
    signal = np.sin(2 * np.pi * 10 * t) + rng.standard_normal(5000) * 0.1

    freqs, psd = compute_psd(signal, sfreq)
    assert len(freqs) == len(psd)
    assert freqs[0] >= 0.5
    assert freqs[-1] <= sfreq / 2

    # Peak should be near 10 Hz
    peak_freq = freqs[np.argmax(psd)]
    assert 9 <= peak_freq <= 11


def test_extract_band_power():
    freqs = np.linspace(0.5, 100, 200)
    psd = np.ones_like(freqs)  # flat spectrum

    bands = {"Alpha": (8, 13), "Gamma": (30, 55)}
    result = extract_band_power(freqs, psd, bands)

    assert "Alpha" in result
    assert "Gamma" in result
    assert result["Alpha"]["absolute"] > 0
    assert 0 < result["Alpha"]["relative"] < 1
    # Gamma band is wider, so should have more absolute power
    assert result["Gamma"]["absolute"] > result["Alpha"]["absolute"]
