"""Tests for spectral analysis modules."""

import numpy as np

from source_analytics.spectral.psd import compute_psd, compute_psd_multiroi
from source_analytics.spectral.band_power import extract_band_power
from source_analytics.spectral.vertex import (
    compute_falff,
    compute_spectral_slope,
    extract_band_power_vertices,
)
from source_analytics.stats.cluster_permutation import hedges_g


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


def test_extract_band_power_vertices_small_scale():
    """Regression test: relative power must not collapse to ~0 for PSDs whose
    integrated total falls below np.finfo(float).eps (~2.2e-16).

    Source-localized PSDs commonly integrate to ~1e-18; an over-aggressive
    np.maximum(total, eps) clamp on the denominator silently rescaled every
    band's relative metric by ~100x and inverted some inter-group directions.
    See FORGE manuscript 2 RERUN_PROPOSAL.md, May 2026.
    """
    freqs = np.linspace(0.5, 110, 220)
    # Flat spectrum at amplitude well below eps — mimics source-localized scale.
    psd = np.full((4, len(freqs)), 1e-20)  # 4 "vertices"

    bands = {"Delta": (1, 4), "Alpha": (8, 13), "Gamma": (30, 55)}
    result = extract_band_power_vertices(freqs, psd, bands, noise_exclude=None)

    # For a flat PSD of amplitude A, band power = A * band_width, total = A * total_width.
    # Relative = band_width / total_width — independent of A.
    total_width = freqs[-1] - freqs[0]
    for band_name, (fmin, fmax) in bands.items():
        rel = result[band_name]["relative"]
        expected = (fmax - fmin) / total_width
        assert np.all(
            np.abs(rel - expected) < 0.01
        ), f"{band_name} relative {rel.mean():.4f} != expected {expected:.4f} — eps-clamp bug regression"


def test_extract_band_power_vertices_sums_to_one():
    """A spectrum entirely covered by named bands should produce relative
    powers summing to 1.0 (allowing small numerical slack)."""
    freqs = np.linspace(1, 100, 200)
    psd = np.ones((3, len(freqs))) * 1e-18  # below-eps amplitude

    # Bands exactly covering 1-100 Hz without gaps
    bands = {"A": (1, 25), "B": (25, 50), "C": (50, 75), "D": (75, 100)}
    result = extract_band_power_vertices(freqs, psd, bands, noise_exclude=None)
    total_rel = sum(result[b]["relative"] for b in bands)
    # Tolerance accounts for trapezoid integration at sub-bin boundaries; the
    # regression test catches the eps-clamp collapse (~0.005), which is two
    # orders of magnitude away from this assertion.
    assert np.all(total_rel > 0.95), (
        f"Relative powers across gap-free bands should sum to ~1.0, got {total_rel}"
    )


def test_compute_falff_small_scale():
    """fALFF must not collapse to ~0 when total integrated power is below eps."""
    freqs = np.linspace(1, 100, 200)
    psd = np.ones((3, len(freqs))) * 1e-18

    # Flat spectrum: gamma (65-100) / total (1-100) = 35 / 99 ≈ 0.354
    falff = compute_falff(freqs, psd, gamma_range=(65, 100), total_range=(1, 100))
    expected = 35 / 99
    assert np.all(np.abs(falff - expected) < 0.01), (
        f"fALFF should be ~{expected:.3f} for flat spectrum, got {falff}"
    )


def test_compute_spectral_slope_small_scale():
    """Spectral slope must recover the true 1/f^alpha exponent even when PSD
    values are below np.finfo(float).eps.

    Pre-fix, np.maximum(psd, eps) floored every value in the log10 spectrum
    to log10(eps) ≈ -15.66, collapsing slope to ~0. Source-localized PSDs
    typically integrate to ~1e-18 with per-bin values 1e-19 to 1e-22 — well
    below eps.
    """
    freqs = np.logspace(0, 2, 200)  # 1 to 100 Hz, log-spaced
    # Construct PSD = scale * f^(-1.5), small absolute scale
    scale = 1e-22
    true_alpha = 1.5
    psd_1d = scale * freqs ** (-true_alpha)
    psd = np.tile(psd_1d, (5, 1))  # 5 vertices, identical

    slope = compute_spectral_slope(freqs, psd, fit_range=(2, 50))
    # Slope should be -true_alpha. Without fix it would be ~0.
    assert np.all(np.abs(slope - (-true_alpha)) < 0.1), (
        f"Slope should recover {-true_alpha:.2f}; got {slope.mean():.3f}"
    )


def test_hedges_g_small_scale():
    """Hedges' g must not collapse when pooled SD is in physical units smaller
    than eps. Without the fix, np.maximum(pooled_std, eps) would deflate g to
    near zero for source-localized data.
    """
    rng = np.random.default_rng(0)
    # Two groups, very different means, tiny SD — should give large |g|.
    a = rng.normal(loc=1e-19, scale=1e-21, size=(20, 4))
    b = rng.normal(loc=5e-19, scale=1e-21, size=(20, 4))

    g = hedges_g(a, b)
    # Expected |g| ~ (4e-19) / (1e-21) * Hedges_correction ≈ several hundred.
    # Pre-fix this collapses to (4e-19) / 2.2e-16 ≈ 0.002.
    assert np.all(np.abs(g) > 50), (
        f"Hedges g should be large for small-scale large-effect data; got {g}"
    )
