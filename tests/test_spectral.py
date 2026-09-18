"""Tests for spectral analysis modules."""

import pytest
import numpy as np

from source_analytics.spectral.psd import compute_psd, compute_psd_multiroi
from source_analytics.spectral.band_power import extract_band_power
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

    # `absolute` is mean power DENSITY: 10*log10(band integral / bandwidth), in
    # dB/Hz. A flat unit PSD therefore reads 0 dB/Hz in EVERY band, however wide
    # — bandwidth no longer inflates wide bands. (This assertion previously read
    # `Gamma > Alpha` "because Gamma is wider", pinning the very artifact the
    # density change removed.)
    assert result["Alpha"]["absolute"] == pytest.approx(0.0, abs=1e-9)
    assert result["Gamma"]["absolute"] == pytest.approx(
        result["Alpha"]["absolute"], abs=1e-9
    )

    # `relative` is still an integral ratio, so it does scale with bandwidth.
    assert 0 < result["Alpha"]["relative"] < 1
    assert result["Gamma"]["relative"] > result["Alpha"]["relative"]


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


def test_aperiodic_default_freq_range_avoids_notch():
    """The package-wide aperiodic fit range must stop below the 57-63 Hz notch.

    Regression guard: vertex_specparam previously defaulted to 1-100 Hz, which
    drags the log-log line fit through the notch and the >80 Hz roll-off. On
    FORGE ROI spectra that collapsed r^2 from ~0.89 to ~0.27 and pulled the
    exponent to ~0, silently invalidating every vertex 1/f map.
    """
    from source_analytics.spectral.aperiodic import (
        DEFAULT_FREQ_RANGE,
        resolve_freq_range,
    )

    assert DEFAULT_FREQ_RANGE[1] <= 50.0, (
        "Default aperiodic fmax must stay below the 57-63 Hz line-noise notch"
    )
    # Lower border must clear the high-pass roll-off AND the theta/alpha peak
    # (Gerster et al. 2022: borders must not cross oscillatory peaks).
    assert DEFAULT_FREQ_RANGE[0] >= 12.0

    # Empty/None config falls back to the safe default.
    assert resolve_freq_range(None) == DEFAULT_FREQ_RANGE
    assert resolve_freq_range({}) == DEFAULT_FREQ_RANGE
    # An explicit range is honoured (so a study can still override deliberately).
    assert resolve_freq_range({"freq_range": [3, 40]}) == (3.0, 40.0)
    # Malformed ranges are rejected rather than silently reinterpreted.
    with pytest.raises(ValueError):
        resolve_freq_range({"freq_range": [1, 2, 3]})


def test_aperiodic_fit_degrades_when_range_spans_notch():
    """Widening the fit range past the notch must measurably hurt the fit.

    Synthesises a 1/f spectrum with a notch at 57-63 Hz and a roll-off above
    80 Hz (the FORGE preprocessing shape) and checks that the default range
    recovers the exponent while the 1-100 Hz range does not.
    """
    from source_analytics.spectral.aperiodic import fit_aperiodic

    freqs = np.arange(1.0, 100.0, 0.5)
    true_exp = 1.5
    psd = 10.0 ** (-true_exp * np.log10(freqs))
    psd[(freqs >= 57) & (freqs <= 63)] *= 1e-3   # notch
    psd[freqs > 80] *= 10.0 ** (-3 * np.log10(freqs[freqs > 80] / 80.0))  # roll-off

    good = fit_aperiodic(freqs, psd, freq_range=(2, 50))
    bad = fit_aperiodic(freqs, psd, freq_range=(1, 100))

    assert good["r_squared"] > bad["r_squared"], (
        f"Notch-spanning range should fit worse; got good={good['r_squared']:.3f} "
        f"bad={bad['r_squared']:.3f}"
    )
    assert abs(good["exponent"] - true_exp) < abs(bad["exponent"] - true_exp), (
        "The clean range must recover the true exponent more closely"
    )


def test_centered_offset_removes_mechanical_slope_coupling():
    """offset_centered must decouple offset from exponent.

    specparam's offset is the intercept at 1 Hz, far below a 12-45 Hz window, so
    a steeper slope mechanically forces a higher intercept. Simulate spectra that
    share a common power level at the window centre but differ in slope: their
    1 Hz offsets must correlate with exponent, and the centred offsets must not.
    """
    from source_analytics.spectral.aperiodic import centered_offset

    rng_win = (12.0, 45.0)
    f_c = np.sqrt(rng_win[0] * rng_win[1])
    exponents = np.linspace(0.5, 2.0, 25)
    level_at_centre = -20.0  # identical broadband level for every spectrum
    offsets = level_at_centre + exponents * np.log10(f_c)

    r_raw = np.corrcoef(offsets, exponents)[0, 1]
    centred = np.array([
        centered_offset(o, e, rng_win) for o, e in zip(offsets, exponents)])

    assert r_raw > 0.99, "raw 1 Hz offset should track exponent mechanically"
    assert np.allclose(centred, level_at_centre), (
        "centred offset must recover the true common level, independent of slope"
    )


def test_resolve_freq_range_rejects_and_warns(caplog):
    """Invalid ranges raise; risky-but-legal ranges warn rather than fail."""
    from source_analytics.spectral.aperiodic import resolve_freq_range

    for bad in ([50, 10], [0, 40], [10, 10]):
        with pytest.raises(ValueError):
            resolve_freq_range({"freq_range": bad})

    with caplog.at_level("WARNING"):
        resolve_freq_range({"freq_range": [2, 50]})
    assert any("high-pass roll-off" in r.getMessage() for r in caplog.records), (
        "a 2 Hz lower border must warn about the roll-off / peak-crossing"
    )

    caplog.clear()
    with caplog.at_level("WARNING"):
        resolve_freq_range({"freq_range": [12, 80]})
    assert any("mains" in r.getMessage() for r in caplog.records)


def test_fit_aperiodic_carries_window_provenance():
    """Every fit records the window it used, so tables are self-documenting."""
    from source_analytics.spectral.aperiodic import fit_aperiodic

    freqs = np.arange(1.0, 100.0, 0.5)
    psd = 10.0 ** (-1.5 * np.log10(freqs))
    out = fit_aperiodic(freqs, psd, freq_range=(12, 45))

    assert out["fit_fmin"] == 12.0 and out["fit_fmax"] == 45.0
    assert "offset_centered" in out
    assert np.isfinite(out["offset_centered"])


# --- Two-fit peak detection / fit-window justification -----------------------

BANDS = {
    "Delta": (1, 4), "Theta": (4, 10), "Alpha": (10, 13), "Beta": (13, 30),
    "Low Gamma": (30, 55), "High Gamma": (65, 80), "Epsilon": (80, 150),
}


def test_band_reachability_marks_unreachable_and_censored():
    from source_analytics.spectral.aperiodic import band_peak_reachability

    reach = band_peak_reachability(BANDS, (12, 45))

    # Wholly outside the window -> peaks are structurally impossible
    for band in ("Delta", "Theta", "High Gamma", "Epsilon"):
        assert not reach[band]["reachable"], f"{band} cannot be reachable at 12-45"
    # Fully inside -> reachable and complete
    assert reach["Beta"]["reachable"] and not reach["Beta"]["censored"]
    assert reach["Beta"]["frac_visible"] == 1.0
    # Partial overlap -> reachable but censored (rates are a lower bound)
    assert reach["Low Gamma"]["reachable"] and reach["Low Gamma"]["censored"]
    assert 0.0 < reach["Low Gamma"]["frac_visible"] < 1.0
