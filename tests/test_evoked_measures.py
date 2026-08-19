"""ERP measures, induced/evoked separation, cycle ramps and tiled extraction.

The measures added in Phase 4. Sign is the thing worth testing hardest: every
existing measure in this package discards it, which is why nobody noticed there
was no ERP measure at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from source_analytics.spectral.evoked import (
    baseline_correct,
    erp_mean_amplitude,
    erp_measures,
    erp_peak,
    evoked_average,
    subtract_evoked,
)
from source_analytics.spectral.tfr import (
    extract_measure_in_band,
    extract_measure_in_tiles,
    resolve_n_cycles,
)

SFREQ = 500.0
XMIN = -0.5


def _epochs_with_erp(n_epochs=40, n_times=1000, amp=-3.0, peak_s=0.12, seed=0):
    """Trials carrying a known negative deflection plus noise."""
    rng = np.random.default_rng(seed)
    t = XMIN + np.arange(n_times) / SFREQ
    component = amp * np.exp(-((t - peak_s) ** 2) / (2 * 0.012 ** 2))
    return rng.standard_normal((n_epochs, n_times)) * 0.35 + component, t


# ----------------------------------------------------------------- ERP peak

def test_peak_recovers_amplitude_and_latency():
    epochs, _ = _epochs_with_erp(amp=-3.0, peak_s=0.12)
    wave = baseline_correct(evoked_average(epochs), SFREQ, (-0.5, 0.0), XMIN)
    peak = erp_peak(wave, SFREQ, (0.05, 0.25), XMIN, polarity="negative")

    assert peak["amplitude"] == pytest.approx(-3.0, abs=0.25)
    assert peak["latency"] == pytest.approx(0.12, abs=0.01)


def test_polarity_selects_the_right_extremum():
    """An N1 is the most negative point, not the largest excursion."""
    n = 600
    wave = np.zeros(n)
    wave[100] = -2.0   # negative peak
    wave[300] = +5.0   # larger positive peak elsewhere in the window

    neg = erp_peak(wave, SFREQ, (0.0, 1.2), 0.0, polarity="negative")
    pos = erp_peak(wave, SFREQ, (0.0, 1.2), 0.0, polarity="positive")
    absolute = erp_peak(wave, SFREQ, (0.0, 1.2), 0.0, polarity="abs")

    assert neg["amplitude"] == pytest.approx(-2.0)
    assert pos["amplitude"] == pytest.approx(5.0)
    assert absolute["amplitude"] == pytest.approx(5.0)


def test_peak_sign_survives():
    """The whole point of the measure: polarity is preserved, not discarded."""
    epochs, _ = _epochs_with_erp(amp=-3.0)
    wave = baseline_correct(evoked_average(epochs), SFREQ, (-0.5, 0.0), XMIN)
    assert erp_peak(wave, SFREQ, (0.05, 0.25), XMIN,
                    polarity="negative")["amplitude"] < 0

    epochs_pos, _ = _epochs_with_erp(amp=+3.0)
    wave_pos = baseline_correct(evoked_average(epochs_pos), SFREQ,
                                (-0.5, 0.0), XMIN)
    assert erp_peak(wave_pos, SFREQ, (0.05, 0.25), XMIN,
                    polarity="positive")["amplitude"] > 0


def test_empty_window_is_nan_not_an_exception():
    wave = np.zeros(100)
    out = erp_peak(wave, SFREQ, (5.0, 6.0), 0.0)
    assert np.isnan(out["amplitude"]) and np.isnan(out["latency"])


def test_unknown_polarity_raises():
    with pytest.raises(ValueError, match="polarity must be"):
        erp_peak(np.zeros(50), SFREQ, (0.0, 0.05), 0.0, polarity="sideways")


# ------------------------------------------------------------ mean amplitude

def test_mean_amplitude_is_the_window_mean():
    wave = np.concatenate([np.zeros(100), np.full(100, 2.0)])
    assert erp_mean_amplitude(wave, SFREQ, (0.2, 0.4), 0.0) == pytest.approx(2.0)


# ------------------------------------------------------------------ baseline

def test_baseline_removes_offset_without_touching_shape():
    wave = np.full(500, 7.0)
    wave[300:320] += 2.0
    corrected = baseline_correct(wave, SFREQ, (-0.5, -0.1), XMIN)
    assert corrected[:100].mean() == pytest.approx(0.0, abs=1e-9)
    assert corrected[305] == pytest.approx(2.0)


def test_empty_baseline_raises():
    with pytest.raises(ValueError, match="baseline"):
        baseline_correct(np.zeros(100), SFREQ, (-0.5, -0.5), XMIN)


# -------------------------------------------------------- induced vs evoked

def test_subtract_evoked_removes_the_phase_locked_part():
    epochs, _ = _epochs_with_erp(n_epochs=200, amp=-4.0)
    induced = subtract_evoked(epochs)

    # the evoked average of the residual is ~zero by construction
    assert np.abs(evoked_average(induced)).max() < 1e-9
    # and the original had a real deflection to remove
    assert np.abs(evoked_average(epochs)).max() > 1.0


def test_induced_needs_more_than_one_trial():
    with pytest.raises(ValueError, match="at least 2 trials"):
        subtract_evoked(np.zeros((1, 100)))


# --------------------------------------------------------------- cycle ramp

def test_cycle_ramp_spans_the_endpoints():
    freqs = np.arange(2, 111, 1.0)
    cycles = resolve_n_cycles(freqs, [1, 30])
    assert cycles[0] == pytest.approx(1.0)
    assert cycles[-1] == pytest.approx(30.0)
    assert np.all(np.diff(cycles) > 0)


def test_scalar_and_adaptive_still_work():
    freqs = np.arange(2, 51, 1.0)
    assert resolve_n_cycles(freqs, 7) == 7.0
    adaptive = resolve_n_cycles(freqs, "adaptive")
    assert adaptive.min() == pytest.approx(3.0)


def test_bad_cycle_specs_raise():
    freqs = np.arange(2, 20, 1.0)
    with pytest.raises(ValueError, match="adaptive"):
        resolve_n_cycles(freqs, "sometimes")
    with pytest.raises(ValueError, match=r"\[lo, hi\]"):
        resolve_n_cycles(freqs, [1, 2, 3])
    with pytest.raises(ValueError, match="positive"):
        resolve_n_cycles(freqs, [0, 30])


# --------------------------------------------------------------- tiled means

def test_one_tile_equals_the_rectangle():
    m = np.arange(60.0).reshape(6, 10)
    freqs = np.arange(6.0)
    rect = extract_measure_in_band(m, freqs, 1.0, (1, 3), (2, 6))
    tiled = extract_measure_in_tiles(
        m, freqs, 1.0, [{"band": (1, 3), "time_window": (2, 6)}])
    assert tiled == pytest.approx(rect)


def test_tiles_track_a_diagonal_a_rectangle_would_miss():
    """A sweeping response is on the diagonal; the box averages in background."""
    n_f, n_t = 40, 100
    m = np.zeros((n_f, n_t))
    freqs = np.arange(n_f, dtype=float)
    # response walks up in frequency as time advances
    for step in range(5):
        f0, f1 = 5 * step, 5 * step + 5
        t0, t1 = 20 * step, 20 * step + 20
        m[f0:f1, t0:t1] = 1.0

    tiles = [{"band": (5 * s, 5 * s + 4), "time_window": (20 * s, 20 * s + 19)}
             for s in range(5)]
    diagonal = extract_measure_in_tiles(m, freqs, 1.0, tiles)
    rectangle = extract_measure_in_band(m, freqs, 1.0, (0, 24), (0, 100))

    assert diagonal == pytest.approx(1.0)
    assert rectangle < 0.35
    assert diagonal > rectangle * 2


def test_tiles_outside_the_map_are_skipped_not_fatal():
    m = np.ones((5, 10))
    freqs = np.arange(5.0)
    out = extract_measure_in_tiles(m, freqs, 1.0, [
        {"band": (0, 2), "time_window": (0, 5)},
        {"band": (99, 100), "time_window": (0, 5)},   # no such frequencies
        {"band": (0, 2), "time_window": (50, 60)},     # past the end
    ])
    assert out == pytest.approx(1.0)


def test_all_tiles_empty_gives_nan():
    m = np.ones((5, 10))
    assert np.isnan(extract_measure_in_tiles(
        m, np.arange(5.0), 1.0, [{"band": (99, 100), "time_window": (0, 5)}]))


# ------------------------------------------------------------ measure driver

def test_erp_measures_emits_amplitude_and_latency():
    epochs, _ = _epochs_with_erp(amp=-3.0, peak_s=0.12)
    out = erp_measures(
        epochs, SFREQ, XMIN, (-0.5, 0.0),
        [{"name": "n1", "time_window": (0.05, 0.25), "polarity": "negative"},
         {"name": "late", "time_window": (0.3, 0.6), "type": "mean"}],
    )
    assert out["n1"] == pytest.approx(-3.0, abs=0.25)
    assert out["n1_latency"] == pytest.approx(0.12, abs=0.01)
    assert "late" in out and "late_latency" not in out


def test_erp_measures_rejects_unknown_type():
    epochs, _ = _epochs_with_erp()
    with pytest.raises(ValueError, match="peak' or 'mean"):
        erp_measures(epochs, SFREQ, XMIN, None,
                     [{"name": "x", "time_window": (0, 0.1), "type": "rms"}])


# --------------------------------------------------------------- ITC debias

def test_debiased_itc_is_zero_for_pure_noise_at_any_trial_count():
    """The bias this removes is a real confound when trial counts differ."""
    from source_analytics.spectral.tfr import debias_itc

    rng = np.random.default_rng(0)
    raw_by_n, debiased_by_n = {}, {}
    for n in (20, 100, 400):
        phases = rng.uniform(0, 2 * np.pi, size=(4000, n))
        raw = float(np.abs(np.exp(1j * phases).mean(axis=1)).mean())
        raw_by_n[n] = raw
        debiased_by_n[n] = float(debias_itc(raw, n))

    # raw ITC of noise falls with trial count — that is the confound
    assert raw_by_n[20] > raw_by_n[400] * 3
    # debiased is ~zero regardless
    for n in (20, 100, 400):
        assert debiased_by_n[n] == pytest.approx(0.0, abs=0.02)


def test_debias_preserves_a_genuinely_phase_locked_response():
    from source_analytics.spectral.tfr import debias_itc

    assert debias_itc(1.0, 50) == pytest.approx(1.0)
    assert debias_itc(0.9, 200) == pytest.approx(0.9, abs=0.01)


def test_debias_clips_at_zero_rather_than_going_imaginary():
    from source_analytics.spectral.tfr import debias_itc

    assert debias_itc(0.0, 10) == pytest.approx(0.0)
    assert np.all(debias_itc(np.array([0.0, 0.01, 0.5]), 10) >= 0.0)


def test_debias_needs_two_trials():
    from source_analytics.spectral.tfr import debias_itc

    with pytest.raises(ValueError, match="at least 2 trials"):
        debias_itc(0.5, 1)


def test_rayleigh_threshold_matches_the_eeglab_formula():
    """rcrit = sqrt(-(1/n)·log(0.5)); reproduced for comparability, not used."""
    from source_analytics.spectral.tfr import itc_rayleigh_threshold

    for n in (10, 50, 200):
        assert itc_rayleigh_threshold(n) == pytest.approx(
            np.sqrt(-(1.0 / n) * np.log(0.5)))
    # it is a threshold, not a bias correction: it does not vanish for noise
    assert itc_rayleigh_threshold(100) > 0.05


def test_needs_induced_gates_on_measure_type():
    """Induced is a second full TFR pass, so it must run only when asked for.

    The call site shipped without the method (`5c3e884`), which failed every
    subject of a 26-subject run while the run still exited 0.
    """
    from source_analytics.analyses.roi_evoked_analysis import ROIEvokedAnalysis

    needs = ROIEvokedAnalysis._needs_induced
    assert needs([{"type": "induced", "name": "induced_gamma"}])
    assert needs([{"type": "induced_stp", "name": "x"}])
    assert not needs([{"type": "itc", "name": "itc_40hz"}])
    assert not needs([])
    # A measure that merely mentions the word is not a request for it.
    assert not needs([{"type": "ersp", "name": "induced_like"}])


def test_all_subjects_failing_raises():
    """An empty run must not report success."""
    import pytest
    from source_analytics.analyses.base import BaseAnalysis

    BaseAnalysis._check_not_all_failed(0, 0)      # nothing to process
    BaseAnalysis._check_not_all_failed(3, 10)     # partial failure is tolerated
    with pytest.raises(RuntimeError, match="All 10 subjects failed"):
        BaseAnalysis._check_not_all_failed(10, 10)
