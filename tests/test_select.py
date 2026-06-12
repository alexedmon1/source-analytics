"""Tests for the --select / --metric / --band sub-output selection mechanism."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from source_analytics.analyses.base import BaseAnalysis
from source_analytics.cli import _parse_selection


class _Dummy(BaseAnalysis):
    """Concrete BaseAnalysis with no-op lifecycle, for unit-testing helpers."""
    name = "dummy"

    def setup(self): ...
    def process_subject(self, subject): ...
    def aggregate(self): ...
    def statistics(self): ...
    def figures(self): ...
    def summary(self): ...


def _bare(name="dummy", selection=None, bands=None):
    """A _Dummy with just the attrs _select / _selected_bands need.

    Skips __init__ (filesystem/atlas setup) — we only exercise the pure
    selection logic.
    """
    obj = _Dummy.__new__(_Dummy)
    obj.name = name
    obj._selection = selection or {}
    if bands is not None:
        obj.config = SimpleNamespace(bands=dict(bands))
    return obj


# --------------------------------------------------------------- _select_norm
def test_select_norm():
    n = BaseAnalysis._select_norm
    assert n("Low Gamma") == "low_gamma"
    assert n("imag-coherence") == "imag_coherence"
    assert n("  PLI ") == "pli"


# -------------------------------------------------------------------- _select
def test_select_no_selection_returns_all():
    a = _bare()
    assert a._select("metric", ["pli", "aec", "dwpli"]) == ["pli", "aec", "dwpli"]


def test_select_filters_and_preserves_order():
    a = _bare(selection={"metric": frozenset({"dwpli", "pli"})})
    # order follows `available`, not the request
    assert a._select("metric", ["pli", "aec", "dwpli"]) == ["pli", "dwpli"]


def test_select_normalized_match():
    a = _bare(selection={"band": frozenset({"low_gamma"})})
    assert a._select("band", ["Low Gamma", "Beta"]) == ["Low Gamma"]


def test_select_unknown_requested_is_ignored(caplog):
    a = _bare(selection={"metric": frozenset({"pli", "nope"})})
    assert a._select("metric", ["pli", "aec"]) == ["pli"]


def test_select_empty_match_raises():
    a = _bare(selection={"metric": frozenset({"nope"})})
    with pytest.raises(ValueError):
        a._select("metric", ["pli", "aec"])


def test_select_dim_not_active_returns_all():
    # a band selection must not affect an unrelated metric dim
    a = _bare(selection={"band": frozenset({"beta"})})
    assert a._select("metric", ["pli", "aec"]) == ["pli", "aec"]


# ------------------------------------------------------------ _selected_bands
def test_selected_bands_all_when_inactive():
    a = _bare(bands={"Delta": (1, 4), "Beta": (13, 30)})
    assert list(a._selected_bands()) == ["Delta", "Beta"]


def test_selected_bands_filtered_and_cached():
    a = _bare(selection={"band": frozenset({"beta"})},
              bands={"Delta": (1, 4), "Beta": (13, 30)})
    assert list(a._selected_bands()) == ["Beta"]
    # second call hits the cache (same object identity)
    assert a._selected_bands() is a._selected_bands()


# ------------------------------------------------------------ _parse_selection
def _args(**kw):
    base = dict(metric=None, band=None, select=None, analysis=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_parse_none_when_empty():
    assert _parse_selection(_args()) is None


def test_parse_metric_shorthand():
    sel = _parse_selection(_args(metric="dwpli,PLI"))
    assert sel == {"metric": frozenset({"dwpli", "pli"})}


def test_parse_band_shorthand_normalized():
    sel = _parse_selection(_args(band="Low Gamma"))
    assert sel == {"band": frozenset({"low_gamma"})}


def test_parse_generic_select_repeatable():
    sel = _parse_selection(_args(select=["metric=pli", "band=beta,delta"]))
    assert sel == {"metric": frozenset({"pli"}),
                   "band": frozenset({"beta", "delta"})}


def test_parse_metric_and_select_merge():
    sel = _parse_selection(_args(metric="aec", select=["metric=pli"]))
    assert sel == {"metric": frozenset({"aec", "pli"})}


def test_parse_bad_format_exits():
    with pytest.raises(SystemExit):
        _parse_selection(_args(select=["metricpli"]))


def test_parse_unknown_dim_exits():
    with pytest.raises(SystemExit):
        _parse_selection(_args(select=["bogus=x"]))


def test_parse_dim_not_selectable_for_target_analysis_exits():
    # roi_psd declares only `band`; requesting metric for it is an error
    with pytest.raises(SystemExit):
        _parse_selection(_args(metric="pli", analysis="roi_psd"))


def test_parse_valid_dim_for_target_analysis():
    sel = _parse_selection(_args(metric="pli", analysis="vertex_connectivity"))
    assert sel == {"metric": frozenset({"pli"})}
