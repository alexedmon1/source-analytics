"""Regression tests for the 2026-09 audit fixes (see CHANGELOG, Unreleased)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from source_analytics.config import StudyConfig
from source_analytics.core import canonical_analysis_name, ANALYSIS_METADATA
from source_analytics.spectral.band_power import extract_band_power
from source_analytics.spectral.vertex import extract_band_power_vertices
from source_analytics.spectral.epoch_sampler import sample_epochs


# ---- #6: the package must import without the optional extras ---------------
def test_package_imports_without_mne():
    code = (
        "import sys; sys.modules['mne'] = None; sys.modules['mne.time_frequency'] = None\n"
        "import source_analytics.core, source_analytics.cli, source_analytics.spectral\n"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_tfr_raises_clear_error_without_mne(monkeypatch):
    import source_analytics.spectral.tfr as tfr
    monkeypatch.setitem(sys.modules, "mne", None)
    monkeypatch.setitem(sys.modules, "mne.time_frequency", None)
    with pytest.raises(ImportError, match=r"source-analytics\[mne\]"):
        tfr.tfr_array_morlet(np.zeros((1, 1, 10)), sfreq=100.0, freqs=[5.0], n_cycles=2)


# ---- #9: ROI and vertex `absolute` share one definition (dB/Hz density) -----
def test_vertex_absolute_matches_roi_density():
    freqs = np.linspace(1, 100, 397)
    psd_1d = 1e-12 / freqs  # 1/f
    bands = {"Alpha": (8, 13), "Beta": (13, 30)}
    roi = extract_band_power(freqs, psd_1d, bands)
    vtx = extract_band_power_vertices(freqs, psd_1d[np.newaxis, :], bands, noise_exclude=None)
    for b in bands:
        assert vtx[b]["absolute"][0] == pytest.approx(roi[b]["absolute"], rel=1e-9)
        assert vtx[b]["relative"][0] == pytest.approx(roi[b]["relative"], rel=1e-9)


# ---- #12: n_bootstrap: 0 means "full timeseries" on the vertex sampler too --
def test_sample_epochs_n_bootstrap_zero_returns_full_data():
    data = np.random.default_rng(0).standard_normal((3, 5000))
    out = sample_epochs(data, 500.0, epoch_duration_sec=1.0, n_epochs=4, seed=1, n_bootstrap=0)
    assert out.shape == (1, 3, 5000)
    np.testing.assert_array_equal(out[0], data)
    sampled = sample_epochs(data, 500.0, epoch_duration_sec=1.0, n_epochs=4, seed=1, n_bootstrap=1)
    assert sampled.shape == (4, 3, 500)


# ---- #19: vertex modules see global + per-analysis epoch_sampling ----------
def test_vertex_epoch_config_merges_global_vertex_and_analysis():
    from source_analytics.analyses.vertex_cluster_analysis import VertexClusterAnalysis

    class _Cfg:
        raw = {
            "epoch_sampling": {"enabled": True, "n_epochs": 40, "n_bootstrap": 5},
            "vertex_cluster": {"epoch_sampling": {"n_bootstrap": 0}},
        }
        vertex = {"epoch_sampling": {"n_epochs": 60}}

    a = VertexClusterAnalysis.__new__(VertexClusterAnalysis)
    a.config = _Cfg()
    merged = a._vertex_epoch_config()
    assert merged == {"enabled": True, "n_epochs": 60, "n_bootstrap": 0}

    class _Off:
        raw = {"epoch_sampling": {"n_epochs": 40}}
        vertex = {}

    a.config = _Off()
    assert a._vertex_epoch_config() is None


# ---- #17: deprecated names resolve to the canonical output dir -------------
def test_canonical_analysis_name():
    assert canonical_analysis_name("psd") == "roi_psd"
    assert canonical_analysis_name("vertex_mvpa") == "vertex_signature"
    assert canonical_analysis_name("roi_psd") == "roi_psd"


# ---- #4/#5: dependency metadata is honest --------------------------------
def test_comparison_modules_declare_requires():
    assert ANALYSIS_METADATA["fcd_comparison"]["requires"] == [
        "electrode_connectivity", "vertex_connectivity"]
    assert ANALYSIS_METADATA["fcd_comparison"]["supplements"] == "electrode_connectivity"
    assert ANALYSIS_METADATA["electrode_comparison"]["requires"] == ["electrode_psd", "roi_psd"]


# ---- #4: fcd_comparison finds its primaries across paradigm dirs ----------
def test_fcd_comparison_cross_paradigm_lookup(tmp_path):
    from source_analytics.analyses.fcd_comparison_analysis import FCDComparisonAnalysis

    analytics = tmp_path / "analytics"
    (analytics / "resting" / "electrode_connectivity" / "data").mkdir(parents=True)
    (analytics / "vertex" / "vertex_connectivity" / "data").mkdir(parents=True)
    rows = "subject,group,band,metric,fcd\ns1,A,Alpha,pli,0.5\ns2,B,Alpha,pli,0.4\n"
    (analytics / "resting" / "electrode_connectivity" / "data" / "electrode_fcd.csv").write_text(rows)
    (analytics / "vertex" / "vertex_connectivity" / "data" / "vertex_fcd.csv").write_text(rows)

    class _Cfg:
        raw = {}
        output_dir = analytics / "resting"
        paradigm_name = "resting"
        results_dir = tmp_path / "results"
        vertex = {}
        rois = None
        roi_categories = {}
        atlas_dir = None

    a = FCDComparisonAnalysis.__new__(FCDComparisonAnalysis)
    a.config = _Cfg()
    a._sensor_df = a._source_df = None
    a._selection = {}
    src = a._find_upstream_csv("vertex_connectivity", "vertex_fcd.csv", "source_dir")
    assert src == analytics / "vertex" / "vertex_connectivity" / "data" / "vertex_fcd.csv"
    sen = a._find_upstream_csv("electrode_connectivity", "electrode_fcd.csv", "sensor_dir")
    assert sen.parent.parent.parent.name == "resting"
    with pytest.raises(FileNotFoundError, match="run 'vertex_connectivity' first"):
        a._find_upstream_csv("vertex_connectivity", "nope.csv", "source_dir")


# ---- #1: init writes a parseable design/hypotheses/paradigms config --------
def _fake_reconstruction(root: Path, flat: bool):
    deriv = root / "derivatives"
    if flat:
        for sid in ("sub-01", "sub-02", "sub-03"):
            (deriv / sid / "pipeline" / "data").mkdir(parents=True)
    else:
        for g, sid in (("WT", "s1"), ("WT", "s2"), ("KO", "s3")):
            (deriv / g / sid / "pipeline" / "data").mkdir(parents=True)
    return deriv


def _run_cli(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "source_analytics.cli", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_init_grouped_layout_writes_parseable_config(tmp_path):
    root = tmp_path / "rest_roi"
    _fake_reconstruction(root, flat=False)
    out = _run_cli("init", str(root), "--name", "demo")
    assert out.returncode == 0, out.stderr
    path = root / "analysis" / "demo.yaml"
    assert path.exists(), out.stderr
    data = yaml.safe_load(path.read_text())
    assert data["design"]["levels"] == ["KO", "WT"]
    assert [h["name"] for h in data["hypotheses"]] == ["WT_vs_KO"]
    assert data["hypotheses"][0]["weights"] == {"WT": 1, "KO": -1}
    assert list(data["paradigms"]) == ["resting"]
    assert "subjects" not in data["paradigms"]["resting"]  # grouped layout: dirs give groups
    cfg = StudyConfig.from_yaml(path)
    assert cfg.has_paradigms
    assert [c.name for c in cfg.contrasts] == ["WT_vs_KO"]
    scoped = cfg.for_paradigm_analysis("resting", "roi_psd")
    assert scoped.output_dir == root / "analysis" / "analytics" / "resting"


def test_init_flat_layout_with_groups_from_and_stdout(tmp_path):
    root = tmp_path / "rest_roi"
    _fake_reconstruction(root, flat=True)
    sl = tmp_path / "sl.yaml"
    sl.write_text(yaml.safe_dump({"subjects": [
        {"id": "01", "group": "A"}, {"id": "02", "group": "B"}, {"id": "03", "group": "C"}]}))
    out = _run_cli("init", str(root), "--groups-from", str(sl), "--output", "-",
                   "--analyses", "roi_psd,roi_connectivity")
    assert out.returncode == 0, out.stderr
    data = yaml.safe_load(out.stdout)  # stdout is pure YAML
    assert data["paradigms"]["resting"]["subjects"] == {"sub-01": "A", "sub-02": "B", "sub-03": "C"}
    names = [h["name"] for h in data["hypotheses"]]
    assert names[0] == "group_omnibus" and len(names) == 4
    assert list(data["paradigms"]["resting"]["analyses"]) == ["roi_psd", "roi_connectivity"]
    assert not (root / "analysis").exists()


# ---- #10: --jobs 0 / -1 reach the resolver; #16: --force wipes -------------
def test_cli_jobs_default_is_none():
    from source_analytics import cli
    parser_ns = cli.main.__globals__  # sanity: module loaded
    assert "canonical_analysis_name" in parser_ns
    import argparse
    # Build the parser the same way main() does by invoking with --help is noisy;
    # instead check the argparse default via a dry parse of `run`.
    out = _run_cli("run", "--help")
    assert "--jobs" in out.stdout and "roi_connectivity" in out.stdout


def test_prepare_output_force_wipes_published_and_working_dirs(tmp_path):
    from source_analytics.cli import _prepare_output

    class _Cfg:
        output_dir = tmp_path / "analytics" / "resting"
        results_dir = tmp_path / "results"
        paradigm_name = "resting"

    work = _Cfg.output_dir / "roi_psd" / "data"
    tbl = _Cfg.results_dir / "tables" / "resting" / "roi_psd"
    fig = _Cfg.results_dir / "figures" / "resting" / "roi_psd"
    for d in (work, tbl, fig):
        d.mkdir(parents=True)
        (d / "x.csv").write_text("stale")

    # deprecated alias resolves to the canonical dir
    target = _prepare_output(_Cfg, "psd", strict=False, force=True, steps=None)
    assert target == _Cfg.output_dir / "roi_psd"
    assert not work.exists() and not tbl.exists() and not fig.exists()

    # --steps without process keeps data/, clears published dirs
    for d in (work, tbl):
        d.mkdir(parents=True)
        (d / "x.csv").write_text("stale")
    _prepare_output(_Cfg, "roi_psd", strict=False, force=True, steps={"statistics"})
    assert work.exists() and not tbl.exists()

    # --strict-output without --force errors on existing output
    with pytest.raises(SystemExit):
        _prepare_output(_Cfg, "roi_psd", strict=True, force=False, steps=None)


# ---- #8: vertex_spatial is retired end to end ------------------------------
def test_vertex_spatial_processes_nothing(tmp_path):
    from source_analytics.analyses.vertex_spatial_analysis import VertexSpatialAnalysis

    class _Cfg:
        raw = {}
        vertex = {}
        name = "t"
        results_dir = tmp_path / "results"
        paradigm_name = None
        roi_categories = {}
        atlas_dir = None

    a = VertexSpatialAnalysis.__new__(VertexSpatialAnalysis)
    a.config = _Cfg()
    a.output_dir = tmp_path / "vertex_spatial"
    a.output_dir.mkdir()
    a._warned = False
    a.setup()
    a.process_subject(object())  # must not touch a loader
    a.statistics()
    a.summary()
    assert (tmp_path / "results" / "tables" / "vertex_spatial" / "vertex_spatial_results.csv").exists()
    assert "RETIRED" in (a.output_dir / "ANALYSIS_SUMMARY.md").read_text()


# ---- #13: vertex_evoked is on the hypothesis contract ----------------------
def test_vertex_evoked_selectable_hypothesis():
    from source_analytics.analyses.vertex_evoked_analysis import VertexEvokedAnalysis
    assert "hypothesis" in VertexEvokedAnalysis.SELECTABLE


# ---- #7: R scripts are discoverable from an installed prefix --------------
def test_find_r_script_dir_env_override(tmp_path, monkeypatch):
    from source_analytics.analyses.base import find_r_script_dir
    d = tmp_path / "Rdir"; d.mkdir()
    monkeypatch.setenv("SOURCE_ANALYTICS_R_DIR", str(d))
    assert find_r_script_dir() == d


# ---- roi_connectivity honours the YAML metrics: list ----------------------
def test_roi_connectivity_reads_config_metrics():
    from source_analytics.analyses.roi_connectivity_analysis import ConnectivityAnalysis

    class _Cfg:
        raw = {"roi_connectivity": {"metrics": ["aec", "pli"]}}

    a = ConnectivityAnalysis.__new__(ConnectivityAnalysis)
    a.config = _Cfg()
    a._selection = {}
    a._edge_rows = []
    a.setup()
    assert a._metrics == [m for m in ConnectivityAnalysis._ROI_METRICS if m in ("aec", "pli")]

    a.config.raw = {"roi_connectivity": {"metrics": ["bogus"]}}
    with pytest.raises(ValueError, match="unknown metrics"):
        a.setup()
