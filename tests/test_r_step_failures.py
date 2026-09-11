"""A failed or timed-out R statistics step fails the run instead of exiting 0.

Found on the FORGE treatment re-run. roi_directed (in all three source arms) and
roi_cross_freq's AAC/PPC tier hit a hard-coded one-hour R timeout. The module
logged it and the command exited 0, so region tables from the previous code
version went on looking current. There is now no default limit (``r_timeout_sec``
sets one), and a failure raises ``RStepFailed``.

Also locks the roi_cross_freq PAC mosaics, whose call named columns the native
hypothesis table does not have, so none was ever drawn.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

import source_analytics.viz.brain_roi as brain_roi
from source_analytics.analyses.base import RStepFailed
from source_analytics.analyses.roi_cross_freq_analysis import ROICrossFreqAnalysis
from source_analytics.analyses.roi_directed_analysis import ROIDirectedAnalysis
from source_analytics.analyses.roi_psd_analysis import ROIPsdAnalysis
from source_analytics.config import StudyConfig


def _module(cls, sample_config_yaml, tmp_path, block=None):
    config = StudyConfig.from_yaml(sample_config_yaml)
    if block is not None:
        config.raw[cls.name] = block
    module = cls(config, tmp_path / "analytics" / cls.name)
    if not hasattr(module, "_sfreq"):
        module._sfreq = None
    return module


def test_there_is_no_default_r_timeout_and_one_can_be_set(sample_config_yaml, tmp_path):
    assert _module(ROIPsdAnalysis, sample_config_yaml, tmp_path)._r_timeout is None
    limited = _module(ROIPsdAnalysis, sample_config_yaml, tmp_path, {"r_timeout_sec": 7200})
    assert limited._r_timeout == 7200.0


def test_a_timed_out_r_step_raises_and_uses_no_limit_by_default(sample_config_yaml, tmp_path, monkeypatch):
    module = _module(ROIDirectedAnalysis, sample_config_yaml, tmp_path)
    (module.output_dir / "data" / "roi_transfer_entropy_edges.csv").write_text("x\n")
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RStepFailed, match="roi_directed: R script timed out"):
        module.summary()
    assert seen["timeout"] is None


def test_a_failing_r_step_raises(sample_config_yaml, tmp_path, monkeypatch):
    module = _module(ROIPsdAnalysis, sample_config_yaml, tmp_path)
    for name in ("band_power.csv", "psd_curves.csv"):
        (module.output_dir / "data" / name).write_text("x\n")
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=2, stdout="", stderr="boom"))
    with pytest.raises(RStepFailed, match="roi_psd: R script failed with exit code 2"):
        module.summary()


def test_cross_freq_attempts_both_tiers_before_failing(sample_config_yaml, tmp_path):
    module = _module(ROICrossFreqAnalysis, sample_config_yaml, tmp_path)
    module._metrics = ["pac", "aac"]
    edges = []
    module._run_pac_r = lambda: False
    module._run_edges_r = lambda metrics: edges.append(metrics) or True
    module._edge_metrics_on_disk = lambda: ["aac"]
    with pytest.raises(RStepFailed, match="PAC"):
        module.summary()
    assert edges == [["aac"]], "the AAC/PPC tier must still run when PAC fails"


def _pac_table(module):
    cols = ["hypothesis", "band", "spatial", "effect_size", "p_value", "q_value", "significant", "dv"]
    path = module.tbl_dir / "roi_pac_posthoc_region.csv"
    path.write_text(",".join(cols) + "\ndisease_effect,Theta-Low Gamma,Motor,0.9,0.001,0.01,TRUE,z_score\n")
    return set(cols)


def test_pac_mosaics_name_columns_the_table_has(sample_config_yaml, tmp_path, monkeypatch):
    module = _module(ROICrossFreqAnalysis, sample_config_yaml, tmp_path)
    columns = _pac_table(module)
    seen = {}
    monkeypatch.setattr(brain_roi, "render_posthoc_mosaics", lambda *a, **kw: seen.update(kw) or [])
    module._render_brain_mosaics()
    named = {seen["effect_col"], seen["roi_col"], seen["p_col"], seen["q_col"], *seen["facet_cols"]}
    assert named <= columns, f"mosaic asks for columns the table lacks: {sorted(named - columns)}"


def test_the_figures_pass_draws_pac_mosaics(sample_config_yaml, tmp_path, monkeypatch):
    module = _module(ROICrossFreqAnalysis, sample_config_yaml, tmp_path)
    _pac_table(module)
    module._metrics = ["pac"]
    module._call_r_figures_only = lambda *a, **kw: None
    module._edge_metrics_on_disk = lambda: []
    drawn = []
    monkeypatch.setattr(brain_roi, "render_posthoc_mosaics", lambda *a, **kw: drawn.append(kw) or [])
    module.figures()
    assert drawn, "a figures-only pass must draw the PAC mosaics"


# ---- the CLI: a failure fails the run, but a batch finishes its other modules --------

def _two_module_study(tmp_path):
    import yaml

    (tmp_path / "data").mkdir()
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump({
        "name": "t", "groups": {"WT_VEH": "WT", "KO_VEH": "KO"}, "bands": {"Theta": [4, 8]},
        "paths": {"analytics": str(tmp_path / "a"), "results": str(tmp_path / "r")},
        "paradigms": {"p1": {"data_dir": str(tmp_path / "data"),
                             "analyses": {"roi_psd": {}, "roi_aperiodic": {}}}},
    }))
    return path


def _run_cli(monkeypatch, argv, fail_on):
    import sys

    from source_analytics import cli

    ran = []

    def fake_run_single(aconfig, name, **kw):
        ran.append(name)
        if name == fail_on:
            raise RStepFailed(f"{name}: R script failed with exit code 2")

    monkeypatch.setattr(cli, "_run_single", fake_run_single)
    monkeypatch.setattr(cli, "_prepare_output", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["source-analytics", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return ran, exit_info.value.code


def test_a_paradigm_run_finishes_the_other_modules_then_exits_1(tmp_path, monkeypatch, capsys):
    study = _two_module_study(tmp_path)
    ran, code = _run_cli(monkeypatch, ["run", "--study", str(study), "--paradigm", "p1"], fail_on="roi_psd")
    # Order follows the config; what matters is that the second module still ran.
    assert sorted(ran) == ["roi_aperiodic", "roi_psd"] and code == 1
    assert "1 R statistics step(s) failed" in capsys.readouterr().out


def test_a_single_module_run_that_fails_exits_1(tmp_path, monkeypatch):
    study = _two_module_study(tmp_path)
    ran, code = _run_cli(monkeypatch, ["run", "--study", str(study), "--paradigm", "p1",
                                       "--analysis", "roi_psd"], fail_on="roi_psd")
    assert ran == ["roi_psd"] and code == 1
