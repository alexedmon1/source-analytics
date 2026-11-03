"""`source-analytics gallery build` (gallery.py): figures into the analysis folder, then the shell."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_results_spec import RECORD, module  # noqa: F401 - the fixture

from source_analytics import gallery as g
from source_analytics import results_spec as rs


def test_figure_kinds_and_captions():
    assert g.figure_kind("effect_size_disease_effect_exponent.png") == "brain"
    assert g.figure_kind("roi_aperiodic_significance_heatmap_dose_iv.png") == "summary"
    assert g.figure_kind("roi_aperiodic_exponent_boxplot.png") == "distribution"
    assert g.figure_kind("something.png") == "other"
    assert g.caption("roi_aperiodic_exponent_boxplot.png", "roi_aperiodic") == "Exponent boxplot"


def test_figures_are_copied_listed_and_checked(module, tmp_path):  # noqa: F811
    pytest.importorskip("neurobox")
    assert rs.write(module, dict(RECORD), "2026-11-03T12:00:00+00:00").ok
    figs = tmp_path / "figs"
    figs.mkdir()
    for name in ("effect_size_disease_effect_exponent.png", "roi_aperiodic_exponent_boxplot.png"):
        (figs / name).write_bytes(b"\x89PNG")
    module.fig_dir = figs
    assert g.attach_figures(module, {"effect_size_disease_effect_exponent.png"}) == 2
    a = json.loads((Path(module.tbl_dir) / "analysis.json").read_text())
    assert [(f["path"], f["kind"], f["drawn_by"]) for f in a["figures"]] == [
        ("figures/effect_size_disease_effect_exponent.png", "brain", "gallery"),
        ("figures/roi_aperiodic_exponent_boxplot.png", "distribution", "analysis")]
    assert (Path(module.tbl_dir) / "figures" / "roi_aperiodic_exponent_boxplot.png").is_file()
    # a rebuild replaces its own figures, nothing else
    (figs / "roi_aperiodic_exponent_boxplot.png").unlink()
    assert g.attach_figures(module) == 1
    a = json.loads((Path(module.tbl_dir) / "analysis.json").read_text())
    assert [f["path"] for f in a["figures"]] == ["figures/effect_size_disease_effect_exponent.png"]


def test_settings_come_from_the_study_config_without_a_gallery_yaml(tmp_path):
    loc = tmp_path / "loc"
    (loc / "qc").mkdir(parents=True)
    (loc / "qc" / "analysis.json").write_text("{}")
    config = SimpleNamespace(name="Study", results_dir=tmp_path / "results", raw={
        "gallery_title": "FORGE", "groups": {"KO": "Knock-out"}, "group_order": ["WT", "KO"],
        "paths": {"localizations": [{"label": "MC", "path": "./loc"}, {"label": "gone", "path": "./nope"}]}})
    s = g.settings_for(config, tmp_path / "study.yaml")
    assert s["name"] == "FORGE" and s["groups"] == {"KO": "Knock-out"} and s["paths"]["gallery"] == "gallery"
    assert s["paths"]["results"] == [str(tmp_path / "results" / "tables"), str((loc / "qc").resolve())]
    (tmp_path / "gallery.yaml").write_text("name: Own\n")
    assert g.settings_for(config, tmp_path / "study.yaml") == {"name": "Own"}
