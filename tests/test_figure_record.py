"""What each figure shows (figure_record.py), recorded where it is drawn and listed by gallery build."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_results_spec import RECORD, module  # noqa: F401 - the fixture

from source_analytics import gallery as g
from source_analytics import results_spec as rs
from source_analytics.figure_record import RECORDS, clear_records, read_records, record_figure
from source_analytics.viz.brain_roi import _record_mosaic


def test_records_are_appended_and_the_latest_wins(tmp_path):
    record_figure(tmp_path, tmp_path / "a.png", measure="exponent", contrast="disease_effect",
                  facets={"band": "Alpha"}, kind="brain")
    record_figure(tmp_path, "b.png", kind="summary", caption="", facets={})
    record_figure(tmp_path, "a.png", measure="offset")
    recs = read_records(tmp_path)
    assert recs == {"a.png": {"measure": "offset"}, "b.png": {"kind": "summary"}}
    with pytest.raises(ValueError, match="unknown figure tags"):
        record_figure(tmp_path, "c.png", colour="red")
    clear_records(tmp_path)
    assert not (tmp_path / RECORDS).exists() and read_records(tmp_path) == {}


def test_a_mosaic_records_its_test_from_its_facet_columns(tmp_path):
    _record_mosaic(tmp_path, tmp_path / "effect_size_disease_effect_Alpha_relative.png",
                   {"hypothesis": "disease_effect", "band": "Alpha", "dv": "relative"},
                   {"hypothesis": "contrast", "dv": "measure", "band": "facet:band"},
                   {"shows": "roi_psd_hypotheses.csv"})
    assert read_records(tmp_path) == {"effect_size_disease_effect_Alpha_relative.png": {
        "measure": "relative", "contrast": "disease_effect", "facets": {"band": "Alpha"}, "kind": "brain",
        "shows": "roi_psd_hypotheses.csv"}}


def test_the_gallery_lists_what_each_figure_shows(module, tmp_path):  # noqa: F811
    pytest.importorskip("lumiere")
    from lumiere import check_analysis

    assert rs.write(module, dict(RECORD), "2026-11-09T12:00:00+00:00").ok
    figs = tmp_path / "figs"
    figs.mkdir()
    for name in ("effect_size_disease_effect_exponent.png", "boxplot.png", "made_up.png"):
        (figs / name).write_bytes(b"\x89PNG")
    _record_mosaic(figs, figs / "effect_size_disease_effect_exponent.png",
                   {"hypothesis": "disease_effect", "dv": "exponent"},
                   {"hypothesis": "contrast", "dv": "measure"}, {"shows": "roi_aperiodic_hypotheses.csv"})
    record_figure(figs, "boxplot.png", kind="distribution", measure="offset", caption="Offset by group",
                  shows="not_a_table.csv")
    module.fig_dir = figs
    assert g.attach_figures(module) == 3
    a = json.loads((Path(module.tbl_dir) / "analysis.json").read_text())
    by = {f["path"]: f for f in a["figures"]}
    mosaic = by["figures/effect_size_disease_effect_exponent.png"]
    assert (mosaic["measure"], mosaic["contrast"], mosaic["kind"], mosaic["shows"]) == (
        "aperiodic_exponent", "disease_effect", "brain", "roi_aperiodic_hypotheses.csv")   # the spec's name
    box = by["figures/boxplot.png"]
    assert (box["caption"], box["kind"], box["measure"]) == ("Offset by group", "distribution", "aperiodic_offset")
    assert "shows" not in box                                   # not one of the analysis's tables
    assert "measure" not in by["figures/made_up.png"]           # no record: as before
    rep = check_analysis(Path(module.tbl_dir))
    assert rep.ok and not [w for w in rep.warnings if "names a test" in w], rep.warnings
