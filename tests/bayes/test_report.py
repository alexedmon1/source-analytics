"""Reading a run without the code (engine plan E4): the results-table schema and report.md.

Fast: a synthetic run directory (table, manifest, job, gate files, scales, posterior predictive) is turned into a
report, and the schema's columns are checked against the R code that writes them. Checked on real runs 2026-11-09
(bayes-sl a97d: a run with no gate, and the gated run whose W set failed one SBC test, shown as "gate failed").
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from source_analytics.bayes import r_dir
from source_analytics.bayes.report import TABLE_COLUMNS, build_report, check_table


def test_every_schema_column_is_written_by_the_engine():
    src = (r_dir() / "bayes" / "readout.R").read_text() + (r_dir() / "bayes" / "analysis.R").read_text()
    for col in TABLE_COLUMNS:
        assert re.search(rf"\b{col}\b", src), col


def test_check_table_rejects_other_tables():
    with pytest.raises(ValueError, match="schema"):
        check_table(pd.DataFrame({"band": [1]}))


def _run(tmp: Path, gate: str = "passed") -> Path:
    work = tmp / "bayes"
    gdir = work / "gate" / "value"
    gdir.mkdir(parents=True)
    rows = []
    for s, est in (("W", -1.2), ("I", -1.0)):
        rows.append({c: None for c in TABLE_COLUMNS} | {
            "hypothesis": "ko", "kind": "contrast", "dv": "value", "prior_set": s, "band": "40 Hz ITC",
            "spatial": "all", "estimate": est, "hdi_lo": est - 0.6, "hdi_hi": est + 0.6, "p_direction": 0.999,
            "p_rope": 0.001, "decision": "effect", "status": "ok",
            "gate_passed": (gate == "passed") if s == "W" else True})
    table = tmp / "toy_hypotheses_bayes.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    job = {"work_dir": str(work), "dv": "value", "spatial_col": None, "informative": {"x_40HzITC_KO": [-0.5, 0.5]},
           "provenance": {"study": {"path": "study.yaml", "sha256": "f" * 64},
                          "informative": {"path": "pri.yaml", "sha256": "e" * 64}}}
    (work / "job_value.json").write_text(json.dumps(job))
    sets = {s: {"gate": {"gate_file": str(gdir / ("gate.json" if s == "W" else "gate_I.json")), "gate_id": "abc",
                         "found_id": "abc", "passed": True},
                "convergence": {"max_rhat_estimands": 1.002, "max_rhat_scale": 1.003, "min_ess_bulk": 3000,
                                "n_divergent": 0}} for s in ("W", "I")}
    man = {"value": {"engine_version": "2026.11.09.1", "coding": "group-levels-2026-10-19",
                     "job_file": str(work / "job_value.json"), "settings": {"rope_sd": 0.1, "chains": 4,
                                                                            "iter": 2000, "warmup": 1000, "seed": 1},
                     "input": {"n_rows_used": 55, "n_subjects": 55, "groups": {"WT": 10, "KO": 45},
                               "bands": ["40 Hz ITC"]}, "sets": sets}}
    (tmp / "toy_hypotheses_bayes_manifest.json").write_text(json.dumps(man))
    rule = lambda ok: {"pass": ok}                                     # noqa: E731
    for s, fname in (("W", "gate.json"), ("I", "gate_I.json")):
        ok = not (s == "W" and gate == "failed")
        (gdir / fname).write_text(json.dumps({"passed": ok, "rules": {
            "prior_pred": {**rule(True), "min_pctile": 0.4, "max_pctile": 0.6},
            "sbc": {**rule(ok), "min_p_roi_average": 0.3 if ok else 0.001, "min_coverage_roi_average": 0.94},
            "null": {**rule(True), "max_claim_effect": 0.03, "max_claim_dir": 0.06},
            "recovery": {**rule(True), "coverage_roi_average": 0.9}}}))
        sfx = "" if s == "W" else "_I"
        pd.DataFrame({"hypothesis": ["ko"], "band": ["40HzITC"], "spatial": ["all"], "claim_effect": [0.03],
                      "claim_dir": [0.06], "claim_bf3": [0.02]}).to_csv(gdir / f"null_summary{sfx}.csv", index=False)
        pd.DataFrame({"stat": ["sd"], "band": ["40HzITC"], "pctile": [0.5]}).to_csv(gdir / f"prior_pred{sfx}.csv",
                                                                                   index=False)
        pd.DataFrame({"set": [s], "band": ["40 Hz ITC"], "group": ["KO"], "stat": ["sd"], "observed": [1.0],
                      "rep_q05": [0.8], "rep_median": [1.0], "rep_q95": [1.2], "pctile": [0.99]}).to_csv(
            work / f"ppc_value_{s}.csv", index=False)
    pd.DataFrame({"band": ["40HzITC"], "center": [0.2], "scale": [0.06]}).to_csv(work / "scale_value.csv",
                                                                                 index=False)
    return table


def test_report_reads_a_gated_run(tmp_path):
    md = build_report(_run(tmp_path))
    assert "`model.group_levels_noroi` v1 (no ROI)" in md and "x_40HzITC_KO N(-0.5, 0.5)" in md
    assert "-1.20 [-1.80, -0.60] (-0.072)" in md                          # SD, then the outcome's own unit
    assert "| effect | 0.030 |" in md                                     # the band's false-effect rate beside it
    assert "- **W:** passed" in md and "- **I:** passed" in md
    assert "40 Hz ITC KO sd 0.990" in md                                  # posterior predictive outside 0.05-0.95
    assert "R-hat max 1.002" in md


def test_report_names_the_uncentred_model_and_lists_one_outcome_whole(tmp_path):
    table = _run(tmp_path)
    jf = tmp_path / "bayes" / "job_value.json"
    job = json.loads(jf.read_text()); job["center"] = False; jf.write_text(json.dumps(job))
    mf = tmp_path / "toy_hypotheses_bayes_manifest.json"
    man = json.loads(mf.read_text()); man["value"]["input"]["bands"] = "40 Hz ITC"; mf.write_text(json.dumps(man))
    md = build_report(table)
    assert "`model.group_levels_uncentred` v1 (no ROI): uncentred" in md
    assert "outcomes (bands): 40 Hz ITC\n" in md


def test_report_marks_a_failed_gate(tmp_path):
    md = build_report(_run(tmp_path, gate="failed"))
    assert "effect (gate failed)" in md and "**failed** (sbc)" in md
