"""The results specification (results_spec.py): analysis.json beside a module's tables.

The tables here are synthetic but have exactly the columns R's hypothesis layer
writes (write_module_hypotheses), with R's ``NA`` for missing values. The
conformance tests need the optional ``results`` extra (neurobox-dev) and skip
without it; the rest check that a missing extra, or a table the writer cannot
describe, leaves the run and its tables alone.

Run: uv run --no-sync pytest tests/test_results_spec.py -q
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from source_analytics import results_spec as rs
from source_analytics.analyses.roi_aperiodic_analysis import ROIAperiodicAnalysis
from source_analytics.core import ANALYSIS_REGISTRY

HEADER = ["hypothesis", "kind", "role", "band", "spatial", "estimate", "SE", "df", "df_num", "stat", "stat_type", "p_value", "estimate_lcl", "estimate_ucl", "effect_size", "effect_size_type", "effect_size_lcl", "effect_size_ucl", "n", "n_a", "n_b", "mde", "mde_percent", "mde_power", "mde_alpha", "mde_note", "label", "test", "q_value", "fdr_family", "significant", "dv", "group_a", "group_b", "margin_used", "equivalent"]


def _family(hyp, dv, n, method="BH"):
    return f"scope=band method={method} key=all|{hyp}|{dv} members=roi[{n}] hash=ab12cd34"


def _rows(spatial=("NA",), method="BH"):
    """Omnibus, contrast and equivalence rows for both parameters, R-style."""
    out = []
    n = 0 if spatial == ("NA",) else len(spatial)
    for dv in ("exponent", "offset"):
        out.append({"hypothesis": "group_omnibus", "kind": "omnibus", "role": "phenotype", "band": "NA",
                        "spatial": "NA", "estimate": "NA", "SE": "NA", "df": "17", "df_num": "1", "stat": "6.1",
                        "stat_type": "F", "p_value": "0.02", "estimate_lcl": "NA", "estimate_ucl": "NA",
                        "effect_size": "0.21", "effect_size_type": "omega2_partial", "effect_size_lcl": "0.01",
                        "effect_size_ucl": "0.48", "n": "20", "n_a": "NA", "n_b": "NA", "mde": "NA",
                        "mde_percent": "NA", "mde_power": "0.8", "mde_alpha": "0.05",
                        "mde_note": "not computed: omnibus F test", "label": "Omnibus", "test": "difference",
                        "q_value": "0.02", "fdr_family": _family("group_omnibus", dv, 0, method),
                        "significant": "TRUE", "dv": dv, "group_a": "NA", "group_b": "NA", "margin_used": "NA",
                        "equivalent": "NA"})
        for sp in spatial:
            common = {"role": "phenotype", "band": "NA", "spatial": sp, "estimate": "0.12", "SE": "0.05", "df": "18",
                          "df_num": "1", "stat_type": "t", "p_value": "0.03", "estimate_lcl": "0.02",
                          "estimate_ucl": "0.22", "effect_size": "0.9", "effect_size_type": "hedges_g",
                          "effect_size_lcl": "0.15", "effect_size_ucl": "1.65", "n": "20", "n_a": "10", "n_b": "10",
                          "mde": "0.15", "mde_percent": "12.5", "mde_power": "0.8", "mde_alpha": "0.05",
                          "mde_note": "", "dv": dv, "group_a": "KO", "group_b": "WT"}
            out.append(dict(common, hypothesis="disease_effect", kind="contrast", stat="2.4",
                            label="Disease effect", test="difference", q_value="0.04",
                            fdr_family=_family("disease_effect", dv, n, method), significant="TRUE",
                            margin_used="NA", equivalent="NA"))
            out.append(dict(common, hypothesis="normalization", kind="equivalence", stat="2.4",
                            stat_type="tost", label="Normalization", test="equivalence",
                            q_value="0.06", fdr_family=_family("normalization", dv, n),
                            significant="FALSE", margin_used="0.1", equivalent="FALSE"))
    return out


def _write_table(path: Path, rows, extra_cols=()):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[*HEADER, *extra_cols], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def module(tmp_path):
    tbl = tmp_path / "results" / "tables" / "rest_roi" / "roi_aperiodic"
    tbl.mkdir(parents=True)
    _write_table(tbl / "roi_aperiodic_global_hypotheses.csv", _rows())
    _write_table(tbl / "roi_aperiodic_hypotheses.csv", _rows(spatial=("V1", "M1")))
    config = SimpleNamespace(paradigm_name="rest_roi", profile_name=None,
                             raw={"pipeline": {"atlas": "allen26"}})
    return SimpleNamespace(name="roi_aperiodic", tbl_dir=tbl, output_dir=tmp_path / "analytics",
                           config=config, RESULTS_SPEC=ROIAperiodicAnalysis.RESULTS_SPEC)


RECORD = {"schema": 1, "analysis": "roi_aperiodic",
          "subjects": {"n": 20, "groups": {"KO": 10, "WT": 10}, "ids": ["s01", "s02"]},
          "source_analytics": {"version": "0.8.2", "git_describe": "v0.8.2-3-gabc"},
          "localization": {"atlas": "allen26"}}


def test_a_module_declares_canonical_measures_and_a_known_domain():
    neurobox = pytest.importorskip("neurobox")
    from neurobox.spec import canonical_measure, domain_vocabulary

    declared = {n: c.RESULTS_SPEC for n, c in ANALYSIS_REGISTRY.items() if c.RESULTS_SPEC is not None}
    assert "roi_aperiodic" in declared
    for name, spec in declared.items():
        assert spec.domain in domain_vocabulary(), name
        for measure in spec.measures.values():
            assert canonical_measure(measure) == (measure, True), (name, measure)
    assert neurobox.SPEC_VERSION >= "0.4"


def test_the_tables_are_described_and_conform(module):
    pytest.importorskip("neurobox")
    from neurobox import check

    rep = rs.write(module, dict(RECORD), "2026-11-03T12:00:00+00:00")
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)
    (again,) = check(module.tbl_dir)
    assert again.ok

    a = json.loads((module.tbl_dir / "analysis.json").read_text())
    assert a["id"] == "eeg/roi/aperiodic/rest_roi"
    assert (a["modality"], a["domain"], a["task"], a["atlas"]) == ("eeg", "aperiodic", "rest", {"name": "allen26"})
    assert a["measures"] == ["aperiodic_exponent", "aperiodic_offset"]
    assert a["design"]["n"] == 20 and a["design"]["test_kinds"] == ["two_group", "equivalence", "custom"]
    assert [t["role"] for t in a["tables"]] == ["tests", "elements"] and a["tables"][0]["headline"]
    assert {f["method"] for f in a["inference"]["families"]} == {"BH"}

    # source-analytics' own provenance fields stay, beside the specification's
    p = json.loads((module.tbl_dir / "provenance.json").read_text())
    assert p["schema"] == 1 and p["subjects"]["ids"] == ["s01", "s02"]
    assert p["spec"] == "neurofaune.results" and p["run"]["status"] == "completed"
    assert p["generated_by"][0] == {"Name": "source-analytics", "Version": "0.8.2",
                                    "Description": "git v0.8.2-3-gabc"}

    # R's NA is the specification's empty cell; measures get their vocabulary names
    with open(module.tbl_dir / "roi_aperiodic_global_hypotheses.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["n_a"] == "" and "NA" not in rows[0].values()
    assert [r["measure"] for r in rows] == ["aperiodic_exponent"] * 3 + ["aperiodic_offset"] * 3

    # writing again (a re-run) changes nothing but the run
    assert rs.write(module, dict(RECORD), "2026-11-03T13:00:00+00:00").ok
    with open(module.tbl_dir / "roi_aperiodic_global_hypotheses.csv") as fh:
        assert list(csv.DictReader(fh)) == rows


def test_families_of_different_kinds_say_per_family(module):
    pytest.importorskip("neurobox")
    _write_table(module.tbl_dir / "roi_aperiodic_hypotheses.csv",
                 _rows(spatial=("V1", "M1"), method="holm"))
    rep = rs.write(module, dict(RECORD), "2026-11-03T12:00:00+00:00")
    assert rep.ok, rep.errors
    d = json.loads((module.tbl_dir / "roi_aperiodic_hypotheses.json").read_text())
    assert d["q_value"]["PKind"] == "per_family"
    methods = {f["method"]: f["p_kind"] for f in
               json.loads((module.tbl_dir / "analysis.json").read_text())["inference"]["families"]}
    assert methods == {"BH": "fdr", "holm": "fwe"}


def test_family_labels():
    fams = rs.families([{"fdr_family": _family("h", "exponent", 26)}] * 3 +
                       [{"fdr_family": _family("h", "offset", 26, "holm")}])
    f = fams[_family("h", "exponent", 26)]
    assert (f["p_kind"], f["method"], f["members"], f["members_hash"]) == ("fdr", "BH", "roi[26]", "ab12cd34")
    assert f["_rows"] == 3 and "3 test(s)" in f["statement"]
    assert fams[_family("h", "offset", 26, "holm")]["p_kind"] == "fwe"
    with pytest.raises(ValueError):
        rs.families([{"fdr_family": "BH over everything"}])


def test_without_the_extra_nothing_is_written(module, monkeypatch):
    monkeypatch.setattr(rs, "neurobox", None)
    before = (module.tbl_dir / "roi_aperiodic_global_hypotheses.csv").read_text()
    assert rs.write(module, dict(RECORD), "2026-11-03T12:00:00+00:00") is None
    assert not (module.tbl_dir / "analysis.json").exists()
    assert (module.tbl_dir / "roi_aperiodic_global_hypotheses.csv").read_text() == before


def test_a_column_nobody_described_writes_no_analysis_and_does_not_raise(module):
    pytest.importorskip("neurobox")
    rows = [dict(r, mystery="1") for r in _rows()]
    _write_table(module.tbl_dir / "roi_aperiodic_global_hypotheses.csv", rows, extra_cols=("mystery",))
    assert rs.write(module, dict(RECORD), "2026-11-03T12:00:00+00:00") is None
    assert not (module.tbl_dir / "analysis.json").exists()


def test_modules_without_a_spec_are_left_alone(module):
    module.RESULTS_SPEC = None
    assert rs.write(module, dict(RECORD), "2026-11-03T12:00:00+00:00") is None
    assert not (module.tbl_dir / "analysis.json").exists()
