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
from source_analytics.analyses.electrode_analysis import ElectrodeAnalysis
from source_analytics.analyses.roi_aperiodic_analysis import ROIAperiodicAnalysis
from source_analytics.analyses.roi_cross_freq_analysis import ROICrossFreqAnalysis
from source_analytics.analyses.roi_psd_analysis import ROIPsdAnalysis
from source_analytics.core import ANALYSIS_REGISTRY

HEADER = ["hypothesis", "kind", "role", "band", "spatial", "estimate", "SE", "df", "df_num", "stat", "stat_type", "p_value", "estimate_lcl", "estimate_ucl", "effect_size", "effect_size_type", "effect_size_lcl", "effect_size_ucl", "n", "n_a", "n_b", "mde", "mde_percent", "mde_power", "mde_alpha", "mde_note", "label", "test", "q_value", "fdr_family", "significant", "dv", "group_a", "group_b", "margin_used", "equivalent"]


def _family(hyp, dv, n, method="BH"):
    return f"scope=band method={method} key=all|{hyp}|{dv} members=roi[{n}] hash=ab12cd34"


def _rows(spatial=("NA",), method="BH", dvs=("exponent", "offset"), band="NA"):
    """Omnibus, contrast and equivalence rows for each measure, R-style."""
    out = []
    n = 0 if spatial == ("NA",) else len(spatial)
    for dv in dvs:
        out.append({"hypothesis": "group_omnibus", "kind": "omnibus", "role": "phenotype", "band": band,
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
            common = {"role": "phenotype", "band": band, "spatial": sp, "estimate": "0.12", "SE": "0.05", "df": "18",
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
    assert {"roi_aperiodic", "roi_psd", "electrode_psd", "roi_cross_freq"} <= set(declared)
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


# ------------------------------------------------- the other declared modules ---
#: module -> (class, its dvs, a band value, the id it gets, its facet name)
MODULES = {
    "roi_psd": (ROIPsdAnalysis, ("absolute", "relative", "delta_ref"), "Alpha", "eeg/roi/psd/rest_roi", "band"),
    "electrode_psd": (ElectrodeAnalysis, ("absolute", "relative"), "Alpha", "eeg/sensor/psd/rest_roi", "band"),
    "roi_cross_freq": (ROICrossFreqAnalysis, ("z_score",), "Theta-Low Gamma", "eeg/roi/cross_freq/rest_roi",
                       "freq_pair"),
}
#: dvs of the AAC / PPC tables (roi_cross_freq_edges_analysis.R: the metric and its _z)
EDGE_DVS = {"aac": ("aac",), "ppc": ("ppc", "ppc_z")}


def _declared_module(tmp_path, name):
    cls, dvs, band, _, _ = MODULES[name]
    tbl = tmp_path / "results" / "tables" / "rest_roi" / name
    tbl.mkdir(parents=True)
    for t in cls.RESULTS_SPEC.tables:
        metric = next((m for m in EDGE_DVS if f"_{m}_" in t.suffix), None)
        table_dvs = EDGE_DVS[metric] if metric else dvs
        extra = ()
        if "directed_edges" in t.suffix:
            spatial, extra = ("V1 -> M1", "M1 -> V1"), ("source", "target")
        elif metric and not t.element:
            spatial = ("(global) -> (global)",)
        else:
            spatial = ("V1", "M1") if t.element else ("NA",)
        rows = _rows(spatial=spatial, dvs=table_dvs, band=band)
        for r in rows:
            if extra:
                r["source"], r["target"] = (r["spatial"].split(" -> ") if r["spatial"] != "NA" else ("NA", "NA"))
        _write_table(tbl / t.filename(name), rows, extra_cols=extra)
    config = SimpleNamespace(paradigm_name="rest_roi", profile_name=None, raw={"pipeline": {"atlas": "allen26"}})
    return SimpleNamespace(name=name, tbl_dir=tbl, output_dir=tmp_path / "analytics", config=config,
                           RESULTS_SPEC=cls.RESULTS_SPEC)


@pytest.mark.parametrize("name", sorted(MODULES))
def test_each_declared_module_conforms(tmp_path, name):
    pytest.importorskip("neurobox")
    module = _declared_module(tmp_path, name)
    rep = rs.write(module, dict(RECORD), "2026-11-04T12:00:00+00:00")
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)

    _, dvs, _, expected_id, facet = MODULES[name]
    spec = module.RESULTS_SPEC
    a = json.loads((module.tbl_dir / "analysis.json").read_text())
    assert a["id"] == expected_id and a["domain"] == spec.domain
    assert [t["path"] for t in a["tables"]] == [t.filename(name) for t in spec.tables]
    assert sum(bool(t.get("headline")) for t in a["tables"]) == 1
    written = set(dvs) | ({dv for d in EDGE_DVS.values() for dv in d} if name == "roi_cross_freq" else set())
    assert set(a["measures"]) == {spec.measures[d] for d in written}
    assert ("atlas" in a) == (name != "electrode_psd")      # an electrode is not a parcel
    first = json.loads((module.tbl_dir / spec.tables[0].filename(name).replace(".csv", ".json")).read_text())
    assert first["band"]["FacetName"] == facet


def test_a_table_named_apart_from_its_module_is_found(tmp_path):
    t = ROICrossFreqAnalysis.RESULTS_SPEC.tables[0]
    assert t.filename("roi_cross_freq") == "roi_pac_global_hypotheses.csv"
    assert ROIPsdAnalysis.RESULTS_SPEC.tables[0].filename("roi_psd") == "roi_psd_global_hypotheses.csv"


# ------------------------------------------- contrasts and variants (spec 0.4.1) ---
def _study_config(paradigm="cartesian_mc", group="Resting", siblings=("cartesian_mc", "shell_mc"), profile=None):
    from source_analytics.config import DesignSpec

    hyps = [{"name": "group_omnibus", "kind": "omnibus", "label": "Omnibus", "role": "phenotype"},
            {"name": "disease_effect", "kind": "contrast", "label": "Disease effect", "role": "phenotype",
             "weights": {"KO": 1, "WT": -1}},
            {"name": "normalization", "kind": "equivalence", "label": "Normalization", "role": "normalization",
             "weights": {"KO": 1, "WT": -1}, "margin": {"mode": "sd", "value": 0.5}},
            {"name": "never_run", "kind": "contrast", "role": "exploratory", "weights": {"KO": 1, "WT": -1}}]
    raw = {"pipeline": {"atlas": "allen26"}, "display": {"group": group, "label": "MC volumetric (cartesian)"},
           "paradigms": {s: {"display": {"group": group}} for s in siblings}}
    return SimpleNamespace(paradigm_name=paradigm, profile_name=profile, raw=raw,
                           design_spec=DesignSpec.from_dict({"hypotheses": hyps}))


def test_declared_contrasts_and_the_variant_are_written(module):
    pytest.importorskip("neurobox")
    module.config = _study_config()
    rep = rs.write(module, dict(RECORD), "2026-11-04T12:00:00+00:00")
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)
    a = json.loads((module.tbl_dir / "analysis.json").read_text())
    assert [c["id"] for c in a["contrasts"]] == ["group_omnibus", "disease_effect", "normalization"]
    assert a["contrasts"][1] == {"id": "disease_effect", "kind": "contrast", "label": "Disease effect",
                                 "group_a": "KO", "group_b": "WT", "tier": "phenotype"}
    assert "group_a" not in a["contrasts"][0] and a["contrasts"][2]["tier"] == "normalization"
    assert a["variant"] == {"of": "eeg/roi/aperiodic", "axis": "reconstruction", "value": "cartesian_mc",
                            "label": "MC volumetric (cartesian)"}


@pytest.mark.parametrize("over", [{"siblings": ("cartesian_mc",)}, {"group": None}, {"profile": "strict"}])
def test_a_paradigm_is_a_variant_only_beside_others_of_its_group(over):
    config = _study_config(**over)
    aid = "eeg/roi/aperiodic/cartesian_mc" + ("/strict" if over.get("profile") else "")
    assert rs.variant(config, aid, "cartesian_mc", config.profile_name) is None
