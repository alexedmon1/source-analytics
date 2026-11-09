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
from source_analytics.analyses.electrode_aperiodic_analysis import ElectrodeAperiodicAnalysis
from source_analytics.analyses.roi_aperiodic_analysis import ROIAperiodicAnalysis
from source_analytics.analyses.roi_cross_freq_analysis import ROICrossFreqAnalysis
from source_analytics.analyses.roi_directed_analysis import ROIDirectedAnalysis
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
    "electrode_aperiodic": (ElectrodeAperiodicAnalysis, ("exponent", "offset"), "NA", "eeg/sensor/aperiodic/rest_roi",
                            None),
    "roi_cross_freq": (ROICrossFreqAnalysis, ("z_score",), "Theta-Low Gamma", "eeg/roi/cross_freq/rest_roi",
                       "freq_pair"),
    "roi_directed": (ROIDirectedAnalysis, ("te", "net_te", "dtf"), "Alpha", "eeg/roi/directed/rest_roi", "band"),
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
    assert ("atlas" in a) == (not name.startswith("electrode_"))      # an electrode is not a parcel
    first = json.loads((module.tbl_dir / spec.tables[0].filename(name).replace(".csv", ".json")).read_text())
    assert first["band"].get("FacetName") == facet           # None: broadband, no band facet
    # spec 0.4.2: every element column says what one element is
    names = set()
    for t in spec.tables:
        cols = json.loads((module.tbl_dir / t.filename(name).replace(".csv", ".json")).read_text())
        if t.element:
            names.add(cols["spatial"]["ElementName"])
        else:
            assert "ElementName" not in cols["spatial"]
    assert names == {"electrode_psd": {"electrode"}, "electrode_aperiodic": {"electrode"}, "roi_aperiodic": {"region", "ROI"},
                     "roi_psd": {"region", "ROI"},
                     "roi_cross_freq": {"region", "region pair", "ROI pair"},
                     "roi_directed": {"region pair", "ROI pair"}}[name]


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


# ---------------------------------------------- roi_graph (Python hypothesis layer) ---
def test_roi_graph_conforms_from_the_python_layer(tmp_path):
    """roi_graph's table comes from the Python tabular adapter: the graph metric is the measure
    (``measure_col``) and the connectivity measure a second facet (``extra_facets``)."""
    pytest.importorskip("neurobox")
    import numpy as np
    import pandas as pd

    from source_analytics.analyses.roi_network_analysis import ROIGraphAnalysis
    from source_analytics.config import DesignSpec, Hypothesis
    from source_analytics.hypothesis.tabular import write_module_hypotheses_tabular

    rng = np.random.default_rng(3)
    groups = ["WT", "KO"]
    rows = [{"subject": f"{g}{i}", "group": g, "band": "Alpha", "conn_metric": cm, "graph_metric": gm,
             "roi": roi, "value": 2.0 + (0.8 if (g == "KO" and roi == "V1") else 0.0) + rng.normal(0, 0.4)}
            for g in groups for i in range(10) for cm in ("coherence", "pli")
            for gm in ("degree", "clustering") for roi in ("V1", "M1")]
    hyps = [Hypothesis(name="group_omnibus", kind="omnibus", role="phenotype"),
            Hypothesis(name="disease_effect", kind="contrast", role="phenotype", label="Disease effect",
                       weights={"KO": 1.0, "WT": -1.0})]
    cfg = SimpleNamespace(design_spec=DesignSpec(factor="group", reference="WT", levels=groups, fdr={},
                                                 hypotheses=hyps),
                          paradigm_name="rest_roi", profile_name=None, raw={"pipeline": {"atlas": "allen26"}})
    tbl = tmp_path / "results" / "tables" / "rest_roi" / "roi_graph"
    tbl.mkdir(parents=True)
    write_module_hypotheses_tabular(pd.DataFrame(rows), cfg, tbl, prefix="roi_graph", value_col="value",
                                    spatial_col="roi", facet_cols=("conn_metric", "graph_metric"),
                                    band_col="band")
    module = SimpleNamespace(name="roi_graph", tbl_dir=tbl, output_dir=tmp_path / "analytics", config=cfg,
                             RESULTS_SPEC=ROIGraphAnalysis.RESULTS_SPEC)
    rep = rs.write(module, dict(RECORD), "2026-11-09T12:00:00+00:00")
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)
    a = json.loads((tbl / "analysis.json").read_text())
    assert a["id"] == "eeg/roi/graph/rest_roi" and set(a["measures"]) == {"degree", "clustering"}
    cols = json.loads((tbl / "roi_graph_hypotheses.json").read_text())
    assert cols["conn_metric"]["FacetName"] == "connectivity" and cols["band"]["FacetName"] == "band"
    assert cols["graph_metric"].get("Standard") is None and cols["measure"]["Standard"] == "measure"
    assert cols["spatial"]["ElementName"] == "ROI"
    with open(tbl / "roi_graph_hypotheses.csv") as fh:
        assert {r["measure"] for r in csv.DictReader(fh)} == {"degree", "clustering"}


def test_roi_graph_statistics_read_the_saved_nodal_metrics(tmp_path):
    """--steps statistics computes no graphs, so the hypotheses come from data/roi_graph_nodal_metrics.csv."""
    import pandas as pd

    from source_analytics.analyses.roi_network_analysis import ROIGraphAnalysis

    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame([{"subject": "WT_s1", "group": "WT", "band": b, "metric": m, "roi": "V1",
                   "degree": 3, "clustering": 0.5, "betweenness": 0.1}
                  for b in ("Alpha", "Beta") for m in ("coherence", "pli")]).to_csv(
        data / "roi_graph_nodal_metrics.csv", index=False)
    module = ROIGraphAnalysis.__new__(ROIGraphAnalysis)
    module.output_dir = tmp_path
    module._selected_bands = lambda: ["Alpha"]
    module._connectivity_metrics = ["pli"]
    rows = module._saved_nodal_rows("group")
    assert len(rows) == 3                                      # one ROI x Alpha x pli x three graph metrics
    assert {r["graph_metric"]: r["value"] for r in rows} == {"degree": 3.0, "clustering": 0.5, "betweenness": 0.1}
    assert rows[0]["group"] == "WT" and rows[0]["conn_metric"] == "pli" and rows[0]["subject"] == "WT_s1"


def test_a_statistics_only_roi_graph_run_writes_its_hypotheses(tmp_path):
    import numpy as np
    import pandas as pd

    from source_analytics.analyses.roi_network_analysis import ROIGraphAnalysis
    from source_analytics.config import DesignSpec, Hypothesis

    rng = np.random.default_rng(5)
    (tmp_path / "data").mkdir()
    pd.DataFrame([{"subject": f"{g}_s{i}", "group": g, "band": "Alpha", "metric": "pli", "roi": roi,
                   "degree": int(rng.integers(1, 6)), "clustering": rng.random(), "betweenness": rng.random()}
                  for g in ("WT", "KO") for i in range(8) for roi in ("V1", "M1")]).to_csv(
        tmp_path / "data" / "roi_graph_nodal_metrics.csv", index=False)
    module = ROIGraphAnalysis.__new__(ROIGraphAnalysis)
    module.output_dir = tmp_path
    module._subject_data, module._subject_groups, module._selection = {}, {}, {}
    module._selected_bands = lambda: ["Alpha"]
    module._connectivity_metrics = ["pli"]
    module._pairwise_contrasts = list
    module.config = SimpleNamespace(paradigm_name="rest_roi", results_dir=tmp_path / "results", design_spec=DesignSpec(
        factor="group", reference="WT", levels=["WT", "KO"], fdr={},
        hypotheses=[Hypothesis(name="disease_effect", kind="contrast", role="phenotype",
                               weights={"KO": 1.0, "WT": -1.0})]))
    module.tbl_dir.mkdir(parents=True)
    module._graph_statistics()
    out = pd.read_csv(module.tbl_dir / "roi_graph_hypotheses.csv")
    assert set(out["graph_metric"]) == {"degree", "clustering", "betweenness"} and set(out["spatial"]) == {"V1", "M1"}
    assert not (module.tbl_dir / "roi_graph_stats.csv").exists()     # the legacy table needs the graphs


# ------------------------------------------ permutation modules: NBS and electrode clusters ---
def _perm_config(groups=("WT", "KO")):
    from source_analytics.config import DesignSpec, Hypothesis

    hyps = [Hypothesis(name="group_omnibus", kind="omnibus", role="phenotype"),
            Hypothesis(name="disease_effect", kind="contrast", role="phenotype", label="Disease effect",
                       weights={"KO": 1.0, "WT": -1.0}),
            Hypothesis(name="normalization", kind="equivalence", role="normalization",
                       weights={"KO": 1.0, "WT": -1.0}, margin={"mode": "sd", "value": 0.5})]
    return SimpleNamespace(design_spec=DesignSpec(factor="group", reference="WT", levels=list(groups), fdr={},
                                                  hypotheses=hyps),
                           paradigm_name="rest_roi", profile_name=None, raw={"pipeline": {"atlas": "allen26"}})


def _write_and_check(tmp_path, name, cls, cfg):
    module = SimpleNamespace(name=name, tbl_dir=tmp_path, output_dir=tmp_path.parent / "analytics", config=cfg,
                             RESULTS_SPEC=cls.RESULTS_SPEC)
    rep = rs.write(module, dict(RECORD), "2026-11-09T12:00:00+00:00")
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)
    return json.loads((tmp_path / "analysis.json").read_text())


def test_roi_nbs_conforms_from_the_edge_layer(tmp_path):
    pytest.importorskip("neurobox")
    import numpy as np

    from source_analytics.analyses.roi_network_analysis import ROINBSAnalysis
    from source_analytics.hypothesis import write_module_hypotheses_edge

    rng = np.random.default_rng(1)
    n, groups = 6, {f"{g}{i}": g for g in ("WT", "KO") for i in range(8)}
    cells = {}
    for band, dv in (("Alpha", "coherence"), ("Alpha", "imag_coherence")):
        mats = {}
        for uid, g in groups.items():
            m = rng.normal(0.3, 0.05, (n, n))
            if g == "KO":
                m[:3, :3] += 0.2                                # a planted sub-network
            m = (m + m.T) / 2
            np.fill_diagonal(m, 0)
            mats[uid] = m
        cells[(band, dv)] = mats
    tbl = tmp_path / "results" / "tables" / "rest_roi" / "roi_nbs"
    tbl.mkdir(parents=True)
    cfg = _perm_config()
    write_module_hypotheses_edge(cells, groups, cfg, tbl, prefix="roi_nbs", nbs_threshold=2.5, n_perms=50,
                                 node_labels=[f"R{i}" for i in range(n)])
    a = _write_and_check(tbl, "roi_nbs", ROINBSAnalysis, cfg)
    assert a["id"] == "eeg/roi/nbs/rest_roi" and set(a["measures"]) == {"coherence", "imcoh"}
    assert a["inference"]["correction"]["method"] == "nbs" and "families" not in a["inference"]
    assert a["effect"]["measure"] is None and a["effect"]["reason"]
    assert [t["role"] for t in a["tables"]] == ["elements", "other"]
    cols = json.loads((tbl / "roi_nbs_hypotheses.json").read_text())
    assert cols["component_id"]["ElementName"] == "component" and cols["component_p"]["PKind"] == "fwe"
    assert cols["peak_stat"]["Standard"] == "stat"


def test_electrode_connectivity_conforms_from_the_cluster_layer(tmp_path):
    pytest.importorskip("neurobox")
    import numpy as np

    from source_analytics.analyses.electrode_connectivity_analysis import (
        ElectrodeConnectivityAnalysis,
    )
    from source_analytics.hypothesis import write_module_hypotheses_perm

    rng = np.random.default_rng(2)
    n_ch, groups = 8, {f"{g}{i}": g for g in ("WT", "KO") for i in range(8)}
    coords = np.array([[i * 10.0, 0.0, 0.0] for i in range(n_ch)])     # a line of electrodes, 10 mm apart
    cells = {}
    for band, dv in (("Alpha", "coherence"), ("Alpha", "aec")):
        cells[(band, dv)] = {uid: rng.normal(0.3, 0.05, n_ch) + (0.15 * (np.arange(n_ch) < 3) if g == "KO" else 0)
                             for uid, g in groups.items()}
    tbl = tmp_path / "results" / "tables" / "rest_roi" / "electrode_connectivity"
    tbl.mkdir(parents=True)
    cfg = _perm_config()
    write_module_hypotheses_perm(cells, groups, coords, cfg, tbl, prefix="electrode_connectivity", n_perms=50,
                                 threshold=2.0, distance_mm=12.0,
                                 node_labels=["Frontal"] * 4 + ["Occipital"] * 4)
    a = _write_and_check(tbl, "electrode_connectivity", ElectrodeConnectivityAnalysis, cfg)
    assert a["id"] == "eeg/sensor/connectivity/rest_roi" and a["measures"] == ["fcd"]
    assert a["inference"]["correction"]["method"] == "cluster" and "atlas" not in a
    cols = json.loads((tbl / "electrode_connectivity_hypotheses.json").read_text())
    assert cols["dv"]["FacetName"] == "connectivity" and cols["measure"]["Standard"] == "measure"
    assert cols["cluster_id"]["ElementName"] == "cluster" and cols["cluster_p"]["Standard"] == "p_value"
    with open(tbl / "electrode_connectivity_hypotheses.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["measure"] for r in rows} == {"fcd"} and {r["dv"] for r in rows} == {"coherence", "aec"}


# ------------------------------------------------------- signature (decoding) modules ---
def test_the_balanced_accuracy_interval_and_bh():
    from source_analytics.stats.signature import (
        balanced_accuracy_interval,
        bh_qvalues,
        wilson_interval,
    )

    lo, hi = wilson_interval(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3          # the textbook Wilson interval
    blo, bhi = balanced_accuracy_interval(0.8, 0.8, 10, 10)
    assert (round(blo, 4), round(bhi, 4)) == (round(lo, 4), round(hi, 4))
    assert balanced_accuracy_interval(1.0, 1.0, 6, 6)[1] == 1.0          # never above 1
    q = bh_qvalues([0.01, 0.04, 0.03, 0.5, float("nan")])
    assert list(q[:4].round(4)) == [0.04, 0.0533, 0.0533, 0.5] and q[4] != q[4]


@pytest.mark.parametrize("name", ["electrode_signature", "roi_signature"])
def test_signature_modules_conform(tmp_path, name):
    pytest.importorskip("neurobox")
    pytest.importorskip("sklearn")
    import numpy as np

    from source_analytics.analyses.electrode_signature_analysis import ElectrodeSignatureAnalysis
    from source_analytics.analyses.roi_signature_analysis import ROISignatureAnalysis

    cls = {"electrode_signature": ElectrodeSignatureAnalysis, "roi_signature": ROISignatureAnalysis}[name]
    rng = np.random.default_rng(4)
    groups = {f"{g}{i}": g for g in ("WT", "KO") for i in range(8)}
    chans = ["c1", "c2", "c3"]
    module = cls.__new__(cls)
    module._subject_order = list(groups)
    module._subject_groups = groups
    module._ch_names = chans
    module._subject_data = {u: {"band_power": {b: {c: rng.normal(1.0 + (0.8 if g == "KO" and c == "c1" else 0), 0.3)
                                                    for c in chans} for b in ("Alpha", "Beta")}}
                            for u, g in groups.items()}
    module._classifiers = ["svm_linear"]
    module._cv_method = "loocv"
    module.output_dir = tmp_path / "analytics" / name
    (module.output_dir / "data").mkdir(parents=True)
    module._n_permutations = 30
    module._signature_results = {}
    module._selected_bands = lambda: ["Alpha", "Beta"]
    module._pairwise_contrasts = lambda: [SimpleNamespace(name="disease_effect", group_a="KO", group_b="WT")]
    cfg = _perm_config()
    cfg.results_dir = tmp_path / "results"
    module.config = cfg
    module.tbl_dir.mkdir(parents=True)
    module.statistics()
    with open(module.tbl_dir / f"{name}_results.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 and {r["n_a"] for r in rows} == {"8"}
    for r in rows:
        assert float(r["balanced_accuracy_lcl"]) <= float(r["balanced_accuracy"]) <= float(r["balanced_accuracy_ucl"])
        assert r["fdr_family"].startswith("scope=contrast method=BH key=disease_effect members=band_x_classifier[2]")
    shim = SimpleNamespace(name=name, tbl_dir=module.tbl_dir, output_dir=tmp_path / "analytics", config=cfg,
                           RESULTS_SPEC=cls.RESULTS_SPEC)
    rep = rs.write(shim, dict(RECORD), "2026-11-09T12:00:00+00:00")
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)
    a = json.loads((module.tbl_dir / "analysis.json").read_text())
    level = "sensor" if name.startswith("electrode") else "roi"
    assert a["id"] == f"eeg/{level}/signature/rest_roi" and a["domain"] == "decoding"
    assert a["measures"] == ["decoding_accuracy"] and a["effect"]["measure"] == "balanced_accuracy"
    assert [f["method"] for f in a["inference"]["families"]] == ["BH"]


def test_signature_statistics_read_the_saved_features(tmp_path):
    """--steps statistics computes no features, so they come from data/<name>_features.csv."""
    import pandas as pd

    from source_analytics.analyses.roi_signature_analysis import ROISignatureAnalysis

    module = ROISignatureAnalysis.__new__(ROISignatureAnalysis)
    module.output_dir = tmp_path
    (tmp_path / "data").mkdir()
    pd.DataFrame([{"subject": f"{g}_s{i}", "group": g, "roi": roi, "band": band, "relative": 0.1}
                  for g in ("WT", "KO") for i in range(2) for roi in ("V1", "M1") for band in ("Alpha", "Beta")]
                 ).to_csv(tmp_path / "data" / "roi_signature_features.csv", index=False)
    pd.DataFrame({"roi": ["M1", "V1"]}).to_csv(tmp_path / "data" / "roi_order.csv", index=False)
    module._subject_data, module._subject_groups, module._subject_order, module._ch_names = {}, {}, [], None
    module._selected_bands = lambda: ["Alpha"]
    assert module._load_features_from_disk()
    assert module._subject_order == ["WT_s0", "WT_s1", "KO_s0", "KO_s1"] and module._ch_names == ["M1", "V1"]
    assert module._subject_data["KO_s1"]["band_power"] == {"Alpha": {"V1": 0.1, "M1": 0.1}}
    assert module._subject_groups["KO_s0"] == "KO"


# ---------------------------------------------------------- roi_connectivity (descriptive) ---
def test_roi_connectivity_writes_descriptive_tables_that_conform(tmp_path):
    pytest.importorskip("neurobox")
    import pandas as pd

    from source_analytics.analyses.roi_connectivity_analysis import ConnectivityAnalysis

    module = ConnectivityAnalysis.__new__(ConnectivityAnalysis)
    module.output_dir = tmp_path / "analytics" / "roi_connectivity"
    (module.output_dir / "data").mkdir(parents=True)
    module.config = SimpleNamespace(paradigm_name="rest_roi", profile_name=None, results_dir=tmp_path / "results",
                                    raw={"pipeline": {"atlas": "allen26"}}, design_spec=None)
    module._metrics = ["coherence", "imag_coherence", "pli"]
    pd.DataFrame([{"subject": f"{g}_s{i}", "group": g, "band": b, "roi1": r1, "roi2": r2,
                   "coherence": 0.5 + 0.1 * i, "imag_coherence": 0.05, "pli": 0.2}
                  for g in ("WT", "KO") for i in range(3) for b in ("Alpha",)
                  for r1, r2 in (("V1", "M1"), ("V1", "S1"), ("M1", "S1"))]).to_csv(
        module.output_dir / "data" / "roi_connectivity_edges.csv", index=False)
    module.statistics()
    summary = pd.read_csv(module.tbl_dir / "roi_connectivity_group_summary.csv")
    assert len(summary) == 2 * 3                                           # groups x measures, one band
    wt = summary[(summary.group == "WT") & (summary.dv == "coherence")].iloc[0]
    assert (wt["n"], round(wt["mean"], 4), round(wt["sd"], 4)) == (3, 0.6, 0.1)
    edges = pd.read_csv(module.tbl_dir / "roi_connectivity_group_edges.csv")
    assert len(edges) == 2 * 3 * 3 and set(edges["spatial"]) == {"V1 -- M1", "V1 -- S1", "M1 -- S1"}
    assert set(edges["measure"]) == {"coherence", "imcoh", "pli"}
    shim = SimpleNamespace(name="roi_connectivity", tbl_dir=module.tbl_dir, output_dir=module.output_dir,
                           config=module.config, RESULTS_SPEC=ConnectivityAnalysis.RESULTS_SPEC)
    rec = dict(RECORD, subjects={"n": 6, "groups": {"WT": 3, "KO": 3}})
    rep = rs.write(shim, rec, "2026-11-09T12:00:00+00:00")
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)
    a = json.loads((module.tbl_dir / "analysis.json").read_text())
    assert a["role"] == "descriptive" and a["effect"]["measure"] is None
    assert a["inference"]["correction"]["p_kind"] == "none"
    assert {t["role"] for t in a["tables"]} == {"descriptives"}


# ------------------------------------------------------- electrode_comparison (diagnostic) ---
def test_electrode_comparison_is_described_as_a_diagnostic(tmp_path):
    pytest.importorskip("neurobox")
    import pandas as pd

    from source_analytics.analyses.electrode_comparison_analysis import ElectrodeComparisonAnalysis

    tbl = tmp_path / "results" / "tables" / "cartesian_mc" / "electrode_comparison"
    tbl.mkdir(parents=True)
    g = {"electrode_hedges_g": 0.4, "electrode_ci_lo": -0.2, "electrode_ci_hi": 1.0}
    pd.DataFrame([{"band": b, "power_type": pt, "contrast": "disease_effect", "correlation_r": 0.7,
                   "correlation_p": 0.001, "n_subjects": 60, **g, "source_hedges_g": 0.6, "source_ci_lo": 0.0,
                   "source_ci_hi": 1.2} for b in ("Alpha", "Beta") for pt in ("relative", "absolute")]
                 ).to_csv(tbl / "comparison_stats.csv", index=False)
    pd.DataFrame([{"band": "Alpha", "power_type": "relative", "contrast": "disease_effect", "region": r, **g,
                   "region_hedges_g": 0.9, "region_ci_lo": 0.2, "region_ci_hi": 1.6, "exceeds_electrode": True}
                  for r in ("Frontal", "Visual")]).to_csv(tbl / "regional_effect_sizes.csv", index=False)
    pd.DataFrame([{"dv": dv, "contrast": "disease_effect", "correlation_r": 0.5, "correlation_p": 0.01,
                   "n_subjects": 60, **g, "source_hedges_g": 0.3, "source_ci_lo": -0.3, "source_ci_hi": 0.9}
                  for dv in ("exponent", "offset")]).to_csv(tbl / "aperiodic_comparison_stats.csv", index=False)
    cfg = SimpleNamespace(paradigm_name="cartesian_mc", profile_name=None, raw={}, design_spec=None)
    shim = SimpleNamespace(name="electrode_comparison", tbl_dir=tbl, output_dir=tmp_path / "analytics", config=cfg,
                           RESULTS_SPEC=ElectrodeComparisonAnalysis.RESULTS_SPEC)
    rep = rs.write(shim, dict(RECORD), "2026-11-09T12:00:00+00:00")
    assert rep is not None and rep.ok and not rep.warnings, (rep and rep.errors, rep and rep.warnings)
    a = json.loads((tbl / "analysis.json").read_text())
    assert a["role"] == "diagnostic" and a["id"] == "eeg/sensor/comparison/cartesian_mc"
    assert set(a["measures"]) == {"relative_power", "band_power", "aperiodic_exponent", "aperiodic_offset"}
    assert [t["path"] for t in a["tables"]] == ["comparison_stats.csv", "regional_effect_sizes.csv",
                                               "aperiodic_comparison_stats.csv"]
    cols = json.loads((tbl / "aperiodic_comparison_stats.json").read_text())
    assert cols["measure"]["Standard"] == "measure" and "power_type" not in cols
