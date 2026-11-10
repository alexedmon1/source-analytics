"""Declared outcome tables (``bayes.tables``, field-method plan F6b): config validation, assembly, jobs.

Checks on real data (2026-11-09; bayes-sl reports/2026-11-09_field_method_library_plan.md, F6b): a97d's field job
built from config reproduces the hand-built one exactly (same 218 rows; same estimates, intervals and decisions under
both prior sets); the deep-midline parcel tables give the same data (436 rows) and estimates within 0.02 SD; FORGE's
L3 region values (csv table) give the same decisions in 20 of 20 regions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from source_analytics.bayes.config import BayesConfigError, parse_bayes_config
from source_analytics.bayes.declared import assemble


def _measures(root: Path, paradigm: str, groups: dict, seed: int = 0):
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for subj, grp in groups.items():
        for roi in ("A", "B", "C"):
            for m in ("itc", "ersp"):
                rows.append({"subject": f"{paradigm}_{subj}", "group": grp, "roi": roi, "measure_name": m,
                             "value": float(rng.normal())})
    d = root / paradigm / "roi_evoked" / "data"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "roi_evoked_measures.csv", index=False)


def _bayes(block: dict, base: Path):
    return parse_bayes_config({"tables": block, "outcomes": {k: [v["value"]] for k, v in block.items()}}, base)


GROUPS = {"s1": "G2", "s2": "G3", "s3": "G6", "s4": "G6", "s5": "G1"}
TABLE = {"field": {"source": "roi_evoked", "paradigms": ["p40", "chirp"], "value": "value", "spatial": "none",
                   "outcomes": {"40 ITC": {"paradigm": "p40", "roi": "A", "measure_name": "itc"},
                                "chirp mean ERSP": {"paradigm": "chirp", "roi": ["A", "B", "C"],
                                                    "measure_name": "ersp", "combine": "mean"}},
                   "groups": {"G2": "KO", "G3": "KO", "G6": "WT"}}}


@pytest.fixture
def tree(tmp_path):
    for p in ("p40", "chirp"):
        _measures(tmp_path, p, GROUPS, seed=len(p))
    return tmp_path


class TestConfig:
    def test_a_valid_table_parses(self, tree):
        b = _bayes(TABLE, tree)
        t = b.tables["field"]
        assert t.spatial is None and t.paradigms == ("p40", "chirp") and b.outcomes["field"] == ("value",)

    @pytest.mark.parametrize("bad,match", [
        ({"csv": "x.csv"}, "exactly one"), ({"value": None}, "value"), ({"outcomes": {}}, "outcomes"),
        ({"nope": 1}, "unknown key"), ({"paradigms": "p40"}, "paradigms"),
        ({"outcomes": {"x": {"roi": "A", "combine": "median"}}}, "combine"), ({"groups": ["G2"]}, "groups")])
    def test_bad_tables_raise(self, tree, bad, match):
        t = {**TABLE["field"], **bad}
        with pytest.raises(BayesConfigError, match=match):
            parse_bayes_config({"tables": {"field": t}}, tree)

    def test_builtin_names_and_wrong_outcome_columns_are_refused(self, tree):
        with pytest.raises(BayesConfigError, match="built-in"):
            parse_bayes_config({"tables": {"roi_psd": TABLE["field"]}}, tree)
        with pytest.raises(BayesConfigError, match="value column"):
            parse_bayes_config({"tables": TABLE, "outcomes": {"field": ["other"]}}, tree)


class TestAssemble:
    def test_selection_pooling_and_combine(self, tree):
        df, prov = assemble(_bayes(TABLE, tree).tables["field"], tree, "group")
        assert list(df.columns) == ["subject", "group", "outcome", "value"]
        assert set(df.group) == {"KO", "WT"} and "p40_s5" not in set(df.subject)       # G1 unmapped: dropped
        assert df.groupby("outcome").size().to_dict() == {"40 ITC": 4, "chirp mean ERSP": 4}
        raw = pd.read_csv(tree / "chirp" / "roi_evoked" / "data" / "roi_evoked_measures.csv")
        want = raw[(raw.subject == "chirp_s1") & (raw.measure_name == "ersp")].value.mean()
        got = df[(df.subject == "chirp_s1") & (df.outcome == "chirp mean ERSP")].value.iloc[0]
        assert got == pytest.approx(want)
        assert set(prov["inputs"]) == {"p40", "chirp"} and all(len(v["sha256"]) == 64 for v in prov["inputs"].values())

    def test_several_rows_per_subject_need_combine(self, tree):
        t = {**TABLE["field"], "outcomes": {"x": {"paradigm": "p40", "measure_name": "itc"}}}
        with pytest.raises(BayesConfigError, match="combine"):
            assemble(_bayes({"field": t}, tree).tables["field"], tree, "group")

    def test_spatial_column_is_kept_as_roi(self, tree):
        t = {**TABLE["field"], "spatial": "roi", "outcomes": {"x": {"paradigm": "p40", "measure_name": "itc"}}}
        df, _ = assemble(_bayes({"field": t}, tree).tables["field"], tree, "group")
        assert list(df.columns) == ["subject", "group", "outcome", "roi", "value"] and set(df.roi) == {"A", "B", "C"}

    def test_empty_selection_raises(self, tree):
        t = {**TABLE["field"], "outcomes": {"x": {"paradigm": "p40", "roi": "Z"}}}
        with pytest.raises(BayesConfigError, match="no rows"):
            assemble(_bayes({"field": t}, tree).tables["field"], tree, "group")

    def test_csv_table_with_subject_groups_and_other_output_dir(self, tree, tmp_path):
        vals = pd.DataFrame({"subject": ["a", "b", "c", "d"] * 2, "region": ["R1"] * 4 + ["R2"] * 4,
                             "y_db": range(8)})
        vals.to_csv(tmp_path / "values.csv", index=False)
        pd.DataFrame({"subject": list("abcd"), "group": ["KO", "KO", "WT", "WT"]}).to_csv(tmp_path / "sg.csv",
                                                                                         index=False)
        t = {"csv": "values.csv", "value": "y_db", "subject_groups": "sg.csv",
             "outcomes": {"R1": {"region": "R1"}, "R2": {"region": "R2"}}}
        df, prov = assemble(_bayes({"vals": t}, tmp_path).tables["vals"], tmp_path, "group")
        assert len(df) == 8 and set(df.group) == {"KO", "WT"} and prov["inputs"]["csv"]["path"].endswith("values.csv")
        other = {**TABLE["field"], "output_dir": str(tree)}
        df2, _ = assemble(_bayes({"field": other}, tmp_path).tables["field"], tmp_path / "elsewhere", "group")
        assert len(df2) == 8


class TestJob:
    def _study(self, tree, priors=None):
        cfg = {"name": "toy", "output_dir": str(tree), "results_dir": str(tree / "results"),
               "groups": {"G2": "", "G3": "", "G6": ""},
               "design": {"factor": "group", "reference": "WT", "levels": ["WT", "KO"]},
               "hypotheses": [{"name": "ko", "kind": "contrast", "weights": {"KO": 1.0, "WT": -1.0}}],
               "bayes": {"prior_sets": ["W", "I"] if priors else ["W"], "tables": TABLE,
                         "outcomes": {"field": ["value"]}, **({"priors": {"I": priors}} if priors else {})}}
        p = tree / "study.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return p

    def test_job_is_built_with_no_roi_and_outcomes_as_bands(self, tree):
        from source_analytics.bayes import build_job
        from source_analytics.config import StudyConfig
        p = self._study(tree)
        cfg = StudyConfig.from_yaml(p)
        job = build_job(cfg, "field", "value", parse_bayes_config(cfg.raw["bayes"], tree), out_dir=tree / "out",
                        study_path=p)
        assert job.spatial_col is None and job.band_col == "outcome" and job.dv == "value"
        assert Path(job.data_csv).exists() and job.provenance["table"]["n_subjects"] == 8
        assert job.table_out.endswith("field_hypotheses_bayes.csv")

    def test_informative_entries_are_kept_per_table(self, tree):
        from source_analytics.bayes import build_job
        from source_analytics.config import StudyConfig
        pri = {"version": 1, "entries": [
            {"dv": "value", "hypothesis": "ko", "band": "40 ITC", "prior": {"mean": -0.5, "sd": 0.5}, "source": "x"},
            {"dv": "value", "hypothesis": "ko", "band": "chirp mean ERSP", "prior": {"mean": -0.4, "sd": 0.5},
             "source": "y"}]}
        (tree / "pri.yaml").write_text(yaml.safe_dump(pri))
        p = self._study(tree, priors="pri.yaml")
        cfg = StudyConfig.from_yaml(p)
        job = build_job(cfg, "field", "value", parse_bayes_config(cfg.raw["bayes"], tree), out_dir=tree / "o2")
        assert job.informative == {"x_40ITC_KO": [-0.5, 0.5], "x_chirpmeanERSP_KO": [-0.4, 0.5]}
        assert job.prior_sets == ["W", "I"]

    def test_unmapped_group_outside_the_design_is_refused(self, tree):
        from source_analytics.bayes import build_job
        from source_analytics.config import StudyConfig
        p = self._study(tree)
        raw = yaml.safe_load(p.read_text())
        raw["bayes"]["tables"]["field"]["groups"] = {"G2": "KO", "G3": "MID", "G6": "WT"}
        p.write_text(yaml.safe_dump(raw))
        cfg = StudyConfig.from_yaml(p)
        with pytest.raises(ValueError, match="design levels"):
            build_job(cfg, "field", "value", parse_bayes_config(cfg.raw["bayes"], tree), out_dir=tree / "o3")


class TestUncentred:
    """``center: false`` (2026-11-10): the job carries it, and set-I entries may then sit on one group's level."""

    def _study(self, tree, center, hyps, priors=None):
        t = {"field": {**TABLE["field"], **({"center": center} if center is not None else {})}}
        cfg = {"name": "toy", "output_dir": str(tree), "results_dir": str(tree / "results"),
               "groups": {"G2": "", "G3": "", "G6": ""},
               "design": {"factor": "group", "reference": "WT", "levels": ["WT", "KO"]}, "hypotheses": hyps,
               "bayes": {"prior_sets": ["W", "I"] if priors else ["W"], "tables": t,
                         "outcomes": {"field": ["value"]}, **({"priors": {"I": priors}} if priors else {})}}
        p = tree / "study.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return p

    def _job(self, tree, p):
        from source_analytics.bayes import build_job
        from source_analytics.config import StudyConfig
        cfg = StudyConfig.from_yaml(p)
        return build_job(cfg, "field", "value", parse_bayes_config(cfg.raw["bayes"], tree), out_dir=tree / "o")

    HYPS = [{"name": "ko_level", "kind": "contrast", "weights": {"KO": 1.0}},
            {"name": "ko", "kind": "contrast", "weights": {"KO": 1.0, "WT": -1.0}}]

    def test_center_defaults_true_and_is_carried_into_the_job(self, tree):
        assert _bayes(TABLE, tree).tables["field"].center is True
        job = self._job(tree, self._study(tree, False, self.HYPS))
        assert job.center is False and job.provenance["table"]["definition"]["center"] is False
        assert self._job(tree, self._study(tree, None, self.HYPS)).center is True

    def test_center_must_be_a_boolean(self, tree):
        with pytest.raises(BayesConfigError, match="center"):
            parse_bayes_config({"tables": {"field": {**TABLE["field"], "center": "no"}}}, tree)

    def test_level_entries_map_to_lv_terms_only_when_uncentred(self, tree):
        pri = {"version": 1, "entries": [
            {"dv": "value", "hypothesis": "ko_level", "band": "40 ITC", "prior": {"mean": 0.3, "sd": 0.9},
             "source": "x"},
            {"dv": "value", "hypothesis": "ko", "band": "chirp mean ERSP", "prior": {"mean": -0.4, "sd": 0.5},
             "source": "y"}]}
        (tree / "pri.yaml").write_text(yaml.safe_dump(pri))
        job = self._job(tree, self._study(tree, False, self.HYPS, priors="pri.yaml"))
        assert job.informative == {"lv_40ITC_KO": [0.3, 0.9], "x_chirpmeanERSP_KO": [-0.4, 0.5]}
        with pytest.raises(BayesConfigError, match="one group minus the reference"):
            self._job(tree, self._study(tree, True, self.HYPS, priors="pri.yaml"))
