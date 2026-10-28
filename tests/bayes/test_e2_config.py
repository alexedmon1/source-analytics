"""The study YAML's bayes: block (engine plan E2): validation before R starts, the informative prior file, and the
acceptance test that FORGE's G1 jobs built from the block alone equal the E0 jobs (``SA_BAYES_E0``)."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import yaml

from source_analytics.bayes import BayesConfigError, build_job, load_informative, parse_bayes_config
from source_analytics.config import StudyConfig

DESIGN = {"factor": "group", "reference": "WT", "levels": ["WT", "KO", "KO_D"], "covariates": []}
HYPS = [{"name": "disease", "kind": "contrast", "weights": {"KO": 1.0, "WT": -1.0}},
        {"name": "rescue", "kind": "contrast", "weights": {"KO_D": 1.0, "KO": -1.0}},
        {"name": "norm", "kind": "equivalence", "weights": {"KO_D": 1.0, "WT": -1.0}}]


def _prior_file(tmp_path, entries):
    p = tmp_path / "I.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "entries": entries}))
    return p


def test_defaults():
    c = parse_bayes_config(None, Path("."))
    assert c.rope_sd == 0.1 and c.prior_sets == ("W",) and c.informative is None   # no set-I file: W only
    assert c.fit.chains == 4 and c.fit.iter == 2000 and c.gate.reps == {"sbc": 100, "null": 100, "recovery": 10, "equivalence": 100}


@pytest.mark.parametrize("block, msg", [
    ({"ropee": {}}, "unknown key"),
    ({"prior_sets": ["W"], "rope": {"value": -1}}, "positive number"),
    ({"prior_sets": ["W"], "rope": {"mode": "raw", "value": 1}}, "only 'sd'"),
    ({"prior_sets": ["W", "X"]}, "prior_sets"),
    ({"prior_sets": ["W", "I"]}, "priors.I is not set"),
    ({"prior_sets": ["W"], "priors": {"I": "missing.yaml"}}, "file not found"),
    ({"prior_sets": ["W"], "fit": {"chains": 0}}, "positive integer"),
    ({"prior_sets": ["W"], "fit": {"iter": 500, "warmup": 500}}, "warmup"),
    ({"prior_sets": ["W"], "gate": {"reps": {"sbc": 10, "nul": 5}}}, "unknown key"),
    ({"prior_sets": ["W"], "gate": {"reduced_bands": "Theta"}}, "list of band names"),
    ({"prior_sets": ["W"], "outcomes": {"roi_pac": ["x"]}}, "not supported"),
    ({"prior_sets": ["W"], "outcomes": {"roi_psd": ["power"]}}, "unknown dv"),
])
def test_bad_blocks_fail_with_a_message(block, msg, tmp_path):
    with pytest.raises(BayesConfigError, match=msg):
        parse_bayes_config(block, tmp_path)


def test_null_reps_key_survives_yaml():
    block = yaml.safe_load("gate: {reps: {sbc: 10, null: 20, recovery: 3}}")
    assert parse_bayes_config(block, Path(".")).gate.reps == {"sbc": 10, "null": 20, "recovery": 3, "equivalence": 100}


def test_relative_prior_path_resolves_against_the_study(tmp_path):
    _prior_file(tmp_path, [])
    c = parse_bayes_config({"priors": {"I": "I.yaml"}}, tmp_path)
    assert c.informative == (tmp_path / "I.yaml").resolve() and c.prior_sets == ("W", "I")


def test_prior_file_maps_hypothesis_and_band_to_terms(tmp_path):
    p = _prior_file(tmp_path, [
        {"dv": "absolute", "hypothesis": "disease", "band": "Low Gamma", "prior": {"mean": 0.5, "sd": 0.8},
         "source": "lit.md"},
        {"dv": "exponent", "hypothesis": "disease", "prior": {"mean": 0.0, "sd": 0.8}, "source": "lit.md"},
        {"dv": "absolute", "hypothesis": "norm", "band": "Delta", "prior": {"mean": 0.1, "sd": 0.5}, "source": "x"}])
    assert load_informative(p, "absolute", DESIGN, HYPS, bands=["Delta", "Low Gamma"]) == \
        {"x_LowGamma_KO": [0.5, 0.8], "x_Delta_KOD": [0.1, 0.5]}
    assert load_informative(p, "exponent", DESIGN, HYPS) == {"x_all_KO": [0.0, 0.8]}


@pytest.mark.parametrize("entry, msg", [
    ({"dv": "absolute", "hypothesis": "rescue", "band": "Delta", "prior": {"mean": 0, "sd": 1}, "source": "x"},
     "minus the reference"),
    ({"dv": "absolute", "hypothesis": "nope", "band": "Delta", "prior": {"mean": 0, "sd": 1}, "source": "x"},
     "not declared"),
    ({"dv": "absolute", "hypothesis": "disease", "band": "Gama", "prior": {"mean": 0, "sd": 1}, "source": "x"},
     "not one of the study's bands"),
    ({"dv": "absolute", "hypothesis": "disease", "band": "Delta", "prior": {"mean": 0, "sd": 1}}, "'source'"),
    ({"dv": "absolute", "hypothesis": "disease", "band": "Delta", "prior": {"mean": 0, "sd": 0}, "source": "x"},
     "positive"),
])
def test_bad_prior_entries_fail(entry, msg, tmp_path):
    with pytest.raises(BayesConfigError, match=msg):
        load_informative(_prior_file(tmp_path, [entry]), "absolute", DESIGN, HYPS, bands=["Delta"])


E0 = os.environ.get("SA_BAYES_E0")


@pytest.mark.skipif(not E0, reason="SA_BAYES_E0 not set (fixtures live outside the repository)")
def test_forge_jobs_from_the_block_equal_e0():
    doc = yaml.safe_load((Path(E0) / "e2_bayes_block.yaml").read_text())
    e2 = doc["e2"]
    study = StudyConfig.from_yaml(e2["study"])
    assert "bayes" not in study.raw, "FORGE's YAML now has its own bayes: block; update this test"
    bayes = parse_bayes_config(doc["bayes"], Path(e2["study"]).parent)
    for analysis, dvs in bayes.outcomes.items():
        cfg = study.for_paradigm_analysis(e2["paradigm"], analysis)
        for dv in dvs:
            got = dataclasses.asdict(build_job(cfg, analysis, dv, bayes, out_dir=Path(e2["out"][analysis])))
            want = json.loads((Path(E0) / e2["cases"][analysis] / "job.json").read_text())
            # `gate` records only whether the CLI ran the gate; R never reads it (the E0 files were last written by
            # their gate runs), so it is not part of what makes a number
            got.pop("gate"), want.pop("gate")
            got = {k: v for k, v in got.items() if k in want}   # fields added since E0 (E3: gate settings, provenance)
            assert got == want, {k: (got.get(k), want.get(k)) for k in set(got) | set(want) if got.get(k) != want.get(k)}
