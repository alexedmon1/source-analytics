"""Config-layer coverage for the declarative hypothesis spec.

Replaces the retired test_hypothesis_contrasts.py (which tested the deleted gating
schema). Covers Hypothesis / DesignSpec parsing, per-kind validation, the legacy
group_a/group_b sugar, and the _contrasts_from_design_spec synthesis bridge.
"""

from __future__ import annotations

import pytest

from source_analytics.config import (
    Contrast,
    DesignSpec,
    Hypothesis,
    StudyConfig,
    _contrasts_from_design_spec,
)


# ---- Hypothesis.from_dict ------------------------------------------------- #
def test_contrast_kind_parses_weights():
    h = Hypothesis.from_dict(
        {"name": "ko_vs_wt", "kind": "contrast",
         "weights": {"KO": 1, "WT": -1}, "role": "phenotype"}
    )
    assert h.kind == "contrast"
    assert h.weights == {"KO": 1.0, "WT": -1.0}
    assert h.role == "phenotype"


def test_legacy_group_ab_sugar_becomes_contrast():
    h = Hypothesis.from_dict({"name": "d", "group_a": "KO", "group_b": "WT"})
    assert h.kind == "contrast"
    assert h.weights == {"KO": 1.0, "WT": -1.0}


def test_omnibus_kind_takes_groups():
    h = Hypothesis.from_dict({"name": "omni", "kind": "omnibus",
                              "groups": ["A", "B", "C"]})
    assert h.kind == "omnibus"
    assert h.groups == ["A", "B", "C"]
    assert h.referenced_groups() == {"A", "B", "C"}


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="invalid kind"):
        Hypothesis.from_dict({"name": "x", "kind": "wat", "weights": {"A": 1, "B": -1}})


def test_contrast_without_weights_raises():
    with pytest.raises(ValueError, match="requires 'weights'"):
        Hypothesis.from_dict({"name": "x", "kind": "contrast"})


def test_regression_requires_predictor():
    with pytest.raises(ValueError, match="requires 'predictor'"):
        Hypothesis.from_dict({"name": "x", "kind": "regression"})


def test_equivalence_requires_margin():
    with pytest.raises(ValueError, match="requires 'margin'"):
        Hypothesis.from_dict({"name": "x", "kind": "equivalence",
                              "weights": {"A": 1, "B": -1}})


def test_equivalence_with_sd_margin_ok():
    h = Hypothesis.from_dict({"name": "n", "kind": "equivalence",
                              "weights": {"T": 1, "WT": -1},
                              "margin": {"mode": "sd", "value": 0.25}})
    assert h.margin == {"mode": "sd", "value": 0.25}


# ---- DesignSpec.from_dict ------------------------------------------------- #
def test_design_spec_parses_blocks():
    spec = DesignSpec.from_dict({
        "design": {"factor": "group", "reference": "WT",
                   "levels": ["WT", "KO"], "covariates": ["n_epochs"]},
        "hypotheses": [
            {"name": "omni", "kind": "omnibus"},
            {"name": "d", "kind": "contrast", "weights": {"KO": 1, "WT": -1}},
        ],
    })
    assert spec.factor == "group" and spec.reference == "WT"
    assert spec.levels == ["WT", "KO"] and spec.covariates == ["n_epochs"]
    assert [h.name for h in spec.hypotheses] == ["omni", "d"]


def test_design_spec_lifts_legacy_contrasts():
    spec = DesignSpec.from_dict({
        "contrasts": [{"name": "d", "group_a": "KO", "group_b": "WT"}]
    })
    assert spec is not None
    assert spec.hypotheses[0].kind == "contrast"


def test_design_spec_none_when_nothing_declared():
    assert DesignSpec.from_dict({"bands": {"Delta": [1, 4]}}) is None


# ---- synthesis bridge ----------------------------------------------------- #
def test_synthesis_maps_pairwise_skips_omnibus():
    spec = DesignSpec.from_dict({"hypotheses": [
        {"name": "omni", "kind": "omnibus"},                                   # skipped
        {"name": "d", "kind": "contrast", "weights": {"KO": 1, "WT": -1}},     # -> Contrast
        {"name": "avg", "kind": "contrast",
         "weights": {"A": 1, "B": -0.5, "C": -0.5}},                           # not pairwise
        {"name": "n", "kind": "equivalence", "weights": {"T": 1, "WT": -1},
         "margin": {"mode": "sd", "value": 0.25}},                             # -> Contrast
    ]})
    contrasts = _contrasts_from_design_spec(spec)
    names = [c.name for c in contrasts]
    assert names == ["d", "n"]
    d = contrasts[0]
    assert isinstance(d, Contrast)
    assert d.group_a == "KO" and d.group_b == "WT"  # positive weight = group_a


# ---- StudyConfig end-to-end ----------------------------------------------- #
def test_study_yaml_synthesizes_contrasts(tmp_path):
    yaml_text = """
name: T
groups: {WT: "WT", KO: "KO"}
group_order: [WT, KO]
design: {factor: group, reference: WT, levels: [WT, KO]}
hypotheses:
  - {name: group_omnibus, kind: omnibus}
  - {name: disease_effect, kind: contrast, weights: {KO: 1, WT: -1}, role: phenotype}
bands: {Delta: [1, 4]}
paths: {analytics: ./a, results: ./r}
discovery: {root_dir: ./d}
"""
    p = tmp_path / "study.yaml"
    p.write_text(yaml_text)
    cfg = StudyConfig.from_yaml(p)
    assert cfg.design_spec is not None
    assert [h.name for h in cfg.design_spec.hypotheses] == ["group_omnibus", "disease_effect"]
    # legacy per-contrast loops still fed: omnibus skipped, pairwise synthesized
    assert [c.name for c in cfg.contrasts] == ["disease_effect"]
    assert cfg.referenced_groups() == {"WT", "KO"}
    assert not hasattr(cfg, "hypothesis_testing")
