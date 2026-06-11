"""Tests for the hypothesis-testing contrast schema (Phase 0).

Covers the Contrast parser, enum/margin validation, and the gating-DAG
validation. The stats engine does not yet act on these fields; these tests
lock the YAML contract.
"""

import pytest

from source_analytics.config import (
    Contrast,
    StudyConfig,
    _validate_contrast_graph,
)


def test_bare_contrast_is_backward_compatible():
    c = Contrast.from_dict({"name": "x", "group_a": "A", "group_b": "B"})
    assert c.role == "exploratory"
    assert c.test == "difference"
    assert c.gate_on == []
    assert c.equivalence_margin is None
    assert c.label is None and c.group is None


def test_full_hypothesis_contrast_parses():
    c = Contrast.from_dict(
        {
            "name": "norm",
            "group_a": "T",
            "group_b": "WT",
            "label": "Normalization",
            "group": "Normalization to WT",
            "role": "normalization",
            "test": "equivalence",
            "gate_on": ["disease", "rescue"],
            "equivalence_margin": {"mode": "gap_fraction", "value": 0.25},
        }
    )
    assert c.role == "normalization"
    assert c.test == "equivalence"
    assert c.gate_on == ["disease", "rescue"]
    assert c.equivalence_margin == {"mode": "gap_fraction", "value": 0.25}


def test_scalar_gate_on_is_normalized_to_list():
    c = Contrast.from_dict(
        {"name": "r", "group_a": "T", "group_b": "K", "gate_on": "disease"}
    )
    assert c.gate_on == ["disease"]


@pytest.mark.parametrize("field,bad", [("role", "bogus"), ("test", "anova")])
def test_invalid_enum_raises(field, bad):
    with pytest.raises(ValueError, match=field):
        Contrast.from_dict({"name": "x", "group_a": "A", "group_b": "B", field: bad})


@pytest.mark.parametrize(
    "margin",
    [
        {"mode": "bogus", "value": 0.25},
        {"mode": "gap_fraction", "value": 0},
        {"mode": "gap_fraction", "value": -1},
        {"mode": "sd"},  # missing value
        "0.25",  # not a mapping
    ],
)
def test_invalid_margin_raises(margin):
    with pytest.raises(ValueError):
        Contrast.from_dict(
            {"name": "x", "group_a": "A", "group_b": "B",
             "test": "equivalence", "equivalence_margin": margin}
        )


def test_unknown_gate_on_target_raises():
    c = Contrast.from_dict(
        {"name": "r", "group_a": "T", "group_b": "K", "gate_on": "ghost"}
    )
    with pytest.raises(ValueError, match="unknown contrast"):
        _validate_contrast_graph([c], {})


def test_equivalence_without_margin_or_default_raises():
    c = Contrast.from_dict(
        {"name": "n", "group_a": "T", "group_b": "WT", "test": "equivalence"}
    )
    with pytest.raises(ValueError, match="equivalence"):
        _validate_contrast_graph([c], {})


def test_equivalence_with_study_default_margin_ok():
    c = Contrast.from_dict(
        {"name": "n", "group_a": "T", "group_b": "WT", "test": "equivalence"}
    )
    # Should not raise — margin resolvable from the study default.
    _validate_contrast_graph(
        [c], {"default_equivalence_margin": {"mode": "gap_fraction", "value": 0.25}}
    )


def test_gate_on_cycle_raises():
    a = Contrast.from_dict({"name": "a", "group_a": "A", "group_b": "B", "gate_on": "b"})
    b = Contrast.from_dict({"name": "b", "group_a": "A", "group_b": "B", "gate_on": "a"})
    with pytest.raises(ValueError, match="[Cc]ycle"):
        _validate_contrast_graph([a, b], {})


def test_valid_dag_passes():
    contrasts = [
        Contrast.from_dict({"name": "disease", "group_a": "KO", "group_b": "WT",
                            "role": "phenotype"}),
        Contrast.from_dict({"name": "rescue", "group_a": "T", "group_b": "KO",
                            "role": "rescue", "gate_on": "disease"}),
        Contrast.from_dict({"name": "norm", "group_a": "T", "group_b": "WT",
                            "role": "normalization", "test": "equivalence",
                            "gate_on": ["disease", "rescue"]}),
    ]
    _validate_contrast_graph(
        contrasts,
        {"default_equivalence_margin": {"mode": "gap_fraction", "value": 0.25}},
    )


def test_study_yaml_round_trips_fields_into_raw(tmp_path):
    """The R-facing study_config.yaml is dumped from config.raw — confirm the
    new hypothesis fields survive into raw and reach R unchanged."""
    cfg_text = """
name: "HT Study"
output_dir: "{out}"
groups:
  WT_VEH: "WT"
  KO_VEH: "KO"
hypothesis_testing:
  default_equivalence_margin: {{ mode: gap_fraction, value: 0.25 }}
  gate_alpha: 0.05
contrasts:
  - name: disease_effect
    group_a: KO_VEH
    group_b: WT_VEH
    role: phenotype
  - name: norm
    group_a: KO_VEH
    group_b: WT_VEH
    role: normalization
    test: equivalence
    gate_on: disease_effect
bands:
  Theta: [4, 8]
discovery:
  root_dir: "{out}"
""".format(out=tmp_path)
    p = tmp_path / "study.yaml"
    p.write_text(cfg_text)

    cfg = StudyConfig.from_yaml(p)
    assert cfg.hypothesis_testing["gate_alpha"] == 0.05
    norm = next(c for c in cfg.contrasts if c.name == "norm")
    assert norm.role == "normalization"
    assert norm.test == "equivalence"
    assert norm.gate_on == ["disease_effect"]

    # raw carries everything R reads (the analysis writers dump dict(config.raw)).
    raw_norm = next(c for c in cfg.raw["contrasts"] if c["name"] == "norm")
    assert raw_norm["role"] == "normalization"
    assert raw_norm["test"] == "equivalence"
    assert cfg.raw["hypothesis_testing"]["gate_alpha"] == 0.05
