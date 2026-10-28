"""Minimum detectable effect on the hypothesis layer's rows (DUET #151).

Checks:
  1. ``mde_from_se`` reproduces the reference values shared with the R helper
     (``tests/test_hypothesis.R``) and with neurofaune / neurovrai;
  2. the reference mean is the design's reference level, else the negative side;
  3. tabular contrast / equivalence rows carry mde from their own SE and df, and
     mde_percent of the reference group's cell mean; omnibus rows carry a note;
  4. cluster-permutation and NBS rows carry a "not computed" note, never a bare NaN.

Run: uv run --no-sync pytest tests/test_power.py -q
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from source_analytics.config import DesignSpec, Hypothesis
from source_analytics.hypothesis.power import (
    MDE_ALPHA,
    MDE_POWER,
    mde_columns,
    mde_from_se,
    reference_weights,
)
from source_analytics.hypothesis.tabular import _adapt_cell, write_module_hypotheses_tabular

MDE_COLS = ["mde", "mde_percent", "mde_power", "mde_alpha", "mde_note"]


@pytest.mark.parametrize("se, df, expected", [
    (1.0, 10, 3.107197),
    (1.0, 26, 2.911195),
    (2.09, 24, 6.104376),
    (1.0, math.inf, 2.801585),
    (0.5, 4, 1.858705),
])
def test_mde_reference_values(se, df, expected):
    assert mde_from_se(se, df) == pytest.approx(expected, abs=1e-6)


def test_mde_falls_back_to_normal_and_refuses_bad_inputs():
    assert mde_from_se(1.0, None) == pytest.approx(2.801585, abs=1e-6)
    assert mde_from_se(1.0, float("nan")) == pytest.approx(2.801585, abs=1e-6)
    for se, df in ((float("nan"), 10), (0.0, 10), (-1.0, 10), (1.0, 0)):
        assert math.isnan(mde_from_se(se, df))
    assert MDE_POWER == 0.80 and MDE_ALPHA == 0.05


def test_mde_columns_note_and_percent():
    c = mde_columns(1.0, 10, ref_mean=-4.0, resid_sd=2.0)
    assert c["mde_percent"] == pytest.approx(100 * 3.107197 / 4.0, abs=1e-4)
    assert c["mde_note"] == "" and c["mde_power"] == 0.8 and c["mde_alpha"] == 0.05
    assert math.isnan(mde_columns(1.0, 10, resid_sd=2.0)["mde_percent"])   # no reference
    assert math.isnan(mde_columns(1.0, 10, ref_mean=4.0)["mde_percent"])   # no residual SD
    bad = mde_columns(float("nan"), 10, ref_mean=4.0, resid_sd=2.0)
    assert math.isnan(bad["mde"]) and bad["mde_note"].startswith("not computed: ")


@pytest.mark.parametrize("ref", [2.0, -2.0])
def test_mde_percent_blank_within_one_residual_sd(ref):
    """|reference mean| < residual SD leaves mde_percent blank; at or above it, computed.
    The mde itself is unaffected either way."""
    sd = 2.0
    just_below = mde_columns(1.0, 10, ref_mean=ref * (1 - 1e-9), resid_sd=sd)
    at = mde_columns(1.0, 10, ref_mean=ref, resid_sd=sd)
    just_above = mde_columns(1.0, 10, ref_mean=ref * (1 + 1e-9), resid_sd=sd)
    assert math.isnan(just_below["mde_percent"]) and just_below["mde_note"] == ""
    assert just_below["mde"] == pytest.approx(3.107197, abs=1e-6)
    assert at["mde_percent"] == pytest.approx(100 * 3.107197 / 2.0, abs=1e-4)
    assert just_above["mde_percent"] == pytest.approx(100 * 3.107197 / 2.0, abs=1e-4)


def test_tabular_percent_blank_for_a_centred_reference():
    """A reference group whose cell mean sits within one pooled SD of zero
    (a change score, a centred measure) gets no mde_percent."""
    rng = np.random.default_rng(9)
    bg = {"WT_VEH": rng.normal(0.1, 1.0, 12), "KO_VEH": rng.normal(1.0, 1.0, 12)}
    hyp = Hypothesis(name="ko_vs_wt", kind="contrast", weights={"KO_VEH": 1.0, "WT_VEH": -1.0})
    row = _adapt_cell(bg, hyp, ["WT_VEH", "KO_VEH"], reference="WT_VEH")
    assert abs(bg["WT_VEH"].mean()) < 1.0 / 2       # well inside one SD of zero
    assert row["mde"] > 0 and math.isnan(row["mde_percent"]) and row["mde_note"] == ""
    bad = mde_columns(float("nan"), 10, ref_mean=4.0)
    assert math.isnan(bad["mde"]) and bad["mde_note"].startswith("not computed: ")


def test_reference_weights():
    g = ["A", "B", "C"]
    assert list(reference_weights({"A": -1, "B": 1}, g, "B")) == [0, 1, 0]
    assert list(reference_weights({"A": 1, "B": -0.5, "C": -0.5}, g, None)) == [0, 0.5, 0.5]
    # the reference outside the contrast falls back to the negative side
    assert list(reference_weights({"B": 1, "C": -1}, g, "A")) == [0, 0, 1]
    assert reference_weights({"A": 1, "B": 1}, g, None) is None


GROUPS = ["WT_VEH", "KO_VEH", "KO_HD_ICV"]


def _by_group(seed=3):
    rng = np.random.default_rng(seed)
    means = {"WT_VEH": 5.0, "KO_VEH": 6.0, "KO_HD_ICV": 5.5}
    return {g: rng.normal(m, 1.0, 10 + i) for i, (g, m) in enumerate(means.items())}


def test_tabular_contrast_row_carries_mde():
    bg = _by_group()
    hyp = Hypothesis(name="ko_vs_wt", kind="contrast", weights={"KO_VEH": 1.0, "WT_VEH": -1.0})
    row = _adapt_cell(bg, hyp, GROUPS, reference="WT_VEH")
    q = st.t.ppf(1 - MDE_ALPHA / 2, row["df"]) + st.t.ppf(MDE_POWER, row["df"])
    assert row["mde"] == pytest.approx(q * row["SE"], rel=1e-12)
    assert row["mde_percent"] == pytest.approx(100 * row["mde"] / abs(bg["WT_VEH"].mean()))
    assert row["mde_note"] == ""


def test_tabular_reference_level_wins_over_negative_side():
    """{WT_VEH: +1, KO_VEH: -1} with reference WT_VEH scales by WT_VEH, not KO_VEH."""
    bg = _by_group()
    hyp = Hypothesis(name="wt_vs_ko", kind="contrast", weights={"WT_VEH": 1.0, "KO_VEH": -1.0})
    row = _adapt_cell(bg, hyp, GROUPS, reference="WT_VEH")
    assert row["mde_percent"] == pytest.approx(100 * row["mde"] / abs(bg["WT_VEH"].mean()))
    row_noref = _adapt_cell(bg, hyp, GROUPS)
    assert row_noref["mde_percent"] == pytest.approx(
        100 * row_noref["mde"] / abs(bg["KO_VEH"].mean()))


def test_tabular_omnibus_and_equivalence_rows():
    bg = _by_group()
    omni = _adapt_cell(bg, Hypothesis(name="o", kind="omnibus"), GROUPS, reference="WT_VEH")
    assert math.isnan(omni["mde"]) and omni["mde_note"] == "not computed: omnibus F test"
    eq = _adapt_cell(bg, Hypothesis(name="e", kind="equivalence",
                                    weights={"KO_HD_ICV": 1.0, "WT_VEH": -1.0},
                                    margin={"mode": "sd", "value": 0.5}),
                     GROUPS, reference="WT_VEH")
    assert eq["stat_type"] == "tost" and eq["mde"] > 0 and eq["mde_percent"] > 0


def test_tabular_csv_has_the_columns(tmp_path):
    rows = []
    for g, vals in _by_group().items():
        for i, v in enumerate(vals):
            rows.append({"subject": f"{g}{i}", "group": g, "band": "B1", "value": v})
    spec = DesignSpec(factor="group", reference="WT_VEH", levels=GROUPS, hypotheses=[
        Hypothesis(name="ko_vs_wt", kind="contrast", weights={"KO_VEH": 1.0, "WT_VEH": -1.0}),
        Hypothesis(name="omni", kind="omnibus"),
    ])
    cfg = SimpleNamespace(design_spec=spec, bands={"B1": [1, 4]})
    write_module_hypotheses_tabular(pd.DataFrame(rows), cfg, tmp_path, "t", value_col="value")
    out = pd.read_csv(tmp_path / "t_hypotheses.csv")
    out["mde_note"] = out["mde_note"].fillna("")      # an empty note reads back as NaN
    assert all(c in out.columns for c in MDE_COLS) and "SE" in out.columns
    con = out[out.kind == "contrast"].iloc[0]
    assert con["mde"] > 0 and con["mde_percent"] > 0 and con["mde_note"] == ""
    om = out[out.kind == "omnibus"].iloc[0]
    assert math.isnan(om["mde"]) and om["mde_note"] == "not computed: omnibus F test"
    assert (out["mde_power"] == 0.8).all() and (out["mde_alpha"] == 0.05).all()


def test_cluster_and_nbs_rows_carry_a_note(tmp_path):
    from source_analytics.hypothesis.edge import write_module_hypotheses_edge
    from source_analytics.hypothesis.permutation import write_module_hypotheses_perm

    rng = np.random.default_rng(5)
    spec = DesignSpec(factor="group", reference="B", levels=["A", "B"], hypotheses=[
        Hypothesis(name="a_vs_b", kind="contrast", weights={"A": 1.0, "B": -1.0}),
        Hypothesis(name="eq", kind="equivalence", weights={"A": 1.0, "B": -1.0},
                   margin={"mode": "sd", "value": 0.5}),
    ])
    cfg = SimpleNamespace(design_spec=spec)
    sgroups = {f"s{i}": ("A" if i < 8 else "B") for i in range(16)}

    maps = {u: rng.normal(0, 1, 20) for u in sgroups}
    coords = np.column_stack([np.arange(20), np.zeros(20), np.zeros(20)]).astype(float)
    perm = write_module_hypotheses_perm(
        {("B1", "dv"): maps}, sgroups, coords, cfg, tmp_path, "vtx",
        n_perms=20, threshold=2.0, distance_mm=1.5)
    assert perm["mde"].isna().all()
    assert set(perm.loc[perm.kind == "contrast", "mde_note"]) == {
        "not computed: cluster permutation test"}
    assert set(perm.loc[perm.kind == "equivalence", "mde_note"]) == {
        "not computed: per-vertex equivalence summary"}

    mats = {}
    for u in sgroups:
        m = np.triu(rng.normal(0, 1, (8, 8)), 1)
        mats[u] = m + m.T
    nbs = write_module_hypotheses_edge(
        {("B1", "coh"): mats}, sgroups, cfg, tmp_path, "nbs",
        nbs_threshold=2.0, n_perms=20)
    assert nbs["mde"].isna().all()
    assert set(nbs.loc[nbs.kind == "contrast", "mde_note"]) == {
        "not computed: network-based statistic component"}
    assert set(nbs.loc[nbs.kind == "equivalence", "mde_note"]) == {
        "not computed: per-edge equivalence summary"}
    for df in (perm, nbs):
        assert (df["mde_power"] == 0.8).all() and (df["mde_alpha"] == 0.05).all()
