"""The benchmark scorer (bayes-sl reports/2026-10-28_stage2_benchmark_plan.md, BM2): synthetic cohorts and claims
tables with known answers, the contract's errors, and (with SA_BAYES_BENCH) parity with the EE3 pilot's scores."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from source_analytics.bayes.benchmark import (BenchmarkError, CohortSet, attach_truth, claims_from_engine, n80,
                                              read_claims, score)

ROIS = ["r1", "r2", "r3"]
BANDS = ["Low Gamma", "Theta"]
REPS = 40
TRUTH = {("r1", "Low Gamma"): 1.0, ("r2", "Low Gamma"): 0.5}     # r1 planted; r2 leaked but real (> 0.1 x 2 dB)


@pytest.fixture(scope="module")
def cs(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohorts") / "v0"
    cohorts = {}
    for cell, eff in (("noise", 0.0), ("F_1.0", 1.0)):
        for r in range(1, REPS + 1):
            d = root / cell / f"rep{r:03d}"
            d.mkdir(parents=True)
            pd.DataFrame([(ro, b, TRUTH.get((ro, b), 0.0) * (eff > 0), 0.0) for ro in ROIS for b in BANDS],
                         columns=["roi", "band", "truth_db", "se_db"]).to_csv(d / "path_truth.csv", index=False)
            pl = pd.DataFrame({"subject": ["m1"], "source": [7], "x_mm": [1.0], "y_mm": [2.0], "z_mm": [3.0],
                               "ux": [1.0], "uy": [0.0], "uz": [0.0], "amplitude": [1.0]})
            (pl if eff > 0 else pl.iloc[:0]).to_csv(d / "plant.csv", index=False)
            cohorts[f"{cell}/rep{r:03d}"] = {}
    (root / "MANIFEST.json").write_text(json.dumps({
        "benchmark": "v0", "plant_band": "Low Gamma", "rois": ROIS, "bands": BANDS,
        "band_units_db": {"Low Gamma": 2.0, "Theta": 4.0}, "cohorts": cohorts,
        "cells": {"noise": {"region": None, "effect_sd": 0.0, "planted_rois": [], "target": None},
                  "F_1.0": {"region": "F", "effect_sd": 1.0, "planted_rois": ["r1"], "target": ["r1"]}}}))
    return CohortSet.load(root)


def _claims(arm, est, half, decide, ns=(8, 16), se=None, cells=("noise", "F_1.0"), seed=0):
    """One row per cohort x N x band x unit; est(truth, rng) -> estimate; decide(row) -> decision."""
    rng = np.random.default_rng(seed)
    rows = []
    for cell in cells:
        eff = cell != "noise"
        for r in range(1, REPS + 1):
            for n in ns:
                for b in BANDS:
                    for u in ["ROI-average", *ROIS]:
                        t = (np.mean([TRUTH.get((x, b), 0.0) for x in ROIS]) if u == "ROI-average"
                             else TRUTH.get((u, b), 0.0)) * eff
                        e = est(t, rng)
                        row = dict(cohort=f"{cell}/rep{r:03d}", n_per_group=n, arm=arm, prior_set="", band=b, unit=u,
                                   estimate_db=e, lo_db=e - half, hi_db=e + half, se_db=se,
                                   decision_matched="excludes_0" if abs(e) > half else "includes_0")
                        row["decision"] = decide(row, t)
                        rows.append(row)
    return pd.DataFrame(rows)


def _by_truth(row, t):
    return "effect_pos" if t > 0.3 else "undecided"


def test_honest_arm_is_calibrated_and_attributed(cs):
    c = _claims("honest", lambda t, g: t + g.normal(0, 0.1), 1.96 * 0.1, _by_truth, se=0.1)
    tabs = score(c, cs)
    cal = tabs["calibration"]
    lg = cal[(cal.cell == "F_1.0") & (cal.band == "Low Gamma")]
    assert (lg.cov_avg > 0.85).all() and (lg.bias_avg_db.abs() < 0.05).all() and (abs(lg.sd_z_avg - 1) < 0.3).all()
    assert (abs(lg.vi_avg - 1) < 0.5).all() and np.allclose(lg.width_avg_db, 0.392)
    assert np.allclose(lg.truth_avg_db, 0.5)                     # ROI-average truth = mean of the ROIs' truths
    at = tabs["attribution"]
    own = at[at.rule == "own"]
    assert np.allclose(own.planted, 0.5) and np.allclose(own.leaked_real, 0.5) and (own.elsewhere == 0).all()
    assert not own.usable_as_location.any()
    nz = tabs["noise"]
    assert (nz.claim_effect == 0).all() and nz["pass"].all()


def test_biased_and_overconfident_arms_fail_calibration(cs):
    biased = score(_claims("biased", lambda t, g: t + 0.5, 0.2, _by_truth, se=0.1), cs)["calibration"]
    assert np.allclose(biased[biased.cell == "F_1.0"].bias_avg_db, 0.5) and (biased.cov_avg == 0).all()
    assert not biased.pass_avg.any()
    over = score(_claims("over", lambda t, g: t + g.normal(0, 1.0), 0.01, _by_truth, se=0.005), cs)["calibration"]
    assert (over.cov_avg < 0.1).all() and (over.sd_z_avg > 50).all() and (over.vi_avg > 1000).all()


def test_wrong_place_and_wrong_sign(cs):
    c = _claims("wrong", lambda t, g: t, 0.1, lambda row, t: "effect_pos" if row["unit"] == "r3" else "undecided")
    at = score(c, cs)["attribution"]
    own = at[at.rule == "own"]
    assert (own.elsewhere == 1).all() and (own.planted == 0).all()
    c = _claims("flip", lambda t, g: t, 0.1, lambda row, t: "effect_neg" if row["unit"] == "r1" else "undecided")
    own = score(c, cs)["attribution"].query("rule == 'own'")
    assert (own.wrong_sign == 1).all()


def test_noise_claims_and_the_matched_rule(cs):
    """An arm that claims an effect in a third of the noise cohorts fails the noise rule; the matched rule
    ("interval excludes 0") is scored from decision_matched, independently of the arm's own rule."""
    def est(t, g):
        return t + (1.0 if g.random() < 1 / 3 else 0.0)
    c = _claims("noisy", est, 0.5, lambda row, t: "effect_pos" if row["estimate_db"] > 0.5 else "undecided", seed=3)
    nz = score(c, cs)["noise"]
    avg = nz[nz.band == "Theta"]
    assert ((avg.claim_effect > 0.2) & (avg.claim_effect < 0.5)).all() and not avg["pass"].any()
    assert np.allclose(avg.claim_effect, avg.claim_matched)


def test_located_arm_distance(cs):
    c = _claims("honest", lambda t, g: t, 0.1, _by_truth, cells=("F_1.0",))
    loc = pd.DataFrame([dict(cohort=f"F_1.0/rep{r:03d}", n_per_group=8, arm="field", prior_set="", band="Low Gamma",
                             unit="located", estimate_db=np.nan, lo_db=np.nan, hi_db=np.nan, decision="undecided",
                             decision_matched="includes_0", x_mm=1.0, y_mm=2.0, z_mm=3.0 + (0.5 if r % 2 else 2.0))
                        for r in range(1, REPS + 1)])
    t = score(pd.concat([c, loc], ignore_index=True), cs)["location"]
    assert len(t) == 1 and np.isclose(t.within_1mm.iloc[0], 0.5) and np.isclose(t.median_mm.iloc[0], 1.25)


def test_paired_comparison(cs):
    a = _claims("a", lambda t, g: t + g.normal(0, 0.2), 0.4, _by_truth, se=0.2, seed=1)
    b = a.assign(arm="b", lo_db=a.estimate_db - 0.3, hi_db=a.estimate_db + 0.3)     # same estimates, narrower
    cmp = score(pd.concat([a, b]), cs, compare=[("a", "b")])["compare_a_vs_b"]
    w = cmp[cmp.metric == "width_db"]
    assert np.allclose(w["diff"], -0.2) and np.allclose(w.lo, -0.2) and np.allclose(w.hi, -0.2)
    e = cmp[cmp.metric == "abs_err_db"]
    assert np.allclose(e["diff"], 0)


def test_n80_recovers_a_known_curve():
    n = np.array([8, 12, 16, 24, 30.0])
    b0, b1 = -6.0, 2.5
    p = 1 / (1 + np.exp(-(b0 + b1 * np.log(n))))
    v, flag = n80(n, p * 1e6, np.full(5, 1e6))
    assert abs(v - np.exp((np.log(4) - b0) / b1)) < 1e-6 and flag == ""
    assert np.isnan(n80(n, np.full(5, 9), np.full(5, 10))[0])                 # already >= 0.8 at the smallest N


@pytest.mark.parametrize("change, msg", [
    (lambda c: c.drop(columns="decision_matched"), "lacks column"),
    (lambda c: c.assign(decision="effect"), "decision must be one of"),
    (lambda c: c.assign(cohort="nope/rep001"), "not in the cohort set"),
    (lambda c: c.assign(unit="r9"), "unknown unit"),
    (lambda c: pd.concat([c, c]), "duplicated"),
    (lambda c: c.assign(lo_db=c.hi_db + 1), "lo_db > hi_db"),
])
def test_contract_errors(cs, change, msg):
    c = _claims("x", lambda t, g: t, 0.1, _by_truth, ns=(8,), cells=("F_1.0",))
    with pytest.raises(BenchmarkError, match=msg):
        attach_truth(change(c), cs)


def test_engine_adapter():
    r = pd.DataFrame({"hypothesis": "effect", "status": "ok", "prior_set": "W", "band": ["Low Gamma", "Theta"],
                      "spatial": "ROI-average", "estimate": [0.5, -0.4], "hdi_lo": [0.2, -0.9], "hdi_hi": [0.8, 0.1],
                      "decision": ["effect", "undecided"], "p_direction": [0.999, 0.9], "log10_bf10": [1.0, -0.2],
                      "rope_hi": 0.1, "rhat": [1.001, 1.06], "max_rhat_scale": 1.01, "post_sd": [0.15, 0.25]})
    c = claims_from_engine(r, "F_1.0/rep001", 8, {"Low Gamma": 2.0, "Theta": 4.0})
    assert c.estimate_db.tolist() == [1.0, -1.6] and c.lo_db.tolist() == [0.4, -3.6] and c.se_db.tolist() == [0.3, 1.0]
    assert c.decision.tolist() == ["effect_pos", "undecided"]
    assert c.decision_matched.tolist() == ["excludes_0", "includes_0"]
    assert c.rope_db.tolist() == [0.2, 0.4] and c.converged.tolist() == [True, False]


BENCH = os.environ.get("SA_BAYES_BENCH")      # the bayes-sl workspace (benchmark/cohorts/v1 and scratch/ee3)


@pytest.mark.skipif(not BENCH, reason="SA_BAYES_BENCH not set (cohorts and EE3 pilot fits live in bayes-sl)")
def test_reproduces_the_ee3_pilot_scores():
    """BM2's acceptance: the engine's EE3 pilot fits, scored by this code against benchmark v1, give EE3's tables
    (scripts/ee3_score.py -> scratch/ee3/score_all), with EE3's leak rule (each row's ROPE in the cohort's unit)."""
    ws = Path(BENCH)
    cs = CohortSet.load(os.environ.get("SA_BAYES_BENCH_COHORTS", ws / "benchmark/cohorts/v1"))
    parts = []
    for f in sorted((ws / "scratch/ee3/fits").glob("*/rep*/N*_*.csv")):
        r = pd.read_csv(f)
        parts.append(claims_from_engine(r, f"{r.cell.iloc[0]}/rep{int(r.rep.iloc[0]):03d}", int(r.n_per_group.iloc[0]),
                                        dict(zip(r.band, r.unit_db)), runtime_s=float(r.fit_seconds.iloc[0])))
    tabs = score(pd.concat(parts, ignore_index=True), cs, leak_rule="row_rope")
    ref = ws / "scratch/ee3/score_all"
    k = ["cell", "prior_set", "n_per_group", "band"]

    def same(ours, theirs, key, cols):
        m = theirs.merge(ours, on=key, suffixes=("_ee3", ""))
        assert len(m) == len(theirs), f"rows: {len(m)} of {len(theirs)}"
        for col in cols:
            a, b = m[f"{col}_ee3"].astype(float), m[col].astype(float)
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), col

    same(tabs["noise"], pd.read_csv(ref / "noise.csv"), k,
         ["reps", "claim_effect", "claim_dir", "claim_bf3", "any_roi_claimed", "roi_claim_rate"])
    same(tabs["calibration"], pd.read_csv(ref / "calibration.csv"), k,
         ["truth_avg_db", "est_avg_db", "bias_avg_db", "cov_avg", "sd_z_avg", "vi_avg", "cov_roi_pooled",
          "cov_roi_median", "cov_roi_min", "sd_z_roi", "vi_roi_median"])
    at = tabs["attribution"].query("rule == 'own'")
    same(at, pd.read_csv(ref / "attribution.csv"), ["cell", "prior_set", "n_per_group"],
         ["planted", "leaked_real", "elsewhere", "wrong_sign", "n_claims", "reps"])
    de = tabs["design"].query("rule == 'own'")
    theirs = pd.read_csv(ref / "design.csv")
    same(de[de.n_per_group != "N80"].astype({"n_per_group": int}),
         theirs[theirs.n_per_group != "N80"].astype({"n_per_group": int}),
         ["cell", "prior_set", "band", "question", "n_per_group"], ["rate", "n_tests"])


def test_cli_scores_and_reports_contract_errors(cs, tmp_path):
    import subprocess
    c = _claims("honest", lambda t, g: t, 0.2, _by_truth, ns=(8,))
    c.to_csv(tmp_path / "a.csv", index=False)
    c.assign(arm="wide", lo_db=c.lo_db - 0.1, hi_db=c.hi_db + 0.1).to_csv(tmp_path / "b.csv", index=False)
    cmd = ["uv", "run", "--no-sync", "source-analytics", "benchmark", "score", "--cohorts", str(cs.root), "--claims",
           str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), "--out", str(tmp_path / "out"), "--compare", "honest:wide"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    w = pd.read_csv(tmp_path / "out" / "compare_honest_vs_wide.csv").query("metric == 'width_db'")
    assert np.allclose(w["diff"], 0.2) and "== calibration ==" in (tmp_path / "out" / "summary.txt").read_text()
    c.assign(decision="maybe").to_csv(tmp_path / "bad.csv", index=False)
    r = subprocess.run(cmd[:9] + [str(tmp_path / "bad.csv"), "--out", str(tmp_path / "o2")], capture_output=True, text=True)
    assert r.returncode == 1 and "decision must be one of" in r.stdout
