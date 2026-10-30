"""The location scorecard on synthetic methods with known answers (bayes-sl location scorecard plan, L1)."""
import json

import numpy as np
import pandas as pd
import pytest

from source_analytics.bayes.benchmark import BenchmarkError, CohortSet
from source_analytics.bayes.benchmark_location import NONE, expand_levels, score_location, validate_probs

H = pd.DataFrame({"roi": ["Aud_L", "Aud_R", "Thal", "Cb"], "compartment": ["cortex", "cortex", "deep", "deep"],
                  "hemisphere": ["left", "right", "midline", "midline"], "group": ["ctx_L", "ctx_R", "sub", "hind"]})
ROIS = H.roi.tolist()
TRUE = {"noise": None, "A": "Aud_R", "B": "Thal"}
REPS = 40


@pytest.fixture
def cs(tmp_path):
    cells = {"noise": {"region": None, "effect_sd": 0.0, "planted_rois": []},
             "A": {"region": "A", "effect_sd": 1.0, "planted_rois": ["Aud_R"]},
             "B": {"region": "B", "effect_sd": 1.0, "planted_rois": ["Thal"]},
             "spread": {"region": "s", "effect_sd": 1.0, "planted_rois": ["Aud_L", "Aud_R", "Thal", "Cb"]}}
    cohorts = {f"{c}/rep{r:03d}": {} for c in cells for r in range(1, REPS + 1)}
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"cells": cells, "cohorts": cohorts, "plant_band": "G",
                                                        "rois": ROIS, "band_units_db": {"G": 1.0}}))
    return CohortSet.load(tmp_path)


def table(arm, fn, cells=("noise", "A", "B")):
    rows = []
    for c in cells:
        for r in range(1, REPS + 1):
            for region, p in fn(c, r).items():
                rows.append(dict(cohort=f"{c}/rep{r:03d}", n_per_group=8, arm=arm, prior_set="", band="G", level="roi",
                                 region=region, prob=p))
    return pd.DataFrame(rows)


def perfect(c, r):
    return {TRUE[c]: 1.0} if TRUE[c] else {NONE: 1.0}


def uniform(c, r):
    return {k: 0.25 for k in ROIS}


def always_thal(c, r):
    return {"Thal": 1.0}


def overconfident(c, r):
    """All its mass on a region that is the truth a quarter of the time."""
    return {ROIS[(r + (ROIS.index(TRUE[c]) if TRUE[c] else 0)) % 4]: 1.0}


def honest(c, r):
    """Says 0.7 on its top region, and that is right 70% of the time (r mod 10 < 7); nothing planted: none."""
    if not TRUE[c]:
        return {NONE: 1.0}
    t = TRUE[c]
    top = t if r % 10 < 7 else next(k for k in ROIS if k != t)
    return {top: 0.7, **{k: 0.1 for k in ROIS if k != top}}


def one(tabs, name, arm, **kw):
    x = tabs[name]
    x = x[x.arm == arm]
    for k, v in kw.items():
        x = x[x[k] == v]
    return x


def test_perfect_method(cs):
    t = score_location(table("p", perfect), cs, H)
    x = one(t, "location", "p", level="roi")
    assert (x.accuracy == 1).all() and (x.p_true == 1).all() and (x.set90_size == 1).all()
    assert (x.bits_gained > 9).all()          # against the floor: its no-plant cohorts put all mass on "none"
    assert (one(t, "noise", "p", level="roi").p_none == 1).all()
    assert set(one(t, "reliable", "p").finest_reliable) == {"roi"}
    assert one(t, "calibration_error", "p", level="roi").calibration_error.iloc[0] == 0


def test_uniform_covers_but_gains_nothing(cs):
    t = score_location(table("u", uniform), cs, H)
    x = one(t, "location", "u", level="roi")
    assert (x.set90_coverage == 1).all() and (x.bits_gained.abs() < 1e-9).all() and (x.set90_size == 4).all()
    r = one(t, "reliable", "u", level="roi")
    assert r.coverage_ok.all() and not r.info_ok.any() and r.noise_ok.all() and r.finest_reliable.isna().all()


def test_always_the_same_place_is_not_information(cs):
    t = score_location(table("c", always_thal), cs, H)
    b = one(t, "location", "c", cell="B", level="roi").iloc[0]
    assert b.accuracy == 1 and abs(b.bits_gained) < 1e-9 and b.accuracy_baseline == 1
    r = one(t, "reliable", "c", level="roi")
    assert not r.reliable.any() and not r.noise_ok.any()   # confident with nothing planted


def test_overconfident_method_fails_calibration(cs):
    t = score_location(table("o", overconfident), cs, H)
    x = one(t, "location", "o", level="roi")
    assert (x.overconfidence > 0.7).all() and (x.set90_coverage < 0.3).all()
    assert one(t, "calibration_error", "o", level="roi").calibration_error.iloc[0] > 0.3
    assert not one(t, "reliable", "o", level="roi").coverage_ok.any()


def test_honest_method_is_calibrated_at_its_stated_confidence(cs):
    t = score_location(table("h", honest), cs, H)
    x = one(t, "location", "h", level="roi")
    assert np.allclose(x.top_prob, 0.7) and np.allclose(x.accuracy, 0.7)
    assert np.allclose(x.overconfidence, 0, atol=1e-9)
    # stated 0.7 on the top region is right 70% of the time
    r = one(t, "reliability", "h", level="roi")
    assert abs(r[r.bin_lo == 0.7].observed.iloc[0] - 0.7) < 0.05
    assert one(t, "calibration_error", "h", level="roi").calibration_error.iloc[0] < 0.1


def test_levels_sum_up_and_coarse_levels_can_be_right_when_roi_is_wrong(cs):
    # always the wrong hemisphere's mirror: wrong at ROI level, right at compartment level
    def mirror(c, r):
        return {"Aud_L": 1.0} if c == "A" else {"Thal": 1.0} if c == "B" else {NONE: 1.0}
    t = score_location(table("m", mirror), cs, H)
    a = one(t, "location", "m", cell="A")
    assert a[a.level == "roi"].accuracy.iloc[0] == 0 and a[a.level == "hemisphere"].accuracy.iloc[0] == 0
    assert a[a.level == "compartment"].accuracy.iloc[0] == 1
    d = expand_levels(validate_probs(table("u", uniform), cs, H), H)
    assert d[(d.level == "compartment") & (d.cohort == "A/rep001")].set_index("region").prob.to_dict() == {"cortex": 0.5, "deep": 0.5}


def test_a_level_the_method_reports_itself_is_kept(cs):
    p = pd.concat([table("u", uniform), table("u", lambda c, r: {"cortex": 0.9, "deep": 0.1}).assign(level="compartment")])
    d = expand_levels(validate_probs(p, cs, H), H)
    assert d[(d.level == "compartment") & (d.cohort == "A/rep001") & (d.region == "cortex")].prob.tolist() == [0.9]


def test_a_truth_covering_every_region_cannot_test_the_level(cs):
    t = score_location(table("u", uniform, cells=("noise", "A", "spread")), cs, H)
    x = one(t, "location", "u", cell="spread")
    assert not x.informative.any()
    assert one(t, "location", "u", cell="A").informative.all()


def test_bad_tables_are_refused(cs):
    p = table("u", uniform)
    with pytest.raises(BenchmarkError, match="sum to 1"):
        validate_probs(p.assign(prob=0.5), cs, H)
    with pytest.raises(BenchmarkError, match="unknown roi"):
        validate_probs(p.assign(region="Nowhere").drop_duplicates(["cohort", "region"]), cs, H)
    with pytest.raises(BenchmarkError, match="not in the cohort set"):
        validate_probs(p.assign(cohort="Z/rep001"), cs, H)


def test_differences(cs):
    p = table("p", perfect)
    rows = []
    for c in ("noise", "A"):
        for r in range(1, REPS + 1):
            rows.append(dict(cohort=f"{c}/rep{r:03d}", n_per_group=8, arm="p", prior_set="", band="G", region="x",
                             p_differs=0.99 if c == "A" else 0.5, p_negligible=0.0 if c == "A" else 0.96))
    d = score_location(p, cs, H, diff=pd.DataFrame(rows))["differ"].set_index("cell")
    assert d.loc["A", "says_differs"] == 1 and d.loc["noise", "says_differs"] == 0
    assert d.loc["noise", "says_negligible"] == 1 and d.loc["A", "brier"] < 0.001
