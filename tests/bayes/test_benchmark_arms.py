"""D10 step 1 (M1): region probabilities from draws, the gain over the no-difference twin, and the arm runner's
bookkeeping, on synthetic inputs with known answers; with SA_BAYES_BENCH (the bayes-sl workspace), parity with
bayes-sl's stored L4 probabilities and its R2 and L6 scorecards, from its stored fits (to 1e-9)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from source_analytics.bayes import arms
from source_analytics.bayes.benchmark import BenchmarkError, CohortSet
from source_analytics.bayes.benchmark_location import (boot_mean, gain_table, read_hierarchy, score_location,
                                                       twin_cell, twin_pairs)
from source_analytics.bayes.benchmark_probs import (build, coarse_sets, coarse_statements, largest_difference,
                                                    probs_from_claims)

H = pd.DataFrame({"roi": ["Aud_L", "Aud_R", "Thal", "Cb"], "compartment": ["cortex", "cortex", "deep", "deep"],
                  "hemisphere": ["left", "right", "midline", "midline"], "group": ["ctx_L", "ctx_R", "sub", "hind"]})
ROIS = H.roi.tolist()
REPS = 20


# --------------------------------------------------------------------------------------- region probabilities
def test_largest_difference_counts_draws_and_none():
    D = pd.DataFrame({"Aud_L": [0.5, 0.05, -0.9, 0.0], "Aud_R": [0.2, 0.02, 0.1, 0.0], "Thal": [0.1, 0.0, 0.3, 0.0]})
    p = largest_difference(D, rope=0.1)
    assert p == pytest.approx({"Aud_L": 0.5, "none": 0.5})      # |-0.9| counts; two draws below the ROPE
    assert "Aud_R" not in p                                     # regions with no draw are left out


def test_tied_draws_are_shared_not_given_to_the_first_column():
    """A singular lme4 fit gives every ROI the same contrast: argmax would put it all on the first column."""
    D = pd.DataFrame({r: [0.4] * 10 for r in ROIS})
    p = largest_difference(D)
    assert p == pytest.approx({r: 0.25 for r in ROIS})


def test_coarse_statements_and_ties_count_half():
    sets = coarse_sets(H)
    assert sets["right_gt_left"] == (["Aud_R"], ["Aud_L"])     # midline left out
    D = pd.DataFrame({"Aud_L": [0.1, 0.1, 1.0], "Aud_R": [0.1, 0.1, 1.0], "Thal": [1.0, 0.0, 0.0], "Cb": [1.0, 0.0, 0.0]})
    s = coarse_statements(D, sets)
    assert s["cortex_gt_deep"] == pytest.approx(2 / 3)          # deep wins draw 1; cortex draws 2 and 3
    assert s["right_gt_left"] == pytest.approx(0.5)             # every draw ties


def test_hierarchy_without_the_coarse_sets_is_refused():
    with pytest.raises(BenchmarkError, match="no 'cortex'"):
        coarse_sets(H.assign(compartment="deep"))


def test_f0_draws_are_seeded_by_cohort_and_scaled_by_the_engine_unit():
    c = pd.DataFrame({"cohort": "A_1.0/rep001", "n_per_group": 8, "arm": "F0", "band": "G", "unit": ROIS,
                      "estimate_db": [4.0, 0.0, 0.0, 0.0], "se_db": [0.01] * 4})
    p1 = probs_from_claims(c, {("A_1.0/rep001", 8): 2.0}, "G")
    p2 = probs_from_claims(c, {("A_1.0/rep001", 8): 2.0}, "G")
    pd.testing.assert_frame_equal(p1, p2)
    assert p1.set_index("region").prob.to_dict() == pytest.approx({"Aud_L": 1.0})
    with pytest.raises(BenchmarkError, match="no unit"):
        probs_from_claims(c, {}, "G")
    with pytest.raises(BenchmarkError, match="se_db"):
        probs_from_claims(c.assign(se_db=np.nan), {("A_1.0/rep001", 8): 2.0}, "G")


def _write_draws(root: Path, cohort: str, n: int, tag: str, D: pd.DataFrame, band="G"):
    f = root / "fits" / cohort / f"N{n:02d}_{tag}_draws.csv.gz"
    f.parent.mkdir(parents=True, exist_ok=True)
    D.rename(columns=lambda r: f"{band}|{r}").assign(**{f"{band}|ROI-average": 0.0}).to_csv(f, index=False)


def test_build_reads_the_runner_layout_in_bayes_sl_order(tmp_path):
    rng = np.random.default_rng(1)
    _write_draws(tmp_path, "A_1.0/rep001", 8, "W", pd.DataFrame(rng.normal(0, 1, (50, 4)), columns=ROIS))
    _write_draws(tmp_path, "A_1.0/rep001", 8, "F1", pd.DataFrame(rng.normal(0, 1, (50, 4)), columns=ROIS))
    out = build({"F1": tmp_path, "engine/W": tmp_path}, "G", hierarchy=H)
    assert out["probs"].arm.drop_duplicates().tolist() == ["engine", "F1"]
    assert out["probs"].groupby("arm").prob.sum().to_numpy() == pytest.approx([1, 1])
    assert set(out["contrasts"].statement) == {"cortex_gt_deep", "right_gt_left"}
    with pytest.raises(BenchmarkError, match="unknown arm"):
        build({"F2": tmp_path}, "G")


# --------------------------------------------------------------------------------------- gain over the twin
@pytest.fixture
def cs(tmp_path):
    cells = {"A_1.0": {"region": "A", "effect_sd": 1.0, "planted_rois": ["Aud_R"], "spacing_mm": 1,
                       "shared_rois": ["Thal"]},
             "A_nodiff": {"region": "A", "effect_sd": 0, "planted_rois": ["Aud_R"], "spacing_mm": 1,
                          "shared_rois": ["Thal"]},
             "B_1.0": {"region": "B", "effect_sd": 1.0, "planted_rois": ["Thal"], "spacing_mm": 2, "twin": "B_twin"},
             "B_twin": {"region": "B", "effect_sd": 0, "planted_rois": ["Thal"], "spacing_mm": 2},
             "noise": {"region": None, "effect_sd": 0.0, "planted_rois": []}}
    cohorts = {f"{c}/rep{r:03d}": {} for c in cells for r in range(1, REPS + 1)}
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"cells": cells, "cohorts": cohorts, "plant_band": "G",
                                                        "rois": ROIS, "band_units_db": {"G": 1.0}}))
    return CohortSet.load(tmp_path)


def probs(cells, fn, arm="m"):
    rows = []
    for c in cells:
        for r in range(1, REPS + 1):
            for region, p in fn(c, r).items():
                rows.append(dict(cohort=f"{c}/rep{r:03d}", n_per_group=8, arm=arm, prior_set="", band="G", level="roi",
                                 region=region, prob=p))
    return pd.DataFrame(rows)


def finds_source(c, r):
    """Names the planted source whether or not it differs: all its hits are the twin's too."""
    return {"Aud_R" if c.startswith("A") else "Thal": 1.0} if c != "noise" else {"none": 1.0}


def finds_difference(c, r):
    """Names the source only where it differs; with no difference, says none."""
    if c in ("A_1.0", "B_1.0"):
        return {"Aud_R" if c == "A_1.0" else "Thal": 1.0}
    return {"none": 1.0}


def test_twin_names(cs):
    assert twin_cell(cs, "A_1.0") == "A_nodiff"
    assert twin_cell(cs, "B_1.0") == "B_twin"                  # the manifest's own twin wins


def test_a_method_that_finds_the_source_gains_nothing(cs):
    cells = list(cs.cells)
    t = score_location(probs(cells, finds_source), cs, H, question="difference", twins=True, group_by=["spacing_mm"])
    g = t["gain"][(t["gain"].level == "roi") & (t["gain"].grouping == "pooled")].iloc[0]
    assert g.accuracy == 1 and g.twin == 1 and g.gain == 0 and g.gain_lo == g.gain_hi == 0
    assert g.reps == 2 * REPS
    assert set(t["gain"].grouping) == {"cell", "spacing_mm", "pooled"}


def test_a_method_that_finds_the_difference_gains_everything(cs):
    t = score_location(probs(list(cs.cells), finds_difference), cs, H, question="difference", twins=True)
    g = t["gain"][(t["gain"].level == "roi") & (t["gain"].grouping == "pooled")].iloc[0]
    assert g.accuracy == 1 and g.twin == 0 and g.gain == 1
    a = t["gain"][(t["gain"].level == "roi") & (t["gain"].value == "A_1.0")].iloc[0]
    assert a.names_shared == 0 and a.twin_names_shared == 0      # A's shared ROI is Thal; B has none
    assert np.isnan(t["gain"][(t["gain"].level == "roi") & (t["gain"].value == "B_1.0")].iloc[0].names_shared)


def test_twins_in_another_cohort_set(cs, tmp_path):
    other = tmp_path / "twins"
    other.mkdir()
    m = json.loads((cs.root / "MANIFEST.json").read_text())
    m["cells"] = {k: v for k, v in m["cells"].items() if k in ("A_nodiff", "B_twin")}
    m["cohorts"] = {k: v for k, v in m["cohorts"].items() if k.split("/")[0] in m["cells"]}
    (other / "MANIFEST.json").write_text(json.dumps(m))
    tw = CohortSet.load(other)
    same = score_location(probs(list(cs.cells), finds_source), cs, H, question="difference", twins=True)
    apart = score_location(probs(["A_1.0", "B_1.0", "noise"], finds_source), cs, H, question="difference",
                           twin_probs=probs(["A_nodiff", "B_twin"], finds_source), twin_cs=tw)
    pd.testing.assert_frame_equal(same["gain"], apart["gain"])


def test_reps_select_and_the_bootstrap_is_reproducible(cs):
    p = probs(list(cs.cells), lambda c, r: finds_difference(c, r) if r % 2 else finds_source(c, r))
    a = score_location(p, cs, H, question="difference", twins=True)["gain"]
    b = score_location(p, cs, H, question="difference", twins=True)["gain"]
    pd.testing.assert_frame_equal(a, b)
    odd = score_location(p, cs, H, question="difference", twins=True, reps=range(1, REPS + 1, 2))["gain"]
    assert (odd[odd.level == "roi"].gain == 1).all()
    x = a[(a.level == "roi") & (a.grouping == "pooled")].iloc[0]
    assert x.gain == pytest.approx(0.5) and x.gain_lo < 0.5 < x.gain_hi


def test_missing_twins_and_twins_with_a_difference_are_refused(cs):
    p = probs(list(cs.cells), finds_source)
    t = score_location(p, cs, H, question="difference")["tables"]
    with pytest.raises(BenchmarkError, match="no twin row"):
        twin_pairs(t, cs, t[~((t.cell == "A_nodiff") & (t.rep == 3))], cs)
    with pytest.raises(BenchmarkError, match="no study cell"):
        twin_pairs(t[~t.cell.isin(["A_nodiff", "B_twin"])], cs)
    bad = json.loads((cs.root / "MANIFEST.json").read_text())
    bad["cells"]["A_nodiff"]["effect_sd"] = 0.5
    (cs.root / "MANIFEST.json").write_text(json.dumps(bad))
    with pytest.raises(BenchmarkError, match="has a difference"):
        twin_pairs(t, CohortSet.load(cs.root))
    with pytest.raises(BenchmarkError, match="grouping field"):
        gain_table(twin_pairs(t, cs), ["no_such_field"])


def test_boot_mean_resamples_reps_within_cells():
    d = pd.DataFrame({"cell": ["a"] * 4 + ["b"] * 4, "rep": [1, 1, 2, 2] * 2, "x": [1, 1, 1, 1, 0, 0, 0, 0]})
    assert boot_mean(d, "x", np.random.default_rng(0)) == (0.5, 0.5)    # each cell keeps its share of rows


# --------------------------------------------------------------------------------------- the runner's bookkeeping
@pytest.fixture
def runnable(tmp_path):
    cells = {"A_1.0": {"region": "A", "effect_sd": 1.0, "planted_rois": ["Aud_R"]}}
    cohorts = {}
    for r in (1, 2, 901):
        c = f"A_1.0/rep{r:03d}"
        cohorts[c] = {}
        (tmp_path / c).mkdir(parents=True)
        (tmp_path / c / "meta.json").write_text(json.dumps({"n_nested": [8, 12]}))
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"cells": cells, "cohorts": cohorts, "plant_band": "Low Gamma",
                                                        "control_band": "Theta", "rois": ROIS,
                                                        "band_units_db": {"Low Gamma": 1.0}}))
    return CohortSet.load(tmp_path)


def test_parse_reps():
    assert arms.parse_reps("1:3,901") == [1, 2, 3, 901]
    assert arms.parse_reps(None) is None


def test_todo_selects_cells_reps_and_n(runnable):
    t = arms.todo(runnable)
    assert len(t) == 6 and t.n_per_group.tolist()[:2] == [8, 12]
    t = arms.todo(runnable, reps=[1, 2], ns=[12])
    assert t.cohort.tolist() == ["A_1.0/rep001", "A_1.0/rep002"] and set(t.n_per_group) == {12}
    assert arms.default_bands(runnable) == ["Low Gamma", "Theta"]
    with pytest.raises(BenchmarkError, match="not in the cohort set"):
        arms.todo(runnable, cells=["B"])
    with pytest.raises(BenchmarkError, match="nothing to run"):
        arms.todo(runnable, reps=[5])


def test_a_failing_part_is_retried_then_reported(tmp_path):
    """The retry for cmdstan CSVs that vanish: each try gets the part's own TMPDIR; a part that keeps failing stops."""
    flag = tmp_path / "tries"
    script = tmp_path / "flaky.py"
    script.write_text("import os, sys, pathlib\n"
                      f"f = pathlib.Path({str(flag)!r}); n = int(f.read_text()) + 1 if f.exists() else 1\n"
                      "f.write_text(str(n)); print(os.environ['TMPDIR']); sys.exit(0 if n >= 2 else 1)\n")
    assert arms._run_part([sys.executable, str(script)], tmp_path / "log", tmp_path / "tmp")
    assert flag.read_text() == "2" and str(tmp_path / "tmp") in (tmp_path / "log").read_text()
    with pytest.raises(RuntimeError, match=r"part\(s\) \[1\] failed 2 times"):
        arms._run_parts([sys.executable, "-c", "import sys; sys.exit(1)"], tmp_path, "x", 1, retries=2)


def test_engine_set_i_needs_its_entries(runnable, tmp_path):
    with pytest.raises(BenchmarkError, match="needs --informative"):
        arms.run_engine(runnable, tmp_path, arms.todo(runnable), prior_set="I")


def test_f0_claims_from_the_roi_psd_tables(runnable, tmp_path):
    """F0's own rule (q per ROI, p for the ROI-average, signed) and the matched rule, from roi_psd's two tables."""
    wd = tmp_path / "A_1.0/rep001/N08"
    (wd / "tbl").mkdir(parents=True)
    (wd / "runtime.json").write_text(json.dumps({"runtime_s": 2.0}))
    base = dict(hypothesis="effect", kind="contrast", dv="absolute", group_a="B", group_b="A", band="Low Gamma")
    roi = pd.DataFrame([dict(base, spatial=r, estimate=e, estimate_lcl=e - 1, estimate_ucl=e + 1, SE=0.5, p_value=p,
                             q_value=q, fdr_family="scope=band method=BH (Low Gamma)")
                        for r, e, p, q in zip(ROIS, [2.0, -2.0, 0.5, 0.0], [.001, .001, .2, .9], [.004, .004, .3, .9])])
    glob = pd.DataFrame([dict(base, spatial="all", estimate=0.4, estimate_lcl=-0.6, estimate_ucl=1.4, SE=0.5,
                              p_value=0.03, q_value=np.nan)])
    roi.to_csv(wd / "tbl" / "roi_psd_hypotheses.csv", index=False)
    glob.to_csv(wd / "tbl" / "roi_psd_global_hypotheses.csv", index=False)
    c = arms.claims_f0(tmp_path, runnable, pd.DataFrame({"cohort": ["A_1.0/rep001"], "n_per_group": [8]}))
    assert c.decision.tolist() == ["effect_pos", "effect_neg", "undecided", "undecided", "effect_pos"]
    assert c.decision_matched.tolist() == ["excludes_0", "excludes_0", "includes_0", "includes_0", "includes_0"]
    assert c.unit.iloc[-1] == "ROI-average" and (c.runtime_s == 2.0).all()
    with pytest.raises(BenchmarkError, match="no F0 run"):
        arms.claims_f0(tmp_path, runnable, pd.DataFrame({"cohort": ["A_1.0/rep002"], "n_per_group": [8]}))


def test_r_scripts_parse():
    from source_analytics.bayes.run import r_dir
    for f in ("bench_fit.R", "bench_lmm.R"):
        r = subprocess.run(["Rscript", "-e", f"invisible(parse(file='{r_dir() / 'bayes' / f}'))"], capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------------------- parity with bayes-sl
BENCH = os.environ.get("SA_BAYES_BENCH")
bench = pytest.mark.skipif(not BENCH, reason="SA_BAYES_BENCH not set (bayes-sl's stored fits and scores)")


def _read(f):
    x = pd.read_csv(f, keep_default_na=False, na_values=[""], low_memory=False)
    if "prior_set" in x:
        x["prior_set"] = x.prior_set.fillna("").astype(str)
    return x


def _same(ref: pd.DataFrame, new: pd.DataFrame, on: list[str], cols: list[str], tol: float = 1e-9):
    m = ref.merge(new, on=on, how="outer", indicator=True, suffixes=("", "_new"))
    assert (m._merge == "both").all(), m._merge.value_counts().to_dict()
    for c in cols:
        a, b = m[c].astype(float), m[c + "_new"].astype(float)
        assert (a.isna() == b.isna()).all(), c
        if a.notna().any():                         # all NaN on both sides (v1_l5's unit_db: no difference) is equal
            assert np.nanmax(np.abs(a - b)) <= tol, (c, np.nanmax(np.abs(a - b)))


def _probs(ws: Path, s: str) -> dict:
    b1 = ws / "scratch" / ("bm_b1" if s == "v1" else f"bm_b1_{s}") / "fits"
    fits = {"engine/W": b1, "engine/I": b1}
    for a, d in (("F1", ws / "scratch/f1" / s / "fits"), ("F1c", ws / "scratch/f1c" / s / "fits")):
        if d.exists():
            fits[a] = d
    return build(fits, "Low Gamma", f0_claims=_read(ws / f"benchmark/claims/{s}/F0/claims.csv"), unit_fits=b1,
                 hierarchy=read_hierarchy(ws / "benchmark/regions_v1.csv"), cohorts_root=ws / "benchmark/cohorts" / s)


@bench
@pytest.mark.parametrize("s", ["r2_v1", "v1_l5", "v1"])
def test_reproduces_bayes_sl_location_probs(s):
    """bayes-sl scripts/bm_l4.py and bm_l4_contrasts.py: every arm's tables, from its stored draws and F0 claims."""
    ws = Path(BENCH)
    out = _probs(ws, s)
    ref = ws / "benchmark/location_probs" / s / "L4"
    _same(_read(ref / "probs.csv"), out["probs"],
          ["cohort", "n_per_group", "arm", "prior_set", "band", "level", "region"], ["prob"])
    if (ref / "contrasts.csv").exists():
        _same(_read(ref / "contrasts.csv"), out["contrasts"], ["cohort", "n_per_group", "arm", "prior_set", "statement"],
              ["prob", "truth_first", "truth_second", "unit_db"])


@bench
def test_reproduces_the_r2_scorecard():
    """bayes-sl R2 (scripts/r2_scorecard.py, accuracy.csv): our probabilities, scored here, give the stored L4 scorer
    tables; with the stored field-model (L3) tables beside them (the bootstrap draws in arm order), the gain table is
    R2's accuracy table, test reps only, to 1e-9."""
    ws = Path(BENCH)
    S = ws / "benchmark/location_scores"
    cs = CohortSet.load(ws / "benchmark/cohorts/r2_v1")
    h = read_hierarchy(ws / "benchmark/regions_v1.csv")
    t4 = score_location(_probs(ws, "r2_v1")["probs"], cs, h, question="difference")["tables"]
    _same(_read(S / "r2_v1_L4/tables.csv"), t4, ["arm", "prior_set", "cohort", "n_per_group", "band", "level"],
          ["p_true", "top_prob", "hit", "p_source", "top_at_source", "entropy_bits"])
    m = twin_pairs(pd.concat([t4, _read(S / "r2_v1_L3/tables.csv")], ignore_index=True), cs)
    g = gain_table(m[m.rep < 900], ["spacing_mm", "pair", "shared_ratio"])
    g["arm"] = g.arm + np.where(g.prior_set != "", "/" + g.prior_set, "")
    g["grouping"] = g.grouping.replace({"cell": "base", "shared_ratio": "ratio"})
    g["value"] = np.where(g.grouping == "base", g.value.astype(str).str.replace(r"_[\d.]+$", "", regex=True),
                          g.value.astype(str))
    g = g.rename(columns={"gain": "excess", "gain_lo": "excess_lo", "gain_hi": "excess_hi"})
    ref = _read(S / "R2/accuracy.csv").assign(value=lambda x: x.value.astype(str))
    _same(ref, g.drop(columns=["prior_set", "band"]), ["arm", "level", "grouping", "value"],
          ["rows", "reps", "accuracy", "twin", "excess", "excess_lo", "excess_hi", "names_shared", "twin_names_shared"])


@bench
def test_reproduces_the_l6_final_scorecard():
    """bayes-sl L6 final (scripts/l6_scorecard.py: accuracy.csv and pairs.csv): accuracy, the no-difference baseline
    and the gain with its interval for every arm with twins (engine/I has none: no L5 fits), against the planted region
    and the path-truth top; then the paired arm contrasts. The generator runs on in L6's order, so the stored
    field-model tables are read beside the readout arms'."""
    ws = Path(BENCH)
    S = ws / "benchmark/location_scores"
    fam = ("L4", "L3", "FMmass")
    bases = ["Frontal_Anterior", "Somatosensory_R", "Hippocampus_Ant_R", "Thalamus", "cortex"]
    runs = {"planted": ("benchmark/cohorts/v1", "benchmark/cohorts/v1_l5", ""),
            "pathtop": ("benchmark/location_truths/v1_pathtop", "benchmark/location_truths/v1_l5_pathtop", "_pathtop")}
    rng = np.random.default_rng(20261104)
    acc, pairs = [], {}
    for truth, (c1, c5, sfx) in runs.items():
        cs, cs5 = CohortSet.load(ws / c1), CohortSet.load(ws / c5)
        t = pd.concat([_read(S / f"v1_{f}{sfx}/tables.csv") for f in fam], ignore_index=True)
        tl = pd.concat([_read(S / f"v1_l5_{f}{sfx}/tables.csv") for f in fam], ignore_index=True)
        t = t[t.cell.str.endswith("_1.0") & pd.Series(list(zip(t.arm, t.prior_set)), index=t.index).isin(
            set(zip(tl.arm, tl.prior_set)))]
        m = twin_pairs(t, cs, tl, cs5)
        m["arm2"] = m.arm + np.where(m.prior_set != "", "/" + m.prior_set, "")
        m["base"] = m.cell.str.replace(r"_[\d.]+$", "", regex=True)
        pairs[truth] = m
        for (arm, lv), x in m.groupby(["arm2", "level"]):
            for b in bases + ["pooled"]:
                y = x[x.base == b] if b != "pooled" else x[x.base != "cortex"] if truth == "planted" else x
                if y.empty:
                    continue
                lo, hi = boot_mean(y, "gain", rng, within="base")
                acc.append(dict(truth=truth, arm=arm, level=lv, cell=b, tables=len(y), accuracy=y.hit.mean(),
                                l5_baseline=y.twin_hit.mean(), excess=y.gain.mean(), excess_lo=lo, excess_hi=hi))
    ref = _read(S / "L6_final/accuracy.csv")
    ref = ref[ref.truth.isin(list(runs)) & ref.l5_baseline.notna()]
    _same(ref, pd.DataFrame(acc), ["truth", "arm", "level", "cell"],
          ["tables", "accuracy", "l5_baseline", "excess", "excess_lo", "excess_hi"])
    # pairs.csv: arm A - arm B on hit and gain, paired by cell x rep x N (the generator continues from accuracy)
    contrasts = [("engine/W", "F1"), ("engine/W", "F1c"), ("engine/W", "F0"), ("F1", "F0"), ("F1c", "F1"),
                 ("FE/W", "engine/W"), ("FF", "F0"), ("FE/W", "F1"), ("FE/W", "FF"), ("FE/I", "FE/W"),
                 ("FE/W", "FMmass"), ("FF", "FMmass")]
    rows = []
    for truth, m in pairs.items():
        if truth == "planted":
            m = m[m.base != "cortex"]
        for lv in ("roi", "compartment", "hemisphere"):
            w = m[m.level == lv].pivot_table(index=["base", "rep", "n_per_group"], columns="arm2", values=["hit", "gain"])
            for a, b in contrasts:
                if a not in w["hit"] or b not in w["hit"]:
                    continue
                for v, name in (("hit", "hit"), ("gain", "excess")):
                    d = (w[v][a] - w[v][b]).rename("d").reset_index()
                    lo, hi = boot_mean(d, "d", rng, within="base")
                    rows.append(dict(truth=truth, level=lv, contrast=f"{a} - {b}", measure=name, diff=d.d.mean(),
                                     lo=lo, hi=hi, n=len(d)))
    _same(_read(S / "L6_final/pairs.csv"), pd.DataFrame(rows), ["truth", "level", "contrast", "measure"],
          ["diff", "lo", "hi", "n"])
