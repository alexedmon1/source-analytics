"""The benchmark's scoring contract (bayes-sl ``reports/2026-10-28_stage2_benchmark_plan.md`` section 5; Stage 2).

Every method ("arm") is scored by this code, against the same frozen cohorts and truth, from one table of claims.
The scorer reads files only: the arms' claims tables and the cohort set (its ``MANIFEST.json`` and each cohort's
``meta.json``, ``path_truth.csv``, ``plant.csv``). It knows nothing about an arm's internals or about the atlas.

**Claims table** (one row per arm x cohort x N x band x unit; CSV):

    cohort            "<cell>/rep<NNN>", as in the cohort set
    n_per_group       the nested N (the first N mice of each group by ``order``)
    arm, prior_set    the method, and its prior set ("" when it has none)
    band              band name as in the cohort's band_power.csv
    unit              "ROI-average", an ROI name, or "located" (an arm that returns positions)
    estimate_db       group B - group A, in dB
    lo_db, hi_db      its 95% interval (HDI or CI), in dB
    decision          the arm's own rule: effect_pos | effect_neg | null | undecided
    decision_matched  the matched rule, the same for every arm: excludes_0 | includes_0
    optional: se_db (posterior SD or SE, dB), p_direction, log10_bf10, rope_db, x_mm, y_mm, z_mm (located),
              runtime_s, converged (bool), rhat, rhat_scale

**Truth.** Quantification is scored against the path truth (the readout's expected difference; ROI-average = the
mean of the ROIs' path truths), attribution against the source truth (the planted ROIs; for located rows, the planted
source positions). A claim in an ROI whose path truth is beyond 0.1 x the band's real-data unit is "leaked but real";
a claim against the sign of a real (or planted) difference is "wrong sign";
``leak_rule="row_rope"`` uses each row's ``rope_db`` instead (the EE3 pilot's rule, for the parity check).

**Pass rules** (fixed before any run; end-to-end plan section 5): noise, ROI-average: claim_effect <= 0.05 and
claim_dir <= 0.10; calibration: ROI-average coverage >= 0.85, per ROI median >= 0.85 and min >= 0.75; a per-ROI claim
is usable as a location only if >= 80% of claims fall in the planted ROI(s).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("cohort", "n_per_group", "arm", "prior_set", "band", "unit", "estimate_db", "lo_db", "hi_db", "decision",
            "decision_matched")
OPTIONAL = ("se_db", "p_direction", "log10_bf10", "rope_db", "x_mm", "y_mm", "z_mm", "runtime_s", "converged", "rhat",
            "rhat_scale")
DECISIONS = {"effect_pos", "effect_neg", "null", "undecided"}
MATCHED = {"excludes_0", "includes_0"}
AVG, LOCATED = "ROI-average", "located"
LEAK_ROPE_SD = 0.1
RULES = {"noise_claim_effect": 0.05, "noise_claim_dir": 0.10, "cov_avg": 0.85, "cov_roi_median": 0.85,
         "cov_roi_min": 0.75, "location_share": 0.80}
KEY = ["arm", "prior_set", "cell", "n_per_group", "band"]


class BenchmarkError(ValueError):
    """A claims table or cohort set breaks the contract."""


@dataclass
class CohortSet:
    """A frozen cohort set: ``<root>/MANIFEST.json`` and ``<root>/<cell>/rep<NNN>/``."""

    root: Path
    manifest: dict
    _truth: dict = field(default_factory=dict, repr=False)
    _plant: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, root: Path | str) -> "CohortSet":
        root = Path(root)
        mf = root / "MANIFEST.json"
        if not mf.exists():
            raise BenchmarkError(f"no MANIFEST.json in {root}")
        m = json.loads(mf.read_text())
        for k in ("cells", "cohorts", "plant_band", "rois", "band_units_db"):
            if k not in m:
                raise BenchmarkError(f"{mf}: no '{k}'")
        return cls(root, m)

    @property
    def cells(self) -> dict:
        return self.manifest["cells"]

    def cell(self, cohort: str) -> dict:
        return self.cells[cohort.split("/")[0]]

    def truth(self, cohort: str) -> pd.DataFrame:
        if cohort not in self._truth:
            t = pd.read_csv(self.root / cohort / "path_truth.csv")
            avg = t.groupby("band", as_index=False).truth_db.mean().assign(roi=AVG)
            self._truth[cohort] = pd.concat([t[["roi", "band", "truth_db"]], avg[["roi", "band", "truth_db"]]])
        return self._truth[cohort]

    def plant(self, cohort: str) -> pd.DataFrame:
        if cohort not in self._plant:
            self._plant[cohort] = pd.read_csv(self.root / cohort / "plant.csv")
        return self._plant[cohort]


def read_claims(paths) -> pd.DataFrame:
    """Every claims CSV under the given files or directories (recursively), concatenated."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.csv")) if p.is_dir() else [p]
    if not files:
        raise BenchmarkError(f"no claims tables in {paths}")
    return pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in files], ignore_index=True)


def validate(claims: pd.DataFrame, cs: CohortSet) -> pd.DataFrame:
    miss = [c for c in REQUIRED if c not in claims]
    if miss:
        raise BenchmarkError(f"claims table lacks column(s) {', '.join(miss)}")
    d = claims.copy()
    d["prior_set"] = d.prior_set.fillna("").astype(str)
    bad = sorted(set(d.decision) - DECISIONS)
    if bad:
        raise BenchmarkError(f"decision must be one of {sorted(DECISIONS)}; got {bad}")
    bad = sorted(set(d.decision_matched) - MATCHED)
    if bad:
        raise BenchmarkError(f"decision_matched must be one of {sorted(MATCHED)}; got {bad}")
    unknown = sorted(set(d.cohort) - set(cs.manifest["cohorts"]))
    if unknown:
        raise BenchmarkError(f"cohort(s) not in the cohort set: {unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
    units = set(cs.manifest["rois"]) | {AVG, LOCATED}
    bad = sorted(set(d.unit) - units)
    if bad:
        raise BenchmarkError(f"unknown unit(s) {bad[:5]} (an ROI of the cohort set, '{AVG}' or '{LOCATED}')")
    q = d[d.unit != LOCATED]
    if (q.lo_db > q.hi_db).any():
        raise BenchmarkError("lo_db > hi_db in some rows")
    loc = d[d.unit == LOCATED]
    if len(loc) and not {"x_mm", "y_mm", "z_mm"} <= set(d):
        raise BenchmarkError("located rows need x_mm, y_mm, z_mm")
    dup = d.duplicated(["arm", "prior_set", "cohort", "n_per_group", "band", "unit"]) & (d.unit != LOCATED)
    if dup.any():
        raise BenchmarkError(f"{int(dup.sum())} duplicated claims (arm, prior_set, cohort, N, band, unit)")
    return d


def attach_truth(claims: pd.DataFrame, cs: CohortSet, leak_rule: str = "truth_unit") -> pd.DataFrame:
    """Claims with their truth and per-row scores (covered, error, width, claims, planted, leaked)."""
    if leak_rule not in ("truth_unit", "row_rope"):
        raise BenchmarkError(f"leak_rule: truth_unit or row_rope, not {leak_rule}")
    d = validate(claims, cs)
    d["cell"] = d.cohort.str.split("/").str[0]
    d["rep"] = d.cohort.str.extract(r"rep(\d+)$")[0].astype(int)
    d["effect_sd"] = d.cell.map(lambda c: float(cs.cells[c]["effect_sd"]))
    truth = pd.concat([cs.truth(c).assign(cohort=c) for c in d.cohort.unique()])
    d = d.merge(truth.rename(columns={"roi": "unit"}), on=["cohort", "unit", "band"], how="left")
    q = d.unit != LOCATED
    if d.loc[q, "truth_db"].isna().any():
        x = d[q & d.truth_db.isna()][["cohort", "band", "unit"]].drop_duplicates().head(3).to_dict("records")
        raise BenchmarkError(f"no path truth for {x}")
    d["kind_cell"] = np.where(d.unit == AVG, AVG, np.where(q, "ROI", LOCATED))
    d["err_db"] = d.estimate_db - d.truth_db
    d["width_db"] = d.hi_db - d.lo_db
    d["covered"] = (d.lo_db <= d.truth_db) & (d.truth_db <= d.hi_db)
    d["z"] = d.err_db / d.se_db if "se_db" in d else np.nan
    d["claim_effect"] = d.decision.isin(["effect_pos", "effect_neg"])
    d["claim_matched"] = d.decision_matched == "excludes_0"
    d["claim_dir"] = d.p_direction >= 0.975 if "p_direction" in d else d.claim_matched
    d["claim_bf3"] = d.log10_bf10 >= np.log10(3) if "log10_bf10" in d else np.nan
    d["claim_sign"] = np.where(d.decision == "effect_pos", 1, np.where(d.decision == "effect_neg", -1, 0))
    d["matched_sign"] = np.where(d.claim_matched, np.sign(d.estimate_db), 0)
    d["planted"] = [u in cs.cells[c]["planted_rois"] for u, c in zip(d.unit, d.cell)]
    if leak_rule == "truth_unit":
        thr = d.band.map(cs.manifest["band_units_db"]) * LEAK_ROPE_SD
    else:
        if "rope_db" not in d:
            raise BenchmarkError("leak_rule row_rope needs rope_db")
        thr = d.rope_db
    d["truth_beyond_rope"] = d.truth_db.abs() > thr
    return d


# ------------------------------------------------------------------------------------------------------- tables
def noise_table(d: pd.DataFrame) -> pd.DataFrame:
    x = d[(d.effect_sd == 0) & (d.kind_cell != LOCATED)]
    if x.empty:
        return pd.DataFrame()
    avg = x[x.kind_cell == AVG].groupby(KEY).agg(
        reps=("rep", "nunique"), claim_effect=("claim_effect", "mean"), claim_matched=("claim_matched", "mean"),
        claim_dir=("claim_dir", "mean"), claim_bf3=("claim_bf3", "mean")).reset_index()
    r = x[x.kind_cell == "ROI"]
    if len(r):
        per = r.groupby(KEY + ["rep"]).agg(any_roi=("claim_effect", "any"), any_roi_matched=("claim_matched", "any"))
        roi = per.groupby(KEY).agg(any_roi_claimed=("any_roi", "mean"),
                                   any_roi_claimed_matched=("any_roi_matched", "mean")).reset_index()
        rate = r.groupby(KEY).claim_effect.mean().rename("roi_claim_rate").reset_index()
        avg = avg.merge(roi, on=KEY, how="left").merge(rate, on=KEY, how="left")
    avg["pass"] = (avg.claim_effect <= RULES["noise_claim_effect"]) & (avg.claim_dir <= RULES["noise_claim_dir"])
    return avg


def calibration_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, x in d[d.kind_cell != LOCATED].groupby(KEY):
        a, r = x[x.kind_cell == AVG], x[x.kind_cell == "ROI"]
        per_roi = r.groupby("unit").covered.mean()
        has_se = "se_db" in x and x.se_db.notna().all()
        vi_avg = a.estimate_db.var(ddof=1) / (a.se_db ** 2).mean() if has_se and len(a) > 1 else np.nan
        vi_roi = (r.groupby("unit")[["err_db", "se_db"]].apply(
            lambda y: y.err_db.var(ddof=1) / (y.se_db ** 2).mean() if len(y) > 1 else np.nan)
            if has_se and len(r) else pd.Series(dtype=float))
        rows.append(dict(zip(KEY, key), reps=a.rep.nunique(), truth_avg_db=a.truth_db.mean(),
                         est_avg_db=a.estimate_db.mean(), bias_avg_db=a.err_db.mean(),
                         rmse_avg_db=float(np.sqrt((a.err_db ** 2).mean())) if len(a) else np.nan,
                         cov_avg=a.covered.mean(), width_avg_db=a.width_db.mean(),
                         sd_z_avg=a.z.std(ddof=1) if has_se else np.nan, vi_avg=vi_avg,
                         cov_roi_pooled=r.covered.mean() if len(r) else np.nan,
                         cov_roi_median=per_roi.median() if len(r) else np.nan,
                         cov_roi_min=per_roi.min() if len(r) else np.nan,
                         cov_roi_min_at=per_roi.idxmin() if len(r) else None,
                         width_roi_median_db=r.width_db.median() if len(r) else np.nan,
                         sd_z_roi=r.z.std(ddof=1) if has_se and len(r) else np.nan,
                         vi_roi_median=vi_roi.median() if len(vi_roi) else np.nan))
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    c["pass_avg"] = c.cov_avg >= RULES["cov_avg"]
    c["pass_roi"] = (c.cov_roi_median >= RULES["cov_roi_median"]) & (c.cov_roi_min >= RULES["cov_roi_min"])
    return c


def attribution_table(d: pd.DataFrame, plant_band: str) -> pd.DataFrame:
    """Where per-ROI claims in the planted band fall, under the arm's own rule and the matched rule."""
    x = d[(d.effect_sd > 0) & (d.kind_cell == "ROI") & (d.band == plant_band)]
    if x.empty:
        return pd.DataFrame()
    g = ["arm", "prior_set", "cell", "n_per_group"]
    out = []
    for rule, claim, sign in (("own", "claim_effect", "claim_sign"), ("matched", "claim_matched", "matched_sign")):
        c = x[x[claim]].copy()
        # wrong sign: against a difference that is real in the readout; a claim where the truth is negligible is
        # placed by where it is (planted, leaked, elsewhere), whatever the sign of that negligible truth
        wrong = (c[sign] != np.sign(c.truth_db)) & (c.truth_beyond_rope | c.planted)
        c["where"] = np.where(wrong, "wrong_sign",
                              np.where(c.planted, "planted", np.where(c.truth_beyond_rope, "leaked_real", "elsewhere")))
        t = (c.groupby(g)["where"].value_counts(normalize=True).unstack(fill_value=0) if len(c)
             else pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=g)))
        t = t.reindex(x.groupby(g).size().index, fill_value=np.nan)
        t["n_claims"] = c.groupby(g).size().reindex(t.index, fill_value=0)
        t["reps"] = x.groupby(g).rep.nunique()
        t = t.reset_index()
        for k in ("planted", "leaked_real", "elsewhere", "wrong_sign"):
            if k not in t:
                t[k] = 0.0
        t.loc[t.n_claims == 0, ["planted", "leaked_real", "elsewhere", "wrong_sign"]] = np.nan
        t["usable_as_location"] = t.planted >= RULES["location_share"]
        out.append(t.assign(rule=rule))
    return pd.concat(out, ignore_index=True)


def location_table(d: pd.DataFrame, cs: CohortSet) -> pd.DataFrame:
    """Located rows: distance to the nearest planted source of the cohort (any mouse), in mm."""
    x = d[(d.kind_cell == LOCATED) & (d.effect_sd > 0)].copy()
    if x.empty:
        return pd.DataFrame()
    dist = []
    for c, xyz in zip(x.cohort, x[["x_mm", "y_mm", "z_mm"]].to_numpy(float)):
        p = cs.plant(c)[["x_mm", "y_mm", "z_mm"]].to_numpy(float)
        dist.append(float(np.sqrt(((p - xyz) ** 2).sum(axis=1)).min()) if len(p) else np.nan)
    x["dist_mm"] = dist
    return x.groupby(["arm", "prior_set", "cell", "n_per_group", "band"]).agg(
        reps=("rep", "nunique"), n_positions=("dist_mm", "size"), median_mm=("dist_mm", "median"),
        q90_mm=("dist_mm", lambda v: v.quantile(0.9)), within_1mm=("dist_mm", lambda v: (v <= 1.0).mean())
    ).reset_index()


def n80(n, k, m):
    """Binomial logistic fit of the decision rate on log N (IRLS); the N at which the fitted rate is 0.8, or NaN with
    the reason. Never read beyond 2x the largest N."""
    n, k, m = (np.asarray(v, float) for v in (n, k, m))
    X = np.column_stack([np.ones(len(n)), np.log(n)])
    p_obs = k / m
    if np.all(p_obs >= 0.8):
        return float("nan"), "reached at the smallest N"
    if np.all(p_obs == p_obs[0]):
        return float("nan"), "flat"
    if np.all((p_obs == 0) | (p_obs == 1)):
        return float("nan"), f"separated: between N {n[p_obs == 0].max():.0f} and {n[p_obs == 1].min():.0f}"
    b = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        W = m * p * (1 - p) + 1e-9
        step = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (k - m * p))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    if b[1] <= 0:
        return float("nan"), "not increasing in N"
    v = float(np.exp((np.log(0.8 / 0.2) - b[0]) / b[1]))
    if v > 2 * n.max():
        return float("nan"), f"beyond 2x the largest N ({v:.0f})"
    return v, "extrapolated" if v > n.max() else ""


def design_table(d: pd.DataFrame) -> pd.DataFrame:
    """Decision rates by N (ROI-average; the planted ROI(s)), own and matched rule, and the N for 80%."""
    x = d[(d.effect_sd > 0) & (d.kind_cell != LOCATED)]
    rows = []
    for (arm, ps, cell, band), y in x.groupby(["arm", "prior_set", "cell", "band"]):
        for q, z in ((AVG, y[y.kind_cell == AVG]), ("planted ROI", y[(y.kind_cell == "ROI") & y.planted])):
            if z.empty:
                continue
            for rule, claim in (("own", "claim_effect"), ("matched", "claim_matched")):
                g = z.groupby("n_per_group")[claim].agg(["sum", "count"]).reset_index()
                for _, r in g.iterrows():
                    rows.append(dict(arm=arm, prior_set=ps, cell=cell, band=band, question=q, rule=rule,
                                     n_per_group=r.n_per_group, rate=r["sum"] / r["count"], n_tests=r["count"]))
                v, flag = n80(g.n_per_group, g["sum"], g["count"])
                rows.append(dict(arm=arm, prior_set=ps, cell=cell, band=band, question=q, rule=rule,
                                 n_per_group="N80", rate=v, n_tests=np.nan, note=flag))
    return pd.DataFrame(rows)


def convergence_table(d: pd.DataFrame) -> pd.DataFrame:
    if "converged" not in d:
        return pd.DataFrame()
    g = d.groupby(["arm", "prior_set", "cohort", "n_per_group"]).agg(
        converged=("converged", "all"), runtime_s=("runtime_s", "first") if "runtime_s" in d else ("converged", "size"))
    return g.reset_index()


def compare_arms(d: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 1) -> pd.DataFrame:
    """b - a, paired by cohort (both arms see the same cohorts): interval width, |error|, coverage, own-rule and
    matched-rule claim rates, at the ROI average and the planted ROI(s), per cell x N x band; 95% bootstrap interval
    over cohorts. a, b: "arm" or "arm/prior_set"."""
    def pick(lbl):
        arm, _, ps = lbl.partition("/")
        y = d[(d.arm == arm) & (d.prior_set == ps if ps else True)]
        if y.empty:
            raise BenchmarkError(f"no claims for arm {lbl}")
        if not ps and y.prior_set.nunique() > 1:
            raise BenchmarkError(f"arm {arm} has prior sets {sorted(y.prior_set.unique())}: name one ({arm}/<set>)")
        return y
    xa, xb = pick(a), pick(b)
    q = lambda y: y[(y.kind_cell == AVG) | ((y.kind_cell == "ROI") & y.planted)].assign(
        question=lambda z: np.where(z.kind_cell == AVG, AVG, "planted ROI"))
    xa, xb = q(xa), q(xb)
    k = ["cohort", "cell", "rep", "n_per_group", "band", "unit", "question"]
    m = xa[k + ["width_db", "err_db", "covered", "claim_effect", "claim_matched"]].merge(
        xb[k + ["width_db", "err_db", "covered", "claim_effect", "claim_matched"]], on=k, suffixes=("_a", "_b"))
    for v in ("width_db", "covered", "claim_effect", "claim_matched"):
        m[v] = m[f"{v}_b"].astype(float) - m[f"{v}_a"].astype(float)
    m["abs_err_db"] = m.err_db_b.abs() - m.err_db_a.abs()
    rng = np.random.default_rng(seed)
    rows = []
    for key, y in m.groupby(["cell", "n_per_group", "band", "question"]):
        per = y.groupby("rep")[["width_db", "abs_err_db", "covered", "claim_effect", "claim_matched"]].mean()
        idx = rng.integers(0, len(per), (n_boot, len(per)))
        for v in per:
            bs = per[v].to_numpy()[idx].mean(axis=1)
            rows.append(dict(zip(["cell", "n_per_group", "band", "question"], key), a=a, b=b, metric=v,
                             reps=len(per), diff=per[v].mean(), lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975)))
    return pd.DataFrame(rows)


def score(claims: pd.DataFrame, cs: CohortSet, out: Path | str | None = None, leak_rule: str = "truth_unit",
          compare: list[tuple[str, str]] | None = None) -> dict[str, pd.DataFrame]:
    """Every table; written to ``out`` (CSV per table and summary.txt) when given."""
    d = attach_truth(claims, cs, leak_rule)
    tabs = {"rows": d, "noise": noise_table(d), "calibration": calibration_table(d),
            "attribution": attribution_table(d, cs.manifest["plant_band"]), "location": location_table(d, cs),
            "design": design_table(d), "convergence": convergence_table(d)}
    for a, b in compare or []:
        tabs[f"compare_{a}_vs_{b}".replace("/", "-")] = compare_arms(d, a, b)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for k, t in tabs.items():
            t.to_csv(out / f"{k}.csv", index=False)
        (out / "summary.txt").write_text(summary(tabs, cs))
    return tabs


def summary(tabs: dict, cs: CohortSet) -> str:
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        lines = [f"Benchmark {cs.manifest.get('benchmark', '?')} ({cs.root}); arms: "
                 f"{', '.join(sorted({f'{a}/{p}' if p else a for a, p in tabs['rows'][['arm', 'prior_set']].drop_duplicates().itertuples(index=False)}))}",
                 f"Rules: {RULES}", ""]
        for k, t in tabs.items():
            if k == "rows" or t.empty:
                continue
            lines += [f"== {k} ==", t.round(3).to_string(index=False), ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------- adapters
def claims_from_engine(readout: pd.DataFrame, cohort: str, n_per_group: int, unit_db: dict[str, float],
                       arm: str = "engine", runtime_s: float | None = None) -> pd.DataFrame:
    """Claims rows from the engine's readout (R/bayes/readout.R: SD units, one row per hypothesis x band x spatial
    cell; a single two-group contrast B - A). unit_db: the fit's unit per band (prep$scale). Needs post_sd (the
    estimand's posterior SD, SD units) for se_db."""
    r = readout[readout.status == "ok"]
    if r.hypothesis.nunique() != 1:
        raise BenchmarkError("claims_from_engine: one hypothesis (the B - A contrast) per readout")
    u = r.band.map(unit_db)
    if u.isna().any():
        raise BenchmarkError(f"no unit for band(s) {sorted(set(r.band[u.isna()]))}")
    dec = np.where(r.decision == "effect", np.where(r.estimate > 0, "effect_pos", "effect_neg"),
                   np.where(r.decision == "null", "null", "undecided"))
    out = pd.DataFrame({
        "cohort": cohort, "n_per_group": n_per_group, "arm": arm, "prior_set": r.prior_set.astype(str),
        "band": r.band, "unit": r.spatial, "estimate_db": r.estimate * u, "lo_db": r.hdi_lo * u,
        "hi_db": r.hdi_hi * u, "decision": dec,
        "decision_matched": np.where((r.hdi_lo > 0) | (r.hdi_hi < 0), "excludes_0", "includes_0"),
        "se_db": r.post_sd * u if "post_sd" in r else np.nan, "p_direction": r.p_direction,
        "log10_bf10": r.log10_bf10, "rope_db": r.rope_hi * u, "rhat": r.rhat, "rhat_scale": r.max_rhat_scale,
        "converged": (r.rhat < 1.05) & (r.max_rhat_scale < 1.05)})
    if runtime_s is not None:
        out["runtime_s"] = runtime_s
    return out.reset_index(drop=True)
