"""The location scorecard (bayes-sl ``reports/2026-10-30_location_scorecard_plan.md``, step L1).

Two questions, scored against a frozen cohort set's truth: where is the signal, and does it differ between the groups?
Like ``benchmark``, this reads files only and knows nothing about a method's internals or the atlas: the regions come
from a hierarchy table, the truth from the cohort set's ``MANIFEST.json`` (a cell's ``planted_rois``, ``effect_sd``).

**Which question the probabilities answer** (``question``, required; there is no default because the truth differs):

    signal       "where is the source?"  Truth: the cell's planted ROIs; "none" where nothing is planted.
    difference   "where do the groups differ?"  Truth: the planted ROIs where effect_sd > 0, else "none". A cell with a
                 source in both groups and no difference (bayes-sl L5, ``<region>_nodiff``) is a no-truth cell here.

No-truth cells are the baseline: Q1's bits, and the noise table. For a no-truth cell that still holds a source (L5 under
``difference``), the noise table also gives how often the top region is the source's (``top_at_source``) and the mean
probability there (``p_source``): a method that finds the source and calls it a difference scores high on both.

**Hierarchy** (CSV, one row per ROI): roi, compartment, hemisphere, group.

**Region probabilities** (Q1; CSV, one row per method x cohort x N x band x level x region):

    cohort, n_per_group, arm, prior_set, band   as in the claims table
    level    compartment | hemisphere | group | roi
    region   a region of that level, or "none" (the method's probability that no planted source is there)
    prob     sums to 1 within method x cohort x N x band x level; regions left out have 0

A method reporting at ROI level is summed up to the coarser levels; a level it reports itself is left alone.

**Group differences** (Q2; CSV): cohort, n_per_group, arm, prior_set, band, region (the source tested), p_differs,
optional p_negligible (the probability that the difference is too small to matter).

**Scored, Q1, per method x cell x N x level:** accuracy (top region in the truth), p_true, set90_size (regions needed to
reach 90%), set90_coverage (how often they include the truth), top_prob, overconfidence (top_prob - accuracy), bits
gained (mean log2 p_true, against the same method's no-plant cohorts scored against this cell's truth, with a bootstrap
interval over cohorts; p_true is floored at EPS, and ``clipped`` is the share floored). Reliability: stated probability
against how often the region is the truth, per bin, on single-truth cells. A cell whose truth covers every region of a
level cannot test that level (``informative`` False).

**Noise:** "no plant" is not "no signal": these cohorts hold real mice, whose own fields a method may locate correctly.
So a confident location here is a claim the plant did not cause, not necessarily a wrong one. Under ``difference``,
"noise" means every no-truth cell, with or without a source.

**Finest reliable level** (rule fixed 2026-10-30, before any method was scored): at that N, in every informative plant
cell, set90_coverage >= 0.85 and the bits' interval above 0; on noise, a confident location (top_prob >= 0.9) at most
0.05 of the time. Each part is reported, so a method failing on noise alone is visible.

**Gain over the no-difference twin** (``gain``; D10 plan step 1, M1; bayes-sl R2 and L6 scorecards). A cell with a
difference can have a twin: the same source(s) in both groups, no difference, the same seeds per rep. The twin is the
cell's manifest ``twin`` if given, else ``<condition>_nodiff`` where ``<condition>`` is the cell's name without its
trailing ``_<effect size>`` (``Thalamus_1.0`` -> ``Thalamus_nodiff``). The twin may be in the same cohort set or in
another (bayes-sl v1 and v1_l5). Per method x N x level, a study's hit is paired with its twin's ``top_at_source`` (the
same answer won with no difference): gain = hit - twin's hit, the hits the difference caused. Its 95% interval
resamples reps within each cell (a rep's N rows share mice). Grouped by cell, by any manifest field given (spacing,
pair, ...), and pooled. Where the manifest names ``shared_rois`` (bayes-sl R2), also the share of ROI-level answers
naming the first shared ROI, in the studies and in their twins.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import BenchmarkError, CohortSet

LEVELS = ("compartment", "hemisphere", "group", "roi")     # coarse to fine
NONE = "none"
EPS = 1e-3
RELIABLE = {"set90_coverage": 0.85, "noise_confident": 0.05}
CONFIDENT, SET_MASS, DIFFERS = 0.9, 0.9, 0.95
TIE = 1e-6          # relative: regions within this of the top probability share the top
METHOD = ["arm", "prior_set", "cohort", "n_per_group", "band", "level"]
BY = ["arm", "prior_set", "cell", "n_per_group", "band", "level"]
PROB_COLS = ("cohort", "n_per_group", "arm", "prior_set", "band", "level", "region", "prob")
DIFF_COLS = ("cohort", "n_per_group", "arm", "prior_set", "band", "region", "p_differs")
QUESTIONS = ("signal", "difference")


def _check_question(question: str) -> None:
    if question not in QUESTIONS:
        raise BenchmarkError(f"question must be one of {QUESTIONS}; got {question!r}")


def no_truth_cells(cs: CohortSet, question: str) -> set[str]:
    """Cells where the question's answer is "none": nothing planted, or (``difference``) no difference planted."""
    _check_question(question)
    return {c for c, v in cs.cells.items()
            if not v["planted_rois"] or (question == "difference" and float(v["effect_sd"]) == 0)}


def read_hierarchy(path: Path | str) -> pd.DataFrame:
    h = pd.read_csv(path, keep_default_na=False)
    miss = [c for c in LEVELS if c not in h]
    if miss:
        raise BenchmarkError(f"hierarchy lacks column(s) {', '.join(miss)}")
    if h.roi.duplicated().any():
        raise BenchmarkError("hierarchy: an ROI appears twice")
    return h


def _level_regions(h: pd.DataFrame, level: str) -> list[str]:
    return sorted(h[level].unique())


def source_regions(cs: CohortSet, h: pd.DataFrame, cell: str, level: str) -> set[str]:
    """The regions of a level that hold the cell's planted ROIs, whatever the question; empty when nothing is planted."""
    planted = cs.cells[cell]["planted_rois"]
    if not planted:
        return set()
    x = h[h.roi.isin(planted)]
    if len(x) != len(set(planted)):
        raise BenchmarkError(f"cell {cell}: planted ROIs missing from the hierarchy")
    return set(x[level])


def truth_regions(cs: CohortSet, h: pd.DataFrame, cell: str, level: str, question: str) -> set[str]:
    """The regions of a level that answer the question for this cell; {"none"} in a no-truth cell."""
    if cell in no_truth_cells(cs, question):
        return {NONE}
    return source_regions(cs, h, cell, level)


def validate_probs(probs: pd.DataFrame, cs: CohortSet, h: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in PROB_COLS if c not in probs]
    if miss:
        raise BenchmarkError(f"region probabilities lack column(s) {', '.join(miss)}")
    d = probs.copy()
    d["prior_set"] = d.prior_set.fillna("").astype(str)
    bad = sorted(set(d.level) - set(LEVELS))
    if bad:
        raise BenchmarkError(f"level must be one of {LEVELS}; got {bad}")
    unknown = sorted(set(d.cohort) - set(cs.manifest["cohorts"]))
    if unknown:
        raise BenchmarkError(f"cohort(s) not in the cohort set: {unknown[:5]}")
    for lv, x in d.groupby("level"):
        bad = sorted(set(x.region) - set(_level_regions(h, lv)) - {NONE})
        if bad:
            raise BenchmarkError(f"unknown {lv} region(s): {bad[:5]}")
    if ((d.prob < 0) | (d.prob > 1 + 1e-9)).any():
        raise BenchmarkError("prob outside [0, 1]")
    if d.duplicated(METHOD + ["region"]).any():
        raise BenchmarkError("duplicated region rows")
    tot = d.groupby(METHOD).prob.sum()
    if ((tot - 1).abs() > 1e-3).any():
        raise BenchmarkError(f"prob does not sum to 1 in {((tot - 1).abs() > 1e-3).sum()} tables, e.g. "
                             f"{tot[(tot - 1).abs() > 1e-3].index[0]}")
    return d


def expand_levels(d: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """ROI-level tables summed up to each coarser level the method did not report itself."""
    roi = d[d.level == "roi"]
    if roi.empty:
        return d
    have = d[METHOD].drop_duplicates()
    out = [d]
    for lv in LEVELS[:-1]:
        x = roi.assign(region=roi.region.map(lambda r: NONE if r == NONE else None).fillna(roi.region.map(h.set_index("roi")[lv])),
                       level=lv)
        x = x.groupby(METHOD + ["region"], as_index=False).prob.sum()
        k = x.merge(have, on=METHOD, how="left", indicator=True)
        out.append(x[(k._merge == "left_only").to_numpy()])
    return pd.concat(out, ignore_index=True)


def _top_share(region: np.ndarray, prob: np.ndarray, regions: set[str]) -> float:
    """The share of the top regions that are in ``regions``: tied tops split the answer, so a tie between a right and a
    wrong region counts half (a singular fit, or rounded probabilities, can tie)."""
    top = prob >= prob.max() * (1 - TIE)
    return float(np.isin(region[top], list(regions)).mean())


def _table_scores(region: np.ndarray, prob: np.ndarray, truth: set[str]) -> dict:
    o = np.argsort(-prob, kind="stable")
    region, prob = region[o], prob[o]
    k = int(np.searchsorted(np.cumsum(prob), SET_MASS - 1e-12) + 1)
    p = prob[prob > 0]
    return dict(p_true=float(prob[np.isin(region, list(truth))].sum()), top_region=region[0], top_prob=float(prob[0]),
                n_top=int((prob >= prob[0] * (1 - TIE)).sum()), hit=_top_share(region, prob, truth), set90_size=k,
                set90_hit=bool(set(region[:k]) & truth), entropy_bits=float(-(p * np.log2(p)).sum()))


def score_tables(d: pd.DataFrame, cs: CohortSet, h: pd.DataFrame, question: str) -> pd.DataFrame:
    """One row per method x cohort x N x band x level, scored against its own cell's truth for the question. Also
    p_source and top_at_source: the probability on the planted source's regions and whether the top region is one of
    them, whatever the question (NaN with nothing planted). ``hit`` and ``top_at_source`` split tied tops (``n_top``
    regions tied); ``top_region`` is the first of them."""
    rows = []
    for key, x in d.groupby(METHOD, sort=False):
        cell = key[2].split("/")[0]
        truth = truth_regions(cs, h, cell, key[5], question)
        src = source_regions(cs, h, cell, key[5])
        s = _table_scores(x.region.to_numpy(), x.prob.to_numpy(), truth)
        rows.append({**dict(zip(METHOD, key)), "cell": cell, "rep": int(key[2].rsplit("rep", 1)[1]), **s,
                     "p_source": float(x.prob[x.region.isin(src)].sum()) if src else np.nan,
                     "top_at_source": _top_share(x.region.to_numpy(), x.prob.to_numpy(), src) if src else np.nan})
    return pd.DataFrame(rows)


def _boot_mean_diff(a: np.ndarray, b: np.ndarray, rng, n: int = 2000) -> tuple[float, float]:
    ia, ib = rng.integers(0, len(a), (n, len(a))), rng.integers(0, len(b), (n, len(b)))
    q = a[ia].mean(1) - b[ib].mean(1)
    return float(np.quantile(q, 0.025)), float(np.quantile(q, 0.975))


def location_table(d: pd.DataFrame, t: pd.DataFrame, cs: CohortSet, h: pd.DataFrame, question: str) -> pd.DataFrame:
    """Q1 per method x cell with a truth x N x level, with the same method's no-truth cohorts as baseline."""
    rng = np.random.default_rng(20261030)
    noise_cells = no_truth_cells(cs, question)
    rows = []
    for key, x in t[~t.cell.isin(noise_cells)].groupby(BY):
        k = dict(zip(BY, key))
        truth = truth_regions(cs, h, k["cell"], k["level"], question)
        lp = np.log2(np.maximum(x.p_true.to_numpy(), EPS))
        base = d[(d.arm == k["arm"]) & (d.prior_set == k["prior_set"]) & (d.n_per_group == k["n_per_group"]) &
                 (d.band == k["band"]) & (d.level == k["level"]) & d.cohort.str.split("/").str[0].isin(noise_cells)]
        r = dict(k, reps=len(x), informative=truth != set(_level_regions(h, k["level"])), accuracy=x.hit.mean(),
                 p_true=x.p_true.mean(), set90_size=x.set90_size.mean(), set90_coverage=x.set90_hit.mean(),
                 top_prob=x.top_prob.mean(), overconfidence=x.top_prob.mean() - x.hit.mean(),
                 clipped=float((x.p_true.to_numpy() < EPS).mean()))
        if len(base):
            bs = base.groupby("cohort").apply(lambda y: y.prob[y.region.isin(truth)].sum(), include_groups=False)
            top = base.groupby("cohort").apply(lambda y: _top_share(y.region.to_numpy(), y.prob.to_numpy(), truth),
                                               include_groups=False)
            lb = np.log2(np.maximum(bs.to_numpy(float), EPS))
            lo, hi = _boot_mean_diff(lp, lb, rng)
            r.update(accuracy_baseline=top.mean(), p_true_baseline=bs.mean(),
                     bits_gained=lp.mean() - lb.mean(), bits_lo=lo, bits_hi=hi)
        else:
            r.update(accuracy_baseline=np.nan, p_true_baseline=np.nan, bits_gained=np.nan, bits_lo=np.nan, bits_hi=np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def noise_table(d: pd.DataFrame, t: pd.DataFrame, cs: CohortSet, question: str) -> pd.DataFrame:
    """What each method says where the question's answer is "none" (per no-truth cell)."""
    x = t[t.cell.isin(no_truth_cells(cs, question))].copy()
    if x.empty:
        return pd.DataFrame()
    x["confident"] = (x.top_prob >= CONFIDENT) & (x.top_region != NONE)
    return x.groupby(BY).agg(reps=("rep", "nunique"), p_none=("p_true", "mean"), top_prob=("top_prob", "mean"),
                             entropy_bits=("entropy_bits", "mean"), set90_size=("set90_size", "mean"),
                             confident=("confident", "mean"), p_source=("p_source", "mean"),
                             top_at_source=("top_at_source", "mean")).reset_index()


def reliability_table(d: pd.DataFrame, cs: CohortSet, h: pd.DataFrame, question: str, bins: int = 10) -> pd.DataFrame:
    """Stated probability against how often the region is the truth, on single-truth cells (a cell with several true
    regions has no one true region per table). Regions a table left out count as stated 0."""
    rows = []
    for (arm, ps, lv), x in d.groupby(["arm", "prior_set", "level"]):
        regions = _level_regions(h, lv) + ([NONE] if (x.region == NONE).any() else [])
        stated, true = [], []
        for coh, y in x.groupby(["cohort", "n_per_group", "band"]):
            tr = truth_regions(cs, h, coh[0].split("/")[0], lv, question)
            if len(tr) != 1:
                continue
            p = y.set_index("region").prob.reindex(regions, fill_value=0.0)
            stated.append(p.to_numpy())
            true.append(np.isin(regions, list(tr)))
        if not stated:
            continue
        s, u = np.concatenate(stated), np.concatenate(true)
        b = np.minimum((s * bins).astype(int), bins - 1)
        for i in np.unique(b):
            m = b == i
            rows.append(dict(arm=arm, prior_set=ps, level=lv, bin_lo=i / bins, n=int(m.sum()), stated=s[m].mean(),
                             observed=u[m].mean()))
    return pd.DataFrame(rows)


def calibration_error(rel: pd.DataFrame) -> pd.DataFrame:
    """Mean |stated - observed| over the bins, weighted by their counts."""
    if rel.empty:
        return rel
    return rel.assign(e=(rel.stated - rel.observed).abs() * rel.n).groupby(["arm", "prior_set", "level"]).apply(
        lambda y: y.e.sum() / y.n.sum(), include_groups=False).rename("calibration_error").reset_index()


def reliable_table(loc: pd.DataFrame, noi: pd.DataFrame) -> pd.DataFrame:
    """Per method x band x N x level, the rule's parts, and each method x band x N's finest level that meets them all.
    Empty when no cell has a truth (e.g. bayes-sl L5 under ``difference``): the rule needs plant cells."""
    if loc.empty:
        return pd.DataFrame()
    x = loc[loc.informative]
    if x.empty:
        return pd.DataFrame()
    k = ["arm", "prior_set", "band", "n_per_group", "level"]
    r = x.groupby(k).agg(cells=("cell", "nunique"), min_coverage=("set90_coverage", "min"),
                         min_bits_lo=("bits_lo", "min")).reset_index()
    if len(noi):
        r = r.merge(noi.groupby(k).confident.max().rename("noise_confident").reset_index(), on=k, how="left")
    else:
        r["noise_confident"] = np.nan
    r["coverage_ok"] = r.min_coverage >= RELIABLE["set90_coverage"]
    r["info_ok"] = r.min_bits_lo > 0
    r["noise_ok"] = r.noise_confident <= RELIABLE["noise_confident"]
    r["reliable"] = r.coverage_ok & r.info_ok & r.noise_ok
    order = {lv: i for i, lv in enumerate(LEVELS)}
    fin = r[r.reliable].assign(o=lambda y: y.level.map(order)).sort_values("o").groupby(k[:-1])["level"].last()
    return r.merge(fin.rename("finest_reliable").reset_index(), on=k[:-1], how="left")


def differ_table(diff: pd.DataFrame, cs: CohortSet) -> pd.DataFrame:
    """Q2: the group's difference at a tested source. A cell's truth is whether its effect is above 0."""
    miss = [c for c in DIFF_COLS if c not in diff]
    if miss:
        raise BenchmarkError(f"group differences lack column(s) {', '.join(miss)}")
    d = diff.copy()
    d["prior_set"] = d.prior_set.fillna("").astype(str)
    if ((d.p_differs < 0) | (d.p_differs > 1)).any():
        raise BenchmarkError("p_differs outside [0, 1]")
    d["cell"] = d.cohort.str.split("/").str[0]
    unknown = sorted(set(d.cell) - set(cs.cells))
    if unknown:
        raise BenchmarkError(f"cell(s) not in the cohort set: {unknown[:5]}")
    d["truly_differs"] = d.cell.map(lambda c: float(cs.cells[c]["effect_sd"]) > 0)
    d["says_differs"] = d.p_differs >= DIFFERS
    d["says_negligible"] = d.p_negligible >= DIFFERS if "p_negligible" in d else np.nan
    d["brier"] = (d.p_differs - d.truly_differs.astype(float)) ** 2
    return d.groupby(["arm", "prior_set", "cell", "n_per_group", "band", "truly_differs"]).agg(
        rows=("cohort", "size"), mean_p_differs=("p_differs", "mean"), says_differs=("says_differs", "mean"),
        says_negligible=("says_negligible", "mean"), brier=("brier", "mean")).reset_index()


GAIN_SEED, GAIN_BOOT = 20261112, 2000           # bayes-sl R2's seed and resamples
PAIR = ["arm", "prior_set", "band", "rep", "n_per_group", "level"]


def twin_cell(cs: CohortSet, cell: str) -> str:
    """The cell's no-difference twin: the manifest's ``twin``, else ``<condition>_nodiff``."""
    return cs.cells[cell].get("twin") or re.sub(r"_[\d.]+$", "", cell) + "_nodiff"


def twin_pairs(t: pd.DataFrame, cs: CohortSet, tw: pd.DataFrame | None = None,
               tw_cs: CohortSet | None = None) -> pd.DataFrame:
    """Each study row (a cell with a difference whose twin was scored) beside its twin's top_at_source. ``t``: the
    scorer's tables; ``tw``/``tw_cs``: the twins' tables and cohort set when they are another run (default: ``t``,
    ``cs``). Adds twin_hit, gain, the cell's manifest fields, and names_shared / twin_names_shared (ROI level, where
    the manifest names shared_rois)."""
    tw, tw_cs = (t, cs) if tw is None else (tw, tw_cs or cs)
    have = set(tw.cell)
    study = t[t.cell.map(lambda c: float(cs.cells[c]["effect_sd"]) > 0 and twin_cell(cs, c) in have)].copy()
    if study.empty:
        raise BenchmarkError("no study cell has its no-difference twin among the scored tables")
    for c in study.cell.unique():
        if float(tw_cs.cells[twin_cell(cs, c)]["effect_sd"]) != 0:
            raise BenchmarkError(f"twin {twin_cell(cs, c)} of {c} has a difference")
    study["twin_cell"] = study.cell.map(lambda c: twin_cell(cs, c))
    x = tw[PAIR + ["cell", "top_at_source", "top_region"]].rename(
        columns={"cell": "twin_cell", "top_at_source": "twin_hit", "top_region": "twin_top"})
    m = study.merge(x, on=PAIR + ["twin_cell"], how="left")
    if m.twin_hit.isna().any():
        k = m[m.twin_hit.isna()].iloc[0]
        raise BenchmarkError(f"{int(m.twin_hit.isna().sum())} study rows have no twin row, e.g. {k.cohort} N{k.n_per_group}"
                             f" {k.level} ({k.arm}{'/' + k.prior_set if k.prior_set else ''})")
    m["gain"] = m.hit - m.twin_hit
    fields = sorted({f for c in m.cell.unique() for f, v in cs.cells[c].items() if np.isscalar(v) or v is None}
                    - set(m.columns))
    for f in fields:
        m[f] = m.cell.map(lambda c: cs.cells[c].get(f))
    shared = m.cell.map(lambda c: (cs.cells[c].get("shared_rois") or [None])[0])
    roi = m.level == "roi"
    m["names_shared"] = np.where(roi & shared.notna(), (m.top_region == shared).astype(float), np.nan)
    m["twin_names_shared"] = np.where(roi & shared.notna(), (m.twin_top == shared).astype(float), np.nan)
    return m


def boot_mean(d: pd.DataFrame, col: str, rng, n: int = GAIN_BOOT, within: str = "cell") -> tuple[float, float]:
    """95% interval of the mean of ``col``, resampling reps within each ``within`` group (a rep's N rows share mice)."""
    tot, cnt = np.zeros(n), np.zeros(n)
    for _, y in d.groupby(within):
        g = y.groupby("rep")[col]
        s, k = g.sum().to_numpy(), g.size().to_numpy()
        i = rng.integers(0, len(s), (n, len(s)))
        tot += s[i].sum(1)
        cnt += k[i].sum(1)
    q = tot / cnt
    return float(np.quantile(q, 0.025)), float(np.quantile(q, 0.975))


def gain_table(m: pd.DataFrame, group_by=(), seed: int = GAIN_SEED, n_boot: int = GAIN_BOOT) -> pd.DataFrame:
    """Per method x band x level x grouping (cell, each field in ``group_by``, pooled): accuracy, the twins' accuracy,
    the gain with its bootstrap interval, and the shared-ROI shares. One generator, in this order, so a table is
    reproduced exactly from the same pairs."""
    miss = [f for f in group_by if f not in m]
    if miss:
        raise BenchmarkError(f"grouping field(s) not in the cohort set's manifest: {miss}")
    rng = np.random.default_rng(seed)
    rows = []
    for (arm, ps, band, lv), x in m.groupby(["arm", "prior_set", "band", "level"]):
        for by in ("cell", *group_by, None):
            for key, y in (x.groupby(by) if by else [("pooled", x)]):
                lo, hi = boot_mean(y, "gain", rng, n_boot)
                rows.append(dict(arm=arm, prior_set=ps, band=band, level=lv, grouping=by or "pooled", value=key,
                                 rows=len(y), reps=y[["cell", "rep"]].drop_duplicates().shape[0], accuracy=y.hit.mean(),
                                 twin=y.twin_hit.mean(), gain=y.gain.mean(), gain_lo=lo, gain_hi=hi,
                                 names_shared=y.names_shared.mean(), twin_names_shared=y.twin_names_shared.mean()))
    return pd.DataFrame(rows)


def score_location(probs: pd.DataFrame, cs: CohortSet, h: pd.DataFrame, *, question: str,
                   diff: pd.DataFrame | None = None, out: Path | str | None = None,
                   twins: bool = False, twin_probs: pd.DataFrame | None = None, twin_cs: CohortSet | None = None,
                   group_by=(), reps=None) -> dict[str, pd.DataFrame]:
    """Every table; written to ``out`` (one CSV per table and summary.txt) when given. ``question`` says what the
    probabilities answer: "signal" (where is the source) or "difference" (where do the groups differ).

    ``twins``: also the gain over each study's no-difference twin (``gain``, and its rows ``gain_pairs``), the twins
    being in this cohort set, or scored from ``twin_probs`` against ``twin_cs``. ``group_by``: manifest fields to
    group the gain by. ``reps``: the reps the gain uses (e.g. bayes-sl R2's test reps, leaving out its pilot)."""
    _check_question(question)
    d = expand_levels(validate_probs(probs, cs, h), h)
    t = score_tables(d, cs, h, question)
    loc, noi = location_table(d, t, cs, h, question), noise_table(d, t, cs, question)
    rel = reliability_table(d, cs, h, question)
    tabs = {"tables": t, "location": loc, "noise": noi, "reliability": rel, "calibration_error": calibration_error(rel),
            "reliable": reliable_table(loc, noi)}
    if diff is not None:
        tabs["differ"] = differ_table(diff, cs)
    if twins or twin_probs is not None:
        tw = None
        if twin_probs is not None:
            if twin_cs is None:
                raise BenchmarkError("twin probabilities need their cohort set")
            tw = score_tables(expand_levels(validate_probs(twin_probs, twin_cs, h), h), twin_cs, h, question)
        m = twin_pairs(t, cs, tw, twin_cs)
        if reps is not None:
            m = m[m.rep.isin(set(reps))]
        tabs["gain_pairs"], tabs["gain"] = m, gain_table(m, group_by)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for k, v in tabs.items():
            v.to_csv(out / f"{k}.csv", index=False)
        lines = ["Location scorecard (source_analytics.bayes.benchmark_location)", f"Question: {question}", ""]
        if len(tabs["reliable"]):
            f = tabs["reliable"].groupby(["arm", "prior_set", "band", "n_per_group"]).finest_reliable.first()
            lines += ["Finest reliable level:", f.fillna("none").to_string(), ""]
        if len(tabs["calibration_error"]):
            lines += ["Calibration error:", tabs["calibration_error"].round(3).to_string(index=False), ""]
        if "gain" in tabs:
            g = tabs["gain"][tabs["gain"].grouping == "pooled"]
            lines += ["Gain over the no-difference twin (hits the difference caused), pooled:",
                      g[["arm", "prior_set", "band", "level", "reps", "accuracy", "twin", "gain", "gain_lo", "gain_hi"]]
                      .round(3).to_string(index=False), ""]
        (out / "summary.txt").write_text("\n".join(lines) + "\n")
    return tabs
