"""Region probabilities from an arm's draws, for the location scorer (bayes-sl L4: ``scripts/bm_l4.py``,
``scripts/bm_l4_contrasts.py``; D10 plan step 1, M1).

Two readout-side statements about the group difference B - A in one band, from an arm's joint draws of the contrast
per ROI (SD units). They locate the difference as the readout sees it, not the planted source, so they answer the
location scorer's ``difference`` question.

**Which ROI differs most** (``largest_difference``): P(r) = the share of draws in which |contrast| is largest at r;
P(none) = the share in which every |contrast| is below the ROPE (the engine's 0.1 SD). Tied draws are shared between
the tied ROIs: a singular lme4 fit gives every ROI the same contrast, and argmax would hand it all to the first column.

**Coarse statements** (``coarse_statements``): each draw reduced to the mean contrast over a set of ROIs (from the
region hierarchy), and P(cortex > deep) = the share of draws with |mean over cortex| > |mean over deep|; P(right >
left) likewise, midline ROIs left out. A tie counts half. The truth is the same reduction of the cohort's path truth;
``unit_db`` is written so that a gap below the ROPE can be scored as no material difference.

**Arms without joint draws** (the frequentist ``roi_psd`` step, F0): 1000 independent normal draws from its estimates
and standard errors per cohort x N, seeded from the cohort and N. This ignores correlation between ROIs. For the
largest-difference table they are put in SD units with the engine's unit from the same cohort and N (F0 has no z scale
of its own); the coarse statements stay in dB (the ratio of two means does not need a unit).

Fit files are the arm runner's (``source_analytics.bayes.arms``): ``<fits>/<cell>/rep<NNN>/N<NN>_<tag>_draws.csv.gz``,
one column per ``<band>|<spatial>``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import AVG, BenchmarkError

ROPE_SD = 0.1
N_DRAWS = 1000
SEED_PROBS, SEED_COARSE = 20261102, 20261103          # bayes-sl's seeds, so its F0 tables are reproduced
TIE = 1e-6
COARSE = {"cortex_gt_deep": ("compartment", "cortex", "deep"), "right_gt_left": ("hemisphere", "right", "left")}


def largest_difference(D: pd.DataFrame, rope: float = ROPE_SD) -> dict[str, float]:
    """D: draws x ROIs (columns the ROI names), SD units. Returns region -> probability, "none" included when any
    draw has every |contrast| below ``rope``; regions with 0 are left out."""
    A = np.abs(D.to_numpy(float))
    big = A.max(1)
    hit = big >= rope
    top = A >= big[:, None] - TIE * np.maximum(big[:, None], 1e-12)
    p = (top / top.sum(1, keepdims=True))[hit].sum(0) / len(A)
    out = {r: float(x) for r, x in zip(D.columns, p) if x > 0}
    if (~hit).any():
        out["none"] = float((~hit).mean())
    return out


def coarse_sets(h: pd.DataFrame) -> dict[str, tuple[list[str], list[str]]]:
    """statement -> (ROIs of the first set, ROIs of the second), from the hierarchy."""
    out = {}
    for k, (col, a, b) in COARSE.items():
        first, second = list(h.roi[h[col] == a]), list(h.roi[h[col] == b])
        if not first or not second:
            raise BenchmarkError(f"hierarchy: no '{a}' or no '{b}' in column {col} (needed for {k})")
        out[k] = (first, second)
    return out


def coarse_statements(D: pd.DataFrame, sets: dict) -> dict[str, float]:
    """statement -> share of draws in which the first set's |mean| beats the second's (a tie counts half)."""
    out = {}
    for k, (a, b) in sets.items():
        x, y = np.abs(D[a].to_numpy(float).mean(1)), np.abs(D[b].to_numpy(float).mean(1))
        tie = np.abs(x - y) <= TIE * np.maximum(np.maximum(x, y), 1e-12)
        out[k] = float(np.where(tie, 0.5, x > y).mean())
    return out


def rois_only(D: pd.DataFrame, band: str) -> pd.DataFrame:
    """The band's per-ROI columns of a draws table (``<band>|<roi>``), renamed to the ROI."""
    x = D[[c for c in D.columns if c.startswith(band + "|") and not c.endswith("|" + AVG)]]
    if x.empty:
        raise BenchmarkError(f"no draws for band {band!r}")
    return x.rename(columns=lambda c: c.split("|", 1)[1])


def fits_dir(path: Path | str) -> Path:
    """An arm's fits directory: ``path/fits`` when the runner's --out is given, else ``path``."""
    p = Path(path)
    return p / "fits" if (p / "fits").is_dir() else p


def draw_files(fits: Path | str, tag: str) -> list[tuple[str, int, Path]]:
    """(cohort, N, path) for every ``<cell>/rep<NNN>/N<NN>_<tag>_draws.csv.gz`` under ``fits``, sorted by path."""
    out = []
    for f in sorted(fits_dir(fits).glob(f"*/rep*/N*_{tag}_draws.csv.gz")):
        out.append((f"{f.parents[1].name}/{f.parent.name}", int(f.name[1:3]), f))
    return out


def _seed(base: int, cohort: str, n: int) -> list[int]:
    return [base, n, *[ord(ch) for ch in cohort]]


def _f0_draws(y: pd.DataFrame, cohort: str, n: int, base: int, n_draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(base, cohort, n))
    return pd.DataFrame(rng.normal(y.estimate_db.to_numpy(), y.se_db.to_numpy(), (n_draws, len(y))),
                        columns=y.unit.to_numpy())


def _claims_rows(claims: pd.DataFrame, band: str, arm: str) -> pd.DataFrame:
    x = claims[(claims.arm == arm) & (claims.band == band) & (claims.unit != AVG)]
    if x.empty:
        raise BenchmarkError(f"no {arm} claims in band {band!r}")
    if x.se_db.isna().any():
        raise BenchmarkError(f"{arm} claims without se_db: normal draws need a standard error")
    return x


def probs_from_draws(fits: Path | str, tag: str, arm: str, prior_set: str, band: str,
                     rope: float = ROPE_SD) -> pd.DataFrame:
    """The largest-difference table for every fit of one arm (``tag``: the file tag, e.g. W, I, F1, F1c)."""
    rows = []
    for cohort, n, f in draw_files(fits, tag):
        for r, p in largest_difference(rois_only(pd.read_csv(f), band), rope).items():
            rows.append(dict(cohort=cohort, n_per_group=n, arm=arm, prior_set=prior_set, band=band, level="roi",
                             region=r, prob=p))
    return pd.DataFrame(rows)


def probs_from_claims(claims: pd.DataFrame, unit_db: dict[tuple[str, int], float], band: str, arm: str = "F0",
                      rope: float = ROPE_SD, n_draws: int = N_DRAWS) -> pd.DataFrame:
    """The largest-difference table for an arm with estimates and SEs only (independent normal draws, dB, divided by
    the engine's unit for the same cohort and N: ``unit_db[(cohort, N)]``)."""
    rows = []
    for (c, n), y in _claims_rows(claims, band, arm).groupby(["cohort", "n_per_group"]):
        if (c, n) not in unit_db:
            raise BenchmarkError(f"no unit for {c} N{n}: run the engine arm on the same cohorts first")
        D = _f0_draws(y, c, n, SEED_PROBS, n_draws) / unit_db[(c, n)]
        for r, p in largest_difference(D, rope).items():
            rows.append(dict(cohort=c, n_per_group=n, arm=arm, prior_set="", band=band, level="roi", region=r,
                             prob=p))
    return pd.DataFrame(rows)


def coarse_from_draws(fits: Path | str, tag: str, arm: str, prior_set: str, band: str, sets: dict) -> pd.DataFrame:
    rows = []
    for cohort, n, f in draw_files(fits, tag):
        for k, p in coarse_statements(rois_only(pd.read_csv(f), band), sets).items():
            rows.append(dict(cohort=cohort, n_per_group=n, arm=arm, prior_set=prior_set, statement=k, prob=p))
    return pd.DataFrame(rows)


def coarse_from_claims(claims: pd.DataFrame, band: str, sets: dict, arm: str = "F0",
                       n_draws: int = N_DRAWS) -> pd.DataFrame:
    rows = []
    for (c, n), y in _claims_rows(claims, band, arm).groupby(["cohort", "n_per_group"]):
        for k, p in coarse_statements(_f0_draws(y, c, n, SEED_COARSE, n_draws), sets).items():
            rows.append(dict(cohort=c, n_per_group=n, arm=arm, prior_set="", statement=k, prob=p))
    return pd.DataFrame(rows)


def coarse_truth(cs_root: Path | str, cohort: str, band: str, sets: dict) -> dict[str, tuple[float, float, float]]:
    """statement -> (|mean path truth| over the first set, over the second, the cohort's unit_db or NaN)."""
    p = pd.read_csv(Path(cs_root) / cohort / "path_truth.csv").query("band == @band").set_index("roi")
    t = p.truth_db
    u = p.unit_db.iloc[0] if "unit_db" in p else np.nan       # no-difference cohorts have none (truth 0 everywhere)
    return {k: (abs(t[a].mean()), abs(t[b].mean()), u) for k, (a, b) in sets.items()}


def add_coarse_truth(d: pd.DataFrame, cs_root: Path | str, band: str, sets: dict) -> pd.DataFrame:
    if d.empty:
        return d
    T = {c: coarse_truth(cs_root, c, band, sets) for c in d.cohort.unique()}
    d = d.copy()
    d["truth_first"] = [T[c][k][0] for c, k in zip(d.cohort, d.statement)]
    d["truth_second"] = [T[c][k][1] for c, k in zip(d.cohort, d.statement)]
    d["unit_db"] = [T[c][k][2] for c, k in zip(d.cohort, d.statement)]
    return d


def engine_units(fits: Path | str, band: str, tag: str = "W") -> dict[tuple[str, int], float]:
    """(cohort, N) -> the engine fit's unit_db in ``band``, from its readout files (``N<NN>_<tag>.csv``)."""
    out = {}
    for f in sorted(fits_dir(fits).glob(f"*/rep*/N*_{tag}.csv")):
        r = pd.read_csv(f, usecols=["band", "unit_db", "cell", "rep", "n_per_group"]).query("band == @band").head(1)
        for x in r.itertuples():
            out[(f"{x.cell}/rep{x.rep:03d}", x.n_per_group)] = x.unit_db
    return out


ARM_TAGS = {"engine/W": ("W", "engine", "W"), "engine/I": ("I", "engine", "I"), "F1": ("F1", "F1", ""),
            "F1c": ("F1c", "F1c", "")}


def build(arms: dict[str, Path | str], band: str, *, f0_claims: pd.DataFrame | None = None,
          unit_fits: Path | str | None = None, hierarchy: pd.DataFrame | None = None,
          cohorts_root: Path | str | None = None, rope: float = ROPE_SD) -> dict[str, pd.DataFrame]:
    """Both tables for a set of arms. ``arms``: label (engine/W, engine/I, F1, F1c) -> fits directory. F0 comes from
    ``f0_claims`` with units from the engine W fits in ``unit_fits``. The coarse table needs ``hierarchy`` (and
    ``cohorts_root`` for its truth columns). Arms are read in bayes-sl's order (engine W, I, F1, F1c, F0)."""
    unknown = sorted(set(arms) - set(ARM_TAGS))
    if unknown:
        raise BenchmarkError(f"unknown arm(s) {unknown}; one of {sorted(ARM_TAGS)} (F0 comes from its claims)")
    probs, coarse = [], []
    sets = coarse_sets(hierarchy) if hierarchy is not None else None
    for label in ARM_TAGS:
        if label not in arms:
            continue
        tag, arm, ps = ARM_TAGS[label]
        probs.append(probs_from_draws(arms[label], tag, arm, ps, band, rope))
        if sets:
            coarse.append(coarse_from_draws(arms[label], tag, arm, ps, band, sets))
    if f0_claims is not None:
        if unit_fits is None:
            raise BenchmarkError("F0's largest-difference table needs the engine's units (unit_fits)")
        probs.append(probs_from_claims(f0_claims, engine_units(unit_fits, band), band, rope=rope))
        if sets:
            coarse.append(coarse_from_claims(f0_claims, band, sets))
    out = {"probs": pd.concat(probs, ignore_index=True) if probs else pd.DataFrame()}
    if sets:
        c = pd.concat(coarse, ignore_index=True) if coarse else pd.DataFrame()
        out["contrasts"] = add_coarse_truth(c, cohorts_root, band, sets) if cohorts_root is not None else c
    return out
