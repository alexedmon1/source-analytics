"""Running the benchmark's arms on a frozen cohort set (D10 plan step 1, M1; ports bayes-sl ``scripts/bm_b1.py``,
``scripts/bm_f0.py``, ``scripts/r2_arms.sh``'s retry, ``R/ee3_fit.R``, ``R/f1_fit.R``, ``R/f1c_fit.R``).

Arms (``source-analytics benchmark run --arm ...``):

    engine   the Bayesian engine, prior set W or I (R/bayes/bench_fit.R): readout + joint draws per fit
    F0       source-analytics' own roi_psd step, called unchanged (R/roi_psd_analysis.R): per band the LMM
             dv ~ group * roi + (1 | subject), emmeans contrasts per ROI with BH across ROIs within each band, and the
             ROI-average contrast from the same run
    F1, F1c  the engine's model structure fitted by lme4 without priors (R/bayes/bench_lmm.R): F1 resamples mice,
             F1c uses the conditional covariance

Every arm works on each cohort's ``band_power.csv`` cut to the first N mice of each group (``order <= N``) for each
N in the cohort's ``meta.json`` ``n_nested``, and resumes: a fit already written is skipped.

Layout under ``--out`` (bayes-sl's, so its stored fits are read the same way):

    engine, F1, F1c   fits/<cell>/rep<NNN>/N<NN>_<tag>.csv and N<NN>_<tag>_draws.csv.gz   (tag W, I, F1, F1c)
    F0                <cell>/rep<NNN>/N<NN>/{data/, tbl/roi_psd_hypotheses.csv, tbl/roi_psd_global_hypotheses.csv,
                      r.log, runtime.json}

**Vanishing temporary files.** cmdstanr's output CSVs on ``/mnt/e`` sometimes vanish right after sampling (bayes-sl,
2026-11-04, 11-13 and 11-14; cause unknown). Each R part therefore gets its own TMPDIR and is retried up to
``RETRIES`` times; fits already written are skipped, so a retry redoes only the fit that failed.

**Claims** (``claims_engine``, ``claims_f0``): the scorer's contract (``source_analytics.bayes.benchmark``). F0's own
decision is q < .05 per ROI and p < .05 (unadjusted, as the pipeline reports it) for the ROI-average, signed by the
estimate, else "undecided" (F0 cannot claim a null); the matched decision is whether the CI excludes 0.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .benchmark import AVG, BenchmarkError, CohortSet, claims_from_engine
from .run import r_dir

ARMS = ("engine", "F0", "F1", "F1c")
RETRIES = 3
ALPHA = 0.05
# Band edges for F0's study config. Only the names reach the R step's fits; FORGE's edges where known.
BAND_EDGES = {"Delta": [1, 4], "Theta": [4, 10], "Alpha": [10, 13], "Beta": [13, 30], "Low Gamma": [30, 55],
              "High Gamma": [65, 80], "Epsilon": [80, 100]}


def default_bands(cs: CohortSet) -> list[str]:
    """The cohort set's planted band and its control band (bayes-sl's engine and F0 arms fitted these two)."""
    return [cs.manifest["plant_band"]] + ([cs.manifest["control_band"]] if cs.manifest.get("control_band") else [])


def todo(cs: CohortSet, cells=None, reps=None, ns=None) -> pd.DataFrame:
    """(cohort, n_per_group) for every cohort of the set in ``cells`` (all if None) with rep in ``reps`` and N in
    ``ns`` (each cohort's ``n_nested`` if None)."""
    if cells:
        unknown = sorted(set(cells) - set(cs.cells))
        if unknown:
            raise BenchmarkError(f"cell(s) not in the cohort set: {unknown[:5]}")
    rows = []
    for c in sorted(cs.manifest["cohorts"]):
        cell, r = c.split("/")
        if cells and cell not in cells:
            continue
        if reps is not None and int(r.removeprefix("rep")) not in set(reps):
            continue
        meta = json.loads((cs.root / c / "meta.json").read_text())
        for n in meta["n_nested"]:
            if ns is None or n in set(ns):
                rows.append((c, int(n)))
    if not rows:
        raise BenchmarkError("nothing to run: no cohort matches the cells, reps and N given")
    return pd.DataFrame(rows, columns=["cohort", "n_per_group"])


def parse_reps(text: str | None) -> list[int] | None:
    """'1:7' or '1,3,901:902' -> the reps; None stays None (all)."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        a, _, b = part.partition(":")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def _run_part(cmd: list[str], log: Path, tmp: Path, retries: int = RETRIES) -> bool:
    log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        tmp.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "TMPDIR": str(tmp), "OMP_NUM_THREADS": "1"}
        with open(log, "a") as f:
            f.write(f"--- try {attempt} {time.strftime('%H:%M:%S')}: {' '.join(cmd)}\n")
            f.flush()
            if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode == 0:
                return True
            f.write(f"--- try {attempt} failed {time.strftime('%H:%M:%S')}\n")
    return False


def _run_parts(base: list[str], out: Path, name: str, parts: int, retries: int = RETRIES) -> None:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(base + ["--part", f"{k}/{parts}"], logs / f"{name}_{k}.log", out / "tmp" / f"{name}_{k}")
            for k in range(1, parts + 1)]
    with ThreadPoolExecutor(parts) as ex:
        ok = list(ex.map(lambda j: _run_part(*j, retries=retries), jobs))
    failed = [k for k, x in enumerate(ok, 1) if not x]
    if failed:
        raise RuntimeError(f"benchmark {name}: part(s) {failed} failed {retries} times; see {logs}/{name}_<k>.log")


def _write_todo(t: pd.DataFrame, out: Path, name: str) -> Path:
    f = out / f"todo_{name}.csv"
    t.to_csv(f, index=False)
    return f


def run_engine(cs: CohortSet, out: Path | str, t: pd.DataFrame, *, prior_set: str = "W", informative: str | None = None,
               bands: list[str] | None = None, parts: int = 4, chains: int = 4, iter: int = 2000, warmup: int = 1000,
               rope: float = 0.1, sa_r: Path | str | None = None, retries: int = RETRIES) -> None:
    """The engine arm: compiles the prior set's template once (on the first cohort at its smallest N; the template's
    data do not reach the fits), then the fits in ``parts`` parallel Rscript processes."""
    if prior_set not in ("W", "I"):
        raise BenchmarkError("prior set W or I")
    if prior_set == "I" and not informative:
        raise BenchmarkError("prior set I needs --informative (a JSON file of x_<band>_<group>: [mean, sd])")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    script = str(r_dir() / "bayes" / "bench_fit.R")
    base = ["Rscript", script, "--cohorts", str(cs.root), "--out", str(out), "--set", prior_set,
            "--bands", ",".join(bands or default_bands(cs)), "--chains", str(chains), "--iter", str(iter),
            "--warmup", str(warmup), "--rope", str(rope)]
    if informative:
        base += ["--informative", str(informative)]
    if sa_r:
        base += ["--sa", str(sa_r)]
    first = t.sort_values("n_per_group").iloc[0]
    tmp = out / "tmp" / f"template_{prior_set}"
    if not _run_part(base + ["--template", "--template-cohort", first.cohort, "--template-n", str(first.n_per_group)],
                     out / "logs" / f"template_{prior_set}.log", tmp, retries):
        raise RuntimeError(f"benchmark engine: template failed; see {out}/logs/template_{prior_set}.log")
    _run_parts(base + ["--todo", str(_write_todo(t, out, f"engine_{prior_set}"))], out, f"engine_{prior_set}", parts,
               retries)


def run_lmm(cs: CohortSet, out: Path | str, t: pd.DataFrame, *, arm: str = "F1", band: str | None = None,
            parts: int = 4, resamples: int = 200, draws: int = 1000, sa_r: Path | str | None = None,
            retries: int = RETRIES) -> None:
    """F1 (mice resampled) or F1c (conditional covariance), on one band (default: the planted band)."""
    if arm not in ("F1", "F1c"):
        raise BenchmarkError("arm F1 or F1c")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = ["Rscript", str(r_dir() / "bayes" / "bench_lmm.R"), "--cohorts", str(cs.root), "--out", str(out),
            "--method", "resample" if arm == "F1" else "conditional", "--band", band or cs.manifest["plant_band"],
            "--b", str(resamples), "--draws", str(draws)]
    if sa_r:
        base += ["--sa", str(sa_r)]
    _run_parts(base + ["--todo", str(_write_todo(t, out, arm))], out, arm, parts, retries)


def f0_config(bands: list[str]) -> dict:
    return {
        "name": "benchmark arm F0",
        "bands": {b: BAND_EDGES.get(b, [0, 0]) for b in bands},
        "dvs": ["absolute"],
        "design": {"factor": "group", "levels": ["A", "B"], "reference": "A", "fdr": {"method": "BH", "scope": "band"}},
        "hypotheses": [{"name": "effect", "kind": "contrast", "label": "B - A", "weights": {"B": 1, "A": -1}}],
        "group_order": ["A", "B"], "groups": {"A": "A", "B": "B"}, "group_colors": {"A": "#000000", "B": "#E0007A"},
    }


def _f0_one(cs: CohortSet, out: Path, cohort: str, n: int, bands: list[str], sa_r: Path) -> str:
    wd = out / cohort / f"N{n:02d}"
    if (wd / "runtime.json").exists():
        return "kept"
    (wd / "data").mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(cs.root / cohort / "band_power.csv")
    d = d[(d.order <= n) & d.band.isin(bands)]
    if d.groupby("group").subject.nunique().to_dict() != {"A": n, "B": n}:
        raise BenchmarkError(f"{cohort} N{n}: groups {d.groupby('group').subject.nunique().to_dict()}")
    d.to_csv(wd / "data" / "band_power.csv", index=False)
    with open(wd / "data" / "study_config.yaml", "w") as f:
        yaml.safe_dump(f0_config(bands), f, sort_keys=False)
    cmd = ["Rscript", str(Path(sa_r) / "roi_psd_analysis.R"), "--data-dir", str(wd / "data"),
           "--config", str(wd / "data" / "study_config.yaml"), "--output-dir", str(wd), "--fig-dir", str(wd / "fig"),
           "--tbl-dir", str(wd / "tbl"), "--no-figures"]
    t0 = time.time()
    with open(wd / "r.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        return f"FAILED rc={rc} ({wd / 'r.log'})"
    (wd / "runtime.json").write_text(json.dumps({"runtime_s": time.time() - t0, "cmd": cmd}, indent=1))
    return "done"


def run_f0(cs: CohortSet, out: Path | str, t: pd.DataFrame, *, bands: list[str] | None = None, workers: int = 4,
           sa_r: Path | str | None = None) -> None:
    out = Path(out)
    bands = bands or default_bands(cs)
    sa_r = Path(sa_r) if sa_r else r_dir()
    jobs = list(t.itertuples(index=False))
    with ThreadPoolExecutor(workers) as ex:
        status = list(ex.map(lambda j: _f0_one(cs, out, j.cohort, int(j.n_per_group), bands, sa_r), jobs))
    failed = [f"{j.cohort} N{j.n_per_group}: {s}" for j, s in zip(jobs, status) if s.startswith("FAILED")]
    if failed:
        raise RuntimeError(f"benchmark F0: {len(failed)} run(s) failed, e.g. {failed[0]}")


# --------------------------------------------------------------------------------------------------- claims
def claims_engine(out: Path | str, prior_set: str | None = None) -> pd.DataFrame:
    """Claims rows for every engine fit under ``out`` (``fits/*/rep*/N*_<W|I>.csv``)."""
    pat = f"N*_{prior_set}.csv" if prior_set else "N*_[WI].csv"
    parts = []
    for f in sorted(Path(out, "fits").glob(f"*/rep*/{pat}")):
        r = pd.read_csv(f)
        parts.append(claims_from_engine(r, f"{r.cell.iloc[0]}/rep{int(r.rep.iloc[0]):03d}", int(r.n_per_group.iloc[0]),
                                        dict(zip(r.band, r.unit_db)), runtime_s=float(r.fit_seconds.iloc[0])))
    if not parts:
        raise BenchmarkError(f"no engine fits under {out}/fits")
    return pd.concat(parts, ignore_index=True)


def _f0_rows(h: pd.DataFrame, cohort: str, n: int, unit, decide_col: str, runtime_s: float) -> pd.DataFrame:
    if not ((h.group_a == "B") & (h.group_b == "A")).all() or set(h.hypothesis) != {"effect"} or set(h.dv) != {"absolute"}:
        raise BenchmarkError(f"{cohort} N{n}: not the B - A contrast on absolute")
    sig = h[decide_col] < ALPHA
    return pd.DataFrame({
        "cohort": cohort, "n_per_group": n, "arm": "F0", "prior_set": "", "band": h.band, "unit": unit,
        "estimate_db": h.estimate, "lo_db": h.estimate_lcl, "hi_db": h.estimate_ucl,
        "decision": np.where(sig, np.where(h.estimate > 0, "effect_pos", "effect_neg"), "undecided"),
        "decision_matched": np.where((h.estimate_lcl > 0) | (h.estimate_ucl < 0), "excludes_0", "includes_0"),
        "se_db": h.SE, "p_value": h.p_value, "q_value": h[decide_col] if decide_col == "q_value" else np.nan,
        "runtime_s": runtime_s})


def f0_runs(out: Path | str) -> pd.DataFrame:
    """(cohort, n_per_group) of every finished F0 run under ``out`` (those with a runtime.json), sorted."""
    rows = sorted((f"{f.parents[2].name}/{f.parents[1].name}", int(f.parent.name[1:]))
                  for f in Path(out).glob("*/rep*/N*/runtime.json"))
    return pd.DataFrame(rows, columns=["cohort", "n_per_group"])


def claims_f0(out: Path | str, cs: CohortSet, t: pd.DataFrame | None = None) -> pd.DataFrame:
    """Claims rows for F0's runs under ``out`` (``t``: the runs to read, each must exist; default every finished
    run there)."""
    out = Path(out)
    t = f0_runs(out) if t is None else t
    if t.empty:
        raise BenchmarkError(f"no finished F0 runs under {out}")
    n_rois = len(cs.manifest["rois"])
    parts = []
    for c, n in t.itertuples(index=False):
        wd = out / c / f"N{n:02d}"
        if not (wd / "runtime.json").exists():
            raise BenchmarkError(f"no F0 run for {c} N{n} under {out}")
        rt = json.loads((wd / "runtime.json").read_text())["runtime_s"]
        roi = pd.read_csv(wd / "tbl" / "roi_psd_hypotheses.csv")
        glob = pd.read_csv(wd / "tbl" / "roi_psd_global_hypotheses.csv")
        roi, glob = roi[roi.kind == "contrast"], glob[glob.kind == "contrast"]
        n_bands = glob.band.nunique()
        if len(roi) != n_rois * n_bands or len(glob) != n_bands or roi.q_value.isna().any():
            raise BenchmarkError(f"{c} N{n}: {len(roi)} ROI rows, {len(glob)} ROI-average rows")
        if not roi.fdr_family.str.startswith("scope=band method=BH").all():
            raise BenchmarkError(f"{c} N{n}: FDR family {roi.fdr_family.unique()}")
        parts += [_f0_rows(roi, c, int(n), roi.spatial.values, "q_value", rt),
                  _f0_rows(glob, c, int(n), AVG, "p_value", rt)]
    return pd.concat(parts, ignore_index=True)


def run(arm: str, cs: CohortSet, out: Path | str, *, cells=None, reps=None, ns=None, **kw) -> pd.DataFrame:
    """Run one arm; returns the cohort x N table it ran."""
    if arm not in ARMS:
        raise BenchmarkError(f"arm must be one of {ARMS}")
    if shutil.which("Rscript") is None:
        raise RuntimeError("benchmark run: Rscript not found")
    t = todo(cs, cells, reps, ns)
    if arm == "engine":
        run_engine(cs, out, t, **kw)
    elif arm == "F0":
        run_f0(cs, out, t, **kw)
    else:
        run_lmm(cs, out, t, arm=arm, **kw)
    return t
