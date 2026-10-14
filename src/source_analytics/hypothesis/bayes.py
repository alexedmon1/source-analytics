"""Bayesian hypothesis adapter: the Python side of the separate ``bayes`` step (docs/methods/BAYES_ENGINE.md).

Thin by design. It resolves where a module's per-subject table lives and where results go, serialises the (profile-
narrowed) design spec into a job file, and calls ``Rscript R/bayes_analysis.R --job <file>``. The model, priors, gate
and readout are in R (``R/bayes_hypothesis.R``). Nothing here imports brms or Stan, and the frequentist outputs are
never touched: results go to ``<analysis>_hypotheses_bayes.csv`` beside ``<analysis>_hypotheses.csv``.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import StudyConfig

R_DIR = Path(__file__).resolve().parents[3] / "R"


@dataclass(frozen=True)
class BayesTable:
    """Where a module's per-subject long table lives and how it is shaped."""

    csv: str
    dvs: tuple[str, ...]
    spatial_col: str = "roi"
    band_col: str | None = "band"


# Modules the adapter knows (G1: FORGE rest). Extend as outcome families arrive (G4).
BAYES_TABLES: dict[str, BayesTable] = {
    "roi_psd": BayesTable("band_power.csv", ("absolute", "relative", "delta_ref")),
    "roi_aperiodic": BayesTable("aperiodic_params.csv", ("exponent", "offset"), band_col=None),
}


@dataclass
class BayesJob:
    """Everything R needs for one run (written to JSON)."""

    data_csv: str
    dv: str
    spatial_col: str
    band_col: str | None
    design: dict
    hypotheses: list[dict]
    prior_sets: list[str]
    informative: dict | None
    rope_sd: float
    table_out: str
    work_dir: str
    gate: bool
    chains: int
    iter: int
    warmup: int
    seed: int


def design_payload(config: StudyConfig) -> tuple[dict, list[dict]]:
    """The design spec as plain dicts, in the shape R's ``parse_design_spec`` reads."""
    spec = config.design_spec
    if spec is None:
        raise ValueError("study has no design:/hypotheses: (or legacy contrasts:) block")
    design = {"factor": spec.factor, "reference": spec.reference, "levels": spec.levels,
              "covariates": spec.covariates}
    hyps = [{k: v for k, v in dataclasses.asdict(h).items() if v is not None} for h in spec.hypotheses]
    return design, hyps


def resolve_paths(config: StudyConfig, analysis: str) -> tuple[Path, Path, Path]:
    """(input CSV, published table dir, working dir for fits) for a paradigm-scoped config."""
    if analysis not in BAYES_TABLES:
        raise ValueError(f"bayes: analysis '{analysis}' not supported (known: {', '.join(BAYES_TABLES)})")
    paradigm = config.paradigm_name or ""
    data_dir = config.output_dir / analysis / "data"
    tbl_dir = config.results_dir / "tables" / paradigm / analysis
    return data_dir / BAYES_TABLES[analysis].csv, tbl_dir, data_dir / "bayes"


def build_job(config: StudyConfig, analysis: str, dv: str, hypotheses: list[str] | None = None,
              prior_sets: tuple[str, ...] = ("W", "I"), informative: Path | None = None, rope_sd: float | None = None,
              gate: bool = False, chains: int = 4, iter: int = 2000, warmup: int = 1000, seed: int = 1,
              out_dir: Path | None = None) -> BayesJob:
    """One job. ``out_dir`` redirects the table and the fits (``<out_dir>/<analysis>_hypotheses_bayes.csv``,
    ``<out_dir>/bayes/``) away from the study tree, e.g. when the study's results are frozen."""
    tab = BAYES_TABLES.get(analysis)
    if tab is None:
        raise ValueError(f"bayes: analysis '{analysis}' not supported (known: {', '.join(BAYES_TABLES)})")
    if dv not in tab.dvs:
        raise ValueError(f"bayes: dv '{dv}' not in {analysis}'s table (known: {', '.join(tab.dvs)})")
    csv, tbl_dir, work = resolve_paths(config, analysis)
    if out_dir is not None:
        tbl_dir, work = Path(out_dir), Path(out_dir) / "bayes"
    design, hyps = design_payload(config)
    if hypotheses:
        unknown = set(hypotheses) - {h["name"] for h in hyps}
        if unknown:
            raise ValueError(f"bayes: unknown hypothesis(es): {', '.join(sorted(unknown))}")
        hyps = [h for h in hyps if h["name"] in hypotheses]
    bayes_cfg = config.raw.get("bayes", {}) or {}
    rope = rope_sd if rope_sd is not None else float((bayes_cfg.get("rope") or {}).get("value", 0.1))
    info = json.loads(Path(informative).read_text()) if informative else None
    sets = [s for s in prior_sets if s != "I" or info is not None]
    return BayesJob(data_csv=str(csv), dv=dv, spatial_col=tab.spatial_col, band_col=tab.band_col, design=design,
                    hypotheses=hyps, prior_sets=sets, informative=info, rope_sd=rope,
                    table_out=str(tbl_dir / f"{analysis}_hypotheses_bayes.csv"), work_dir=str(work), gate=gate,
                    chains=chains, iter=iter, warmup=warmup, seed=seed)


def run_job(job: BayesJob, timeout: float | None = None) -> subprocess.CompletedProcess:
    if not Path(job.data_csv).exists():
        raise FileNotFoundError(f"bayes: input table not found: {job.data_csv} (run the module's process step first)")
    if shutil.which("Rscript") is None:
        raise RuntimeError("bayes: Rscript not found")
    work = Path(job.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    Path(job.table_out).parent.mkdir(parents=True, exist_ok=True)
    job_file = work / f"job_{job.dv}.json"
    job_file.write_text(json.dumps(dataclasses.asdict(job), indent=2))
    cmd = ["Rscript", str(R_DIR / "bayes_analysis.R"), "--job", str(job_file)]
    return subprocess.run(cmd, text=True, timeout=timeout, check=False)


GATE_MODES = ("prior_pred", "sbc", "null", "recovery", "summarize")


def run_gate(job: BayesJob, bands: list[str] | None = None, reps: tuple[int, int, int] = (100, 50, 10),
             workers: int = 5, timeout: float | None = None) -> int:
    """The gate in order (R/bayes_gate.R), for each of the job's prior sets: prior predictive, SBC and null at
    ``bands`` (a reduced design; None = all), recovery at the full design, then the verdict
    (``<work_dir>/gate/<dv>/gate.json``; ``gate_I.json`` for set I). Finished replications are skipped on a re-run.
    Hours at a real design."""
    if not Path(job.data_csv).exists():
        raise FileNotFoundError(f"bayes: input table not found: {job.data_csv}")
    work = Path(job.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    job_file = work / f"job_{job.dv}.json"
    job_file.write_text(json.dumps(dataclasses.asdict(job), indent=2))
    n = {"sbc": reps[0], "null": reps[1], "recovery": reps[2]}
    for prior_set in job.prior_sets:
        for mode in GATE_MODES:
            cmd = ["Rscript", str(R_DIR / "bayes_gate.R"), "--job", str(job_file), "--mode", mode,
                   "--workers", str(workers), "--set", prior_set]
            if mode in n:
                cmd += ["--reps", str(n[mode])]
            if bands and mode in ("sbc", "null"):
                cmd += ["--bands", ",".join(bands)]
            res = subprocess.run(cmd, text=True, timeout=timeout, check=False)
            if res.returncode != 0:
                return res.returncode
    return 0
