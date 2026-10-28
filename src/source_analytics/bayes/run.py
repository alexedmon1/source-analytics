"""Running the engine's R side: one fit job (``R/bayes/analysis.R``) or the gate (``R/bayes/gate.R``)."""
from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

from .job import BayesJob

GATE_MODES = ("prior_pred", "sbc", "null", "equivalence", "recovery", "summarize")


def r_dir() -> Path:
    """The package's R/ directory (``SOURCE_ANALYTICS_R_DIR``, the checkout, or the wheel's data files); the engine is
    in its ``bayes/`` subdirectory."""
    from ..analyses.base import find_r_script_dir  # lazy: the engine depends on the analyses only for this lookup

    return find_r_script_dir()


def _write_job(job: BayesJob) -> Path:
    work = Path(job.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    job_file = work / f"job_{job.dv}.json"
    job_file.write_text(json.dumps(dataclasses.asdict(job), indent=2))
    return job_file


def run_job(job: BayesJob, timeout: float | None = None) -> subprocess.CompletedProcess:
    if not Path(job.data_csv).exists():
        raise FileNotFoundError(f"bayes: input table not found: {job.data_csv} (run the module's process step first)")
    if shutil.which("Rscript") is None:
        raise RuntimeError("bayes: Rscript not found")
    job_file = _write_job(job)
    Path(job.table_out).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["Rscript", str(r_dir() / "bayes" / "analysis.R"), "--job", str(job_file)]
    return subprocess.run(cmd, text=True, timeout=timeout, check=False)


def check_env(manifest: Path | None = None) -> int:
    """``source-analytics bayes --check-env``: print what R/bayes/check_env.R reports; its exit status (1 if a required
    R package or CmdStan is missing)."""
    if shutil.which("Rscript") is None:
        print('{"missing": ["Rscript"]}')
        return 1
    cmd = ["Rscript", str(r_dir() / "bayes" / "check_env.R")]
    if manifest is not None:
        cmd += ["--against", str(manifest)]
    return subprocess.run(cmd, text=True, check=False).returncode


def run_gate(job: BayesJob, bands: list[str] | None = None, reps: dict[str, int] | None = None,
             workers: int = 5, timeout: float | None = None) -> int:
    """The gate in order (R/bayes/gate.R), for each of the job's prior sets: prior predictive, SBC, null and
    equivalence (each equivalence contrast at its margin; skipped by R without equivalence hypotheses) at
    ``bands`` (a reduced design; None = all), recovery at the full design with the real fit's sampler settings
    (the job's chains/iter/warmup; the Conductor 2026-10-19: 2-chain recovery fits missed R-hat < 1.05), then the verdict
    (``<work_dir>/gate/<dv>/gate.json``; ``gate_I.json`` for set I). Finished replications are skipped on a re-run.
    Hours at a real design."""
    if not Path(job.data_csv).exists():
        raise FileNotFoundError(f"bayes: input table not found: {job.data_csv}")
    job_file = _write_job(job)
    gate_r = r_dir() / "bayes" / "gate.R"
    n = {"sbc": 100, "null": 100, "recovery": 10, "equivalence": 100}             # engine defaults, then the study's
    n.update({k: v for k, v in (job.gate_settings or {}).items() if k in n and v is not None})
    n.update(reps or {})
    for prior_set in job.prior_sets:
        for mode in GATE_MODES:
            cmd = ["Rscript", str(gate_r), "--job", str(job_file), "--mode", mode,
                   "--workers", str(workers), "--set", prior_set]
            if mode in n:
                cmd += ["--reps", str(n[mode])]
            if mode == "recovery":
                cmd += ["--chains", str(job.chains), "--iter", str(job.iter), "--warmup", str(job.warmup)]
            if bands and mode in ("sbc", "null", "equivalence"):
                cmd += ["--bands", ",".join(bands)]
            res = subprocess.run(cmd, text=True, timeout=timeout, check=False)
            if res.returncode != 0:
                return res.returncode
    return 0
