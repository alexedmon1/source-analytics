"""Bayesian hypothesis engine (docs/methods/BAYES_ENGINE.md): the Python side of the separate ``bayes`` step.

Thin by design. It resolves where a module's per-subject table lives and where results go, serialises the (profile-
narrowed) design spec into a job file, and calls R (``R/bayes/analysis.R`` for fits, ``R/bayes/gate.R`` for the gate).
The model, priors, gate and readout are in R (``R/bayes/engine.R``). Nothing here imports brms or Stan, and the
frequentist outputs are never touched: results go to ``<analysis>_hypotheses_bayes.csv`` beside
``<analysis>_hypotheses.csv``.

    tables.py  outcome-table registry (BAYES_TABLES)
    job.py     BayesJob, build_job, design_payload, resolve_paths
    run.py     run_job, run_gate, r_dir
"""
from .job import BayesJob, build_job, design_payload, resolve_paths
from .run import GATE_MODES, r_dir, run_gate, run_job
from .tables import BAYES_TABLES, BayesTable

__all__ = ["BAYES_TABLES", "BayesJob", "BayesTable", "GATE_MODES", "build_job", "design_payload", "r_dir",
           "resolve_paths", "run_gate", "run_job"]
