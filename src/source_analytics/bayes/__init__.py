"""Bayesian hypothesis engine (docs/methods/BAYES_ENGINE.md): the Python side of the separate ``bayes`` step.

Thin by design. It resolves where a module's per-subject table lives and where results go, serialises the (profile-
narrowed) design spec into a job file, and calls R (``R/bayes/analysis.R`` for fits, ``R/bayes/gate.R`` for the gate).
The model, priors, gate and readout are in R (``R/bayes/engine.R``). Nothing here imports brms or Stan, and the
frequentist outputs are never touched: results go to ``<analysis>_hypotheses_bayes.csv`` beside
``<analysis>_hypotheses.csv``.

    config.py  the study YAML's bayes: block (BayesConfig, parse_bayes_config)
    priors.py  informative prior files (load_informative)
    tables.py  outcome-table registry (BAYES_TABLES)
    job.py     BayesJob, build_job, design_payload, resolve_paths
    run.py     run_job, run_gate, check_env, r_dir
"""
from .config import BayesConfig, BayesConfigError, FitSettings, GateSettings, parse_bayes_config
from .job import BayesJob, build_job, design_payload, resolve_paths
from .run import GATE_MODES, check_env, r_dir, run_gate, run_job
from .priors import load_informative
from .tables import BAYES_TABLES, BayesTable

__all__ = ["BAYES_TABLES", "BayesConfig", "BayesConfigError", "BayesJob", "BayesTable", "FitSettings", "GATE_MODES",
           "GateSettings", "build_job", "check_env", "design_payload", "load_informative", "parse_bayes_config", "r_dir",
           "resolve_paths", "run_gate", "run_job"]
