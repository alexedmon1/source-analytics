"""Building one engine job: the input table, the design spec, prior sets and sampler settings, written as JSON for R."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import StudyConfig
from .tables import BAYES_TABLES


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
