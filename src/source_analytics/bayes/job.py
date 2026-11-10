"""Building one engine job: the input table, the design spec, prior sets and sampler settings, written as JSON for R."""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..config import StudyConfig
from .config import BayesConfig, parse_bayes_config
from .priors import load_informative
from .tables import BAYES_TABLES


@dataclass
class BayesJob:
    """Everything R needs for one run (written to JSON)."""

    data_csv: str
    dv: str
    spatial_col: str | None                                         # None: no ROI in the model
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
    gate_settings: dict = dataclasses.field(default_factory=dict)   # bayes.gate: part of the gate id (E3)
    provenance: dict = dataclasses.field(default_factory=dict)      # package, study and prior files (the manifest)
    center: bool = True                                             # False: uncentred outcome (declared tables)


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


def build_job(config: StudyConfig, analysis: str, dv: str, bayes: BayesConfig | None = None,
              hypotheses: list[str] | None = None, prior_sets: tuple[str, ...] | None = None, gate: bool = False,
              out_dir: Path | None = None, study_path: Path | None = None) -> BayesJob:
    """One job. Everything that changes a number comes from ``bayes`` (the study's ``bayes:`` block, bayes/config.py;
    default: parsed from ``config.raw`` with paths relative to the working directory). The other arguments say what
    to run and where: ``hypotheses`` and ``prior_sets`` select from what the study declares; ``out_dir`` redirects the
    table and the fits (``<out_dir>/<analysis>_hypotheses_bayes.csv``, ``<out_dir>/bayes/``) away from the study
    tree, e.g. when the study's results are frozen. Set I is dropped when its file has no entry for ``dv``.
    ``study_path`` (the study YAML) is hashed into the job's provenance for the manifest."""
    if bayes is None:
        bayes = parse_bayes_config(config.raw.get("bayes"), Path.cwd())
    if analysis in bayes.tables:
        return _build_declared_job(config, bayes.tables[analysis], dv, bayes, hypotheses, prior_sets, gate, out_dir,
                                   study_path)
    tab = BAYES_TABLES.get(analysis)
    if tab is None:
        raise ValueError(f"bayes: analysis '{analysis}' not supported (known: {', '.join(BAYES_TABLES)})")
    if dv not in tab.dvs:
        raise ValueError(f"bayes: dv '{dv}' not in {analysis}'s table (known: {', '.join(tab.dvs)})")
    csv, tbl_dir, work = resolve_paths(config, analysis)
    if out_dir is not None:
        tbl_dir, work = Path(out_dir), Path(out_dir) / "bayes"
    design, hyps = design_payload(config)
    info = None
    if bayes.informative is not None:
        bands = list(config.bands) if tab.band_col and config.bands else None
        info = load_informative(bayes.informative, dv, design, hyps, bands=bands) or None
    if hypotheses:
        unknown = set(hypotheses) - {h["name"] for h in hyps}
        if unknown:
            raise ValueError(f"bayes: unknown hypothesis(es): {', '.join(sorted(unknown))}")
        hyps = [h for h in hyps if h["name"] in hypotheses]
    sets = list(bayes.prior_sets)
    if prior_sets is not None:
        extra = set(prior_sets) - set(sets)
        if extra:
            raise ValueError(f"bayes: prior set(s) {', '.join(sorted(extra))} not in the study's bayes.prior_sets")
        sets = [s for s in sets if s in prior_sets]
    sets = [s for s in sets if s != "I" or info is not None]
    f, g = bayes.fit, bayes.gate
    gate_settings = {"sbc": g.sbc, "null": g.null, "recovery": g.recovery, "equivalence": g.equivalence,
                     "reduced_bands": list(g.reduced_bands) if g.reduced_bands and tab.band_col else None}
    provenance = {"source_analytics": _package(),
                  "study": _file_record(study_path),
                  "informative": _file_record(bayes.informative) if "I" in sets else None}
    return BayesJob(data_csv=str(csv), dv=dv, spatial_col=tab.spatial_col, band_col=tab.band_col, design=design,
                    hypotheses=hyps, prior_sets=sets, informative=info, rope_sd=bayes.rope_sd,
                    table_out=str(tbl_dir / f"{analysis}_hypotheses_bayes.csv"), work_dir=str(work), gate=gate,
                    chains=f.chains, iter=f.iter, warmup=f.warmup, seed=f.seed, gate_settings=gate_settings,
                    provenance=provenance)


def _select(bayes: BayesConfig, hyps: list[dict], hypotheses, prior_sets, info) -> tuple[list[dict], list[str]]:
    if hypotheses:
        unknown = set(hypotheses) - {h["name"] for h in hyps}
        if unknown:
            raise ValueError(f"bayes: unknown hypothesis(es): {', '.join(sorted(unknown))}")
        hyps = [h for h in hyps if h["name"] in hypotheses]
    sets = list(bayes.prior_sets)
    if prior_sets is not None:
        extra = set(prior_sets) - set(sets)
        if extra:
            raise ValueError(f"bayes: prior set(s) {', '.join(sorted(extra))} not in the study's bayes.prior_sets")
        sets = [s for s in sets if s in prior_sets]
    return hyps, [s for s in sets if s != "I" or info is not None]


def _safe(x: str) -> str:
    """The engine's name for a band or group (R/bayes/model.R .safe: alphanumerics only)."""
    return "".join(ch for ch in str(x) if ch.isalnum())


def _build_declared_job(config: StudyConfig, spec, dv: str, bayes: BayesConfig, hypotheses, prior_sets, gate: bool,
                        out_dir, study_path) -> BayesJob:
    """A job on a declared table (bayes.tables; bayes/declared.py): the table is assembled into the work dir."""
    from .declared import assemble
    if dv != spec.value:
        raise ValueError(f"bayes: a declared table's outcome is its value column ({spec.value}), got {dv!r}")
    tbl_dir = Path(out_dir) if out_dir is not None else config.results_dir / "tables" / "bayes" / spec.name
    work = Path(out_dir) / "bayes" if out_dir is not None else config.output_dir / "bayes" / spec.name
    design, hyps = design_payload(config)
    data, table_prov = assemble(spec, config.output_dir, design["factor"], config.paradigm_name)
    levels = design.get("levels")
    if levels and not set(data[design["factor"]]) <= set(levels):
        extra = sorted(set(data[design["factor"]]) - set(levels))
        raise ValueError(f"bayes.tables.{spec.name}: groups {extra} are not design levels {levels}; map them in groups")
    work.mkdir(parents=True, exist_ok=True)
    csv = work / f"{spec.name}_data.csv"
    data.to_csv(csv, index=False)
    info = None
    if bayes.informative is not None:
        # one set-I file may serve several declared tables: check every entry's band against all of them (a misspelt
        # band still fails), then keep this table's entries only
        every = sorted({o for t in bayes.tables.values() for o in t.outcomes})
        info = load_informative(bayes.informative, dv, design, hyps, bands=every, levels=not spec.center) or {}
        mine = tuple(f"{p}_{_safe(o)}_" for o in spec.outcomes for p in ("x", "lv"))
        info = {k: v for k, v in info.items() if k.startswith(mine)} or None
    hyps, sets = _select(bayes, hyps, hypotheses, prior_sets, info)
    f, g = bayes.fit, bayes.gate
    gate_settings = {"sbc": g.sbc, "null": g.null, "recovery": g.recovery, "equivalence": g.equivalence,
                     "reduced_bands": list(g.reduced_bands) if g.reduced_bands else None}
    provenance = {"source_analytics": _package(), "study": _file_record(study_path),
                  "informative": _file_record(bayes.informative) if "I" in sets else None,
                  "table": {**table_prov, "data_csv_sha256": _file_record(csv)["sha256"]}}
    return BayesJob(data_csv=str(csv), dv=dv, spatial_col="roi" if spec.spatial else None, band_col="outcome",
                    design=design, hypotheses=hyps, prior_sets=sets, informative=info, rope_sd=bayes.rope_sd,
                    table_out=str(tbl_dir / f"{spec.name}_hypotheses_bayes.csv"), work_dir=str(work), gate=gate,
                    chains=f.chains, iter=f.iter, warmup=f.warmup, seed=f.seed, gate_settings=gate_settings,
                    provenance=provenance, center=spec.center)


def _file_record(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _package() -> dict:
    from .. import __version__
    from .._version import git_describe

    return {"version": __version__, "git_describe": git_describe()}
