"""The study YAML's ``bayes:`` block: everything that changes a number (engine plan E2, principle 1).

    bayes:
      rope: {mode: sd, value: 0.1}            # ROPE for plain contrasts, between-subject SD units
      prior_sets: [W, I]                      # default: [W, I] with priors.I, else [W]; I needs priors.I
      priors:
        I: priors/forge_informative.yaml      # relative to the study YAML (bayes/priors.py)
      fit: {chains: 4, iter: 2000, warmup: 1000, seed: 1}
      gate:
        reps: {sbc: 100, null: 100, recovery: 10, equivalence: 100}   # equivalence: only with equivalence hypotheses
        reduced_bands: [Theta, Low Gamma]     # SBC, null and equivalence; recovery runs at the full design
      tables:                                 # declared outcome tables (field-method plan F6b; bayes/declared.py)
        deficit_along_field:
          source: roi_evoked                  # an analysis's per-subject table in each paradigm's output ...
          # csv: path/values.csv              # ... or an external table (relative to the study YAML)
          paradigms: [assr_40hz, chirp]       # source tables: which paradigms' outputs to read
          # output_dir: ../other/analytics    # source tables: another output tree (default: the study's)
          value: value                        # the outcome column
          spatial: none                       # none (no ROI in the model) or a column used as the engine's ROI
          outcomes:                           # engine outcome ("band") label -> rows (column: value or [values])
            40 Hz ASSR ITC: {paradigm: assr_40hz, roi: assr40_D, measure_name: itc_40hz}
            deep midline ITC: {paradigm: assr_40hz, roi: [Brainstem_Tectum, Cerebellum, Thalamus],
                               measure_name: itc_40hz, combine: mean}
          groups: {G2_D1pre: KO, G6_D1pre: WT}  # table group -> design level; rows of other groups are dropped
      outcomes:                               # what `source-analytics bayes` runs when --analysis/--dv are not given
        roi_psd: [absolute]
        roi_aperiodic: [exponent]
        deficit_along_field: [value]          # a declared table: its value column

Every key is optional; absent keys take the engine defaults below. Unknown keys and bad values raise
``BayesConfigError`` before R is started. CLI flags say only what to run and where (the engine plan, section 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .tables import BAYES_TABLES

PRIOR_SETS = ("W", "I")


class BayesConfigError(ValueError):
    """The ``bayes:`` block or a file it points to is invalid."""


@dataclass(frozen=True)
class FitSettings:
    chains: int = 4
    iter: int = 2000
    warmup: int = 1000
    seed: int = 1


@dataclass(frozen=True)
class GateSettings:
    sbc: int = 100
    null: int = 100
    recovery: int = 10
    equivalence: int = 100     # replications with each equivalence contrast at its margin (engine E5)
    reduced_bands: tuple[str, ...] | None = None

    @property
    def reps(self) -> dict[str, int]:
        return {"sbc": self.sbc, "null": self.null, "recovery": self.recovery, "equivalence": self.equivalence}


@dataclass(frozen=True)
class TableSpec:
    """A declared outcome table (``bayes.tables.<name>``): where its rows come from and how they become outcomes."""

    name: str
    value: str
    outcomes: dict
    source: str | None = None
    csv: Path | None = None
    paradigms: tuple[str, ...] | None = None
    spatial: str | None = None                  # None: no ROI in the model
    groups: dict | None = None
    subject: str = "subject"
    group: str = "group"
    subject_groups: Path | None = None          # csv tables without a group column: subject,group file
    output_dir: Path | None = None              # source tables: read this output tree instead of the study's


@dataclass(frozen=True)
class BayesConfig:
    rope_sd: float = 0.1
    prior_sets: tuple[str, ...] = PRIOR_SETS
    informative: Path | None = None
    fit: FitSettings = field(default_factory=FitSettings)
    gate: GateSettings = field(default_factory=GateSettings)
    outcomes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    tables: dict[str, TableSpec] = field(default_factory=dict)


def _check_keys(block: dict, allowed: set[str], where: str) -> None:
    if not isinstance(block, dict):
        raise BayesConfigError(f"{where}: expected a mapping, got {type(block).__name__}")
    unknown = set(block) - allowed
    if unknown:
        raise BayesConfigError(f"{where}: unknown key(s) {', '.join(sorted(unknown))} "
                               f"(allowed: {', '.join(sorted(allowed))})")


def _pos_int(v, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise BayesConfigError(f"{where}: expected a positive integer, got {v!r}")
    return v


def parse_bayes_config(block: dict | None, base_dir: Path) -> BayesConfig:
    """Validate a ``bayes:`` block; relative paths resolve against ``base_dir`` (the study YAML's directory)."""
    block = block or {}
    _check_keys(block, {"rope", "prior_sets", "priors", "fit", "gate", "outcomes", "tables"}, "bayes")

    rope = block.get("rope") or {}
    _check_keys(rope, {"mode", "value"}, "bayes.rope")
    if rope.get("mode", "sd") != "sd":
        raise BayesConfigError(f"bayes.rope.mode: only 'sd' is implemented, got {rope['mode']!r}")
    rope_sd = rope.get("value", 0.1)
    if isinstance(rope_sd, bool) or not isinstance(rope_sd, (int, float)) or rope_sd <= 0:
        raise BayesConfigError(f"bayes.rope.value: expected a positive number, got {rope_sd!r}")

    priors = block.get("priors") or {}
    sets = block.get("prior_sets", list(PRIOR_SETS) if priors.get("I") is not None else ["W"])
    if isinstance(sets, str) or not isinstance(sets, list) or not sets or any(s not in PRIOR_SETS for s in sets) \
            or len(set(sets)) != len(sets):
        raise BayesConfigError(f"bayes.prior_sets: expected a list from {list(PRIOR_SETS)}, got {sets!r}")

    _check_keys(priors, {"W", "I"}, "bayes.priors")
    if priors.get("W", "default") != "default":
        raise BayesConfigError("bayes.priors.W: only the engine default ('default') is implemented")
    informative = None
    if priors.get("I") is not None:
        informative = Path(priors["I"])
        if not informative.is_absolute():
            informative = (base_dir / informative).resolve()
        if not informative.is_file():
            raise BayesConfigError(f"bayes.priors.I: file not found: {informative}")
    if "I" in sets and informative is None:
        raise BayesConfigError("bayes.prior_sets includes I but bayes.priors.I is not set (set I needs entries "
                               "derived from a logged literature search; BAYES_ENGINE.md section 5)")

    fit = block.get("fit") or {}
    _check_keys(fit, {"chains", "iter", "warmup", "seed"}, "bayes.fit")
    f = FitSettings(**{k: _pos_int(v, f"bayes.fit.{k}") for k, v in fit.items()})
    if f.warmup >= f.iter:
        raise BayesConfigError(f"bayes.fit: warmup ({f.warmup}) must be below iter ({f.iter})")

    gate = block.get("gate") or {}
    _check_keys(gate, {"reps", "reduced_bands"}, "bayes.gate")
    reps = gate.get("reps") or {}
    if isinstance(reps, dict):   # YAML reads the key `null` as None, not the string: `reps: {null: 100}` means null
        reps = {("null" if k is None else k): v for k, v in reps.items()}
    _check_keys(reps, {"sbc", "null", "recovery", "equivalence"}, "bayes.gate.reps")
    bands = gate.get("reduced_bands")
    if bands is not None and (isinstance(bands, str) or not isinstance(bands, list) or not bands
                              or not all(isinstance(b, str) for b in bands)):
        raise BayesConfigError(f"bayes.gate.reduced_bands: expected a list of band names, got {bands!r}")
    g = GateSettings(**{k: _pos_int(v, f"bayes.gate.reps.{k}") for k, v in reps.items()},
                     reduced_bands=tuple(bands) if bands else None)

    tables = _parse_tables(block.get("tables"), base_dir)
    outcomes = {}
    oc = block.get("outcomes") or {}
    if not isinstance(oc, dict):
        raise BayesConfigError("bayes.outcomes: expected a mapping analysis -> [dv, ...]")
    for analysis, dvs in oc.items():
        if analysis in tables:
            if dvs != [tables[analysis].value]:
                raise BayesConfigError(f"bayes.outcomes.{analysis}: a declared table's outcome is its value column "
                                       f"([{tables[analysis].value}]), got {dvs!r}")
            outcomes[analysis] = tuple(dvs)
            continue
        if analysis not in BAYES_TABLES:
            raise BayesConfigError(f"bayes.outcomes: analysis {analysis!r} not supported "
                                   f"(known: {', '.join(BAYES_TABLES)})")
        if isinstance(dvs, str) or not isinstance(dvs, list) or not dvs:
            raise BayesConfigError(f"bayes.outcomes.{analysis}: expected a list of outcome columns")
        bad = [d for d in dvs if d not in BAYES_TABLES[analysis].dvs]
        if bad:
            raise BayesConfigError(f"bayes.outcomes.{analysis}: unknown dv(s) {', '.join(bad)} "
                                   f"(known: {', '.join(BAYES_TABLES[analysis].dvs)})")
        outcomes[analysis] = tuple(dvs)

    return BayesConfig(rope_sd=float(rope_sd), prior_sets=tuple(sets), informative=informative, fit=f, gate=g,
                       outcomes=outcomes, tables=tables)


_TABLE_KEYS = {"source", "csv", "paradigms", "value", "spatial", "outcomes", "groups", "subject", "group",
               "subject_groups", "output_dir"}


def _parse_tables(block, base_dir: Path) -> dict[str, TableSpec]:
    """``bayes.tables``: validated table declarations (module docstring)."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise BayesConfigError("bayes.tables: expected a mapping name -> table")
    out = {}
    for name, t in block.items():
        w = f"bayes.tables.{name}"
        if name in BAYES_TABLES:
            raise BayesConfigError(f"{w}: the name is a built-in analysis; choose another")
        _check_keys(t, _TABLE_KEYS, w)
        if bool(t.get("source")) == bool(t.get("csv")):
            raise BayesConfigError(f"{w}: give exactly one of source (an analysis) or csv (a file)")
        if not t.get("value"):
            raise BayesConfigError(f"{w}.value: the outcome column is required")
        oc = t.get("outcomes")
        if not isinstance(oc, dict) or not oc:
            raise BayesConfigError(f"{w}.outcomes: expected a mapping label -> row selection")
        for label, sel in oc.items():
            if not isinstance(sel, dict) or not sel:
                raise BayesConfigError(f"{w}.outcomes.{label}: expected a mapping column -> value(s)")
            if sel.get("combine", "none") not in ("none", "mean"):
                raise BayesConfigError(f"{w}.outcomes.{label}.combine: none or mean")
        paradigms = t.get("paradigms")
        if paradigms is not None and (isinstance(paradigms, str) or not isinstance(paradigms, list)):
            raise BayesConfigError(f"{w}.paradigms: expected a list")
        spatial = t.get("spatial", "none")
        groups = t.get("groups")
        if groups is not None and not isinstance(groups, dict):
            raise BayesConfigError(f"{w}.groups: expected a mapping table group -> design level")
        csv = (base_dir / t["csv"]).resolve() if t.get("csv") else None
        odir = (base_dir / t["output_dir"]).resolve() if t.get("output_dir") else None
        if odir is not None and not odir.is_dir():
            raise BayesConfigError(f"{w}.output_dir: not a directory: {odir}")
        sg = (base_dir / t["subject_groups"]).resolve() if t.get("subject_groups") else None
        for path, key in ((csv, "csv"), (sg, "subject_groups")):
            if path is not None and not path.exists():
                raise BayesConfigError(f"{w}.{key}: file not found: {path}")
        out[name] = TableSpec(name=name, value=str(t["value"]), outcomes=dict(oc), source=t.get("source"), csv=csv,
                              paradigms=tuple(paradigms) if paradigms else None,
                              spatial=None if spatial in (None, "none") else str(spatial),
                              groups={str(k): str(v) for k, v in groups.items()} if groups else None,
                              subject=str(t.get("subject", "subject")), group=str(t.get("group", "group")),
                              subject_groups=sg, output_dir=odir)
    return out
