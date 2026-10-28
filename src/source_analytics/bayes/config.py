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
      outcomes:                               # what `source-analytics bayes` runs when --analysis/--dv are not given
        roi_psd: [absolute]
        roi_aperiodic: [exponent]

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
class BayesConfig:
    rope_sd: float = 0.1
    prior_sets: tuple[str, ...] = PRIOR_SETS
    informative: Path | None = None
    fit: FitSettings = field(default_factory=FitSettings)
    gate: GateSettings = field(default_factory=GateSettings)
    outcomes: dict[str, tuple[str, ...]] = field(default_factory=dict)


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
    _check_keys(block, {"rope", "prior_sets", "priors", "fit", "gate", "outcomes"}, "bayes")

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

    outcomes = {}
    oc = block.get("outcomes") or {}
    if not isinstance(oc, dict):
        raise BayesConfigError("bayes.outcomes: expected a mapping analysis -> [dv, ...]")
    for analysis, dvs in oc.items():
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
                       outcomes=outcomes)
