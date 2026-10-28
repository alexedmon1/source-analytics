"""Informative prior files (set I), in the user's terms: hypothesis x band, each entry with its source.

    version: 1
    entries:
      - dv: absolute                       # outcome column; entries for other outcomes are not used
        hypothesis: disease_effect         # a declared contrast of one group against the reference (+1 / -1)
        band: Low Gamma                    # omit for band-less outcomes (e.g. the aperiodic exponent)
        prior: {mean: 0.5, sd: 0.8}        # between-subject SD units, the contrast's prior
        source: priors/lit/2026-10-07_fmr1_resting_state.md   # required: the logged search it comes from
        derivation: "pooled rule: median of sign-agreeing regional entries"

The engine's model terms (``x_<band>_<group>``, R/bayes/model.R) are internal; this module maps entries onto them. A
JSON file ``{term: [mean, sd]}`` (the pre-E2 format) is still read as terms, unchecked against the design.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from .config import BayesConfigError


def _safe(x: str) -> str:
    """R/bayes/model.R's .safe(): drop every character that is not a letter or digit."""
    return re.sub(r"[^A-Za-z0-9]", "", x)


def _num(v, where: str, positive: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (positive and v <= 0):
        raise BayesConfigError(f"{where}: expected a {'positive ' if positive else ''}number, got {v!r}")
    return float(v)


def load_informative(path: Path, dv: str, design: dict, hypotheses: list[dict],
                     bands: list[str] | None = None) -> dict[str, list[float]]:
    """Entries of ``path`` for outcome ``dv`` as engine terms ``{x_<band>_<group>: [mean, sd]}``, in file order.

    ``design`` and ``hypotheses`` are the job's design payload (job.design_payload); ``bands`` (the study's band
    names), when given, catches a misspelt band instead of letting the engine ignore it."""
    path = Path(path)
    if path.suffix == ".json":
        return {k: [float(v[0]), float(v[1])] for k, v in json.loads(path.read_text()).items()}
    doc = yaml.safe_load(path.read_text()) or {}
    where = f"informative priors {path.name}"
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("entries"), list):
        raise BayesConfigError(f"{where}: expected 'version: 1' and a list 'entries'")
    ref = design["reference"]
    by_name = {h["name"]: h for h in hypotheses}
    out: dict[str, list[float]] = {}
    for i, e in enumerate(doc["entries"]):
        w = f"{where}, entry {i + 1}"
        if not isinstance(e, dict):
            raise BayesConfigError(f"{w}: expected a mapping")
        unknown = set(e) - {"dv", "hypothesis", "band", "prior", "source", "derivation", "note"}
        if unknown:
            raise BayesConfigError(f"{w}: unknown key(s) {', '.join(sorted(unknown))}")
        for k in ("dv", "hypothesis", "prior", "source"):
            if not e.get(k):
                raise BayesConfigError(f"{w}: '{k}' is required")
        if e["dv"] != dv:
            continue
        h = by_name.get(e["hypothesis"])
        if h is None:
            raise BayesConfigError(f"{w}: hypothesis {e['hypothesis']!r} is not declared in the study "
                                   f"(declared: {', '.join(by_name)})")
        wts = {g: float(x) for g, x in (h.get("weights") or {}).items() if float(x) != 0}
        groups = [g for g, x in wts.items() if x == 1.0]
        if h.get("kind") not in ("contrast", "equivalence") or len(wts) != 2 or wts.get(ref) != -1.0 \
                or len(groups) != 1:
            raise BayesConfigError(f"{w}: hypothesis {e['hypothesis']!r} must be one group minus the reference "
                                   f"({ref}) to carry a set-I prior (weights {h.get('weights')})")
        band = e.get("band")
        if band is not None and bands is not None and band not in bands:
            raise BayesConfigError(f"{w}: band {band!r} is not one of the study's bands ({', '.join(bands)})")
        p = e["prior"]
        if not isinstance(p, dict) or set(p) != {"mean", "sd"}:
            raise BayesConfigError(f"{w}: prior must be {{mean, sd}}")
        term = f"x_{_safe(band) if band is not None else 'all'}_{_safe(groups[0])}"
        if term in out:
            raise BayesConfigError(f"{w}: a second entry for {e['hypothesis']} / {band or '(no band)'}")
        out[term] = [_num(p["mean"], f"{w}: prior.mean"), _num(p["sd"], f"{w}: prior.sd", positive=True)]
    return out
