"""Reading a Bayesian engine run without the code: the output schema and ``report.md`` (engine plan E4).

A run writes ``<analysis>_hypotheses_bayes.csv`` (one row per hypothesis x band x spatial cell x prior set; the
columns are :data:`TABLE_COLUMNS`, schema :data:`TABLE_SCHEMA`), its manifest beside it, and its fits and job in the
work directory. :func:`write_report` turns those into one Markdown page per outcome:

- the run: study, inputs and their hashes, the model and every prior by catalogue id and version, ROPE, sampler;
- the results, prior sets side by side, each estimate also in the outcome's own unit, and beside each row the
  gate's false-claim rate for that band when there is no effect (how often a claim like it arises by chance at this
  design);
- the gate per prior set (passed, failed or absent, with its rules), convergence;
- the prior predictive (from the gate) and the posterior predictive (``R/bayes/ppc.R`` on the saved fits): where the
  observed spread and mean of each band fall among data replicated from the model.

The report reads files only. Running the posterior predictive needs R and the saved fits (``ppc=True``).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import catalogue as C
from . import r_dir

__all__ = ["TABLE_SCHEMA", "TABLE_COLUMNS", "check_table", "run_ppc", "build_report", "write_report"]

TABLE_SCHEMA = 1
#: The columns of ``<analysis>_hypotheses_bayes.csv``, schema 1 (R/bayes/readout.R and analysis.R).
TABLE_COLUMNS = ("hypothesis", "kind", "role", "label", "dv", "prior_set", "band", "spatial", "estimate", "hdi_lo",
                 "hdi_hi", "p_direction", "p_rope", "rope_lo", "rope_hi", "log10_bf10", "decision", "rhat",
                 "ess_bulk", "status", "fdr_family", "max_rhat_scale", "max_rhat", "n_divergent", "gate_passed",
                 "gate_id")



def _as_list(x) -> list:
    """A manifest list that JSON may have unboxed to a scalar (one band: "Low Gamma", not ["Low Gamma"])."""
    return [] if x is None else [x] if isinstance(x, str) else list(x)

def check_table(df: pd.DataFrame) -> None:
    """Raise if a results table does not have schema 1's columns."""
    missing = [c for c in TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"not a schema-{TABLE_SCHEMA} Bayesian results table: missing {missing}")


def _manifest(table_csv: Path) -> dict:
    p = table_csv.with_name(table_csv.stem + "_manifest.json")
    return json.loads(p.read_text()) if p.exists() else {}


def run_ppc(job_file: Path, sets, ndraws: int = 1000) -> dict:
    """Run ``R/bayes/ppc.R`` for each prior set whose fit exists; returns set -> CSV path."""
    job = json.loads(Path(job_file).read_text())
    out = {}
    for s in sets:
        fit = Path(job["work_dir"]) / f"fit_{job['dv']}_{s}.rds"
        if not fit.exists():
            continue
        csv = Path(job["work_dir"]) / f"ppc_{job['dv']}_{s}.csv"
        r = subprocess.run(["Rscript", str(r_dir() / "bayes" / "ppc.R"), "--job", str(job_file), "--set", s, "--out",
                            str(csv), "--ndraws", str(ndraws)], capture_output=True, text=True)
        if r.returncode == 0:
            out[s] = csv
    return out


def _fmt(x, nd=2):
    return "" if pd.isna(x) else f"{x:+.{nd}f}"


def _gate_dir(man_dv: dict, prior_set: str):
    g = ((man_dv.get("sets") or {}).get(prior_set) or {}).get("gate") or {}
    return Path(g["gate_file"]).parent if g.get("gate_file") else None


def build_report(table_csv, dv: str | None = None, ppc_files: dict | None = None) -> str:
    """The Markdown report of one outcome (``dv``; default: the table's only one)."""
    table_csv = Path(table_csv)
    df = pd.read_csv(table_csv)
    check_table(df)
    dvs = sorted(df.dv.unique())
    dv = dv or (dvs[0] if len(dvs) == 1 else None)
    if dv is None:
        raise ValueError(f"table has several outcomes {dvs}; name one")
    df = df[df.dv == dv]
    man = _manifest(table_csv).get(dv, {})
    job = json.loads(Path(man["job_file"]).read_text()) if man.get("job_file") and Path(man["job_file"]).exists() \
        else {}
    sets = sorted(df.prior_set.unique(), key=lambda s: ("W", "I").index(s) if s in ("W", "I") else 9)
    work = Path(job.get("work_dir", table_csv.parent))
    scale = {}
    sc = work / f"scale_{dv}.csv"
    if sc.exists():
        s = pd.read_csv(sc)
        scale = dict(zip(s.band.astype(str), s.scale))
    safe = lambda b: "".join(ch for ch in str(b) if ch.isalnum())     # noqa: E731
    E = C.catalogue_entries()
    no_roi = job.get("spatial_col") is None and bool(job)
    uncentred = job.get("center") is False
    model_id = ("model.group_levels_uncentred" if uncentred else "model.group_levels_noroi" if no_roi
                else "model.group_levels")
    prov = job.get("provenance") or {}
    settings = man.get("settings") or {}
    inp = man.get("input") or {}
    L = [f"# Bayesian engine report: {table_csv.stem.replace('_hypotheses_bayes', '')} / {dv}", "",
         f"Written {datetime.now().isoformat(timespec='seconds')} from `{table_csv.name}` (results table schema "
         f"{TABLE_SCHEMA}). Engine {man.get('engine_version', '?')}, coding {man.get('coding', '?')}.", "",
         "## Run", ""]
    study = prov.get("study") or {}
    if isinstance(study, dict) and study:
        L.append(f"- **Study:** `{study.get('path')}` (sha256 {str(study.get('sha256'))[:12]})")
    elif study:
        L.append(f"- **Study:** {study}")
    tab = prov.get("table")
    if tab:
        ins = "; ".join(f"`{v['path']}` ({v['sha256'][:12]})" for v in tab["inputs"].values())
        L.append(f"- **Declared table** `{tab['name']}`: {ins}; {tab['n_rows']} rows, {tab['n_subjects']} subjects")
    if inp:
        L.append(f"- **Input:** {inp.get('n_rows_used')} rows, {inp.get('n_subjects')} subjects; groups "
                 f"{inp.get('groups')}; outcomes (bands): {', '.join(map(str, _as_list(inp.get('bands'))))}")
    L.append(f"- **Model:** `{model_id}` v{E[model_id].version}" + (" (no ROI)" if no_roi else "")
             + (": uncentred, so a hypothesis whose weights do not sum to 0 is a group's level against 0"
                if uncentred else ""))
    pri = [p for p in E[model_id].rests_on if p.startswith("prior.")]
    inf = prov.get("informative")
    inf_src = inf.get("path") if isinstance(inf, dict) else inf
    L.append("- **Priors:** " + ", ".join(f"`{p}` v{E[p].version}" for p in pri if p != "prior.i.contrast_entry")
             + (f"; set I: `prior.i.contrast_entry` v{E['prior.i.contrast_entry'].version} from "
                f"`{inf_src}`" if "I" in sets else ""))
    if job.get("informative"):
        L.append("- **Set-I entries** (contrast vs the reference, between-subject SD): "
                 + "; ".join(f"{k} N({v[0]}, {v[1]})" for k, v in job["informative"].items()))
    L.append(f"- **ROPE:** +-{settings.get('rope_sd', '?')} between-subject SD (`rule.contrast_decision` "
             f"v{E['rule.contrast_decision'].version}); sampler {settings.get('chains')} chains x "
             f"{settings.get('iter', 0) - settings.get('warmup', 0) if settings else '?'} draws after "
             f"{settings.get('warmup')} warm-up, seed {settings.get('seed')}")
    L += ["", "## Results", "",
          "Posterior median [95% HDI] of each contrast in between-subject SD (the unit of a per-cell Hedges g); in "
          "brackets after it, the median in the outcome's own unit. P(dir): probability of the sign; P(ROPE): "
          "probability inside the ROPE. *False effect* is the gate's rate of 'effect' claims for that band when the "
          "simulated groups do not differ.", ""]
    null = {}
    for s in sets:
        gd = _gate_dir(man, s)
        f = gd / f"null_summary{'' if s == 'W' else '_' + s}.csv" if gd else None
        if f is not None and f.exists():
            n = pd.read_csv(f)
            null[s] = {(str(r.band), str(r.hypothesis)): r.claim_effect for r in n.itertuples()}
    head = "| hypothesis | band | cell | " + " | ".join(
        f"{s}: median [HDI] | {s}: P(dir) | {s}: P(ROPE) | {s}: decision | {s}: false effect" for s in sets) + " |"
    L += [head, "|" + "---|" * (3 + 5 * len(sets))]
    for (h, b, cell), g in df.groupby(["hypothesis", "band", "spatial"], sort=False):
        cells = []
        for s in sets:
            r = g[g.prior_set == s]
            if r.empty:
                cells += [""] * 5
                continue
            r = r.iloc[0]
            sc_b = scale.get(safe(b))
            nat = f" ({r.estimate * sc_b:+.3g})" if sc_b else ""
            gate_note = "" if r.gate_passed is True or str(r.gate_passed) == "True" else (
                " (gate failed)" if str(r.gate_passed) == "False" else " (no gate)")
            fe = null.get(s, {}).get((safe(b), str(h)))
            cells += [f"{_fmt(r.estimate)} [{_fmt(r.hdi_lo)}, {_fmt(r.hdi_hi)}]{nat}", f"{r.p_direction:.3f}",
                      f"{r.p_rope:.3f}", f"{r.decision}{gate_note}", "" if fe is None else f"{fe:.3f}"]
        L.append(f"| {h} | {b} | {cell} | " + " | ".join(cells) + " |")
    L += ["", "## Gate (simulation checks at this design)", ""]
    for s in sets:
        gd = _gate_dir(man, s)
        gf = gd / ("gate.json" if s == "W" else f"gate_{s}.json") if gd else None
        rec = ((man.get("sets") or {}).get(s) or {}).get("gate") or {}
        if gf is None or not gf.exists():
            L.append(f"- **{s}:** no gate ({rec.get('reason', 'not run')}); its rows are not covered by a gate.")
            continue
        v = json.loads(gf.read_text())
        ru = v.get("rules", {})
        bits = []
        if "prior_pred" in ru:
            bits.append(f"prior predictive {ru['prior_pred']['min_pctile']:.2f}-{ru['prior_pred']['max_pctile']:.2f}")
        if "sbc" in ru:
            bits.append(f"SBC min p {ru['sbc']['min_p_roi_average']:.4f}, coverage {ru['sbc']['min_coverage_roi_average']:.2f}")
        if "null" in ru:
            bits.append(f"null: effect claims <= {ru['null']['max_claim_effect']:.3f}, direction <= {ru['null']['max_claim_dir']:.3f}")
        if "recovery" in ru:
            bits.append(f"recovery coverage {ru['recovery']['coverage_roi_average']:.2f}")
        failed = [k for k, r in ru.items() if not r.get("pass")]
        verdict = "passed" if v.get("passed") else f"**failed** ({', '.join(failed) or 'incomplete'})"
        same = rec.get("found_id") == rec.get("gate_id")
        L.append(f"- **{s}:** {verdict}; " + "; ".join(bits) + ("" if same else
                 " (gate id differs from this fit's: the verdict does not cover it)"))
    L += ["", f"Rules (`gate.rules` v{E['gate.rules'].version}): prior predictive 5th-95th percentile; SBC p above "
          "0.01 / estimands and coverage >= 0.85; false effect claims <= 0.05 and direction claims <= 0.10 with no "
          "effect; recovery coverage >= 0.85; R-hat < 1.05.", "", "## Convergence", ""]
    for s in sets:
        c = ((man.get("sets") or {}).get(s) or {}).get("convergence") or {}
        if c:
            L.append(f"- **{s}:** R-hat max {c.get('max_rhat_estimands', float('nan')):.3f} (estimands), "
                     f"{c.get('max_rhat_scale', float('nan')):.3f} (scales); min bulk ESS "
                     f"{c.get('min_ess_bulk', float('nan')):.0f}; {c.get('n_divergent')} divergent transitions")
    L += ["", "## Prior and posterior predictive", "",
          "Percentile of the observed statistic among data replicated from the prior (gate) or the posterior "
          "(ppc.R); values below 0.05 or above 0.95 are marked *.", ""]
    for s in sets:
        gd = _gate_dir(man, s)
        pp = gd / f"prior_pred{'' if s == 'W' else '_' + s}.csv" if gd else None
        if pp is not None and pp.exists():
            p = pd.read_csv(pp)
            L.append(f"- **{s}, prior predictive:** " + "; ".join(
                f"{r.band} {r.stat} {r.pctile:.2f}{'*' if not 0.05 <= r.pctile <= 0.95 else ''}"
                for r in p.itertuples()))
        f = (ppc_files or {}).get(s) or (work / f"ppc_{dv}_{s}.csv")
        if Path(f).exists():
            q = pd.read_csv(f)
            flag = q[(q.pctile < 0.05) | (q.pctile > 0.95)]
            L.append(f"- **{s}, posterior predictive:** {len(q)} statistics (band x group x SD/MAD/mean); "
                     + ("all within 0.05-0.95" if flag.empty else "outside 0.05-0.95: " + "; ".join(
                         f"{r.band} {r.group} {r.stat} {r.pctile:.3f}" for r in flag.itertuples())))
    L += ["", "## Files", "", f"- results: `{table_csv}`", f"- manifest: `{table_csv.stem}_manifest.json`",
          f"- job: `{man.get('job_file')}`", f"- work directory (fits, priors, scales, ppc): `{work}`"]
    return "\n".join(L) + "\n"


def write_report(table_csv, dv: str | None = None, ppc: bool = True) -> Path:
    """Write ``<table stem>_<dv>_report.md`` beside the table; run the posterior predictive first if ``ppc``."""
    table_csv = Path(table_csv)
    df = pd.read_csv(table_csv)
    dv = dv or sorted(df.dv.unique())[0]
    files = {}
    man = _manifest(table_csv).get(dv, {})
    if ppc and man.get("job_file"):
        files = run_ppc(Path(man["job_file"]), sorted(df[df.dv == dv].prior_set.unique()))
    out = table_csv.with_name(f"{table_csv.stem}_{dv}_report.md")
    out.write_text(build_report(table_csv, dv, files))
    return out
