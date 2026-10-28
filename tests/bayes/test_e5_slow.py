"""Slow engine tests (engine plan E5): a tiny portable fit, rerun determinism, the template guard on a compiled fit,
single-band and multi-band outcomes, and a gate smoke run with an equivalence hypothesis through the CLI. Simulated
data only, so they run on any machine with the R toolchain (``source-analytics bayes --check-env``).

Marked slow (tests/bayes/conftest.py): ``SA_BAYES_SLOW=1``; ~10 min together.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from source_analytics.bayes import r_dir

ENGINE = r_dir() / "bayes" / "engine.R"
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")]


def _root(tmp_path: Path, name: str) -> Path:
    root = Path(os.environ.get("SA_BAYES_SCRATCH", tmp_path)) / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def _env(root: Path) -> dict:
    return {**os.environ, "TMPDIR": str(root)}       # cmdstanr writes its CSVs under TMPDIR


_FIT = """
suppressMessages({{ source('{engine}'); library(digest) }})
set.seed(5)
subj <- setNames(rnorm(12), paste0('s', 1:12))
mk <- function(bands) {{
  d <- expand.grid(subject = paste0('s', 1:12), roi = c('r1', 'r2', 'r3'), band = bands, stringsAsFactors = FALSE)
  d$group <- c('WT', 'KO', 'TR')[(as.integer(sub('s', '', d$subject)) - 1) %/% 4 + 1]
  d$value <- subj[d$subject] + 0.8 * (d$group == 'KO') + rnorm(nrow(d), 0, 0.5)
  d }}
hyps <- list(list(name = 'disease', kind = 'contrast', weights = list(KO = 1, WT = -1)),
             list(name = 'tr_norm', kind = 'equivalence', weights = list(TR = 1, WT = -1),
                  margin = list(mode = 'sd', value = 0.25)))
spec <- parse_design_spec(list(design = list(factor = 'group', reference = 'WT', levels = list('WT', 'KO', 'TR')),
                               hypotheses = hyps))
dg <- function(f) digest(as.data.frame(as_draws_df(f)), algo = 'sha256')
out <- list()
for (case in c('single', 'multi')) {{
  d <- if (case == 'single') mk('x')[, c('subject', 'roi', 'group', 'value')] else mk(c('Alpha', 'Low Gamma'))
  bc <- if (case == 'single') NULL else 'band'
  pW <- bayes_prepare(d, spec, 'value', 'roi', bc, 'W')
  a <- bayes_fit(pW, chains = 2, iter = 600, warmup = 300, seed = 1, model_dir = paste0('{root}/stan_', case))
  b <- bayes_fit(pW, chains = 2, iter = 600, warmup = 300, seed = 1, template = a)
  c2 <- bayes_fit(pW, chains = 2, iter = 600, warmup = 300, seed = 2, template = a)
  inf <- setNames(lapply(pW$bands, function(x) c(0, 0.3)), sprintf('x_%s_TR', pW$bands))
  pI <- bayes_prepare(d, spec, 'value', 'roi', bc, 'I', inf)
  guard <- try(bayes_fit(pI, chains = 2, iter = 600, warmup = 300, seed = 1, template = a), silent = TRUE)
  ro <- bayes_readout(a, pW, spec, rope_sd = 0.1)
  sc <- pW$scale$scale[match(pW$bands, pW$scale$band)]
  dis <- ro[ro$hypothesis == 'disease' & ro$spatial == 'ROI-average', ]
  out[[case]] <- list(bands = pW$bands, same_seed_same_draws = dg(a) == dg(b), other_seed_differs = dg(a) != dg(c2),
                      guard = inherits(guard, 'try-error') && grepl('differs from these priors', guard),
                      n_rows = nrow(ro), decisions = unique(ro$decision), kinds = unique(ro$kind),
                      truth = 0.8 / sc, lo = dis$hdi_lo, hi = dis$hdi_hi, rhat = max(ro$rhat),
                      n_divergent = ro$n_divergent[1], cols = names(ro))
}}
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA))
"""


def test_tiny_fit_determinism_guard_single_and_multi_band(tmp_path):
    root = _root(tmp_path, "e5_fit")
    code = _FIT.format(engine=ENGINE, root=root)
    r = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, env=_env(root), timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    for case, n_bands in (("single", 1), ("multi", 2)):
        o = out[case]
        assert len(o["bands"] if isinstance(o["bands"], list) else [o["bands"]]) == n_bands
        assert o["same_seed_same_draws"], f"{case}: same seed, different draws"
        assert o["other_seed_differs"]
        assert o["guard"], f"{case}: the template guard let a set-I fit run on a W template"
        assert o["n_rows"] == 2 * n_bands * (1 + 3)            # hypotheses x bands x (ROI-average + 3 ROIs)
        dec = o["decisions"] if isinstance(o["decisions"], list) else [o["decisions"]]
        assert set(dec) <= {"effect", "null", "undecided", "equivalent", "not_shown"}
        assert {"hdi_lo", "hdi_hi", "p_rope", "log10_bf10", "rhat", "ess_bulk", "max_rhat_scale"} <= set(o["cols"])
        lo, hi, t = (np.atleast_1d(o[k]) for k in ("lo", "hi", "truth"))
        assert ((lo <= t) & (t <= hi)).all(), f"{case}: the KO - WT HDI misses the simulated effect"
        assert o["rhat"] < 1.05


def test_gate_smoke_with_equivalence_through_the_cli(tmp_path):
    """A multi-band study with a contrast, an equivalence hypothesis and both prior sets through
    `source-analytics bayes --gate`, then a fit: the verdict has the equivalence rule (every replication at the
    margin), is complete, and the fit records it under its own gate id."""
    root = _root(tmp_path, "e5_gate")
    rng = np.random.default_rng(11)
    rows = ["subject,group,roi,band,absolute,relative,delta_ref"]
    for i in range(12):
        g = ["WT", "KO", "TR"][i // 4]
        base = rng.normal(0, 1) + (0.8 if g == "KO" else 0)
        for roi in ("r1", "r2", "r3"):
            for band in ("Alpha", "Low Gamma", "Beta"):
                rows.append(f"s{i},{g},{roi},{band},{base + rng.normal(0, 0.5):.6f},0.1,0.0")
    prior = root / "I.yaml"
    prior.write_text(yaml.safe_dump({"version": 1, "entries": [
        {"dv": "absolute", "hypothesis": "disease", "band": b, "prior": {"mean": 0.5, "sd": 0.5},
         "source": "test (simulated)"} for b in ("Alpha", "Low Gamma", "Beta")]}))
    doc = {"name": "e5", "design": {"factor": "group", "reference": "WT", "levels": ["WT", "KO", "TR"]},
           "hypotheses": [{"name": "disease", "kind": "contrast", "weights": {"KO": 1, "WT": -1}},
                          {"name": "tr_norm", "kind": "equivalence", "weights": {"TR": 1, "WT": -1},
                           "margin": {"mode": "sd", "value": 0.25}}],
           "paths": {"analytics": str(root / "analytics"), "results": str(root / "results")},
           "paradigms": {"arm": {"data_dir": str(root / "d"), "analyses": {"roi_psd": {}}}},
           "bayes": {"priors": {"I": str(prior)}, "fit": {"chains": 2, "iter": 400, "warmup": 200, "seed": 1},
                     "gate": {"reps": {"sbc": 3, "null": 3, "recovery": 1, "equivalence": 4},
                              "reduced_bands": ["Alpha", "Low Gamma"]},
                     "outcomes": {"roi_psd": ["absolute"]}}}
    sp = root / "study.yaml"
    sp.write_text(yaml.safe_dump(doc))

    from source_analytics.bayes import build_job, parse_bayes_config
    from source_analytics.config import StudyConfig

    job = build_job(StudyConfig.from_yaml(sp).for_paradigm_analysis("arm", "roi_psd"), "roi_psd", "absolute",
                    parse_bayes_config(doc["bayes"], root))
    Path(job.data_csv).parent.mkdir(parents=True)
    Path(job.data_csv).write_text("\n".join(rows) + "\n")
    assert job.prior_sets == ["W", "I"]
    cli = ["uv", "run", "--no-sync", "source-analytics", "bayes", "--study", str(sp), "--paradigm", "arm",
           "--workers", "2"]
    for extra in (["--gate"], []):
        r = subprocess.run(cli + extra, capture_output=True, text=True, env=_env(root), timeout=3600)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    gdir = Path(job.work_dir) / "gate" / "absolute"
    t = pd.read_csv(job.table_out)
    for sfx, set_ in (("", "W"), ("_I", "I")):
        g = json.loads((gdir / f"gate{sfx}.json").read_text())
        assert set(g["rules"]) == {"prior_pred", "sbc", "null", "equivalence", "recovery"}
        assert g["complete"]["equivalence"] == {"found": 4, "wanted": 4}
        e = g["rules"]["equivalence"]
        assert e["n_reps"] == 4 and e["min_n_at_margin"] == 4 and 0 <= e["max_false_equiv"] <= 1
        assert e["shown_when_equal_min"] is not None            # the null replications score equivalence too
        s = pd.read_csv(gdir / f"equivalence_summary{sfx}.csv")
        assert set(s["hypothesis"]) == {"tr_norm"} and set(s["band"]) == {"Alpha", "LowGamma"}
        ts = t[t["prior_set"] == set_]
        assert (ts["gate_id"] == g["gate_id"]).all() and ts["gate_passed"].notna().all()
        assert (ts["gate_passed"] == g["passed"]).all()
