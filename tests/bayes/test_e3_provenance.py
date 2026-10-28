"""Provenance (engine plan E3): gate ids, the gate check a real fit makes, the manifest, --check-env.

Fast tests use R without sampling. ``test_end_to_end_gate_fit_manifest`` runs a tiny gate and fit through the CLI
(compiles Stan, a few minutes); it runs only with ``SA_BAYES_SLOW=1``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from source_analytics.bayes import r_dir

ENGINE = r_dir() / "bayes" / "engine.R"
pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")

_SETUP = f"""
suppressMessages(source('{ENGINE}'))
d <- expand.grid(subject = paste0('s', 1:6), roi = c('r1', 'r2'), stringsAsFactors = FALSE)
d$group <- ifelse(d$subject %in% c('s1', 's2', 's3'), 'WT', 'KO'); set.seed(1); d$value <- rnorm(nrow(d))
hyps <- list(list(name = 'disease', kind = 'contrast', weights = list(KO = 1, WT = -1)))
design <- list(factor = 'group', reference = 'WT', levels = list('WT', 'KO'))
spec <- parse_design_spec(list(design = design, hypotheses = hyps))
job <- list(hypotheses = hyps, rope_sd = 0.1, chains = 4, iter = 2000, warmup = 1000,
            gate_settings = list(sbc = 100, null = 100, recovery = 10))
inf <- list(x_all_KO = c(0.5, 0.8))
prep <- function(set = 'W', dd = d) bayes_prepare(dd, spec, 'value', 'roi', NULL, set, inf)
id <- function(p = prep(), j = job) bayes_gate_id(p, j)$id
"""


def _r(code: str) -> list[str]:
    r = subprocess.run(["Rscript", "-e", _SETUP + code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_gate_id_moves_with_each_input_and_only_those():
    out = _r("""
    j2 <- job; j2$gate_settings$null <- 50
    j3 <- job; j3$rope_sd <- 0.2
    j4 <- job; j4$hypotheses[[1]]$weights <- list(KO = -1, WT = 1)
    j5 <- job; j5$iter <- 3000                                   # recovery runs at the real fit's settings
    j6 <- job; j6$seed <- 99                                     # the fit's seed is not the gate's business
    d2 <- d; d2$value[1] <- d2$value[1] + 1e-9                   # the outcome (prior predictive reads it)
    cat(id() == id(), id() != id(prep('I')), id() != id(j = j2), id() != id(j = j3), id() != id(j = j4),
        id() != id(j = j5), id() == id(j = j6), id() != id(prep('W', d2)), nchar(id()))""")
    assert out == ["TRUE"] * 8 + ["16"]


def test_fit_records_a_gate_only_with_its_own_id(tmp_path):
    out = _r(f"""
    gf <- '{tmp_path}/gate.json'
    g <- bayes_gate_id(prep(), job)
    write(jsonlite::toJSON(list(passed = TRUE, gate_id = g$id, gate_inputs = g$components), auto_unbox = TRUE), gf)
    a <- bayes_gate_check(gf, prep(), job)
    j2 <- job; j2$rope_sd <- 0.3
    b <- bayes_gate_check(gf, prep(), j2)
    ci <- bayes_gate_check(gf, prep("I"), job)
    write(jsonlite::toJSON(list(passed = TRUE, coding = BAYES_CODING), auto_unbox = TRUE), gf)   # pre-E3 verdict
    e <- bayes_gate_check(gf, prep(), job)
    f <- bayes_gate_check('{tmp_path}/none.json', prep(), job)
    cat(a$passed, is.na(b$passed), grepl('ROPE', b$reason), is.na(ci$passed), grepl('prior table', ci$reason),
        grepl('prior set', ci$reason), is.na(e$passed), grepl('predates', e$reason), is.na(f$passed))""")
    assert out == ["TRUE"] * 9


def test_check_env_reports_and_compares(tmp_path):
    from source_analytics.bayes import check_env

    man = tmp_path / "m.json"
    man.write_text(json.dumps({"versions": {"R": "0.0", "brms": "0.0", "cmdstanr": "0.0", "posterior": "0.0",
                                            "digest": "0.0", "cmdstan": "0.0"}}))
    r = subprocess.run(["Rscript", str(r_dir() / "bayes" / "check_env.R"), "--against", str(man)],
                       capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    rep = json.loads(r.stdout)
    assert set(rep["packages"]) >= {"brms", "cmdstanr", "posterior", "digest"}
    assert set(rep["differs_from_record"]) == {"R", "brms", "cmdstanr", "posterior", "digest", "cmdstan"}
    assert check_env() == (1 if rep["missing"] else 0)


SLOW = os.environ.get("SA_BAYES_SLOW")


@pytest.mark.skipif(not SLOW, reason="slow (compiles and samples); set SA_BAYES_SLOW=1")
def test_end_to_end_gate_fit_manifest(tmp_path):
    """A tiny study through the CLI: gate, then a fit that records that gate (its id); then the study's gate settings
    change and the same fit records NA; the manifest names every input."""
    root = Path(os.environ.get("SA_BAYES_SCRATCH", tmp_path)) / "e3_e2e"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    rows = ["subject,group,roi,exponent"]
    for i in range(12):
        g = ["WT", "KO", "TR"][i // 4]
        base = rng.normal(0, 1) + (0.8 if g == "KO" else 0)
        rows += [f"s{i},{g},{r},{base + rng.normal(0, 0.5):.6f}" for r in ("r1", "r2", "r3")]
    csv_text = "\n".join(rows) + "\n"

    def study(null_reps):
        doc = {"name": "e3", "design": {"factor": "group", "reference": "WT", "levels": ["WT", "KO", "TR"]},
               "hypotheses": [{"name": "disease", "kind": "contrast", "weights": {"KO": 1, "WT": -1}}],
               "paths": {"analytics": str(root / "analytics"), "results": str(root / "results")},
               "paradigms": {"arm": {"data_dir": str(root / "d"), "analyses": {"roi_aperiodic": {}}}},
               "bayes": {"prior_sets": ["W"], "fit": {"chains": 2, "iter": 400, "warmup": 200, "seed": 1},
                         "gate": {"reps": {"sbc": 3, "null": null_reps, "recovery": 1}},
                         "outcomes": {"roi_aperiodic": ["exponent"]}}}
        p = root / "study.yaml"
        p.write_text(yaml.safe_dump(doc))
        return p

    from source_analytics.bayes import build_job, parse_bayes_config
    from source_analytics.config import StudyConfig

    sp = study(3)
    job = build_job(StudyConfig.from_yaml(sp).for_paradigm_analysis("arm", "roi_aperiodic"), "roi_aperiodic",
                    "exponent", parse_bayes_config(yaml.safe_load(sp.read_text())["bayes"], root))
    data = Path(job.data_csv).parent
    data.mkdir(parents=True)
    Path(job.data_csv).write_text(csv_text)
    env = {**os.environ, "TMPDIR": str(root)}
    cli = ["uv", "run", "--no-sync", "source-analytics", "bayes", "--paradigm", "arm", "--workers", "1"]

    def run(*extra):
        r = subprocess.run(cli + ["--study", str(study(extra[0]))] + list(extra[1:]), capture_output=True,
                           text=True, env=env, timeout=1800)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r

    run(3, "--gate")
    gate = json.loads((Path(job.work_dir) / "gate" / "exponent" / "gate.json").read_text())
    assert gate["gate_id"] and gate["complete"]["null"] == {"found": 3, "wanted": 3}
    run(3)
    tbl = Path(job.table_out)
    import pandas as pd

    t = pd.read_csv(tbl)
    assert (t["gate_id"] == gate["gate_id"]).all() and t["gate_passed"].notna().all()
    man = json.loads(tbl.with_name("roi_aperiodic_hypotheses_bayes_manifest.json").read_text())["exponent"]
    assert man["input"]["data_csv"].endswith("aperiodic_params.csv") and len(man["input"]["sha256"]) == 64
    assert man["study"]["path"].endswith("study.yaml") and man["job_sha256"] and man["table"]["sha256"]
    assert man["sets"]["W"]["gate"]["gate_id"] == gate["gate_id"] and man["versions"]["cmdstan"]
    assert man["sets"]["W"]["fit_sha256"] and man["sets"]["W"]["priors_sha256"]
    r = run(4)                                      # the study now asks for 4 null reps: the old gate is not this fit's
    t = pd.read_csv(tbl)
    assert t["gate_passed"].isna().all() and (t["gate_id"] != gate["gate_id"]).all()
    assert "has no gate of its own (gate_passed NA): the gate was made for different gate settings" in r.stdout
