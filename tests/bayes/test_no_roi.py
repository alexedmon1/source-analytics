"""The engine with no ROI (spatial_col = NULL; bayes-sl L3): one value per subject, e.g. its power along a located field.

Fast: the coding, formula, priors and Stan code (no sampling), the ROI design's formula unchanged, and the gate's simulator.
Slow (SA_BAYES_SLOW=1): a fit whose group contrast matches a plain linear model's, and a readout with one spatial cell.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from source_analytics.bayes import r_dir

ENGINE = r_dir() / "bayes" / "engine.R"
GATE = r_dir() / "bayes" / "gate.R"
pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")

_SETUP = f"""
suppressMessages(source('{ENGINE}'))
gate_fns <- new.env(parent = globalenv())
for (e in parse('{GATE}')) if (is.call(e) && identical(e[[1]], as.name('<-')) && is.call(e[[3]]) &&
                              identical(e[[3]][[1]], as.name('function'))) eval(e, gate_fns)
spec <- parse_design_spec(list(design = list(factor = 'group', reference = 'A', levels = list('A', 'B')),
                               hypotheses = list(list(name = 'effect', kind = 'contrast', weights = list(B = 1, A = -1)))))
set.seed(3)
one <- data.frame(subject = paste0('m', 1:24), group = rep(c('A', 'B'), each = 12))
one$value <- rnorm(24, 0, 2) + 1.5 * (one$group == 'B')
roi <- merge(one[, c('subject', 'group')], data.frame(roi = c('r1', 'r2', 'r3')))
roi$value <- rnorm(nrow(roi))
"""


def _r(code: str, env=None) -> list[str]:
    r = subprocess.run(["Rscript", "-e", _SETUP + code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    return [ln.split("=", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("OUT=")]


def test_roi_formula_is_unchanged():
    out = _r("""
p <- bayes_prepare(roi, spec, 'value', 'roi', NULL)
cat('OUT=', deparse1(bayes_formula(p)$formula), '\\n', sep = '')
cat('OUT=', p$no_roi, p$subject_re, '\\n', sep = '')
""")
    assert out[0] == "z ~ 0 + lv_all_A + lv_all_B + (0 + x_all_B || roi) + (1 | roi:band) + (0 + bd_all || subject)"
    assert out[1] == "FALSETRUE"


def test_no_roi_drops_the_roi_terms_and_the_unidentified_subject_term():
    out = _r("""
p <- bayes_prepare(one, spec, 'value', NULL, NULL)
cat('OUT=', deparse1(bayes_formula(p)$formula), '\\n', sep = '')
pr <- bayes_priors(p)
cat('OUT=', paste(sort(unique(pr$class)), collapse = ','), '\\n', sep = '')
sc <- stancode(bayes_formula(p), data = p$data, family = gaussian(), prior = pr)
cat('OUT=', grepl('sd_', sc), '\\n', sep = '')
# repeated rows within a band keep the subject term
two <- rbind(one, transform(one, value = value + rnorm(24, 0, 0.3)))
cat('OUT=', deparse1(bayes_formula(bayes_prepare(two, spec, 'value', NULL, NULL))$formula), '\\n', sep = '')
# a scale in between-subject SD units: the pooled within-group SD of the 24 values
s <- sqrt(mean(tapply(one$value, one$group, var)))
cat('OUT=', abs(p$scale$scale - s) < 1e-12, '\\n', sep = '')
""")
    assert out[0] == "z ~ 0 + lv_all_A + lv_all_B"
    assert out[1] == "b"                       # only the fixed effects and sigma's (class b, dpar sigma)
    assert out[2] == "FALSE"                   # no group-level SDs in the Stan code
    assert out[3] == "z ~ 0 + lv_all_A + lv_all_B + (0 + bd_all || subject)"
    assert out[4] == "TRUE"


def test_a_missing_spatial_column_is_an_error_not_no_roi():
    r = subprocess.run(["Rscript", "-e", _SETUP + "bayes_prepare(one, spec, 'value', 'roi', NULL)"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "no spatial column 'roi'" in r.stderr


def test_gate_simulates_without_roi_or_subject_terms():
    out = _r("""
p <- bayes_prepare(one, spec, 'value', NULL, NULL)
set.seed(1); pa <- gate_fns$draw_params(p)
set.seed(2); z <- gate_fns$simulate_z(p, pa)
mu <- p$data$lv_all_A * pa$b[['lv_all_A']] + p$data$lv_all_B * pa$b[['lv_all_B']]
# with no ROI or subject term, z - mu is the residual alone: its SD is the drawn sigma
cat('OUT=', length(z) == 24 && abs(sd(z - mu) / pa$sigma[[1]] - 1) < 0.5, '\\n', sep = '')
set.seed(2); e <- rnorm(24)
cat('OUT=', isTRUE(all.equal(z, as.numeric(mu + pa$sigma[[1]] * e))), '\\n', sep = '')
""")
    assert out == ["TRUE", "TRUE"]


@pytest.mark.slow
def test_no_roi_fit_matches_a_linear_model(tmp_path):
    root = os.environ.get("SA_BAYES_SCRATCH", str(tmp_path))
    out = _r(f"""
p <- bayes_prepare(one, spec, 'value', NULL, NULL)
f <- bayes_fit(p, chains = 2, iter = 2000, warmup = 1000, seed = 1, model_dir = '{root}/stan_noroi')
r <- bayes_readout(f, p, spec)
cat('OUT=', nrow(r), r$spatial, '\\n', sep = ' ')
# conjugate check: each group level ~ N(0, 1/2) with 12 mice at residual variance ~1 (z units) is shrunk by 12 / (12 + 2)
ols <- coef(lm(value ~ group, one))[['groupB']] / p$scale$scale
cat('OUT=', abs(r$estimate - ols * 12 / 14), '\\n', sep = '')
cat('OUT=', r$rhat < 1.01, '\\n', sep = '')
""", env={**os.environ, "TMPDIR": root})
    assert out[0].split() == ["1", "all"]
    assert float(out[1]) < 0.05                # OLS shrunk as the conjugate normal model shrinks it
    assert out[2].strip() == "TRUE"


def test_residual_sd_prior_is_centred_at_1_only_without_roi_and_subject_terms():
    out = _r("""
sig <- function(p) { pr <- as.data.frame(bayes_priors(p)); pr$prior[pr$dpar == 'sigma'] }
cat('OUT=', sig(bayes_prepare(one, spec, 'value', NULL, NULL)), '\\n', sep = '')
cat('OUT=', sig(bayes_prepare(roi, spec, 'value', 'roi', NULL)), '\\n', sep = '')
two <- rbind(one, transform(one, value = value + rnorm(24, 0, 0.3)))
cat('OUT=', sig(bayes_prepare(two, spec, 'value', NULL, NULL)), '\\n', sep = '')
# the gate draws sigma from the same prior
p <- bayes_prepare(one, spec, 'value', NULL, NULL)
set.seed(1); s <- replicate(4000, gate_fns$draw_params(p)$sigma[[1]])
cat('OUT=', abs(median(log(s))) < 0.05, '\\n', sep = '')
""")
    assert out[0] == "normal(0, 0.5)"
    assert out[1] == out[2] == "normal(-0.693147, 0.5)"
    assert out[3] == "TRUE"
