"""Uncentred outcomes (``center: false``, 2026-11-10; bayes-sl SL learning effects): a group's level against 0.

Fast: the coding (scaled, not centred), the ROI x band prior, the level-hypothesis guard, set-I level entries, the
gate's null and simulator, the readout's level arithmetic, and the gate id. Slow (SA_BAYES_SLOW=1): a fit at a ROI
design recovers known levels (one group learning, one not) and reads them off per ROI cell.
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

# 12 mice per group, 6 ROIs, two bands; true levels (raw units, SD 2): KO 1.2 in band s and 0 in band u, WT 0 in both
_SETUP = f"""
suppressMessages(source('{ENGINE}'))
gate_fns <- new.env(parent = globalenv())
for (e in parse('{GATE}')) if (is.call(e) && identical(e[[1]], as.name('<-')) && is.call(e[[3]]) &&
                              identical(e[[3]][[1]], as.name('function'))) eval(e, gate_fns)
spec <- parse_design_spec(list(design = list(factor = 'group', reference = 'WT', levels = list('WT', 'KO')),
  hypotheses = list(list(name = 'ko_level', kind = 'contrast', weights = list(KO = 1)),
                    list(name = 'wt_level', kind = 'contrast', weights = list(WT = 1)),
                    list(name = 'ko_vs_wt', kind = 'contrast', weights = list(KO = 1, WT = -1)))))
set.seed(5)
mice <- data.frame(subject = paste0('m', 1:24), group = rep(c('WT', 'KO'), each = 12))
d <- merge(merge(mice, data.frame(roi = paste0('r', 1:6))), data.frame(band = c('s', 'u')))
d$value <- 1.2 * (d$group == 'KO' & d$band == 's') + rnorm(nrow(d), 0, 2)
"""


def _r(code: str, env=None) -> list[str]:
    r = subprocess.run(["Rscript", "-e", _SETUP + code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    return [ln.split("=", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("OUT=")]


def test_uncentred_z_is_scaled_not_centred_and_centred_is_unchanged():
    out = _r("""
pc <- bayes_prepare(d, spec, 'value', 'roi', 'band')
pu <- bayes_prepare(d, spec, 'value', 'roi', 'band', center = FALSE)
cat('OUT=', all(pu$scale$center == 0), isTRUE(all.equal(pu$scale$scale, pc$scale$scale)), '\\n', sep = '')
i <- match(pu$data$band, pu$scale$band)
cat('OUT=', isTRUE(all.equal(pu$data$z, pu$data$value / pu$scale$scale[i])), '\\n', sep = '')
cat('OUT=', abs(mean(pc$data$z[pc$data$band == 's'])) < 1e-12, pc$centred, pu$centred, '\\n', sep = '')
rb <- function(p) { pr <- as.data.frame(bayes_priors(p)); pr$prior[pr$group == 'roi:band' & pr$class == 'sd'] }
cat('OUT=', rb(pc), ';', rb(pu), '\\n', sep = '')
cat('OUT=', identical(deparse1(bayes_formula(pc)$formula), deparse1(bayes_formula(pu)$formula)), '\\n', sep = '')
""")
    assert out[0] == "TRUETRUE" and out[1] == "TRUE" and out[2] == "TRUETRUEFALSE"
    assert out[3] == "normal(0, 3);normal(0, 0.5)"
    assert out[4] == "TRUE"                       # same model terms; only the outcome's centre and one prior differ


def test_a_level_hypothesis_needs_an_uncentred_outcome():
    r = subprocess.run(["Rscript", "-e", _SETUP + """
pc <- bayes_prepare(d, spec, 'value', 'roi', 'band')
.is_level(weight_vector(spec$hypotheses[[1]], pc$levels), pc, 'ko_level')"""], capture_output=True, text=True)
    assert r.returncode != 0 and "center: false" in r.stderr
    out = _r("""
pu <- bayes_prepare(d, spec, 'value', 'roi', 'band', center = FALSE)
cat('OUT=', sapply(spec$hypotheses, function(h) .is_level(weight_vector(h, pu$levels), pu, h$name)), '\\n')
""")
    assert out[0].split() == ["TRUE", "TRUE", "FALSE"]


def test_set_I_level_entries_replace_that_levels_prior_and_only_when_uncentred():
    out = _r("""
inf <- list(lv_s_KO = c(0.3, 0.9))
pu <- bayes_prepare(d, spec, 'value', 'roi', 'band', set = 'I', informative = inf, center = FALSE)
f <- pu$fixed
cat('OUT=', f$prior_mean[f$col == 'lv_s_KO'], f$prior_sd[f$col == 'lv_s_KO'], f$prior_sd[f$col == 'lv_u_KO'], '\\n')
e <- tryCatch(bayes_prepare(d, spec, 'value', 'roi', 'band', set = 'I', informative = inf), error = conditionMessage)
cat('OUT=', grepl('uncentred', e), '\\n', sep = '')
e <- tryCatch(bayes_prepare(d, spec, 'value', 'roi', 'band', set = 'I', center = FALSE,
                            informative = c(inf, list(x_s_KO = c(0, 1)))), error = conditionMessage)
cat('OUT=', grepl('both a level entry and a difference entry', e), '\\n', sep = '')
""")
    assert [float(x) for x in out[0].split()] == pytest.approx([0.3, 0.9, 2 ** -0.5])
    assert out[1] == out[2] == "TRUE"


def test_gate_null_puts_every_level_at_0_and_the_simulator_returns_the_roi_band_levels():
    out = _r("""
pu <- bayes_prepare(d, spec, 'value', 'roi', 'band', center = FALSE)
pc <- bayes_prepare(d, spec, 'value', 'roi', 'band')
set.seed(1); nu <- gate_fns$draw_params(pu, null = TRUE)
set.seed(1); nc <- gate_fns$draw_params(pc, null = TRUE)
cat('OUT=', all(nu$b == 0), all(nc$b != 0), '\\n', sep = '')
cat('OUT=', all(replicate(200, gate_fns$draw_params(pu)$sd_rb) < 2), '\\n', sep = '')    # half-N(0, 0.5)
set.seed(2); z <- gate_fns$simulate_z(pu, nu, keep_rb = TRUE)
rb <- attr(z, 'rb')
cat('OUT=', length(rb), all(sort(names(rb)) == sort(unique(paste(pu$data$roi, pu$data$band)))), '\\n')
# keep_rb changes nothing else: the same draws, and no attribute by default
set.seed(2); z2 <- gate_fns$simulate_z(pu, nu)
cat('OUT=', is.null(attributes(z2)), identical(as.numeric(z), z2), '\\n', sep = '')
""")
    assert out[0] == "TRUETRUE" and out[1] == "TRUE"
    assert out[2].split() == ["12", "TRUE"]
    assert out[3] == "TRUETRUE"


def test_contrast_draws_add_the_roi_band_level_only_for_levels():
    out = _r("""
pu <- bayes_prepare(d, spec, 'value', 'roi', 'band', center = FALSE)
B <- matrix(c(0.1, 0.5, 0, 0), 1, dimnames = list(NULL, c('lv_s_WT', 'lv_s_KO', 'lv_u_WT', 'lv_u_KO')))
R <- matrix(c(0.2, 0), 1, dimnames = list(NULL, c('x_s_KO', 'x_u_KO')))
w_lv <- c(WT = 0, KO = 1); w_c <- c(WT = -1, KO = 1)
cat('OUT=', .contrast_draws(B, NULL, pu, 's', w_lv), .contrast_draws(B, R, pu, 's', w_lv, 0.3),
    .contrast_draws(B, R, pu, 's', w_c, 0.3), '\\n')
""")
    assert [float(x) for x in out[0].split()] == pytest.approx([0.5, 1.0, 0.6])


def test_centring_is_part_of_the_gate_id():
    out = _r("""
job <- list(hypotheses = list(), rope_sd = 0.1, chains = 4, iter = 2000, warmup = 1000)
a <- bayes_gate_id(bayes_prepare(d, spec, 'value', 'roi', 'band'), job)
b <- bayes_gate_id(bayes_prepare(d, spec, 'value', 'roi', 'band', center = FALSE), job)
cat('OUT=', a$components$centring, b$components$centring, a$id != b$id, '\\n')
""")
    assert out[0].split() == ["centred", "uncentred", "TRUE"]


@pytest.mark.slow
def test_uncentred_fit_recovers_known_levels(tmp_path):
    root = os.environ.get("SA_BAYES_SCRATCH", str(tmp_path))
    out = _r(f"""
pu <- bayes_prepare(d, spec, 'value', 'roi', 'band', center = FALSE)
f <- bayes_fit(pu, chains = 4, iter = 2000, warmup = 1000, seed = 1, model_dir = '{root}/stan_uncentred')
r <- bayes_readout(f, pu, spec)
a <- r[r$spatial == 'ROI-average', ]
for (k in seq_len(nrow(a))) {{
  s <- pu$scale$scale[pu$scale$band == a$band[k]]
  t <- c(ko_level = 1.2, wt_level = 0, ko_vs_wt = 1.2)[[a$hypothesis[k]]] * (a$band[k] == 's') / s
  cat('OUT=', a$hypothesis[k], a$band[k], a$hdi_lo[k] <= t && t <= a$hdi_hi[k], a$decision[k], '\\n')
}}
cells <- r[r$spatial != 'ROI-average' & r$hypothesis == 'ko_level' & r$band == 's', ]
cat('OUT=cells', nrow(cells), all(is.finite(cells$estimate)), mean(cells$estimate > 0) == 1, '\\n')
cat('OUT=conv', max(r$rhat) < 1.05, r$n_divergent[1], '\\n')
# the gate scores a level in a ROI cell against b + r + rb (estimand_table): one simulated replication, refitted
job <- list(rope_sd = 0.1); L <- 100
environment(gate_fns$estimand_table) <- environment()
set.seed(11); p <- gate_fns$draw_params(pu); z <- gate_fns$simulate_z(pu, p, keep_rb = TRUE); p$rb <- attr(z, 'rb')
pr <- pu; pr$data$z <- as.numeric(z)
f2 <- update(f, newdata = pr$data, recompile = FALSE, chains = 2, iter = 1000, warmup = 500, seed = 2, refresh = 0)
s <- gate_fns$estimand_table(pr, p, f2)
x <- s[s$hypothesis == 'ko_level' & s$band == 's' & s$spatial == 'r3', ]
want <- p$b[['lv_s_KO']] + p$r['r3', 'x_s_KO'] + p$rb[['r3 s']]
y <- s[s$hypothesis == 'ko_vs_wt' & s$band == 's' & s$spatial == 'r3', ]
cat('OUT=gate', abs(x$truth - want) < 1e-12, abs(y$truth - (p$b[['lv_s_KO']] - p$b[['lv_s_WT']] + p$r['r3', 'x_s_KO'])) < 1e-12,
    '\\n')
""", env={**os.environ, "TMPDIR": root})
    rows = {tuple(x.split()[:2]): x.split()[2:] for x in out if not x.startswith(("cells", "conv", "gate"))}
    assert all(v[0] == "TRUE" for v in rows.values()), rows          # every truth inside its 95% interval
    assert rows[("ko_level", "s")][1] == "effect"                    # 0.54 [0.24, 0.84] SD at seed 5 (truth 0.60)
    assert all(rows[(h, "u")][1] != "effect" for h in ("ko_level", "wt_level", "ko_vs_wt"))
    cells = next(x for x in out if x.startswith("cells")).split()
    assert cells[1:] == ["6", "TRUE", "TRUE"]                         # ROI cells read, with the roi:band levels
    conv = next(x for x in out if x.startswith("conv")).split()
    assert conv[1] == "TRUE" and conv[2] == "0"
    assert next(x for x in out if x.startswith("gate")).split()[1:] == ["TRUE", "TRUE"]
