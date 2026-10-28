"""The gate's equivalence rule (engine plan E5): equivalence hypotheses are scored by their own decision rule, at
their margin, so the gate measures how often the engine wrongly calls a contrast "equivalent".

Fast: the boundary draw (R, no sampling). The tiny gate run is in test_e5_slow.py.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from source_analytics.bayes import r_dir

GATE = r_dir() / "bayes" / "gate.R"
ENGINE = r_dir() / "bayes" / "engine.R"
pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")

# gate.R's function definitions without running the script (as tests/bayes/e0_replay.R does)
_SETUP = f"""
suppressMessages(source('{ENGINE}'))
gate_fns <- new.env(parent = globalenv())
for (e in parse('{GATE}')) if (is.call(e) && identical(e[[1]], as.name('<-')) && is.call(e[[3]]) &&
                              identical(e[[3]][[1]], as.name('function'))) eval(e, gate_fns)
d <- expand.grid(subject = paste0('s', 1:12), roi = c('r1', 'r2'), band = c('Alpha', 'Beta'), stringsAsFactors = FALSE)
d$group <- c('WT', 'KO', 'TA', 'TB')[(as.integer(sub('s', '', d$subject)) - 1) %/% 3 + 1]
set.seed(1); d$value <- rnorm(nrow(d))
eq <- function(name, g, m) list(name = name, kind = 'equivalence', weights = setNames(list(1, -1), c(g, 'WT')),
                                margin = list(mode = 'sd', value = m))
contrast <- list(name = 'disease', kind = 'contrast', weights = list(KO = 1, WT = -1))
design <- list(factor = 'group', reference = 'WT', levels = list('WT', 'KO', 'TA', 'TB'))
inf <- list(x_Alpha_TA = c(0.3, 0.2))
truth <- function(prep, p, h, b) {{
  w <- weight_vector(h, prep$levels)
  .contrast_draws(matrix(p$b[prep$fixed$col], 1, dimnames = list(NULL, prep$fixed$col)), NULL, prep, b, w) }}
"""


def _r(code: str) -> list[str]:
    r = subprocess.run(["Rscript", "-e", _SETUP + code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


@pytest.mark.parametrize("prior_set", ["W", "I"])
def test_boundary_draw_puts_every_equivalence_contrast_at_its_margin(prior_set):
    """Independent equivalence contrasts (FORGE's shape: each treated group vs WT) are all placed at once, at
    +-margin x the drawn sigma of each band, under both codings (set I re-expresses TA's Alpha level as WT + dx)."""
    out = _r(f"""
    hyps <- list(contrast, eq('ta', 'TA', 0.25), eq('tb', 'TB', 0.5))
    spec <- parse_design_spec(list(design = design, hypotheses = hyps)); assign('spec', spec, globalenv())
    prep <- bayes_prepare(d, spec, 'value', 'roi', 'band', '{prior_set}', inf)
    ok <- c(); signs <- c()
    for (r in 1:20) {{
      set.seed(40000 + r); p <- gate_fns$boundary_params(prep, r)
      for (b in prep$bands) for (h in hyps[2:3]) {{
        t0 <- truth(prep, p, h, b); m <- h$margin$value * p$sigma[[b]]
        ok <- c(ok, abs(abs(t0) - m) < 1e-10); signs <- c(signs, sign(t0)) }}
      ok <- c(ok, setequal(p$placed, c('ta', 'tb')), all(p$tau == 0))
    }}
    cat(all(ok), any(signs > 0), any(signs < 0))""")
    assert out == ["TRUE", "TRUE", "TRUE"]


def test_dependent_equivalence_contrasts_are_placed_one_at_a_time():
    """TA - WT, TB - WT and TA - TB cannot all sit at their margins at once; one hypothesis per replication, in
    turn, and only that one is marked as placed."""
    out = _r("""
    tatb <- list(name = 'tatb', kind = 'equivalence', weights = list(TA = 1, TB = -1), margin = list(mode = 'sd', value = 0.25))
    hyps <- list(eq('ta', 'TA', 0.25), eq('tb', 'TB', 0.25), tatb)
    spec <- parse_design_spec(list(design = design, hypotheses = hyps)); assign('spec', spec, globalenv())
    prep <- bayes_prepare(d, spec, 'value', 'roi', 'band', 'W', NULL)
    placed <- c(); ok <- c()
    for (r in 1:6) {
      set.seed(40000 + r); p <- gate_fns$boundary_params(prep, r); placed <- c(placed, p$placed)
      h <- hyps[[match(p$placed, c('ta', 'tb', 'tatb'))]]
      for (b in prep$bands) ok <- c(ok, abs(abs(truth(prep, p, h, b)) - 0.25 * p$sigma[[b]]) < 1e-10)
    }
    cat(all(ok), placed)""")
    assert out == ["TRUE", "ta", "tb", "tatb", "ta", "tb", "tatb"]
