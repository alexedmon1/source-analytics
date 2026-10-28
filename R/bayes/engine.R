# R/bayes/engine.R — the Bayesian hypothesis engine (brms), loaded as one unit (docs/methods/BAYES_ENGINE.md)
# ============================================================================
# The fourth adapter of the hypothesis layer. Same design:/hypotheses: spec and the same per-subject long tables as the
# emmeans adapter (R/hypothesis.R, whose parser it reuses); answers each hypothesis with a posterior instead of a
# p-value. Runs only from the separate `bayes` step (R/bayes/analysis.R, R/bayes/gate.R).
#
#   source("R/bayes/engine.R")    # loads R/hypothesis.R (unless parse_design_spec exists), then:
#     model.R    bayes_prepare(data, spec, dv_col, spatial_col, band_col, set, informative) -> coded data + term map
#                bayes_formula(prep) / bayes_priors(prep)  -> brms formula / prior table; engine constants
#     fit.R      bayes_check_template(), bayes_fit(prep, ...) -> brmsfit (cmdstanr backend)
#     readout.R  bayes_readout(fit, prep, spec, rope_sd) -> one row per hypothesis x band x spatial cell, and the
#                decision rules (.decide, .decide_equivalence)
# ============================================================================

local({
  here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) file.path("R", "bayes"))
  if (!exists("parse_design_spec", envir = globalenv())) source(file.path(here, "..", "hypothesis.R"))
  for (f in c("model.R", "fit.R", "readout.R")) source(file.path(here, f))
})
