# R/bayes/provenance.R — what a number came from (engine plan E3; part of R/bayes/engine.R)
# ============================================================================
# Gate id (principle 3): a gate belongs to one design, outcome, hypothesis set, ROPE, prior table, engine version and
# gate settings (with its thresholds). Its id is a hash of exactly those, computed here by the gate (written into gate.json) and by a real fit
# (compared with it); a fit records gate_passed only for a gate with its own id, NA otherwise.
# Every component is hashed as text (serialize = FALSE), so ids do not move with R's serialisation format.
#
#   bayes_gate_settings(job)            the job's gate settings, engine defaults filled in
#   bayes_gate_id(prep, job)            list(id, components)
#   bayes_versions()                    R, packages, CmdStan
#   bayes_sha_file(path)
# ============================================================================

suppressMessages(library(digest))

#' The engine's behaviour version: bump it when the model, priors, readout or gate change what they compute. It is
#' part of every gate id, so a gate from an older engine stops counting. (The package version and git commit are
#' recorded in the manifest, not in the id: a commit that changes only docs must not invalidate gates.)
BAYES_ENGINE_VERSION <- "2026.11.09.1"   # residual-SD prior centred at log 1 with no ROI and no subject term
# (2026.10.28.2: the gate scores equivalence hypotheses by their own rule, E5)

#' Gate thresholds, locked engine defaults (decision B3); part of the gate id.
BAYES_GATE_RULES <- list(prior_pred_lo = 0.05, prior_pred_hi = 0.95, sbc_alpha = 0.01, sbc_min_coverage = 0.85,
                         null_max_claim_effect = 0.05, null_max_claim_dir = 0.10, recovery_min_coverage = 0.85,
                         rhat_max = BAYES_RHAT_MAX, equiv_max_false = 0.05)

BAYES_GATE_DEFAULTS <- list(sbc = 100L, null = 100L, recovery = 10L, equivalence = 100L, reduced_bands = NULL)

.sha_text <- function(x) digest(paste(x, collapse = "\n"), algo = "sha256", serialize = FALSE)
bayes_sha_file <- function(path) digest(path, algo = "sha256", file = TRUE)

#' The job's gate settings (bayes.gate in the study YAML, via the job file), defaults filled in. Recovery runs with the
#' real fit's sampler settings, so those are part of them.
bayes_gate_settings <- function(job) {
  g <- job$gate_settings %||% list()
  out <- BAYES_GATE_DEFAULTS
  for (k in intersect(names(g), names(out))) out[[k]] <- g[[k]]
  out$reduced_bands <- if (length(out$reduced_bands)) as.character(unlist(out$reduced_bands)) else NULL
  for (k in c("sbc", "null", "recovery", "equivalence")) out[[k]] <- as.integer(out[[k]])
  out$recovery_chains <- as.integer(job$chains); out$recovery_iter <- as.integer(job$iter)
  out$recovery_warmup <- as.integer(job$warmup)
  out
}

#' Gate id of prep's design and prior set under the job's hypotheses and gate settings. prep is the full design (the
#' real fit's), whatever subset of bands SBC and null run on.
bayes_gate_id <- function(prep, job) {
  d <- prep$data
  pr <- as.data.frame(bayes_priors(prep))[, c("prior", "class", "coef", "group", "dpar")]
  comp <- list(
    engine = BAYES_ENGINE_VERSION, coding = BAYES_CODING, prior_set = prep$set,
    design = .sha_text(paste(d$subject, d$.grp, d$roi, d$band, sep = "\t")),
    outcome = .sha_text(sprintf("%.17g", d[[prep$dv]])),
    hypotheses = .sha_text(as.character(jsonlite::toJSON(job$hypotheses, auto_unbox = TRUE, digits = NA))),
    rope = sprintf("%.17g", as.numeric(job$rope_sd)),                 # the null rule's claims use it
    priors = .sha_text(do.call(paste, c(pr, sep = "\t"))),
    gate = .sha_text(as.character(jsonlite::toJSON(bayes_gate_settings(job), auto_unbox = TRUE, null = "null"))),
    rules = .sha_text(as.character(jsonlite::toJSON(BAYES_GATE_RULES, auto_unbox = TRUE, digits = NA))))
  list(id = substr(.sha_text(paste(names(comp), unlist(comp), sep = "=")), 1, 16), components = comp)
}

.COMPONENT_WORDS <- c(engine = "engine version", coding = "coding", prior_set = "prior set", design = "design rows",
                      outcome = "outcome values", hypotheses = "hypotheses", rope = "ROPE", priors = "prior table",
                      gate = "gate settings", rules = "gate thresholds")

bayes_versions <- function() {
  pv <- function(p) if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else NA_character_
  cs <- tryCatch(as.character(cmdstanr::cmdstan_version()), error = function(e) NA_character_)
  list(R = paste(R.version$major, R.version$minor, sep = "."), brms = pv("brms"), cmdstanr = pv("cmdstanr"),
       posterior = pv("posterior"), digest = pv("digest"), cmdstan = cs)
}

#' The verdict a real fit records for prep's set: TRUE/FALSE from a gate with this fit's id, NA otherwise (with the
#' reason). gf: the gate.json path.
bayes_gate_check <- function(gf, prep, job) {
  want <- bayes_gate_id(prep, job)
  if (!file.exists(gf)) return(list(passed = NA, gate_id = NA_character_, want = want$id, reason = "no gate has run"))
  g <- jsonlite::fromJSON(gf)
  if (is.null(g$gate_id))
    return(list(passed = NA, gate_id = NA_character_, want = want$id,
                reason = "the gate predates gate ids (engine E3, 2026-10-28); rerun its summarize"))
  if (!identical(g$gate_id, want$id)) {
    diff <- names(want$components)[!vapply(names(want$components), function(k)
      identical(g$gate_inputs[[k]], want$components[[k]]), logical(1))]
    return(list(passed = NA, gate_id = g$gate_id, want = want$id,
                reason = sprintf("the gate was made for different %s", paste(.COMPONENT_WORDS[diff], collapse = ", "))))
  }
  list(passed = isTRUE(g$passed), gate_id = g$gate_id, want = want$id, reason = "")
}
