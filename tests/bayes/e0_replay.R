#!/usr/bin/env Rscript
# tests/bayes/e0_replay.R — regression check against the engine plan's E0 fixtures (FORGE G1 fits and gate verdicts,
# frozen by bayes-sl R/e0_freeze.R; kept outside this repository because they are study results).
#
#   Rscript tests/bayes/e0_replay.R --fixtures <dir> --engine R/bayes/engine.R --mode check --scratch <dir>
#   Rscript tests/bayes/e0_replay.R --fixtures <dir> --engine <old>/R/bayes_hypothesis.R --mode write_sim
#
# check: (1) every saved fit's Stan code equals the engine's for its set; (2) its readout, replayed, equals the frozen
# readout digit for digit (17 significant digits); (3) the gate's summarize, rerun on copies of the frozen replications,
# writes the frozen verdict; (4) the gate's simulator (draw_params, simulate_z) gives the frozen data for fixed seeds.
# Prints "E0 PASS" or stops. write_sim records (4) with a given engine (done once, with the engine E0 was frozen on).

suppressMessages({ library(optparse); library(jsonlite); library(digest) })
opt <- parse_args(OptionParser(option_list = list(
  make_option("--fixtures"), make_option("--engine"), make_option("--mode", default = "check"),
  make_option("--scratch", default = NULL), make_option("--cases", default = NULL))))
eng <- normalizePath(opt$engine)
if (basename(eng) == "bayes_hypothesis.R") source(file.path(dirname(eng), "hypothesis.R"))   # pre-E1 layout
source(eng)
gate_r <- if (basename(eng) == "bayes_hypothesis.R") file.path(dirname(eng), "bayes_gate.R") else
  file.path(dirname(eng), "gate.R")

# the gate script's function definitions, without running it
gate_fns <- new.env(parent = globalenv())
for (e in parse(gate_r)) if (is.call(e) && identical(e[[1]], as.name("<-")) && is.call(e[[3]]) &&
                             identical(e[[3]][[1]], as.name("function"))) eval(e, gate_fns)

man <- fromJSON(file.path(opt$fixtures, "MANIFEST_fits.json"), simplifyVector = FALSE)
cases <- names(man$cases)
if (!is.null(opt$cases)) cases <- intersect(cases, strsplit(opt$cases, ",")[[1]])
full <- function(df) {
  for (k in names(df)) if (is.double(df[[k]])) df[[k]] <- ifelse(is.na(df[[k]]), NA, sprintf("%.17g", df[[k]]))
  df
}
fail <- character(0)

load_case <- function(cs) {
  job <- fromJSON(file.path(opt$fixtures, cs, "job.json"), simplifyVector = FALSE)
  list(job = job, spec = parse_design_spec(list(design = job$design, hypotheses = job$hypotheses)),
       dat = read.csv(job$data_csv, check.names = FALSE, stringsAsFactors = FALSE),
       inf = if (is.null(job$informative)) NULL else lapply(job$informative, function(v) as.numeric(unlist(v))))
}

sim_hashes <- function(L) {
  out <- list()
  for (set in unlist(L$job$prior_sets)) {
    prep <- bayes_prepare(L$dat, L$spec, L$job$dv, L$job$spatial_col, L$job$band_col, set, L$inf)
    for (mode in c("sbc", "null", "recovery")) for (r in 1:2) {
      set.seed(10000 * match(mode, c("sbc", "null", "recovery")) + r)       # the gate's seeding (run_reps)
      p <- gate_fns$draw_params(prep, null = mode == "null")
      z <- gate_fns$simulate_z(prep, p)
      out[[sprintf("%s_%s_%d", set, mode, r)]] <- digest(z, algo = "sha256")
    }
  }
  out
}

for (cs in cases) {
  L <- load_case(cs)
  if (identical(opt$mode, "write_sim")) {
    write(toJSON(sim_hashes(L), auto_unbox = TRUE, pretty = TRUE), file.path(opt$fixtures, cs, "gate_sim.json"))
    cat("wrote", file.path(opt$fixtures, cs, "gate_sim.json"), "\n")
    next
  }
  # (1)-(2) readouts
  for (set in names(man$cases[[cs]]$sets)) {
    m <- man$cases[[cs]]$sets[[set]]
    prep <- bayes_prepare(L$dat, L$spec, L$job$dv, L$job$spatial_col, L$job$band_col, set, L$inf)
    fit <- readRDS(m$fit_file)
    if (!identical(digest(m$fit_file, algo = "sha256", file = TRUE), m$fit_sha256))
      stop(sprintf("%s %s: the fit file changed since E0 (%s)", cs, set, m$fit_file))
    want_code <- stancode(bayes_formula(prep), data = prep$data, family = gaussian(), prior = bayes_priors(prep),
                          threads = fit$threads)
    if (!identical(as.character(stancode(fit)), as.character(want_code)))
      fail <- c(fail, sprintf("%s %s: Stan code differs from the engine's", cs, set))
    got <- full(bayes_readout(fit, prep, L$spec, rope_sd = L$job$rope_sd))
    want <- read.csv(file.path(opt$fixtures, cs, m$readout_csv), check.names = FALSE, stringsAsFactors = FALSE,
                     colClasses = "character")
    got[] <- lapply(got, function(v) ifelse(is.na(v), NA_character_, as.character(v)))
    want[] <- lapply(want, function(v) ifelse(is.na(v) | v == "NA", NA_character_, v))
    if (!identical(names(got), names(want)) || nrow(got) != nrow(want)) {
      fail <- c(fail, sprintf("%s %s: readout columns/rows differ", cs, set))
    } else {
      bad <- names(got)[!mapply(identical, got, want)]
      if (length(bad)) fail <- c(fail, sprintf("%s %s: readout differs in %s", cs, set, paste(bad, collapse = ",")))
    }
    cat(sprintf("%s %s: readout %d rows checked\n", cs, set, nrow(got)))
  }
  # (3) gate verdicts (only cases whose frozen gate is current)
  gd <- file.path(opt$fixtures, cs, "gate", L$job$dv)
  if (identical(man$cases[[cs]]$rules, "current") && dir.exists(gd) && !is.null(opt$scratch)) {
    wd <- file.path(opt$scratch, cs); dir.create(file.path(wd, "gate"), recursive = TRUE, showWarnings = FALSE)
    file.copy(gd, file.path(wd, "gate"), recursive = TRUE)
    jb <- L$job; jb$work_dir <- normalizePath(wd)
    jf <- file.path(wd, "job.json"); write(toJSON(jb, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA), jf)
    for (set in unlist(L$job$prior_sets)) {
      sfx <- if (set == "W") "" else paste0("_", set)
      out <- system2("Rscript", c(gate_r, "--job", jf, "--mode", "summarize", "--set", set), stdout = TRUE, stderr = TRUE)
      # the verdict's rules and pass/fail, byte for byte; provenance fields added since E0 (E3: gate_id,
      # gate_inputs, engine_version, versions, complete) are not part of the fixture
      strip <- function(f) { x <- fromJSON(f, simplifyVector = FALSE)
        x[setdiff(names(x), c("gate_id", "gate_inputs", "engine_version", "versions", "complete"))] }
      a <- strip(file.path(wd, "gate", L$job$dv, sprintf("gate%s.json", sfx)))
      b <- strip(file.path(gd, sprintf("stored_gate%s.json", sfx)))
      if (!identical(toJSON(a, auto_unbox = TRUE, digits = NA), toJSON(b, auto_unbox = TRUE, digits = NA)))
        fail <- c(fail, sprintf("%s %s: gate verdict differs", cs, set))
      cat(sprintf("%s %s: gate verdict checked\n", cs, set))
    }
  }
  # (4) gate simulator
  sf <- file.path(opt$fixtures, cs, "gate_sim.json")
  if (file.exists(sf)) {
    want <- fromJSON(sf); got <- sim_hashes(L)
    bad <- names(want)[!vapply(names(want), function(k) identical(want[[k]], got[[k]]), logical(1))]
    if (length(bad)) fail <- c(fail, sprintf("%s: gate simulator differs at %s", cs, paste(bad, collapse = ",")))
    cat(sprintf("%s: gate simulator %d draws checked\n", cs, length(want)))
  }
}
if (identical(opt$mode, "check")) {
  if (length(fail)) { cat(paste("E0 FAIL:", fail), sep = "\n"); quit(status = 1) }
  cat("E0 PASS\n")
}
