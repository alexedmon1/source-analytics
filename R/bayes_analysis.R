#!/usr/bin/env Rscript
# R/bayes_analysis.R — entry point of the separate `bayes` step (src/source_analytics/hypothesis/bayes.py writes the
# job file). One outcome (dv) per run; each prior set in the job is fitted and read out, and the rows for this dv are
# written into <analysis>_hypotheses_bayes.csv (rows of other dvs in that file are kept).
#
#   Rscript R/bayes_analysis.R --job <job.json>

suppressMessages({ library(optparse); library(jsonlite) })
script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "R"
})
source(file.path(script_dir, "hypothesis.R"))
source(file.path(script_dir, "bayes_hypothesis.R"))

opt <- parse_args(OptionParser(option_list = list(make_option("--job", type = "character"))))
job <- fromJSON(opt$job, simplifyVector = FALSE)
spec <- parse_design_spec(list(design = job$design, hypotheses = job$hypotheses))
dat <- read.csv(job$data_csv, check.names = FALSE, stringsAsFactors = FALSE)
prep <- bayes_prepare(dat, spec, job$dv, job$spatial_col, job$band_col)
cat(sprintf("bayes: %s, %d rows, %d subjects, %d bands, %d group-effect terms\n", job$dv, nrow(prep$data),
            nlevels(prep$data$subject), length(prep$bands), nrow(prep$terms)))

# the gate (R/bayes_gate.R) writes gate.json; a real fit records its verdict (NA when no gate has run at this design)
gate_passed <- NA
gf <- file.path(job$work_dir, "gate", job$dv, "gate.json")
if (file.exists(gf)) {
  g <- fromJSON(gf)
  gate_passed <- if (identical(as.integer(g$design$n_rows), nrow(dat))) isTRUE(g$passed) else NA
}
if (!isTRUE(gate_passed)) cat("bayes: WARNING: no passed gate for this design (gate_passed =", gate_passed, ")\n")

informative <- NULL
if (!is.null(job$informative)) informative <- lapply(job$informative, function(v) as.numeric(unlist(v)))
dir.create(job$work_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(prep$scale, file.path(job$work_dir, sprintf("scale_%s.csv", job$dv)), row.names = FALSE)

rows <- list()
for (set in unlist(job$prior_sets)) {
  pr <- bayes_priors(prep, set, informative)
  write.csv(as.data.frame(pr)[, c("prior", "class", "coef", "group", "dpar")],
            file.path(job$work_dir, sprintf("priors_%s_%s.csv", job$dv, set)), row.names = FALSE)
  fit <- bayes_fit(prep, set, informative, chains = job$chains, iter = job$iter, warmup = job$warmup,
                   seed = job$seed, model_dir = file.path(job$work_dir, "stan"))
  saveRDS(fit, file.path(job$work_dir, sprintf("fit_%s_%s.rds", job$dv, set)))
  r <- bayes_readout(fit, prep, spec, set, informative, rope_sd = job$rope_sd)
  r$gate_passed <- gate_passed
  rows[[length(rows) + 1]] <- r
  cat(sprintf("bayes: %s set %s done (max rhat %.3f, %d divergent)\n", job$dv, set, r$max_rhat[1], r$n_divergent[1]))
}
out <- dplyr::bind_rows(rows)
if (file.exists(job$table_out)) {
  old <- read.csv(job$table_out, check.names = FALSE, stringsAsFactors = FALSE)
  out <- dplyr::bind_rows(old[old$dv != job$dv, , drop = FALSE], out)
}
write.csv(out, job$table_out, row.names = FALSE)
cat("bayes: wrote", job$table_out, "\n")
