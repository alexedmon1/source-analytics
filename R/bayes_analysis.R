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
prep <- bayes_prepare(dat, spec, job$dv, job$spatial_col, job$band_col)   # W coding: data summary and scale
cat(sprintf("bayes: %s, %d rows, %d subjects, %d bands, %d groups\n", job$dv, nrow(prep$data),
            nlevels(prep$data$subject), length(prep$bands), length(prep$levels)))

# the gate (R/bayes_gate.R) writes gate.json (W) / gate_I.json (I); a real fit records the verdict of its own prior set
# (NA when no gate has run at this design and coding)
gate_verdict <- function(set) {
  gf <- file.path(job$work_dir, "gate", job$dv, sprintf("gate%s.json", if (set == "W") "" else paste0("_", set)))
  if (!file.exists(gf)) return(NA)
  g <- fromJSON(gf)
  if (identical(as.integer(g$design$n_rows), nrow(dat)) && identical(g$coding, BAYES_CODING)) isTRUE(g$passed) else NA
}

informative <- NULL
if (!is.null(job$informative)) informative <- lapply(job$informative, function(v) as.numeric(unlist(v)))
dir.create(job$work_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(prep$scale, file.path(job$work_dir, sprintf("scale_%s.csv", job$dv)), row.names = FALSE)

rows <- list()
for (set in unlist(job$prior_sets)) {
  prep <- bayes_prepare(dat, spec, job$dv, job$spatial_col, job$band_col, set, informative)   # the set's coding
  pr <- bayes_priors(prep)
  write.csv(as.data.frame(pr)[, c("prior", "class", "coef", "group", "dpar")],
            file.path(job$work_dir, sprintf("priors_%s_%s.csv", job$dv, set)), row.names = FALSE)
  fit <- bayes_fit(prep, chains = job$chains, iter = job$iter, warmup = job$warmup,
                   seed = job$seed, model_dir = file.path(job$work_dir, "stan"))
  saveRDS(fit, file.path(job$work_dir, sprintf("fit_%s_%s.rds", job$dv, set)))
  r <- bayes_readout(fit, prep, spec, rope_sd = job$rope_sd)
  r$gate_passed <- gate_verdict(set)
  if (!isTRUE(r$gate_passed[1])) cat("bayes: WARNING: no passed gate for set", set, "at this design\n")
  rows[[length(rows) + 1]] <- r
  ok <- r$status == "ok"
  cat(sprintf("bayes: %s set %s done (R-hat: estimands max %.3f, scales %.3f, raw %.3f; %d divergent)\n", job$dv, set,
              max(r$rhat[ok]), r$max_rhat_scale[1], r$max_rhat[1], r$n_divergent[1]))
  if (max(r$rhat[ok]) >= BAYES_RHAT_MAX || r$max_rhat_scale[1] >= BAYES_RHAT_MAX)
    cat("bayes: WARNING: R-hat of an estimand or scale >=", BAYES_RHAT_MAX, "for set", set, "\n")
}
out <- dplyr::bind_rows(rows)
if (file.exists(job$table_out)) {
  old <- read.csv(job$table_out, check.names = FALSE, stringsAsFactors = FALSE)
  out <- dplyr::bind_rows(old[old$dv != job$dv, , drop = FALSE], out)
}
write.csv(out, job$table_out, row.names = FALSE)
cat("bayes: wrote", job$table_out, "\n")
