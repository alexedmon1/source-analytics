#!/usr/bin/env Rscript
# R/bayes/analysis.R — entry point of the separate `bayes` step (src/source_analytics/bayes/ writes the
# job file). One outcome (dv) per run; each prior set in the job is fitted and read out, and the rows for this dv are
# written into <analysis>_hypotheses_bayes.csv (rows of other dvs in that file are kept).
#
#   Rscript R/bayes/analysis.R --job <job.json>

suppressMessages({ library(optparse); library(jsonlite) })
script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "R"
})
source(file.path(script_dir, "engine.R"))   # loads ../hypothesis.R, then the engine

opt <- parse_args(OptionParser(option_list = list(make_option("--job", type = "character"))))
job <- fromJSON(opt$job, simplifyVector = FALSE)
spec <- parse_design_spec(list(design = job$design, hypotheses = job$hypotheses))
dat <- read.csv(job$data_csv, check.names = FALSE, stringsAsFactors = FALSE)
prep <- bayes_prepare(dat, spec, job$dv, job$spatial_col, job$band_col)   # W coding: data summary and scale
cat(sprintf("bayes: %s, %d rows, %d subjects, %d bands, %d groups\n", job$dv, nrow(prep$data),
            nlevels(prep$data$subject), length(prep$bands), length(prep$levels)))

# the gate (R/bayes/gate.R) writes gate.json (W) / gate_I.json (I); a real fit records the verdict of a gate with its
# own id (bayes_gate_check: same design, outcome, hypotheses, priors, engine version and gate settings), NA otherwise
gate_file <- function(set) file.path(job$work_dir, "gate", job$dv, sprintf("gate%s.json", if (set == "W") "" else
  paste0("_", set)))

informative <- NULL
if (!is.null(job$informative)) informative <- lapply(job$informative, function(v) as.numeric(unlist(v)))
dir.create(job$work_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(prep$scale, file.path(job$work_dir, sprintf("scale_%s.csv", job$dv)), row.names = FALSE)

rows <- list(); sets_man <- list()
for (set in unlist(job$prior_sets)) {
  prep <- bayes_prepare(dat, spec, job$dv, job$spatial_col, job$band_col, set, informative)   # the set's coding
  pr <- bayes_priors(prep)
  write.csv(as.data.frame(pr)[, c("prior", "class", "coef", "group", "dpar")],
            file.path(job$work_dir, sprintf("priors_%s_%s.csv", job$dv, set)), row.names = FALSE)
  fit <- bayes_fit(prep, chains = job$chains, iter = job$iter, warmup = job$warmup,
                   seed = job$seed, model_dir = file.path(job$work_dir, "stan"))
  saveRDS(fit, file.path(job$work_dir, sprintf("fit_%s_%s.rds", job$dv, set)))
  fit_file <- file.path(job$work_dir, sprintf("fit_%s_%s.rds", job$dv, set))
  r <- bayes_readout(fit, prep, spec, rope_sd = job$rope_sd)
  gc <- bayes_gate_check(gate_file(set), prep, job)
  r$gate_passed <- gc$passed
  r$gate_id <- gc$want
  if (is.na(gc$passed)) cat("bayes: WARNING: set", set, "has no gate of its own (gate_passed NA):", gc$reason, "\n")
  else if (!gc$passed) cat("bayes: WARNING: set", set, "failed its gate\n")
  rows[[length(rows) + 1]] <- r
  ok <- r$status == "ok"
  pf <- file.path(job$work_dir, sprintf("priors_%s_%s.csv", job$dv, set))
  sets_man[[set]] <- list(
    gate = list(gate_id = gc$want, gate_file = gate_file(set), found_id = gc$gate_id, passed = gc$passed,
                reason = gc$reason),
    priors_csv = pf, priors_sha256 = bayes_sha_file(pf), fit_file = fit_file, fit_sha256 = bayes_sha_file(fit_file),
    convergence = list(max_rhat_estimands = max(r$rhat[ok]), max_rhat_scale = r$max_rhat_scale[1],
                       max_rhat_raw = r$max_rhat[1], min_ess_bulk = min(r$ess_bulk[ok]), n_divergent = r$n_divergent[1]))
  cat(sprintf("bayes: %s set %s done (R-hat: estimands max %.3f, scales %.3f, raw %.3f; %d divergent)\n", job$dv, set,
              max(r$rhat[ok]), r$max_rhat_scale[1], r$max_rhat[1], r$n_divergent[1]))
  if (max(r$rhat[ok]) >= BAYES_RHAT_MAX || r$max_rhat_scale[1] >= BAYES_RHAT_MAX)
    cat("bayes: WARNING: R-hat of an estimand or scale >=", BAYES_RHAT_MAX, "for set", set, "\n")
}
out <- dplyr::bind_rows(rows)
if (file.exists(job$table_out)) {   # keep other dvs' rows; a CSV round trip can change a column's type (an all-NA
  old <- read.csv(job$table_out, check.names = FALSE, stringsAsFactors = FALSE)       # column reads as logical)
  old <- old[old$dv != job$dv, , drop = FALSE]
  if (nrow(old)) {
    for (k in intersect(names(old), names(out)))
      if (!identical(class(old[[k]])[1], class(out[[k]])[1])) old[[k]] <- methods::as(old[[k]], class(out[[k]])[1])
    out <- dplyr::bind_rows(old, out)
  }
}
write.csv(out, job$table_out, row.names = FALSE)
cat("bayes: wrote", job$table_out, "\n")

# manifest (principle 2): every input, setting, version and gate this dv's rows rely on; in the work dir, and merged
# by dv beside the table (<analysis>_hypotheses_bayes_manifest.json)
g_tab <- table(prep$data$.grp[!duplicated(prep$data$subject)])
man <- list(
  schema = 1L, written = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), dv = job$dv,
  engine_version = BAYES_ENGINE_VERSION, coding = BAYES_CODING, versions = bayes_versions(),
  source_analytics = job$provenance$source_analytics, study = job$provenance$study,
  informative = job$provenance$informative,
  job_file = normalizePath(opt$job), job_sha256 = bayes_sha_file(opt$job),
  input = list(data_csv = job$data_csv, sha256 = bayes_sha_file(job$data_csv), n_rows_file = nrow(dat),
               n_rows_used = nrow(prep$data), n_subjects = nlevels(prep$data$subject),
               groups = as.list(setNames(as.integer(g_tab), names(g_tab))), bands = unname(prep$band_labels)),
  settings = list(rope_sd = job$rope_sd, chains = job$chains, iter = job$iter, warmup = job$warmup, seed = job$seed,
                  gate = bayes_gate_settings(job), gate_rules = BAYES_GATE_RULES, rhat_max = BAYES_RHAT_MAX),
  sets = sets_man,
  table = list(path = job$table_out, sha256 = bayes_sha_file(job$table_out)))
mj <- function(x) jsonlite::toJSON(x, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = NA)
write(mj(man), file.path(job$work_dir, sprintf("manifest_%s.json", job$dv)))
tm <- sub("\\.csv$", "_manifest.json", job$table_out)
all_man <- if (file.exists(tm)) jsonlite::fromJSON(tm, simplifyVector = FALSE) else list()
all_man[[job$dv]] <- man
write(mj(all_man), tm)
cat("bayes: wrote", tm, "\n")
