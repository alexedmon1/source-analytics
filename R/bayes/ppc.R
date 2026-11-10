#!/usr/bin/env Rscript
# R/bayes/ppc.R — posterior predictive summary of a finished fit (engine plan E4: reading a run). Read only: it loads
# the saved fit and the job's data, changes nothing the engine computed, and is not part of any gate id.
#
#   Rscript R/bayes/ppc.R --job <job.json> --set W --out <ppc.csv> [--ndraws 1000]
#
# Per band (and per group): the observed spread of the standardised outcome z (SD, MAD) and its mean, against 1000
# replicated data sets from the posterior; ``pctile`` is the observed value's percentile among the replicates (the
# same statistics as the gate's prior predictive check, now after fitting). Values near 0 or 1 say the model does not
# reproduce that feature of the data.

suppressMessages({ library(optparse); library(jsonlite) })
script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "R"
})
source(file.path(script_dir, "engine.R"))

opt <- parse_args(OptionParser(option_list = list(
  make_option("--job"), make_option("--set", default = "W"), make_option("--out"),
  make_option("--ndraws", type = "integer", default = 1000), make_option("--seed", type = "integer", default = 1))))
job <- fromJSON(opt$job, simplifyVector = FALSE)
spec <- parse_design_spec(list(design = job$design, hypotheses = job$hypotheses))
dat <- read.csv(job$data_csv, check.names = FALSE, stringsAsFactors = FALSE)
informative <- if (is.null(job$informative)) NULL else lapply(job$informative, function(v) as.numeric(unlist(v)))
prep <- bayes_prepare(dat, spec, job$dv, job$spatial_col, job$band_col, opt$set, informative,
                      center = !isFALSE(job$center))
fit <- readRDS(file.path(job$work_dir, sprintf("fit_%s_%s.rds", job$dv, opt$set)))
set.seed(opt$seed)
Yrep <- posterior_predict(fit, newdata = prep$data, ndraws = opt$ndraws)      # draws x rows, z units
d <- prep$data
rows <- list()
stat <- function(x, s) switch(s, sd = sd(x), mad = mad(x), mean = mean(x))
for (b in levels(d$band)) for (g in c("all", levels(d$.grp))) {
  i <- which(d$band == b & (g == "all" | d$.grp == g))
  if (length(i) < 2) next
  for (s in c("sd", "mad", "mean")) {
    obs <- stat(d$z[i], s)
    rep <- apply(Yrep[, i, drop = FALSE], 1, stat, s = s)
    rows[[length(rows) + 1]] <- data.frame(set = opt$set, band = b, group = g, stat = s, observed = obs,
                                           rep_q05 = unname(quantile(rep, 0.05)), rep_median = median(rep),
                                           rep_q95 = unname(quantile(rep, 0.95)), pctile = mean(rep <= obs))
  }
}
out <- do.call(rbind, rows)
labels <- prep$band_labels
out$band <- ifelse(out$band %in% names(labels) & !is.na(labels[out$band]), labels[out$band], out$band)
write.csv(out, opt$out, row.names = FALSE)
cat(sprintf("ppc: %s set %s, %d draws -> %s\n", job$dv, opt$set, opt$ndraws, opt$out))
