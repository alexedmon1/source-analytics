#!/usr/bin/env Rscript
# R/bayes_gate.R — the gate of the Bayesian adapter (docs/methods/BAYES_ENGINE.md section 6), generated from the spec,
# run at the study's actual design before any real fit. The outcome is read only for the prior predictive spread.
#
#   Rscript R/bayes_gate.R --job <job.json> --mode prior_pred
#   Rscript R/bayes_gate.R --job <job.json> --mode sbc      --reps 100 --workers 5 [--bands "Theta,Low Gamma"]
#   Rscript R/bayes_gate.R --job <job.json> --mode null     --reps 50  --workers 5 [--bands ...]
#   Rscript R/bayes_gate.R --job <job.json> --mode recovery --reps 10  --workers 5 [--chains 4]   (full design)
#   Rscript R/bayes_gate.R --job <job.json> --mode summarize
# Add --set I to gate prior set I (parameters drawn from I; outputs and verdict suffixed _I: gate_I.json).
#
# sbc / null may run at a reduced design (a subset of bands, every subject and ROI): the model's structure per band is
# the same, so calibration transfers; recovery runs at the full design to confirm (the Conductor, 2026-10-14).
# Parameters are drawn in R from the stated priors (independently of Stan), data simulated at the design, the
# template refitted; ranks of the truth among 100 thinned draws (SBC), HDI coverage, and decision-rule rates when every
# group effect is 0 (null). Outputs in <work_dir>/gate/; summarize writes gate.json (pass/fail per rule).

suppressMessages({ library(optparse); library(jsonlite); library(parallel) })
script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "R"
})
source(file.path(script_dir, "hypothesis.R"))
source(file.path(script_dir, "bayes_hypothesis.R"))

opt <- parse_args(OptionParser(option_list = list(
  make_option("--job"), make_option("--mode"), make_option("--reps", type = "integer", default = 100),
  make_option("--workers", type = "integer", default = 5), make_option("--bands", default = NULL),
  make_option("--iter", type = "integer", default = 1500), make_option("--warmup", type = "integer", default = 750),
  make_option("--chains", type = "integer", default = 2),   # per replication fit (cores = chains)
  make_option("--set", default = "W"))))
job <- fromJSON(opt$job, simplifyVector = FALSE)
spec <- parse_design_spec(list(design = job$design, hypotheses = job$hypotheses))
dat <- read.csv(job$data_csv, check.names = FALSE, stringsAsFactors = FALSE)
informative <- if (is.null(job$informative)) NULL else lapply(job$informative, function(v) as.numeric(unlist(v)))
GDIR <- file.path(job$work_dir, "gate", job$dv)
dir.create(GDIR, recursive = TRUE, showWarnings = FALSE)
L <- 100
SET <- opt$set
if (SET == "I" && is.null(informative)) stop("bayes gate: --set I needs the job's informative entries")
SFX <- if (SET == "W") "" else paste0("_", SET)          # W keeps the original file names (gate.json, prior_pred.csv)

design_rows <- function(bands = NULL) {
  d <- dat
  if (!is.null(bands) && !is.null(job$band_col)) d <- d[as.character(d[[job$band_col]]) %in% bands, , drop = FALSE]
  bayes_prepare(d, spec, job$dv, job$spatial_col, job$band_col)
}

#' One parameter draw from prior set `set` (mirrors bayes_priors()). null zeroes every group effect.
draw_params <- function(prep, set = "W", null = FALSE) {
  cols <- prep$terms$col
  hn <- function(s, n = 1) abs(rnorm(n, 0, s))
  b <- sapply(cols, function(cl) {
    ms <- if (identical(set, "I") && !is.null(informative[[cl]])) informative[[cl]] else c(0, BAYES_W_EFFECT_SD)
    rnorm(1, ms[1], ms[2])
  })
  tau <- setNames(hn(0.25, length(cols)), cols)
  if (null) { b[] <- 0; tau[] <- 0 }
  rois <- levels(prep$data$roi)
  list(b_band = setNames(rnorm(length(prep$bands), 0, 1), prep$bands), b = b, tau = tau,
       r = sapply(cols, function(cl) setNames(rnorm(length(rois), 0, tau[[cl]]), rois)),
       sd_rb = hn(BAYES_SD_ROI_BAND), sd_sub = setNames(hn(0.5, length(prep$bands)), prep$bands),
       sigma = setNames(exp(rnorm(length(prep$bands), log(0.5), 0.5)), prep$bands))
}

simulate_z <- function(prep, p) {
  d <- prep$data
  bnd <- as.character(d$band); roi <- as.character(d$roi)
  mu <- p$b_band[bnd]
  for (cl in prep$terms$col) mu <- mu + d[[cl]] * (p$b[[cl]] + p$r[roi, cl])
  rb <- paste(roi, bnd); urb <- unique(rb)
  mu <- mu + setNames(rnorm(length(urb), 0, p$sd_rb), urb)[rb]
  sb <- paste(d$subject, bnd); usb <- unique(sb)
  sd_sb <- p$sd_sub[sub("^.* ", "", usb)]
  mu <- mu + setNames(rnorm(length(usb), 0, sd_sb), usb)[sb]
  as.numeric(mu + p$sigma[bnd] * rnorm(nrow(d)))
}

#' Truth and posterior draws of every contrast/equivalence estimand: band x (ROI-average, each ROI).
estimand_table <- function(prep, p, fit) {
  dr <- as_draws_df(fit)
  cols <- prep$terms$col
  B <- as.matrix(dr[, paste0("b_", cols)]); colnames(B) <- cols
  rois <- levels(prep$data$roi)
  P0 <- list(b = matrix(p$b, 1, dimnames = list(NULL, cols)), r = p$r)
  set.seed(1); pd <- .prior_effect_draws(prep, SET, informative, n = 20000)
  keep <- round(seq(1, nrow(B), length.out = L))
  out <- list()
  for (hyp in spec$hypotheses) {
    if (!hyp$kind %in% c("contrast", "equivalence")) next
    w <- weight_vector(hyp, prep$levels)
    for (b in prep$bands) for (sp in c("all", rois)) {
      R <- if (sp == "all") NULL else {
        m <- as.matrix(dr[, sprintf("r_roi[%s,%s]", sp, cols)]); colnames(m) <- cols; m }
      x <- .contrast_draws(B, R, prep, b, w)
      t0 <- .contrast_draws(P0$b, if (sp == "all") NULL else matrix(p$r[sp, ], 1, dimnames = list(NULL, cols)),
                            prep, b, w)
      x0 <- .contrast_draws(pd$b, if (sp == "all") NULL else pd$r, prep, b, w)
      dec <- .decide(x, .dens0(x0), job$rope_sd)
      out[[length(out) + 1]] <- data.frame(hypothesis = hyp$name, band = b, spatial = sp, truth = t0,
                                           rank = sum(x[keep] < t0), mean = mean(x), sd = sd(x),
                                           covered = as.numeric(t0 >= dec$hdi_lo & t0 <= dec$hdi_hi),
                                           claim_effect = as.numeric(dec$decision == "effect"),
                                           claim_dir = as.numeric(dec$p_direction >= 0.975),
                                           claim_bf3 = as.numeric(dec$log10_bf10 >= log10(3)))
    }
  }
  np <- nuts_params(fit)
  res <- dplyr::bind_rows(out)
  res$max_rhat <- max(rhat(fit), na.rm = TRUE); res$n_div <- sum(np$Value[np$Parameter == "divergent__"])
  res
}

run_reps <- function(mode, bands) {
  prep <- design_rows(bands)
  tag <- if (mode == "recovery") "full" else paste(.safe(bands %||% "all"), collapse = "-")
  odir <- file.path(GDIR, sprintf("%s_%s%s", mode, tag, SFX)); dir.create(odir, showWarnings = FALSE)
  tpl_f <- file.path(GDIR, sprintf("template_%s%s.rds", tag, SFX))     # one per set: brms writes priors into the code
  if (!file.exists(tpl_f)) {
    set.seed(999); prep$data$z <- simulate_z(prep, draw_params(prep, SET))
    saveRDS(bayes_fit(prep, SET, informative, chains = 2, iter = 400, warmup = 200, seed = 1,
                      model_dir = file.path(GDIR, "stan")), tpl_f)
  }
  tpl <- readRDS(tpl_f)
  todo <- Filter(function(r) !file.exists(file.path(odir, sprintf("rep%04d.rds", r))), seq_len(opt$reps))
  mclapply(todo, function(r) {
    set.seed(10000 * match(mode, c("sbc", "null", "recovery")) + r)
    p <- draw_params(prep, SET, null = mode == "null")
    pr <- prep; pr$data$z <- simulate_z(prep, p)
    fit <- update(tpl, newdata = pr$data, recompile = FALSE, chains = opt$chains, cores = opt$chains,
                  iter = opt$iter, warmup = opt$warmup, seed = r, refresh = 0)
    s <- estimand_table(pr, p, fit); s$rep <- r; s$mode <- mode
    saveRDS(s, file.path(odir, sprintf("rep%04d.rds", r)))
    cat(sprintf("gate %s rep %d done\n", mode, r))
    NULL
  }, mc.cores = opt$workers, mc.preschedule = FALSE) -> res
  err <- Filter(function(x) inherits(x, "try-error"), res)
  if (length(err)) stop(sprintf("%d %s replication(s) failed; first: %s", length(err), mode, err[[1]]))
  invisible(NULL)
}

bands <- if (is.null(opt$bands)) NULL else trimws(strsplit(opt$bands, ",")[[1]])
if (opt$mode == "prior_pred") {
  prep <- design_rows()
  real <- prep$data$z
  out <- do.call(rbind, lapply(c("sd", "mad"), function(st) {
    fs <- match.fun(st)
    rs <- tapply(real, prep$data$band, fs)
    set.seed(7)
    sims <- do.call(rbind, lapply(1:500, function(i) {    # 500 x bands; stays a matrix for a single band
      z <- simulate_z(prep, draw_params(prep, SET)); c(tapply(z, prep$data$band, fs))
    }))
    data.frame(stat = st, band = prep$bands, real = as.numeric(rs[prep$bands]),
               sim_q05 = apply(sims, 2, quantile, 0.05), sim_median = apply(sims, 2, median),
               sim_q95 = apply(sims, 2, quantile, 0.95),
               pctile = sapply(prep$bands, function(b) mean(sims[, b] < rs[[b]])), row.names = NULL)
  }))
  write.csv(out, file.path(GDIR, sprintf("prior_pred%s.csv", SFX)), row.names = FALSE)
  print(out, digits = 3)
} else if (opt$mode %in% c("sbc", "null")) {
  run_reps(opt$mode, bands)
} else if (opt$mode == "recovery") {
  run_reps("recovery", NULL)
} else if (opt$mode == "summarize") {
  read_mode <- function(pat) {
    ds <- list.dirs(GDIR, recursive = FALSE); ds <- ds[grepl(pat, basename(ds))]
    ds <- if (SET == "W") ds[!grepl("_[A-Z]+$", basename(ds))] else ds[endsWith(basename(ds), SFX)]
    fs <- unlist(lapply(ds, list.files, full.names = TRUE))
    if (length(fs)) dplyr::bind_rows(lapply(fs, readRDS)) else NULL
  }
  verdict <- list(dv = job$dv, prior_set = SET, design = list(n_rows = nrow(dat)), rules = list())
  pp <- file.path(GDIR, sprintf("prior_pred%s.csv", SFX))
  if (file.exists(pp)) {
    x <- read.csv(pp)
    verdict$rules$prior_pred <- list(min_pctile = min(x$pctile), max_pctile = max(x$pctile),
                                     pass = all(x$pctile >= 0.05 & x$pctile <= 0.95))
  }
  sbc <- read_mode("^sbc_")
  if (!is.null(sbc)) {
    u <- do.call(rbind, lapply(split(sbc, list(sbc$hypothesis, sbc$band, sbc$spatial), drop = TRUE), function(x) {
      bins <- cut(x$rank, breaks = seq(-0.5, L + 0.5, length.out = 11))
      data.frame(hypothesis = x$hypothesis[1], band = x$band[1], spatial = x$spatial[1], n = nrow(x),
                 chisq_p = suppressWarnings(chisq.test(table(bins))$p.value), coverage = mean(x$covered))
    }))
    write.csv(u, file.path(GDIR, sprintf("sbc_summary%s.csv", SFX)), row.names = FALSE)
    avg <- u[u$spatial == "all", ]
    verdict$rules$sbc <- list(n_reps = max(table(sbc$hypothesis)) / length(unique(paste(sbc$band, sbc$spatial))),
                              n_estimands = nrow(u), min_p_roi_average = min(avg$chisq_p),
                              bonferroni = 0.01 / nrow(avg), min_coverage_roi_average = min(avg$coverage),
                              median_coverage_cells = median(u$coverage[u$spatial != "all"]),
                              max_rhat = max(sbc$max_rhat), divergent_fit_frac = mean(tapply(sbc$n_div > 0, sbc$rep, any)),
                              pass = min(avg$chisq_p) > 0.01 / nrow(avg) && min(avg$coverage) >= 0.85)
  }
  nul <- read_mode("^null_")
  if (!is.null(nul)) {
    f <- aggregate(cbind(claim_effect, claim_dir, claim_bf3) ~ hypothesis + band + spatial, data = nul, FUN = mean)
    write.csv(f, file.path(GDIR, sprintf("null_summary%s.csv", SFX)), row.names = FALSE)
    avg <- f[f$spatial == "all", ]
    verdict$rules$null <- list(max_claim_effect = max(avg$claim_effect), max_claim_dir = max(avg$claim_dir),
                               max_claim_bf3 = max(avg$claim_bf3),
                               cells_mean_claim_dir = mean(f$claim_dir[f$spatial != "all"]),
                               pass = max(avg$claim_effect) <= 0.05 && max(avg$claim_dir) <= 0.10)
  }
  rec <- read_mode("^recovery_")
  if (!is.null(rec)) {
    avg <- rec[rec$spatial == "all", ]
    verdict$rules$recovery <- list(n_reps = length(unique(rec$rep)), coverage_roi_average = mean(avg$covered),
                                   coverage_cells = mean(rec$covered[rec$spatial != "all"]),
                                   mean_abs_error = mean(abs(avg$mean - avg$truth)), max_rhat = max(rec$max_rhat),
                                   pass = mean(avg$covered) >= 0.85 && max(rec$max_rhat) < 1.05)
  }
  verdict$passed <- length(verdict$rules) == 4 && all(vapply(verdict$rules, function(r) isTRUE(r$pass), logical(1)))
  write(toJSON(verdict, auto_unbox = TRUE, pretty = TRUE, digits = 4), file.path(GDIR, sprintf("gate%s.json", SFX)))
  cat(toJSON(verdict, auto_unbox = TRUE, pretty = TRUE, digits = 4), "\n")
}
