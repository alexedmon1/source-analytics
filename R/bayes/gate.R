#!/usr/bin/env Rscript
# R/bayes/gate.R — the gate of the Bayesian adapter (docs/methods/BAYES_ENGINE.md section 6), generated from the spec,
# run at the study's actual design before any real fit. The outcome is read only for the prior predictive spread.
#
#   Rscript R/bayes/gate.R --job <job.json> --mode prior_pred
#   Rscript R/bayes/gate.R --job <job.json> --mode sbc      --reps 100 --workers 5 [--bands "Theta,Low Gamma"]
#   Rscript R/bayes/gate.R --job <job.json> --mode null     --reps 100 --workers 5 [--bands ...]
#   Rscript R/bayes/gate.R --job <job.json> --mode equivalence --reps 100 --workers 5 [--bands ...]
#                                                           (only when the spec has equivalence hypotheses)
#   Rscript R/bayes/gate.R --job <job.json> --mode recovery --reps 10  --workers 5 --chains 4 --iter 2000 --warmup 1000
#                                                           (full design; the CLI passes the job's real-fit settings)
#   Rscript R/bayes/gate.R --job <job.json> --mode summarize
# Add --set I to gate prior set I (parameters drawn from I; outputs and verdict suffixed _I: gate_I.json).
#
# sbc / null may run at a reduced design (a subset of bands, every subject and ROI): the model's structure per band is
# the same, so calibration transfers; recovery runs at the full design to confirm (the Conductor, 2026-10-14).
# Parameters are drawn in R from the stated priors (independently of Stan), data simulated at the design, the
# template refitted; ranks of the truth among 100 thinned draws (SBC), HDI coverage, decision-rule rates when every
# group effect is 0 (null), and how often an equivalence hypothesis is called "equivalent" when its contrast lies
# exactly at the margin (equivalence: the false-equivalence rate, engine E5). Outputs in <work_dir>/gate/; summarize
# writes gate.json (pass/fail per rule).

suppressMessages({ library(optparse); library(jsonlite); library(parallel) })
script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "R"
})
source(file.path(script_dir, "engine.R"))   # loads ../hypothesis.R, then the engine

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
  bayes_prepare(d, spec, job$dv, job$spatial_col, job$band_col, SET, informative)
}

#' One parameter draw from prep's prior set (mirrors bayes_priors()). null: every group of a band at one common
#' level (drawn from the reference level's prior), set-I differences 0, no ROI deviations.
draw_params <- function(prep, null = FALSE) {
  cols <- prep$terms$col; fx <- prep$fixed
  hn <- function(s, n = 1) abs(rnorm(n, 0, s))
  b <- setNames(rnorm(nrow(fx), fx$prior_mean, fx$prior_sd), fx$col)
  tau <- setNames(hn(0.25, length(cols)), cols)
  if (null) {
    tau[] <- 0
    for (bd in prep$bands) {
      k <- fx$band == bd
      b[fx$col[k]] <- ifelse(startsWith(fx$col[k], "dx_"), 0, b[fx$col[k & fx$group == prep$ref]])
    }
  }
  rois <- levels(prep$data$roi)
  list(b = b, tau = tau,
       r = sapply(cols, function(cl) setNames(rnorm(length(rois), 0, tau[[cl]]), rois)),
       sd_rb = hn(BAYES_SD_ROI_BAND), sd_sub = setNames(hn(0.5, length(prep$bands)), prep$bands),
       sigma = setNames(exp(rnorm(length(prep$bands), log(0.5), 0.5)), prep$bands))
}

#' Equivalence at its margin (mode equivalence): the null draw, then each band's group levels moved so that every
#' equivalence contrast sits exactly at +-(margin x the drawn sigma), signs drawn per hypothesis, the shift of least
#' size. When the hypotheses' weights are linearly dependent (they cannot all be placed at once), one hypothesis per
#' replication, in turn. p$placed names the hypotheses at their margin.
boundary_params <- function(prep, r) {
  eq <- Filter(function(h) h$kind == "equivalence", spec$hypotheses)
  p <- draw_params(prep, null = TRUE)
  W <- do.call(rbind, lapply(eq, function(h) weight_vector(h, prep$levels)[prep$levels]))
  if (qr(W)$rank < nrow(W)) { k <- (r - 1) %% length(eq) + 1; eq <- eq[k]; W <- W[k, , drop = FALSE] }
  s <- sample(c(-1, 1), length(eq), replace = TRUE)
  mg <- vapply(eq, function(h) as.numeric(h$margin$value), numeric(1))
  for (b in prep$bands) {
    dl <- t(W) %*% solve(W %*% t(W), s * mg * p$sigma[[b]])      # least-size level shift with W dl = target
    M <- prep$lmap[[b]][prep$levels, , drop = FALSE]                # groups x this band's fixed parameters
    p$b[colnames(M)] <- p$b[colnames(M)] + as.numeric(solve(M, dl))
  }
  p$placed <- vapply(eq, `[[`, "", "name")
  p
}

simulate_z <- function(prep, p) {
  d <- prep$data
  bnd <- as.character(d$band); roi <- as.character(d$roi)
  mu <- 0
  for (cl in prep$fixed$col) mu <- mu + d[[cl]] * p$b[[cl]]
  for (cl in prep$terms$col) mu <- mu + d[[cl]] * p$r[roi, cl]
  rb <- paste(roi, bnd); urb <- unique(rb)
  mu <- mu + setNames(rnorm(length(urb), 0, p$sd_rb), urb)[rb]
  sb <- paste(d$subject, bnd); usb <- unique(sb)
  sd_sb <- p$sd_sub[sub("^.* ", "", usb)]
  mu <- mu + setNames(rnorm(length(usb), 0, sd_sb), usb)[sb]
  as.numeric(mu + p$sigma[bnd] * rnorm(nrow(d)))
}

#' Truth and posterior draws of every contrast/equivalence estimand: band x (ROI-average, each ROI). Equivalence
#' hypotheses are scored by both rules: the contrast rule (claim_*, read by the null rule) and their own
#' (.decide_equivalence, margin per draw = margin x posterior sigma: claim_equiv; the truth's margin uses the drawn
#' sigma: margin_true; at_margin = placed at its margin by boundary_params).
estimand_table <- function(prep, p, fit) {
  dr <- as.data.frame(as_draws_df(fit))
  cols <- prep$terms$col; fx <- prep$fixed$col
  B <- as.matrix(dr[, paste0("b_", fx)]); colnames(B) <- fx
  sig <- exp(as.matrix(dr[, paste0("b_sigma_bd_", prep$bands), drop = FALSE])); colnames(sig) <- prep$bands
  rois <- levels(prep$data$roi)
  P0 <- list(b = matrix(p$b[fx], 1, dimnames = list(NULL, fx)), r = p$r)
  set.seed(1); pd <- .prior_effect_draws(prep, n = 20000)
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
      is_eq <- hyp$kind == "equivalence"
      if (is_eq && !identical(hyp$margin$mode %||% "sd", "sd")) stop("bayes gate: equivalence margin mode 'sd' only")
      eqd <- if (is_eq) .decide_equivalence(x, .dens0(x0), hyp$margin$value * sig[, b]) else NULL
      cv <- .estimand_conv(x, dr$.chain)
      out[[length(out) + 1]] <- data.frame(hypothesis = hyp$name, band = b, spatial = sp, truth = t0,
                                           rank = sum(x[keep] < t0), mean = mean(x), sd = sd(x),
                                           rhat = cv[["rhat"]], ess_bulk = cv[["ess_bulk"]],
                                           covered = as.numeric(t0 >= dec$hdi_lo & t0 <= dec$hdi_hi),
                                           claim_effect = as.numeric(dec$decision == "effect"),
                                           claim_dir = as.numeric(dec$p_direction >= 0.975),
                                           claim_bf3 = as.numeric(dec$log10_bf10 >= log10(3)),
                                           claim_equiv = if (is_eq) as.numeric(eqd$decision == "equivalent") else NA,
                                           margin_true = if (is_eq) hyp$margin$value * p$sigma[[b]] else NA,
                                           at_margin = as.numeric(hyp$name %in% p$placed))
    }
  }
  np <- nuts_params(fit)
  res <- dplyr::bind_rows(out)
  sc <- .scale_conv(fit)
  res$max_rhat_scale <- sc[["max_rhat_scale"]]; res$max_rhat <- sc[["max_rhat_raw"]]   # max_rhat: raw, reported only
  res$n_div <- sum(np$Value[np$Parameter == "divergent__"])
  res
}

run_reps <- function(mode, bands) {
  prep <- design_rows(bands)
  tag <- if (mode == "recovery") "full" else paste(.safe(bands %||% "all"), collapse = "-")
  odir <- file.path(GDIR, sprintf("%s_%s%s", mode, tag, SFX)); dir.create(odir, showWarnings = FALSE)
  tpl_f <- file.path(GDIR, sprintf("template_%s%s.rds", tag, SFX))     # one per set: brms writes priors into the code
  if (!file.exists(tpl_f)) {
    set.seed(999); prep$data$z <- simulate_z(prep, draw_params(prep))
    saveRDS(bayes_fit(prep, chains = 2, iter = 400, warmup = 200, seed = 1,
                      model_dir = file.path(GDIR, "stan")), tpl_f)
  }
  bayes_check_template(readRDS(tpl_f), prep, bayes_priors(prep))
  todo <- Filter(function(r) !file.exists(file.path(odir, sprintf("rep%04d.rds", r))), seq_len(opt$reps))
  if (!length(todo)) return(invisible(NULL))
  # Workers are separate R processes (PSOCK), not forks. With cmdstanr fits in forked mclapply workers, this process
  # was SIGKILLed together with its whole process group as it exited (3 of 4 trials), which took the calling queue
  # down at the end of a stage (2026-10-14 to 10-16). Each worker reads the template itself.
  nw <- max(1, min(opt$workers, length(todo), floor(detectCores() / opt$chains)))   # workers x chains <= cores
  cl <- makePSOCKcluster(nw, outfile = "")
  on.exit(stopCluster(cl), add = TRUE)
  clusterExport(cl, ls(globalenv(), all.names = TRUE), envir = globalenv())
  clusterExport(cl, "tpl_f", envir = environment())
  clusterEvalQ(cl, { suppressMessages({ library(brms); library(posterior) }); tpl <- readRDS(tpl_f); NULL })
  one_rep <- function(r) try({
    set.seed(10000 * match(mode, c("sbc", "null", "recovery", "equivalence")) + r)
    p <- if (mode == "equivalence") boundary_params(prep, r) else draw_params(prep, null = mode == "null")
    pr <- prep; pr$data$z <- simulate_z(prep, p)
    fit <- update(tpl, newdata = pr$data, recompile = FALSE, chains = opt$chains, cores = opt$chains,
                  iter = opt$iter, warmup = opt$warmup, seed = r, refresh = 0)
    s <- estimand_table(pr, p, fit); s$rep <- r; s$mode <- mode
    saveRDS(s, file.path(odir, sprintf("rep%04d.rds", r)))
    cat(sprintf("gate %s rep %d done\n", mode, r))
    NULL
  })
  environment(one_rep) <- list2env(list(prep = prep, odir = odir, mode = mode), parent = globalenv())
  res <- parLapplyLB(cl, todo, one_rep, chunk.size = 1)
  err <- Filter(function(x) inherits(x, "try-error"), res)
  if (length(err)) stop(sprintf("%d %s replication(s) failed; first: %s", length(err), mode, err[[1]]))
  invisible(NULL)
}

bands <- if (is.null(opt$bands)) NULL else trimws(strsplit(opt$bands, ",")[[1]])
GS <- bayes_gate_settings(job)                 # the study's bayes.gate (E3); absent in pre-E3 job files
LEGACY <- is.null(job$gate_settings)
HAS_EQ <- any(vapply(spec$hypotheses, function(h) identical(h$kind, "equivalence"), logical(1)))
if (!LEGACY) {
  if (opt$mode %in% c("sbc", "null", "recovery", "equivalence") && opt$reps != GS[[opt$mode]])
    stop(sprintf("bayes gate: --reps %d differs from the study's bayes.gate.reps.%s (%d)", opt$reps, opt$mode,
                 GS[[opt$mode]]))
  if (opt$mode %in% c("sbc", "null", "equivalence") && !identical(bands, GS$reduced_bands))
    stop("bayes gate: --bands differs from the study's bayes.gate.reduced_bands")
}
if (opt$mode == "prior_pred") {
  prep <- design_rows()
  real <- prep$data$z
  out <- do.call(rbind, lapply(c("sd", "mad"), function(st) {
    fs <- match.fun(st)
    rs <- tapply(real, prep$data$band, fs)
    set.seed(7)
    sims <- do.call(rbind, lapply(1:500, function(i) {    # 500 x bands; stays a matrix for a single band
      z <- simulate_z(prep, draw_params(prep)); c(tapply(z, prep$data$band, fs))
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
} else if (opt$mode == "equivalence") {
  if (HAS_EQ) run_reps("equivalence", bands) else cat("bayes gate: no equivalence hypotheses; nothing to run\n")
} else if (opt$mode == "recovery") {
  run_reps("recovery", NULL)
} else if (opt$mode == "summarize") {
  RU <- BAYES_GATE_RULES   # thresholds, locked defaults (B3), part of the gate id
  red_tag <- paste(.safe(GS$reduced_bands %||% "all"), collapse = "-")
  read_mode <- function(pat) {
    ds <- list.dirs(GDIR, recursive = FALSE); ds <- ds[grepl(pat, basename(ds))]
    if (!LEGACY) {                     # exactly the configured design's directory, not every run left in GDIR
      tag <- if (pat == "^recovery_") "full" else red_tag
      ds <- ds[basename(ds) == paste0(sub("^\\^", "", pat), tag, SFX)]
    }
    ds <- if (SET == "W") ds[!grepl("_[A-Z]+$", basename(ds))] else ds[endsWith(basename(ds), SFX)]
    fs <- unlist(lapply(ds, list.files, full.names = TRUE))
    if (length(fs)) dplyr::bind_rows(lapply(fs, readRDS)) else NULL
  }
  gid <- bayes_gate_id(design_rows(), job)
  verdict <- list(dv = job$dv, prior_set = SET, coding = BAYES_CODING, design = list(n_rows = nrow(dat)), rules = list())
  pp <- file.path(GDIR, sprintf("prior_pred%s.csv", SFX))
  if (file.exists(pp)) {
    x <- read.csv(pp)
    verdict$rules$prior_pred <- list(min_pctile = min(x$pctile), max_pctile = max(x$pctile),
                                     pass = all(x$pctile >= RU$prior_pred_lo & x$pctile <= RU$prior_pred_hi))
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
                              bonferroni = RU$sbc_alpha / nrow(avg), min_coverage_roi_average = min(avg$coverage),
                              median_coverage_cells = median(u$coverage[u$spatial != "all"]),
                              max_rhat_raw = max(sbc$max_rhat),
                              max_rhat_estimands = if ("rhat" %in% names(sbc)) max(sbc$rhat) else NA,
                              divergent_fit_frac = mean(tapply(sbc$n_div > 0, sbc$rep, any)),
                              pass = min(avg$chisq_p) > RU$sbc_alpha / nrow(avg) && min(avg$coverage) >= RU$sbc_min_coverage)
  }
  nul <- read_mode("^null_")
  if (!is.null(nul)) {
    f <- aggregate(cbind(claim_effect, claim_dir, claim_bf3) ~ hypothesis + band + spatial, data = nul, FUN = mean)
    write.csv(f, file.path(GDIR, sprintf("null_summary%s.csv", SFX)), row.names = FALSE)
    avg <- f[f$spatial == "all", ]
    verdict$rules$null <- list(max_claim_effect = max(avg$claim_effect), max_claim_dir = max(avg$claim_dir),
                               max_claim_bf3 = max(avg$claim_bf3),
                               cells_mean_claim_dir = mean(f$claim_dir[f$spatial != "all"]),
                               pass = max(avg$claim_effect) <= RU$null_max_claim_effect && max(avg$claim_dir) <= RU$null_max_claim_dir)
  }
  eqv <- if (HAS_EQ) read_mode("^equivalence_") else NULL
  if (!is.null(eqv)) {
    # at the margin, "equivalent" is the wrong call: its rate per hypothesis x band (ROI-average) is gated. Reported:
    # the cells, and (from null replications written since E5) how often equivalence is shown when the groups are equal
    m <- eqv[eqv$at_margin == 1, ]
    f <- aggregate(claim_equiv ~ hypothesis + band + spatial, data = m, FUN = function(v) c(rate = mean(v), n = length(v)))
    f <- do.call(data.frame, f); names(f) <- sub("^claim_equiv\\.", "", names(f))
    write.csv(f, file.path(GDIR, sprintf("equivalence_summary%s.csv", SFX)), row.names = FALSE)
    avg <- f[f$spatial == "all", ]
    shown <- if (!is.null(nul) && "claim_equiv" %in% names(nul)) {
      z <- nul[nul$spatial == "all" & !is.na(nul$claim_equiv), ]
      if (nrow(z)) range(tapply(z$claim_equiv, paste(z$hypothesis, z$band), mean)) else c(NA, NA)
    } else c(NA, NA)
    verdict$rules$equivalence <- list(n_reps = length(unique(eqv$rep)), min_n_at_margin = min(avg$n),
                                      max_false_equiv = max(avg$rate), cells_mean_false_equiv =
                                        mean(f$rate[f$spatial != "all"]),
                                      shown_when_equal_min = shown[1], shown_when_equal_max = shown[2],
                                      pass = max(avg$rate) <= RU$equiv_max_false)
  }
  rec <- read_mode("^recovery_")
  if (!is.null(rec)) {
    # R-hat on what is decided (the Conductor, 2026-10-21): every estimand (ROI-average and cells) and the scales; the
    # max over every parameter is reported, not gated (.estimand_conv). Reps from before the rule have no estimand
    # R-hat and are refused rather than read as passing.
    if (!all(c("rhat", "max_rhat_scale") %in% names(rec)) || anyNA(rec$rhat))
      stop("bayes gate: recovery reps without per-estimand R-hat (written before 2026-10-21); delete them and rerun")
    avg <- rec[rec$spatial == "all", ]
    verdict$rules$recovery <- list(n_reps = length(unique(rec$rep)), coverage_roi_average = mean(avg$covered),
                                   coverage_cells = mean(rec$covered[rec$spatial != "all"]),
                                   mean_abs_error = mean(abs(avg$mean - avg$truth)),
                                   max_rhat_estimands = max(rec$rhat), max_rhat_roi_average = max(avg$rhat),
                                   max_rhat_scale = max(rec$max_rhat_scale), min_ess_bulk_estimands = min(rec$ess_bulk),
                                   max_rhat_raw = max(rec$max_rhat), rhat_max = BAYES_RHAT_MAX,
                                   pass = mean(avg$covered) >= RU$recovery_min_coverage && max(rec$rhat) < RU$rhat_max &&
                                     max(rec$max_rhat_scale) < RU$rhat_max)
  }
  # every configured replication present (a resumable gate summarized half way must not pass)
  found <- c(sbc = if (is.null(sbc)) 0L else length(unique(sbc$rep)), null = if (is.null(nul)) 0L else
    length(unique(nul$rep)), recovery = if (is.null(rec)) 0L else length(unique(rec$rep)),
    equivalence = if (is.null(eqv)) 0L else length(unique(eqv$rep)))
  wanted <- GS[c("sbc", "null", "recovery", "equivalence")]
  if (!HAS_EQ) wanted$equivalence <- 0L                 # no equivalence hypotheses: the rule does not apply
  complete <- lapply(c(sbc = "sbc", null = "null", recovery = "recovery", equivalence = "equivalence"), function(m)
    list(found = found[[m]], wanted = wanted[[m]]))
  is_complete <- all(vapply(complete, function(x) x$found >= x$wanted, logical(1)))
  verdict$passed <- length(verdict$rules) == 4 + HAS_EQ && all(vapply(verdict$rules, function(r) isTRUE(r$pass), logical(1))) &&
    is_complete
  verdict$complete <- complete
  verdict$gate_id <- gid$id
  verdict$gate_inputs <- gid$components
  verdict$engine_version <- BAYES_ENGINE_VERSION
  verdict$versions <- bayes_versions()
  write(toJSON(verdict, auto_unbox = TRUE, pretty = TRUE, digits = 4), file.path(GDIR, sprintf("gate%s.json", SFX)))
  cat(toJSON(verdict, auto_unbox = TRUE, pretty = TRUE, digits = 4), "\n")
}
