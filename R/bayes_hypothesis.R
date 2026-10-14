# R/bayes_hypothesis.R — Bayesian hypothesis adapter (brms), G1
# ============================================================================
# The fourth adapter of the hypothesis layer (docs/methods/BAYES_ENGINE.md). Same design:/hypotheses: spec and the
# same per-subject long tables as the emmeans adapter (R/hypothesis.R, whose parser it reuses); answers each
# hypothesis with a posterior instead of a p-value. Runs only from the separate `bayes` step.
#
#   bayes_prepare(data, spec, dv_col, spatial_col, band_col)  -> coded data + term names
#   bayes_formula(prep) / bayes_priors(prep, set)             -> brms formula / prior table, generated from the spec
#   bayes_fit(prep, set, ...)                                  -> brmsfit (cmdstanr backend)
#   bayes_readout(fit, prep, spec, set, rope_sd)               -> one row per hypothesis x band x spatial cell
#
# Model (bands pooled; a band-less outcome such as the aperiodic exponent is one "band"):
#   z ~ 0 + band + x_<band>_<group> ...                 band level of the reference group + group effect per band
#       + (0 + x_<band>_<group> ... || roi)             ROI deviation of each band's group effect (partial pooling)
#       + (1 | roi:band) + (0 + band || subject),         ROI level per band; subject level per band
#   sigma ~ 0 + band,  family = gaussian()
# x_<band>_<group> = 1 for rows of that band and (non-reference) group. z is the outcome standardised per band by the
# within-group between-subject SD within ROI x band (pooled over ROIs and groups), so effects, priors, the ROPE and `mode: sd` margins are in
# between-subject SD units (the unit of a per-cell Hedges g).
# ============================================================================

suppressMessages({ library(brms); library(posterior) })
if (!exists("parse_design_spec")) {
  .here <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "R")
  source(file.path(.here, "hypothesis.R"))
}
`%||%` <- function(a, b) if (is.null(a)) b else a

BAYES_SETS <- c("W", "I")
BAYES_ROPE_SD <- 0.1      # default ROPE for plain contrasts, between-subject SD units (the Conductor, 2026-10-14)
BAYES_HDI <- 0.95

.safe <- function(x) gsub("[^A-Za-z0-9]", "", x)

#' Code a long table for the model. Returns the data plus the term map.
bayes_prepare <- function(data, spec, dv_col, spatial_col = "roi", band_col = "band") {
  d <- as.data.frame(data)
  d <- d[!is.na(d[[dv_col]]), , drop = FALSE]
  levels <- spec$levels %||% sort(unique(as.character(d[[spec$factor]])))
  ref <- spec$reference %||% levels[1]
  d$.grp <- factor(as.character(d[[spec$factor]]), levels = levels)
  d <- d[!is.na(d$.grp), , drop = FALSE]
  has_band <- !is.null(band_col) && band_col %in% names(d)
  raw_band <- if (has_band) as.character(d[[band_col]]) else rep("all", nrow(d))
  band_labels <- if (has_band) setNames(unique(raw_band), .safe(unique(raw_band))) else c(all = NA_character_)
  d$band <- factor(.safe(raw_band))
  d$roi <- factor(as.character(d[[spatial_col]]))
  d$subject <- factor(as.character(d$subject))
  # standardise per band by the pooled within-group between-subject SD within roi x band (the Hedges g unit)
  scale_tab <- do.call(rbind, lapply(levels(d$band), function(b) {
    x <- d[d$band == b, ]
    cell <- interaction(x$roi, x$.grp, drop = TRUE)
    n <- tapply(x[[dv_col]], cell, length); v <- tapply(x[[dv_col]], cell, var)
    ok <- n > 1 & !is.na(v)
    data.frame(band = b, center = mean(x[[dv_col]]), scale = sqrt(sum((n[ok] - 1) * v[ok]) / sum(n[ok] - 1)))
  }))
  i <- match(d$band, scale_tab$band)
  d$z <- (d[[dv_col]] - scale_tab$center[i]) / scale_tab$scale[i]
  # group-effect indicators per band
  terms <- expand.grid(band = levels(d$band), group = setdiff(levels, ref), stringsAsFactors = FALSE)
  terms$col <- sprintf("x_%s_%s", terms$band, .safe(terms$group))
  for (k in seq_len(nrow(terms)))
    d[[terms$col[k]]] <- as.numeric(d$band == terms$band[k] & d$.grp == terms$group[k])
  covs <- intersect(spec$covariates, names(d))
  for (cv in covs) d[[paste0("cov_", .safe(cv))]] <- d[[cv]] - mean(d[[cv]], na.rm = TRUE)
  list(data = d, terms = terms, levels = levels, ref = ref, bands = levels(d$band), band_labels = band_labels,
       covs = if (length(covs)) paste0("cov_", .safe(covs)) else character(0), scale = scale_tab,
       dv = dv_col, spatial = spatial_col)
}

bayes_formula <- function(prep) {
  x <- paste(prep$terms$col, collapse = " + ")
  cv <- if (length(prep$covs)) paste(" +", paste(prep$covs, collapse = " + ")) else ""
  sub <- if (length(prep$bands) > 1) "(0 + band || subject)" else "(1 | subject)"
  bf(as.formula(paste0("z ~ 0 + band + ", x, cv, " + (0 + ", x, " || roi) + (1 | roi:band) + ", sub)),
     sigma ~ 0 + band)
}

#' Prior sets, generated from the spec (docs/methods/BAYES_ENGINE.md section 5). `informative` is a named list
#' col -> c(mean, sd) for set I (derived by script from a logged literature search); terms without an entry stay W.
bayes_priors <- function(prep, set = "W", informative = NULL) {
  p <- c(set_prior("normal(0, 1)", class = "b"),                       # band levels (reference group)
         do.call(c, lapply(prep$terms$col, function(cl) set_prior("normal(0, 0.5)", class = "b", coef = cl))),
         set_prior("normal(0, 0.25)", class = "sd", group = "roi"),
         set_prior("normal(0, 0.5)", class = "sd", group = "roi:band"),
         set_prior("normal(0, 0.5)", class = "sd", group = "subject"),
         set_prior(sprintf("normal(%g, 0.5)", log(0.5)), class = "b", dpar = "sigma"))
  if (identical(set, "I") && length(informative))
    for (cl in names(informative))
      p <- c(p, set_prior(sprintf("normal(%g, %g)", informative[[cl]][1], informative[[cl]][2]), class = "b", coef = cl))
  p
}

bayes_fit <- function(prep, set = "W", informative = NULL, chains = 4, iter = 2000, warmup = 1000, seed = 1,
                      backend = "cmdstanr", threads = NULL, template = NULL, ...) {
  pr <- bayes_priors(prep, set, informative)
  if (!is.null(template))
    return(update(template, newdata = prep$data, prior = pr, recompile = FALSE, chains = chains, iter = iter,
                  warmup = warmup, seed = seed, refresh = 0, ...))
  brm(bayes_formula(prep), data = prep$data, family = gaussian(), prior = pr, chains = chains, iter = iter,
      warmup = warmup, cores = chains, seed = seed, backend = backend, refresh = 0,
      threads = if (is.null(threads)) NULL else threading(threads), ...)
}

#' Prior draws of the group-effect parameters (for Savage-Dickey densities at 0): b ~ N(m, s) per term, ROI deviation
#' r ~ N(0, tau) with tau ~ half-N(0, 0.25).
.prior_effect_draws <- function(prep, set, informative, n = 20000) {
  cols <- prep$terms$col
  b <- sapply(cols, function(cl) {
    ms <- if (identical(set, "I") && !is.null(informative[[cl]])) informative[[cl]] else c(0, 0.5)
    rnorm(n, ms[1], ms[2])
  })
  r <- sapply(cols, function(cl) rnorm(n, 0, abs(rnorm(n, 0, 0.25))))
  list(b = matrix(b, n, dimnames = list(NULL, cols)), r = matrix(r, n, dimnames = list(NULL, cols)))
}

.hdi <- function(x, mass = BAYES_HDI) {
  x <- sort(x); n <- length(x); k <- floor(mass * n)
  i <- which.min(x[(k + 1):n] - x[1:(n - k)]); c(x[i], x[i + k])
}
.dens0 <- function(x) {
  bw <- tryCatch(bw.SJ(x), error = function(e) bw.nrd0(x)); mean(dnorm(-x / bw)) / bw
}

#' Weighted group-effect contrast per draw: sum_g w_g * (b_g [+ r_g]). The reference group's effect is 0; band levels
#' and ROI x band levels cancel when the weights sum to 0 (required here).
.contrast_draws <- function(B, R, prep, band, w) {
  out <- 0
  for (g in names(w)[w != 0]) {
    if (g == prep$ref) next
    cl <- sprintf("x_%s_%s", band, .safe(g))
    out <- out + w[[g]] * (B[, cl] + if (is.null(R)) 0 else R[, cl])
  }
  out
}

.decide <- function(x, prior0, rope) {
  h <- .hdi(x)
  bf10 <- prior0 / max(.dens0(x), 1e-12)
  data.frame(estimate = median(x), hdi_lo = h[1], hdi_hi = h[2], p_direction = max(mean(x > 0), mean(x < 0)),
             p_rope = mean(abs(x) < rope), rope_lo = -rope, rope_hi = rope, log10_bf10 = log10(bf10),
             decision = if (h[1] > rope || h[2] < -rope) "effect" else if (h[1] > -rope && h[2] < rope) "null"
                        else "undecided")
}

#' Read every requested hypothesis off one fit. One row per hypothesis x band x spatial cell ("all" = ROI-average,
#' i.e. the band's population effect). Omnibus: a placeholder row (deferred).
bayes_readout <- function(fit, prep, spec, set = "W", informative = NULL, rope_sd = BAYES_ROPE_SD,
                          hypotheses = NULL) {
  dr <- as_draws_df(fit)
  cols <- prep$terms$col
  B <- as.matrix(dr[, paste0("b_", cols)]); colnames(B) <- cols
  rois <- levels(prep$data$roi)
  Rr <- lapply(setNames(rois, rois), function(ro) {
    m <- as.matrix(dr[, sprintf("r_roi[%s,%s]", ro, cols)]); colnames(m) <- cols; m
  })
  sig <- exp(as.matrix(dr[, paste0("b_sigma_band", prep$bands)])); colnames(sig) <- prep$bands
  set.seed(1)
  pd <- .prior_effect_draws(prep, set, informative)
  np <- nuts_params(fit)
  diag <- list(max_rhat = max(rhat(fit), na.rm = TRUE), n_divergent = sum(np$Value[np$Parameter == "divergent__"]))
  hyps <- spec$hypotheses
  if (!is.null(hypotheses)) hyps <- hyps[hypotheses]
  rows <- list()
  for (hyp in hyps) {
    base <- data.frame(hypothesis = hyp$name, kind = hyp$kind, role = hyp$role, label = hyp$label, dv = prep$dv,
                       prior_set = set)
    if (hyp$kind == "omnibus" || hyp$kind == "regression") {
      rows[[length(rows) + 1]] <- cbind(base, band = NA, spatial = NA, status = if (hyp$kind == "omnibus")
        "not_implemented_bayes (omnibus deferred)" else "not_implemented_bayes (regression: G1 follow-up)")
      next
    }
    w <- weight_vector(hyp, prep$levels)
    if (abs(sum(w)) > 1e-9) stop(sprintf("hypothesis '%s': weights must sum to 0 for the Bayes adapter", hyp$name))
    for (b in prep$bands) {
      for (sp in c("all", rois)) {
        R <- if (sp == "all") NULL else Rr[[sp]]
        x <- .contrast_draws(B, R, prep, b, w)
        x0 <- .contrast_draws(pd$b, if (sp == "all") NULL else pd$r, prep, b, w)
        if (hyp$kind == "equivalence") {
          mg <- hyp$margin
          if (!identical(mg$mode %||% "sd", "sd")) stop("Bayes adapter: equivalence margin mode 'sd' only (G1)")
          m <- mg$value * sig[, b]                                      # per draw, between-subject-SD units x sigma
          h <- .hdi(x)
          res <- data.frame(estimate = median(x), hdi_lo = h[1], hdi_hi = h[2],
                            p_direction = max(mean(x > 0), mean(x < 0)), p_rope = mean(abs(x) < m),
                            rope_lo = -median(m), rope_hi = median(m), log10_bf10 = log10(.dens0(x0) /
                                                                                      max(.dens0(x), 1e-12)),
                            decision = if (h[1] > -median(m) && h[2] < median(m)) "equivalent" else "not_shown")
        } else {
          res <- .decide(x, .dens0(x0), rope_sd)
        }
        rows[[length(rows) + 1]] <- cbind(base, band = prep$band_labels[[b]] %||% b,
                                          spatial = if (sp == "all") "ROI-average" else sp, res, status = "ok")
      }
    }
  }
  out <- dplyr::bind_rows(rows)
  out$fdr_family <- "none (partial pooling across ROIs)"
  out$max_rhat <- diag$max_rhat; out$n_divergent <- diag$n_divergent
  out
}
