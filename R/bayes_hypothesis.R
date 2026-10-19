# R/bayes_hypothesis.R — Bayesian hypothesis adapter (brms), G1
# ============================================================================
# The fourth adapter of the hypothesis layer (docs/methods/BAYES_ENGINE.md). Same design:/hypotheses: spec and the
# same per-subject long tables as the emmeans adapter (R/hypothesis.R, whose parser it reuses); answers each
# hypothesis with a posterior instead of a p-value. Runs only from the separate `bayes` step.
#
#   bayes_prepare(data, spec, dv_col, spatial_col, band_col, set, informative) -> coded data + term map for one prior set
#   bayes_formula(prep) / bayes_priors(prep)                   -> brms formula / prior table, generated from the spec
#   bayes_fit(prep, ...)                                       -> brmsfit (cmdstanr backend)
#   bayes_readout(fit, prep, spec, rope_sd)                    -> one row per hypothesis x band x spatial cell
#
# Model (bands pooled; a band-less outcome such as the aperiodic exponent is one "band"):
#   z ~ 0 + lv_<band>_<group> ... [+ dx_<band>_<group> ...]  group level per band (every group, reference included)
#       + (0 + x_<band>_<group> ... || roi)               ROI deviation of each band's group effect (partial pooling)
#       + (1 | roi:band) + (0 + bd_<band> ... || subject), ROI level per band; subject level per band
#   sigma ~ 0 + bd_<band> ...,  family = gaussian()       (bd_<band> = 1 on rows of that band: numeric, so one band works)
# x_<band>_<group> = 1 for rows of that band and (non-reference) group. z is the outcome standardised per band by the
# within-group between-subject SD within ROI x band (pooled over ROIs and groups), so effects, priors, the ROPE and `mode: sd` margins are in
# between-subject SD units (the unit of a per-cell Hedges g).
#
# Group levels, not reference-coded effects (the Conductor, 2026-10-19; bayes-sl reports/G1_forge_exponent.md): with
# N(0, 1) on each effect against the reference, the effects' priors jointly pooled the reference level toward the other
# groups' levels, shrinking every contrast against the reference by up to 0.18 SD in FORGE. Now each level is
# N(0, 1/sqrt(2)) (W), so every pairwise contrast is a priori N(0, 1) and no group is privileged. A set-I entry
# x_<band>_<group> (group - reference) re-expresses that group's level as reference level + dx_<band>_<group>, with
# dx ~ the entry: that contrast's prior is exactly the entry, and no other group is tied to the reference.
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
BAYES_W_EFFECT_SD <- 1  # W: every pairwise group contrast a priori N(0, 1), Track A's weak default (the Conductor, 2026-10-14; was 0.5)
BAYES_W_LEVEL_SD <- BAYES_W_EFFECT_SD / sqrt(2)   # ... from independent group levels N(0, 1/sqrt(2)) (2026-10-19)
BAYES_CODING <- "group-levels-2026-10-19"   # fixed-effect parametrisation; a gate verdict attaches only to fits of the same
BAYES_SD_ROI_BAND <- 3  # ROI level per band, half-N(0, 3): FORGE source power ROI means spread 0.9-3.0 SD (the Conductor, 2026-10-15; was 0.5)

.safe <- function(x) gsub("[^A-Za-z0-9]", "", x)

#' Code a long table for the model, for one prior set (the set decides the fixed-effect parametrisation, see the
#' header). `informative` is a named list x_<band>_<group> -> c(mean, sd) (group - reference) for set I; terms without
#' an entry stay W, and entries for terms not in this design are unused. Returns the data plus the term map.
bayes_prepare <- function(data, spec, dv_col, spatial_col = "roi", band_col = "band", set = "W", informative = NULL) {
  if (!set %in% BAYES_SETS) stop(sprintf("bayes: unknown prior set '%s'", set))
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
  # band indicators (numeric, so one band works the same as seven) and group-effect indicators per band
  for (b in levels(d$band)) d[[paste0("bd_", b)]] <- as.numeric(d$band == b)
  terms <- expand.grid(band = levels(d$band), group = setdiff(levels, ref), stringsAsFactors = FALSE)
  terms$col <- sprintf("x_%s_%s", terms$band, .safe(terms$group))
  for (k in seq_len(nrow(terms)))
    d[[terms$col[k]]] <- as.numeric(d$band == terms$band[k] & d$.grp == terms$group[k])
  # fixed effects: one parameter per band x group; lmap[[band]] (groups x parameters) gives the group levels
  fixed <- list(); lmap <- list()
  for (b in levels(d$band)) {
    anch <- Filter(function(g) g != ref && identical(set, "I") && !is.null(informative[[sprintf("x_%s_%s", b, .safe(g))]]),
                   levels)
    cols <- ifelse(levels %in% anch, sprintf("dx_%s_%s", b, .safe(levels)), sprintf("lv_%s_%s", b, .safe(levels)))
    M <- diag(length(levels)); dimnames(M) <- list(levels, cols)
    M[anch, cols[levels == ref]] <- 1                      # anchored level = reference level + dx
    lmap[[b]] <- M
    ms <- t(sapply(levels, function(g)
      if (g %in% anch) informative[[sprintf("x_%s_%s", b, .safe(g))]] else c(0, BAYES_W_LEVEL_SD)))
    fixed[[b]] <- data.frame(col = cols, band = b, group = levels, prior_mean = ms[, 1], prior_sd = ms[, 2],
                             row.names = NULL)
    for (p in cols) d[[p]] <- as.numeric(d$band == b) * M[as.character(d$.grp), p]
  }
  covs <- intersect(spec$covariates, names(d))
  for (cv in covs) d[[paste0("cov_", .safe(cv))]] <- d[[cv]] - mean(d[[cv]], na.rm = TRUE)
  list(data = d, terms = terms, fixed = do.call(rbind, unname(fixed)), lmap = lmap, set = set,
       levels = levels, ref = ref, bands = levels(d$band), band_labels = band_labels,
       covs = if (length(covs)) paste0("cov_", .safe(covs)) else character(0), scale = scale_tab,
       dv = dv_col, spatial = spatial_col)
}

bayes_formula <- function(prep) {
  fx <- paste(prep$fixed$col, collapse = " + ")
  x <- paste(prep$terms$col, collapse = " + ")
  bd <- paste(paste0("bd_", prep$bands), collapse = " + ")
  cv <- if (length(prep$covs)) paste(" +", paste(prep$covs, collapse = " + ")) else ""
  bf(as.formula(paste0("z ~ 0 + ", fx, cv, " + (0 + ", x, " || roi) + (1 | roi:band) + (0 + ", bd,
                       " || subject)")),
     as.formula(paste0("sigma ~ 0 + ", bd)))
}

#' Prior table of prep's set, generated from the spec (docs/methods/BAYES_ENGINE.md section 5): group levels
#' N(0, BAYES_W_LEVEL_SD), set-I differences dx ~ their entry (derived by script from a logged literature search).
bayes_priors <- function(prep) {
  fx <- prep$fixed
  c(set_prior("normal(0, 1)", class = "b"),                            # covariates
    do.call(c, lapply(seq_len(nrow(fx)), function(k)
      set_prior(sprintf("normal(%g, %g)", fx$prior_mean[k], fx$prior_sd[k]), class = "b", coef = fx$col[k]))),
    set_prior("normal(0, 0.25)", class = "sd", group = "roi"),
    set_prior(sprintf("normal(0, %g)", BAYES_SD_ROI_BAND), class = "sd", group = "roi:band"),
    set_prior("normal(0, 0.5)", class = "sd", group = "subject"),
    set_prior(sprintf("normal(%g, 0.5)", log(0.5)), class = "b", dpar = "sigma"))
}

#' A template's priors are in its compiled code: brms writes prior values into the Stan program, and
#' update(..., recompile = FALSE) runs them whatever `prior =` says. Stops unless `template` was compiled with
#' exactly the prior table `pr` (and this prep's formula); returns the template.
bayes_check_template <- function(template, prep, pr) {
  want <- stancode(bayes_formula(prep), data = prep$data, family = gaussian(), prior = pr, threads = template$threads)
  if (!identical(as.character(stancode(template)), as.character(want)))
    stop("bayes: the template's compiled code differs from these priors and formula (a template from another prior ",
         "set or an older prior?); update() would keep the template's priors. Refit the template.")
  template
}

#' model_dir: where cmdstanr keeps the compiled executable, so a saved fit can serve as an update() template in a
#' later process (brms' default is a session temp dir, gone when the process ends).
bayes_fit <- function(prep, chains = 4, iter = 2000, warmup = 1000, seed = 1,
                      backend = "cmdstanr", threads = NULL, template = NULL, model_dir = NULL, ...) {
  pr <- bayes_priors(prep)
  if (!is.null(template))
    return(update(bayes_check_template(template, prep, pr), newdata = prep$data, recompile = FALSE, chains = chains,
                  iter = iter, warmup = warmup, seed = seed, refresh = 0, ...))
  brm(bayes_formula(prep), data = prep$data, family = gaussian(), prior = pr, chains = chains, iter = iter,
      warmup = warmup, cores = chains, seed = seed, backend = backend, refresh = 0,
      threads = if (is.null(threads)) NULL else threading(threads),
      stan_model_args = if (is.null(model_dir)) list() else { dir.create(model_dir, FALSE, TRUE); list(dir = model_dir) },
      ...)
}

#' Prior draws of the group parameters (for Savage-Dickey densities at 0): each fixed parameter (level or set-I
#' difference) ~ N(m, s), ROI deviation r ~ N(0, tau) with tau ~ half-N(0, 0.25).
.prior_effect_draws <- function(prep, n = 20000) {
  fx <- prep$fixed; cols <- prep$terms$col
  b <- sapply(seq_len(nrow(fx)), function(k) rnorm(n, fx$prior_mean[k], fx$prior_sd[k]))
  r <- sapply(cols, function(cl) rnorm(n, 0, abs(rnorm(n, 0, 0.25))))
  list(b = matrix(b, n, dimnames = list(NULL, fx$col)), r = matrix(r, n, dimnames = list(NULL, cols)))
}

.hdi <- function(x, mass = BAYES_HDI) {
  x <- sort(x); n <- length(x); k <- floor(mass * n)
  i <- which.min(x[(k + 1):n] - x[1:(n - k)]); c(x[i], x[i + k])
}
.dens0 <- function(x) {
  bw <- tryCatch(bw.SJ(x), error = function(e) bw.nrd0(x)); mean(dnorm(-x / bw)) / bw
}

#' Weighted group contrast per draw: sum_g w_g * (level_g [+ r_g]). B holds the fixed parameters (columns
#' prep$fixed$col); levels = B %*% t(lmap). The reference group has no ROI deviation; ROI x band levels cancel when the
#' weights sum to 0 (required here).
.contrast_draws <- function(B, R, prep, band, w) {
  M <- prep$lmap[[band]]
  out <- as.numeric(B[, colnames(M), drop = FALSE] %*% (t(M) %*% w[rownames(M)]))
  if (!is.null(R)) for (g in names(w)[w != 0]) {
    if (g == prep$ref) next
    out <- out + w[[g]] * R[, sprintf("x_%s_%s", band, .safe(g))]
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
bayes_readout <- function(fit, prep, spec, rope_sd = BAYES_ROPE_SD, hypotheses = NULL) {
  set <- prep$set
  dr <- as.data.frame(as_draws_df(fit))            # plain data frame: column subsets keep no draws_df metadata
  cols <- prep$terms$col
  B <- as.matrix(dr[, paste0("b_", prep$fixed$col)]); colnames(B) <- prep$fixed$col
  rois <- levels(prep$data$roi)
  Rr <- lapply(setNames(rois, rois), function(ro) {
    m <- as.matrix(dr[, sprintf("r_roi[%s,%s]", ro, cols)]); colnames(m) <- cols; m
  })
  sig <- exp(as.matrix(dr[, paste0("b_sigma_bd_", prep$bands), drop = FALSE])); colnames(sig) <- prep$bands
  set.seed(1)
  pd <- .prior_effect_draws(prep)
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
