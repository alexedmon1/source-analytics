# R/bayes/readout.R — reading hypotheses off a fit, and the engine's decision rules (part of R/bayes/engine.R).
# .decide (HDI against the ROPE) and .decide_equivalence (HDI inside the margin) are the only decision rules; the
# gate (R/bayes/gate.R) uses the same functions.

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

#' Convergence of what is decided (the Conductor, 2026-10-21): R-hat and bulk ESS of one estimand's draws, split by
#' chain. The max R-hat over every parameter includes directions the data do not identify: a per-band constant trades
#' between the group levels and the (1 | roi:band) offsets, held only by the prior (FORGE power recovery rep 9: raw max
#' 1.076 on Low Gamma's offsets and levels, every pairwise contrast <= 1.02; bayes-sl R/g1_power_rhat_directions.R).
#' No contrast sees that direction, so the raw max is reported, not gated.
.estimand_conv <- function(x, chain) {
  m <- do.call(cbind, split(x, chain))                     # iterations x chains
  c(rhat = posterior::rhat(m), ess_bulk = posterior::ess_bulk(m))
}

#' Max R-hat of the scale parameters (every SD and the per-band log sigma), and over every parameter (reported only).
.scale_conv <- function(fit) {
  rh <- rhat(fit)
  c(max_rhat_scale = max(rh[grepl("^(sd_|b_sigma_)", names(rh))], na.rm = TRUE), max_rhat_raw = max(rh, na.rm = TRUE))
}

.decide <- function(x, prior0, rope) {
  h <- .hdi(x)
  bf10 <- prior0 / max(.dens0(x), 1e-12)
  data.frame(estimate = median(x), hdi_lo = h[1], hdi_hi = h[2], p_direction = max(mean(x > 0), mean(x < 0)),
             p_rope = mean(abs(x) < rope), rope_lo = -rope, rope_hi = rope, log10_bf10 = log10(bf10),
             decision = if (h[1] > rope || h[2] < -rope) "effect" else if (h[1] > -rope && h[2] < rope) "null"
                        else "undecided")
}

#' Equivalence: "equivalent" when the HDI lies inside the margin (its median over draws); m is the margin per draw.
.decide_equivalence <- function(x, prior0, m) {
  h <- .hdi(x)
  data.frame(estimate = median(x), hdi_lo = h[1], hdi_hi = h[2], p_direction = max(mean(x > 0), mean(x < 0)),
             p_rope = mean(abs(x) < m), rope_lo = -median(m), rope_hi = median(m),
             log10_bf10 = log10(prior0 / max(.dens0(x), 1e-12)),
             decision = if (h[1] > -median(m) && h[2] < median(m)) "equivalent" else "not_shown")
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
  sc <- .scale_conv(fit)
  diag <- list(max_rhat = sc[["max_rhat_raw"]], max_rhat_scale = sc[["max_rhat_scale"]],
               n_divergent = sum(np$Value[np$Parameter == "divergent__"]))
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
          res <- .decide_equivalence(x, .dens0(x0), mg$value * sig[, b])  # margin per draw: SD units x sigma
        } else {
          res <- .decide(x, .dens0(x0), rope_sd)
        }
        cv <- .estimand_conv(x, dr$.chain)
        res$rhat <- cv[["rhat"]]; res$ess_bulk <- cv[["ess_bulk"]]
        rows[[length(rows) + 1]] <- cbind(base, band = prep$band_labels[[b]] %||% b,
                                          spatial = if (sp == "all") "ROI-average" else sp, res, status = "ok")
      }
    }
  }
  out <- dplyr::bind_rows(rows)
  out$fdr_family <- "none (partial pooling across ROIs)"
  out$max_rhat_scale <- diag$max_rhat_scale
  out$max_rhat <- diag$max_rhat; out$n_divergent <- diag$n_divergent   # max_rhat: over every parameter, reported only
  out
}
