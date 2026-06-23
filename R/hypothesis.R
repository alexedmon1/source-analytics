# R/hypothesis.R — declarative hypothesis layer (emmeans adapter)
# ============================================================================
# Shared inference layer (peer to stats_utils.R / the Python stats/ package), NOT
# a registry analysis module. Sourced by *_analysis.R modules from their
# statistics step. See DESIGN_SPEC.md.
#
#   parse_design_spec(config)            -> design + named hypotheses
#   run_hypothesis(data, hyp, spec, ...) -> tidy per-cell result, ONE hypothesis
#
# Kinds: omnibus | contrast | regression | equivalence. This file is the EMMEANS
# adapter (R LMM modules); the permutation adapter lives in Python. The legacy
# group_a/group_b contrast form is accepted as sugar for a pairwise weights map.
# ============================================================================

suppressMessages({
  library(dplyr); library(lme4); library(lmerTest)
  library(emmeans); library(effectsize); library(readr)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

VALID_KINDS <- c("omnibus", "contrast", "regression", "equivalence")

# ---- Spec parsing ----------------------------------------------------------

#' Lift a legacy contrast (group_a/group_b) to a hypothesis def.
.contrast_to_hypothesis <- function(c) {
  kind <- if (identical(c$test, "equivalence")) "equivalence" else "contrast"
  w <- list(); w[[c$group_a]] <- 1; w[[c$group_b]] <- -1
  list(name = c$name, label = c$label, role = c$role, kind = kind,
       weights = w, test = c$test, margin = c$equivalence_margin)
}

#' Normalize one raw hypothesis mapping (from YAML) into a parsed def.
parse_hypothesis <- function(h) {
  if (is.null(h$name)) stop("hypothesis missing 'name'")
  # legacy sugar: group_a/group_b in place of weights
  if (is.null(h$kind) && is.null(h$weights) &&
      !is.null(h$group_a) && !is.null(h$group_b)) {
    return(parse_hypothesis(.contrast_to_hypothesis(h)))
  }
  kind <- h$kind %||% "contrast"
  if (!kind %in% VALID_KINDS)
    stop(sprintf("hypothesis '%s': invalid kind '%s' (expected %s)",
                 h$name, kind, paste(VALID_KINDS, collapse = "/")))
  weights <- NULL
  if (!is.null(h$weights)) {
    weights <- vapply(h$weights, as.numeric, numeric(1))
    names(weights) <- names(h$weights)
  }
  list(
    name      = h$name,
    label     = h$label %||% h$name,
    role      = h$role %||% "exploratory",
    kind      = kind,
    weights   = weights,
    groups    = if (!is.null(h$groups)) as.character(unlist(h$groups)) else NULL,
    predictor = h$predictor,
    by        = h$by,
    test      = h$test %||% (if (kind == "equivalence") "equivalence" else "difference"),
    margin    = h$margin %||% h$equivalence_margin
  )
}

#' Parse the design: + hypotheses: blocks from a read_yaml'd study config.
#' Falls back to lifting a legacy contrasts: block when hypotheses: is absent.
parse_design_spec <- function(config) {
  design <- config$design %||% list()
  raw_hyps <- config$hypotheses
  if (is.null(raw_hyps) && !is.null(config$contrasts))
    raw_hyps <- lapply(config$contrasts, .contrast_to_hypothesis)
  hyps <- lapply(raw_hyps %||% list(), parse_hypothesis)
  names(hyps) <- vapply(hyps, function(h) h$name, character(1))
  list(
    factor     = design$factor %||% "group",
    reference  = design$reference,
    levels     = if (!is.null(design$levels)) as.character(unlist(design$levels)) else NULL,
    covariates = if (!is.null(design$covariates)) as.character(unlist(design$covariates)) else character(0),
    hypotheses = hyps
  )
}

# ---- Helpers ---------------------------------------------------------------

#' Align a hypothesis's weights to factor `levels` (0-fill), in level order.
weight_vector <- function(hyp, levels) {
  wv <- setNames(numeric(length(levels)), levels)
  if (!is.null(hyp$weights)) {
    miss <- setdiff(names(hyp$weights), levels)
    if (length(miss))
      stop(sprintf("hypothesis '%s': weights reference level(s) not in the fit: %s",
                   hyp$name, paste(miss, collapse = ", ")))
    wv[names(hyp$weights)] <- as.numeric(hyp$weights)
  }
  wv
}

#' Which groups enter the model fit, given kind + fit_scope.
.fit_groups <- function(hyp, spec, scope) {
  if (hyp$kind %in% c("omnibus", "regression"))
    return(hyp$groups %||% spec$levels)
  if (identical(scope, "per_contrast"))
    return(names(hyp$weights)[hyp$weights != 0])
  spec$levels %||% names(hyp$weights)
}

#' Fit dv ~ grp [* spatial] [+ covariates] + (1|subject) on `groups`.
.hyp_fit <- function(data, spec, dv_col, spatial_col, groups) {
  d <- data[as.character(data[[spec$factor]]) %in% groups, , drop = FALSE]
  d$.grp <- factor(d[[spec$factor]], levels = groups)
  if (!is.null(spec$reference) && spec$reference %in% groups)
    d$.grp <- relevel(d$.grp, ref = spec$reference)
  d$.sp <- factor(d[[spatial_col]])
  d$.dv <- d[[dv_col]]
  covs <- intersect(spec$covariates, names(d))
  for (cv in covs) if (is.numeric(d[[cv]])) d[[cv]] <- d[[cv]] - mean(d[[cv]], na.rm = TRUE)
  cov_terms <- if (length(covs)) paste(" +", paste(covs, collapse = " + ")) else ""
  has_sp <- nlevels(d$.sp) > 1
  sp_term <- if (has_sp) " * .sp" else ""
  f <- as.formula(paste0(".dv ~ .grp", sp_term, cov_terms, " + (1 | subject)"))
  list(fit = suppressMessages(lmer(f, data = d)), data = d, has_sp = has_sp)
}

# ---- Kind adapters (emmeans) ----------------------------------------------

#' Omnibus F for the group term (partial omega^2 effect size). One row.
.adapt_omnibus <- function(fo) {
  aov <- anova(fo$fit, type = 3)
  if (!".grp" %in% rownames(aov)) stop("omnibus: no group term in model")
  fF <- aov[".grp", "F value"]; p <- aov[".grp", "Pr(>F)"]
  df1 <- aov[".grp", "NumDF"]; df2 <- aov[".grp", "DenDF"]
  o2 <- tryCatch(as.numeric(effectsize::F_to_omega2(fF, df1, df2)[1]),
                 error = function(e) NA_real_)
  data.frame(spatial = NA_character_, estimate = NA_real_, SE = NA_real_,
             df = df2, df_num = df1, stat = fF, stat_type = "F", p_value = p,
             estimate_lcl = NA_real_, estimate_ucl = NA_real_,
             effect_size = o2, effect_size_type = "omega2_partial",
             stringsAsFactors = FALSE)
}

#' Weighted linear contrast over group means, per spatial cell. Hedges g.
.adapt_contrast <- function(fo, hyp) {
  glev <- levels(fo$data$.grp)
  wv <- weight_vector(hyp, glev)
  emm <- if (fo$has_sp) emmeans(fo$fit, ~ .grp | .sp) else emmeans(fo$fit, ~ .grp)
  con <- contrast(emm, method = setNames(list(wv), hyp$name))
  cd <- as.data.frame(summary(con, infer = c(TRUE, TRUE)))
  resid_sd <- sigma(fo$fit)
  data.frame(
    spatial = if (fo$has_sp) as.character(cd$.sp) else NA_character_,
    estimate = cd$estimate, SE = cd$SE, df = cd$df, df_num = 1,
    stat = cd$t.ratio, stat_type = "t", p_value = cd$p.value,
    estimate_lcl = cd$lower.CL, estimate_ucl = cd$upper.CL,
    effect_size = cd$estimate / resid_sd, effect_size_type = "hedges_g",
    stringsAsFactors = FALSE
  )
}

#' Slope of dv on a continuous predictor (standardized beta). emtrends.
.adapt_regression <- function(data, spec, dv_col, hyp, groups) {
  pred <- hyp$predictor
  if (is.null(pred) || !pred %in% names(data))
    stop(sprintf("regression '%s': predictor '%s' not found", hyp$name, pred %||% "NULL"))
  d <- data[as.character(data[[spec$factor]]) %in% groups, , drop = FALSE]
  d$.grp <- factor(d[[spec$factor]], levels = groups)
  d$.dv <- d[[dv_col]]; d$.x <- as.numeric(d[[pred]])
  covs <- intersect(spec$covariates, names(d))
  for (cv in covs) if (is.numeric(d[[cv]])) d[[cv]] <- d[[cv]] - mean(d[[cv]], na.rm = TRUE)
  cov_terms <- if (length(covs)) paste(" +", paste(covs, collapse = " + ")) else ""
  by_grp <- !is.null(hyp$by)
  rhs <- if (by_grp) ".x * .grp" else ".x"
  fit <- suppressMessages(lmer(as.formula(
    paste0(".dv ~ ", rhs, cov_terms, " + (1 | subject)")), data = d))
  sdx <- sd(d$.x, na.rm = TRUE); sdy <- sd(d$.dv, na.rm = TRUE)
  et <- if (by_grp) emtrends(fit, ~ .grp, var = ".x") else emtrends(fit, ~ 1, var = ".x")
  cd <- as.data.frame(summary(et, infer = c(TRUE, TRUE)))
  slope <- cd[[".x.trend"]]
  tstat <- slope / cd$SE
  data.frame(
    spatial = if (by_grp) as.character(cd$.grp) else NA_character_,
    estimate = slope, SE = cd$SE, df = cd$df, df_num = 1,
    stat = tstat, stat_type = "t",
    p_value = 2 * pt(-abs(tstat), cd$df),
    estimate_lcl = cd$lower.CL, estimate_ucl = cd$upper.CL,
    effect_size = slope * sdx / sdy, effect_size_type = "std_beta",
    stringsAsFactors = FALSE
  )
}

#' Equivalence (TOST) on a weighted contrast. Reuses stats_utils primitives.
#' `ref_estimates`: named vector spatial->phenotype estimate (gap_fraction only).
.adapt_equivalence <- function(fo, hyp, ref_estimates = NULL) {
  base <- .adapt_contrast(fo, hyp)
  resid_sd <- sigma(fo$fit)
  base$margin_used <- NA_real_; base$equivalent <- NA
  for (i in seq_len(nrow(base))) {
    pheno <- if (!is.null(ref_estimates)) ref_estimates[[base$spatial[i] %||% ""]] else NA_real_
    m <- .equivalence_margin(hyp$margin, pheno, resid_sd)
    base$margin_used[i] <- m
    base$equivalent[i] <- tost_equivalent(base$estimate[i], base$SE[i], base$df[i], m)
  }
  base$stat_type <- "tost"
  base
}

# ---- Runner ----------------------------------------------------------------

#' Run ONE hypothesis over a (possibly multi-band) long data.frame.
#'
#' @param data long df: columns subject, <factor>, <spatial_col>, <dv_col>,
#'   optional <band_col> and covariates.
#' @param hyp a parsed hypothesis def, or a name resolved against spec$hypotheses.
#' @param spec parsed design spec (parse_design_spec()).
#' @param fit_scope "shared" (all design levels in one fit; required for omnibus)
#'   or "per_contrast" (subset to the contrast's groups — reproduces legacy SEs).
#' @return tidy df: one row per band x spatial cell, with estimate/SE/df/CI/stat/
#'   p_value/q_value/effect_size + hypothesis/kind/role/label/test/band metadata.
#'   FDR (within-run, across the band x spatial family) in q_value/significant.
run_hypothesis <- function(data, hyp, spec,
                           dv_col = "dv", spatial_col = "roi",
                           band_col = "band", bands = NULL,
                           fit_scope = "shared", fdr_method = "BH",
                           ref_estimates = NULL) {
  if (is.character(hyp)) {
    hyp <- spec$hypotheses[[hyp]]
    if (is.null(hyp)) stop("unknown hypothesis")
  }
  if (is.null(spec$levels))
    spec$levels <- sort(unique(as.character(data[[spec$factor]])))

  has_band <- !is.null(band_col) && band_col %in% names(data)
  band_vals <- if (has_band) (bands %||% unique(as.character(data[[band_col]]))) else NA

  out <- list()
  for (bn in band_vals) {
    bdata <- if (has_band) data[as.character(data[[band_col]]) == bn, , drop = FALSE] else data
    if (nrow(bdata) == 0) next
    groups <- intersect(.fit_groups(hyp, spec, fit_scope),
                        unique(as.character(bdata[[spec$factor]])))
    if (hyp$kind != "regression" && length(groups) < 2) next

    res <- tryCatch({
      if (hyp$kind == "regression") {
        .adapt_regression(bdata, spec, dv_col, hyp, groups %||% spec$levels)
      } else {
        fo <- .hyp_fit(bdata, spec, dv_col, spatial_col, groups)
        switch(hyp$kind,
               omnibus     = .adapt_omnibus(fo),
               contrast    = .adapt_contrast(fo, hyp),
               equivalence = .adapt_equivalence(fo, hyp, ref_estimates))
      }
    }, error = function(e) {
      message("  hypothesis '", hyp$name, "'",
              if (has_band) paste0(" [", bn, "]") else "", ": ", conditionMessage(e))
      NULL
    })
    if (!is.null(res) && nrow(res) > 0) {
      res$band <- if (has_band) bn else NA_character_
      out[[length(out) + 1]] <- res
    }
  }

  df <- bind_rows(out)
  if (nrow(df) == 0) return(df)
  df$hypothesis <- hyp$name; df$kind <- hyp$kind
  df$role <- hyp$role; df$label <- hyp$label; df$test <- hyp$test
  ok <- !is.na(df$p_value)
  df$q_value <- NA_real_
  df$q_value[ok] <- p.adjust(df$p_value[ok], method = fdr_method)
  df$significant <- !is.na(df$q_value) & df$q_value < 0.05
  df$fdr_family <- sprintf("%s: %d cells, %s", hyp$name, sum(ok), fdr_method)

  front <- c("hypothesis", "kind", "role", "band", "spatial")
  df[, c(front, setdiff(names(df), front)), drop = FALSE]
}

# ---- Legacy-schema compat shim --------------------------------------------

#' Add legacy-posthoc column aliases to a hypotheses data.frame.
#'
#' Lets the existing figure/table consumers (figure_registry, summary_figures,
#' render_posthoc_mosaics) read `<module>_hypotheses.csv` in place of the retired
#' `<module>_posthoc_*.csv` with NO code change — the native hypothesis columns
#' remain the source of truth; these are duplicates under the old names. Dropped
#' once the figure layer is migrated to the native schema.
#'
#' Aliases: contrast<-label/name, roi<-spatial, power_type<-dv,
#' hedges_g<-effect_size (only where effect_size_type=="hedges_g", so omnibus
#' omega^2 / regression beta are NOT mislabelled), p_fdr<-q_value.
.add_legacy_aliases <- function(df) {
  if (!"contrast" %in% names(df))
    df$contrast <- ifelse(is.na(df$label) | df$label == "", df$hypothesis, df$label)
  if ("spatial" %in% names(df) && !"roi" %in% names(df)) df$roi <- df$spatial
  if ("dv" %in% names(df) && !"power_type" %in% names(df)) df$power_type <- df$dv
  if ("effect_size" %in% names(df) && !"hedges_g" %in% names(df))
    df$hedges_g <- ifelse(df$effect_size_type == "hedges_g", df$effect_size, NA_real_)
  if ("q_value" %in% names(df) && !"p_fdr" %in% names(df)) df$p_fdr <- df$q_value
  df
}

# ---- Module convenience wrapper -------------------------------------------

#' Run every declared hypothesis for a module and write <prefix>_hypotheses.csv.
#'
#' The one-call wiring an *_analysis.R module uses (the Tier-1 emmeans sweep, see
#' DESIGN_SPEC.md). Loops hypotheses x dv_cols, runs each via run_hypothesis()
#' (fit_scope "shared"), tags the DV, binds, and writes the additive table.
#' Honors a --hypothesis name filter. Band-less modules pass band_col = NULL.
#'
#' @param df long data.frame (subject, <factor>, <spatial_col>, dv columns, opt band).
#' @param config read_yaml'd study config (design:/hypotheses: or legacy contrasts:).
#' @param tbl_dir output tables directory.
#' @param prefix output basename (e.g. "roi_psd" -> roi_psd_hypotheses.csv).
#' @param dv_cols character vector of dependent-variable column names to test.
#' @param spatial_col spatial unit column (default "roi").
#' @param band_col band column, or NULL for band-less modules (e.g. aperiodic).
#' @param hypothesis optional comma-separated name filter (the --hypothesis arg).
#' @param fit_scope "shared" (default) or "per_contrast".
#' @return (invisibly) the combined hypotheses data.frame.
write_module_hypotheses <- function(df, config, tbl_dir, prefix, dv_cols,
                                    spatial_col = "roi", band_col = "band",
                                    hypothesis = NULL, fit_scope = "shared") {
  spec <- parse_design_spec(config)
  if (length(spec$hypotheses) == 0) {
    message("  No hypotheses/contrasts declared — skipping ", prefix, " hypotheses.")
    return(invisible(NULL))
  }
  hyp_names <- names(spec$hypotheses)
  if (!is.null(hypothesis)) {
    want <- trimws(strsplit(hypothesis, ",")[[1]])
    hyp_names <- intersect(hyp_names, want)
    if (length(hyp_names) == 0)
      message("  No declared hypothesis matches --hypothesis '", hypothesis, "'")
  }
  has_band <- !is.null(band_col) && band_col %in% names(df)
  bands <- if (has_band) names(config$bands) else NULL
  out <- list()
  for (hn in hyp_names) for (dv in dv_cols) {
    res <- tryCatch(
      run_hypothesis(df, hn, spec, dv_col = dv, spatial_col = spatial_col,
                     band_col = if (has_band) band_col else NULL,
                     bands = bands, fit_scope = fit_scope),
      error = function(e) { message("  ", hn, "/", dv, ": ", conditionMessage(e)); NULL })
    if (!is.null(res) && nrow(res) > 0) { res$dv <- dv; out[[paste(hn, dv)]] <- res }
  }
  hyp_df <- bind_rows(out)
  if (nrow(hyp_df) > 0) {
    hyp_df <- .add_legacy_aliases(hyp_df)
    path <- file.path(tbl_dir, paste0(prefix, "_hypotheses.csv"))
    write_csv(hyp_df, path)
    n_sig <- sum(hyp_df$significant, na.rm = TRUE)
    message("  Saved: ", basename(path), " (", nrow(hyp_df), " rows, ",
            length(hyp_names), " hypotheses x ", length(dv_cols), " DV; ", n_sig, " sig cells)")
  }
  invisible(hyp_df)
}
