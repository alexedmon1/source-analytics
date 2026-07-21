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
       weights = w, test = c$test, margin = c$equivalence_margin, fdr = c$fdr)
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
    margin    = h$margin %||% h$equivalence_margin,
    # Per-hypothesis FDR override (list with $scope and/or $method), or NULL to
    # inherit the design-level default. See .resolve_fdr / .apply_fdr.
    fdr       = h$fdr
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
    # Study-level FDR default (list with $scope and/or $method); per-hypothesis
    # fdr: overrides it field-by-field. Absent -> {scope=hypothesis, method=BH}.
    fdr        = design$fdr %||% list(),
    hypotheses = hyps
  )
}

#' Derive the legacy pairwise contrast list from a parsed design spec.
#'
#' The R mirror of config.py's `_contrasts_from_design_spec` /
#' `Hypothesis.pairwise_endpoints`. Returns a list of `{name, group_a, group_b,
#' label, role}` records — one per clean two-group (pairwise) contrast/equivalence
#' hypothesis (group_a = positive-weight level, group_b = negative-weight level).
#' Omnibus/regression and non-pairwise (>2-level weighted) hypotheses are skipped.
#'
#' Lets a module feed the KEPT `run_omnibus_lmm*` DIAGNOSTIC (and any other
#' contrast-iterating legacy code) now that `config$contrasts` is no longer
#' populated post design-spec migration — without re-introducing the bridge.
contrasts_from_spec <- function(spec) {
  out <- list()
  for (hyp in spec$hypotheses) {
    if (!hyp$kind %in% c("contrast", "equivalence")) next
    w <- hyp$weights
    if (is.null(w)) next
    pos <- names(w)[w > 0]; neg <- names(w)[w < 0]
    if (length(pos) == 1 && length(neg) == 1)
      out[[length(out) + 1]] <- list(name = hyp$name, group_a = pos, group_b = neg,
                                     label = hyp$label, role = hyp$role)
  }
  out
}

# ---- Multiple-comparison control -------------------------------------------

.FDR_SCOPES  <- c("hypothesis", "band", "spatial", "none")

#' Resolve the effective {method, scope} for one hypothesis.
#'
#' Precedence: per-hypothesis `fdr:` > design-level `fdr:` > built-in default
#' ({scope=hypothesis, method=BH} — i.e. correct across the whole band x spatial
#' grid, matching the pre-toggle behaviour). `default_method` lets a caller pass
#' a fallback (the run_hypothesis fdr_method arg) below the spec defaults.
.resolve_fdr <- function(hyp, spec, default_method = "BH") {
  sp <- spec$fdr %||% list()
  hp <- hyp$fdr %||% list()
  method <- hp$method %||% sp$method %||% default_method
  scope  <- hp$scope  %||% sp$scope  %||% "hypothesis"
  if (!method %in% p.adjust.methods)
    stop(sprintf("fdr method '%s' not one of %s", method,
                 paste(p.adjust.methods, collapse = "/")))
  if (!scope %in% .FDR_SCOPES)
    stop(sprintf("fdr scope '%s' not one of %s", scope,
                 paste(.FDR_SCOPES, collapse = "/")))
  list(method = method, scope = scope)
}

#' Apply multiple-comparison correction within scope-defined families.
#'
#' `scope` partitions the (band x spatial) result rows into the families across
#' which p-values are corrected: "hypothesis" = one family (all rows, the
#' conservative default); "band" = a family per band/freq_pair; "spatial" = a
#' family per spatial cell; "none" = no correction (each row its own family,
#' q = p). Aggressiveness is driven by FAMILY SIZE, not just the method.
#' Sets q_value/significant/fdr_family on `df` (rows with NA p are left out).
.apply_fdr <- function(df, method, scope, hypothesis = NA_character_,
                       dv = NA_character_, spatial_name = "spatial") {
  ok <- !is.na(df$p_value)
  n <- nrow(df)
  bandv <- if ("band" %in% names(df)) as.character(df$band) else rep(NA_character_, n)
  spatv <- if ("spatial" %in% names(df)) as.character(df$spatial) else rep(NA_character_, n)

  fam <- rep("all", n)
  if (scope == "band" && "band" %in% names(df)) {
    fam <- bandv
  } else if (scope == "spatial" && "spatial" %in% names(df)) {
    fam <- spatv
  } else if (scope == "none") {
    fam <- as.character(seq_len(n))          # every row alone -> q == p
  }
  fam[is.na(fam)] <- "all"

  df$q_value <- NA_real_
  df$fdr_family <- NA_character_
  hyp_lbl <- if (length(hypothesis) >= 1 && !is.na(hypothesis[1])) hypothesis[1] else "NA"
  dv_lbl  <- if (length(dv) >= 1 && !is.na(dv[1])) dv[1] else "NA"
  for (f in unique(fam[ok])) {
    idx <- which(ok & fam == f)
    df$q_value[idx] <- p.adjust(df$p_value[idx], method = method)
    df$fdr_family[idx] <- .fdr_family_label(
      method = method, scope = scope, family = f,
      hypothesis = hyp_lbl, dv = dv_lbl,
      bands = bandv[idx], spatial = spatv[idx], spatial_name = spatial_name)
  }
  df$significant <- !is.na(df$q_value) & df$q_value < 0.05
  df
}

#' Build the fully-qualified `fdr_family` label for one FDR family (W1).
#'
#' Encodes the family's fixed coordinates (`key` = the scope-fixed axis value,
#' hypothesis, dv) AND the IDENTITY of the corrected member set (axis, count,
#' and an 8-hex md5 over the sorted member strings). The hash means a 20-ROI
#' family is never mistaken for a 32-ROI one even when scope/method/band match,
#' so q-values from different profiles are provably non-comparable (REPORT_PLAN
#' §10b). q-values themselves are unchanged — this is label-only.
#'
#' Kept byte-parity with Python `_fdr_family_label`: same member selection, same
#' `sort(method="radix")` (C/codepoint order, matching Python `sorted`), same
#' `md5(paste(members, collapse="|"))`.
.fdr_family_label <- function(method, scope, family, hypothesis, dv,
                              bands, spatial, spatial_name) {
  # Normalize missing coords to the literal "NA" (matches Python str(None)->"NA")
  bands   <- ifelse(is.na(bands), "NA", as.character(bands))
  spatial <- ifelse(is.na(spatial), "NA", as.character(spatial))
  if (length(family) >= 1 && is.na(family[1])) family <- "NA"
  if (scope == "band") {
    members <- sort(unique(spatial), method = "radix"); axis <- spatial_name
    key <- paste(family, hypothesis, dv, sep = "|")
  } else if (scope == "spatial") {
    members <- sort(unique(bands), method = "radix"); axis <- "band"
    key <- paste(family, hypothesis, dv, sep = "|")
  } else if (scope == "none") {
    members <- paste(bands, spatial, sep = ":"); axis <- "row"
    key <- paste(bands[1], spatial[1], hypothesis, dv, sep = "|")
  } else {                                    # "hypothesis": whole band x spatial grid
    members <- sort(unique(paste(bands, spatial, sep = ":")), method = "radix"); axis <- "cell"
    key <- paste("all", hypothesis, dv, sep = "|")
  }
  members <- members[!is.na(members) & members != "NA"]
  h <- substr(digest::digest(paste(members, collapse = "|"),
                             algo = "md5", serialize = FALSE), 1, 8)
  sprintf("scope=%s method=%s key=%s members=%s[%d] hash=%s",
          scope, method, key, axis, length(members), h)
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

# emmeans falls back to ASYMPTOTIC (z) inference when n_obs exceeds the
# lmerTest/pbkrtest limit (default 3000): summary() then carries z.ratio /
# asymp.LCL / asymp.UCL and df=Inf instead of t.ratio / lower.CL / upper.CL.
# These accessors read whichever form is present, so high-cardinality spatial
# fits (e.g. 100 region-pairs x 60 subjects = 6000 obs) don't crash the adapter
# on a missing column. A z-test is the df=Inf limit of a t-test, so stat_type
# stays "t" and the t_ratio legacy alias keeps populating.
.emm_stat <- function(cd) if ("t.ratio" %in% names(cd)) cd$t.ratio else cd$z.ratio
.emm_df   <- function(cd) if ("df" %in% names(cd)) cd$df else rep(Inf, nrow(cd))
.emm_lcl  <- function(cd) if ("lower.CL" %in% names(cd)) cd$lower.CL else cd$asymp.LCL
.emm_ucl  <- function(cd) if ("upper.CL" %in% names(cd)) cd$upper.CL else cd$asymp.UCL

#' Weighted linear contrast over group means, per spatial cell. Hedges g.
#' marginal=TRUE collapses the spatial dimension: the contrast is taken on the
#' group means MARGINALIZED over .sp (one row), reproducing the legacy "global"
#' marginal-over-ROI contrast (group*spatial fit, emmeans ~ group).
.adapt_contrast <- function(fo, hyp, marginal = FALSE) {
  glev <- levels(fo$data$.grp)
  wv <- weight_vector(hyp, glev)
  per_sp <- fo$has_sp && !marginal
  emm <- if (per_sp) emmeans(fo$fit, ~ .grp | .sp) else emmeans(fo$fit, ~ .grp)
  con <- contrast(emm, method = setNames(list(wv), hyp$name))
  cd <- as.data.frame(summary(con, infer = c(TRUE, TRUE)))
  resid_sd <- sigma(fo$fit)
  # Pairwise endpoints (positive- vs negative-weight group) for legacy figures
  # that draw between-group brackets. NA when the contrast is not a clean pair.
  pos <- glev[wv > 0]; neg <- glev[wv < 0]
  data.frame(
    spatial = if (per_sp) as.character(cd$.sp) else NA_character_,
    estimate = cd$estimate, SE = cd$SE, df = .emm_df(cd), df_num = 1,
    stat = .emm_stat(cd), stat_type = "t", p_value = cd$p.value,
    estimate_lcl = .emm_lcl(cd), estimate_ucl = .emm_ucl(cd),
    effect_size = cd$estimate / resid_sd, effect_size_type = "hedges_g",
    group_a = if (length(pos) == 1) pos else NA_character_,
    group_b = if (length(neg) == 1) neg else NA_character_,
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
  cd_df <- .emm_df(cd)
  tstat <- slope / cd$SE
  data.frame(
    spatial = if (by_grp) as.character(cd$.grp) else NA_character_,
    estimate = slope, SE = cd$SE, df = cd_df, df_num = 1,
    stat = tstat, stat_type = "t",
    p_value = 2 * pt(-abs(tstat), cd_df),
    estimate_lcl = .emm_lcl(cd), estimate_ucl = .emm_ucl(cd),
    effect_size = slope * sdx / sdy, effect_size_type = "std_beta",
    stringsAsFactors = FALSE
  )
}

#' Equivalence (TOST) on a weighted contrast. Reuses stats_utils primitives.
#' `ref_estimates`: named vector spatial->phenotype estimate (gap_fraction only).
.adapt_equivalence <- function(fo, hyp, ref_estimates = NULL, marginal = FALSE) {
  base <- .adapt_contrast(fo, hyp, marginal = marginal)
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
                           ref_estimates = NULL, marginal = FALSE) {
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
    # Skip all-NA (band x dv) families (e.g. delta_ref on its excluded reference
    # band). The fit would error; skipping keeps the log clean and emits no row.
    if (dv_col %in% names(bdata) && all(is.na(bdata[[dv_col]]))) next
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
               contrast    = .adapt_contrast(fo, hyp, marginal = marginal),
               equivalence = .adapt_equivalence(fo, hyp, ref_estimates, marginal = marginal))
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
  # Multiple-comparison correction within scope-defined families (declarative
  # fdr: spec; default = whole band x spatial grid, method BH).
  fdr <- .resolve_fdr(hyp, spec, default_method = fdr_method)
  df <- .apply_fdr(df, fdr$method, fdr$scope,
                   hypothesis = hyp$name, dv = dv_col, spatial_name = spatial_col)

  front <- c("hypothesis", "kind", "role", "band", "spatial")
  df[, c(front, setdiff(names(df), front)), drop = FALSE]
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
#' @param bands optional band-level override. When NULL (default) the band axis
#'   is taken from names(config$bands) — correct for power/aperiodic modules. PAC
#'   and other modules whose band axis is NOT the study bands (e.g. freq_pair)
#'   pass the explicit level vector here.
#' @return (invisibly) the combined hypotheses data.frame.
write_module_hypotheses <- function(df, config, tbl_dir, prefix, dv_cols,
                                    spatial_col = "roi", band_col = "band",
                                    hypothesis = NULL, fit_scope = "shared",
                                    marginal = FALSE, bands = NULL) {
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
  band_levels <- if (has_band) (bands %||% names(config$bands)) else NULL
  out <- list()
  for (hn in hyp_names) for (dv in dv_cols) {
    res <- tryCatch(
      run_hypothesis(df, hn, spec, dv_col = dv, spatial_col = spatial_col,
                     band_col = if (has_band) band_col else NULL,
                     bands = band_levels, fit_scope = fit_scope, marginal = marginal),
      error = function(e) { message("  ", hn, "/", dv, ": ", conditionMessage(e)); NULL })
    if (!is.null(res) && nrow(res) > 0) { res$dv <- dv; out[[paste(hn, dv)]] <- res }
  }
  hyp_df <- bind_rows(out)
  if (nrow(hyp_df) > 0) {
    path <- file.path(tbl_dir, paste0(prefix, "_hypotheses.csv"))
    write_csv(hyp_df, path)
    n_sig <- sum(hyp_df$significant, na.rm = TRUE)
    message("  Saved: ", basename(path), " (", nrow(hyp_df), " rows, ",
            length(hyp_names), " hypotheses x ", length(dv_cols), " DV; ", n_sig, " sig cells)")
  }
  invisible(hyp_df)
}

# ---- Directed-edge adapter (asymmetric, mass-univariate) -------------------
# The emmeans adapter (run_hypothesis) puts the spatial axis INTO the model
# (group * spatial), which is infeasible for directed connectivity: a fully
# directed ROI graph has up to N*(N-1) ordered source->target edges (~10^3 for an
# atlas), far too many factor levels for one joint fit. The directed-edge adapter
# instead makes the edge the ITERATE, not a model factor: ONE small model per
# (band x edge) on all design levels (fit_scope "shared"), reused across every
# hypothesis, then correction across the edge family. Directed: source->target and
# target->source are distinct edges, tested independently.

#' Apply a kind adapter to ONE per-cell one-way model fit `lm(.dv ~ .grp)`.
#'
#' Mirrors the emmeans kind-adapters (.adapt_omnibus/.adapt_contrast/
#' .adapt_equivalence) but for a model where the spatial cell is the iterate
#' rather than a `.sp` factor, so the contrast is always over the (marginal) cell
#' means. The contrast is computed ANALYTICALLY from the cell means + pooled
#' residual SD, which for a balanced/unbalanced one-way `lm` is identical to
#' emmeans (cell means are mutually independent, cov = sigma^2 * diag(1/n_g)) but
#' ~100x cheaper — the adapter is called ~10^4 times (edges x bands), so emmeans'
#' reference-grid overhead is prohibitive. `tests/test_hypothesis.R` pins this
#' against a hand-fit emmeans contrast. `fit` is an `lm` (one obs per subject per
#' cell -> a (1|subject) term would be singular); `d` is that cell's data with
#' `.dv`/`.grp`; `glev` the group levels in fit order.
.adapt_cell <- function(fit, hyp, glev, d) {
  resid_sd <- sigma(fit)
  if (hyp$kind == "omnibus") {
    aov <- anova(fit)
    rn <- if (".grp" %in% rownames(aov)) ".grp" else rownames(aov)[1]
    fF <- aov[rn, "F value"]; p <- aov[rn, "Pr(>F)"]
    df1 <- aov[rn, "Df"]; df2 <- aov["Residuals", "Df"]
    o2 <- tryCatch(as.numeric(effectsize::F_to_omega2(fF, df1, df2)[1]),
                   error = function(e) NA_real_)
    return(data.frame(estimate = NA_real_, SE = NA_real_, df = df2, df_num = df1,
                      stat = fF, stat_type = "F", p_value = p,
                      estimate_lcl = NA_real_, estimate_ucl = NA_real_,
                      effect_size = o2, effect_size_type = "omega2_partial",
                      group_a = NA_character_, group_b = NA_character_,
                      stringsAsFactors = FALSE))
  }
  wv <- weight_vector(hyp, glev)
  gm <- tapply(d$.dv, d$.grp, mean)[glev]
  gn <- tapply(d$.dv, d$.grp, length)[glev]
  df_res <- df.residual(fit)
  est <- sum(wv * gm)
  se  <- resid_sd * sqrt(sum(wv^2 / gn))
  tval <- est / se
  pval <- 2 * pt(-abs(tval), df_res)
  tcrit <- qt(0.975, df_res)
  pos <- glev[wv > 0]; neg <- glev[wv < 0]
  res <- data.frame(
    estimate = est, SE = se, df = df_res, df_num = 1,
    stat = tval, stat_type = "t", p_value = pval,
    estimate_lcl = est - tcrit * se, estimate_ucl = est + tcrit * se,
    effect_size = est / resid_sd, effect_size_type = "hedges_g",
    group_a = if (length(pos) == 1) pos else NA_character_,
    group_b = if (length(neg) == 1) neg else NA_character_,
    stringsAsFactors = FALSE)
  if (hyp$kind == "equivalence") {
    m <- .equivalence_margin(hyp$margin, NA_real_, resid_sd)
    res$margin_used <- m
    res$equivalent <- tost_equivalent(res$estimate, res$SE, res$df, m)
    res$stat_type <- "tost"
  }
  res
}

#' Run hypotheses across asymmetric DIRECTED edges (mass-univariate).
#'
#' @param data long df: subject, <factor>, <source_col>, <target_col>, <dv_col>,
#'   optional <band_col>.
#' @param hyps character vector of hypothesis names, or list of parsed hyp defs.
#'   `regression` kind is skipped (no per-edge predictor contract). Fitting ONCE
#'   per edge and applying all hypotheses requires every hypothesis to share the
#'   same fit -> fit_scope "shared" only.
#' @param spec parsed design spec.
#' @param dv_col dependent-variable column (e.g. "te").
#' @param source_col,target_col directed-edge endpoint columns.
#' @param band_col band column, or NULL.
#' @param bands optional band-level override.
#' @param min_per_group skip an edge unless every fitted group has >= this many
#'   finite observations (default 3).
#' @return tidy df: one row per band x edge x hypothesis, edge id in `spatial`
#'   (+ `source`/`target`), FDR (declarative scope, per hypothesis across its
#'   band x edge family) in q_value/significant. Same schema as run_hypothesis().
run_directed_edges <- function(data, hyps, spec,
                               dv_col = "te",
                               source_col = "source_roi", target_col = "target_roi",
                               band_col = "band", bands = NULL,
                               fit_scope = "shared", fdr_method = "BH",
                               min_per_group = 3) {
  if (!identical(fit_scope, "shared"))
    stop("run_directed_edges supports fit_scope='shared' only")
  if (is.character(hyps)) hyps <- lapply(hyps, function(h) spec$hypotheses[[h]])
  hyps <- Filter(Negate(is.null), hyps)
  hyps <- Filter(function(h) !identical(h$kind, "regression"), hyps)
  if (length(hyps) == 0) return(data.frame())
  if (is.null(spec$levels))
    spec$levels <- sort(unique(as.character(data[[spec$factor]])))

  fac <- spec$factor
  has_band <- !is.null(band_col) && band_col %in% names(data)
  band_vals <- if (has_band) (bands %||% unique(as.character(data[[band_col]]))) else NA
  data$.edge <- paste(data[[source_col]], "->", data[[target_col]])

  rows <- list()
  for (bn in band_vals) {
    bdata <- if (has_band) data[as.character(data[[band_col]]) == bn, , drop = FALSE] else data
    if (nrow(bdata) == 0) next
    for (eg in unique(bdata$.edge)) {
      ed <- bdata[bdata$.edge == eg, , drop = FALSE]
      groups <- intersect(spec$levels, unique(as.character(ed[[fac]])))
      if (length(groups) < 2) next
      d <- ed[as.character(ed[[fac]]) %in% groups, , drop = FALSE]
      d$.dv <- d[[dv_col]]
      d <- d[is.finite(d$.dv), , drop = FALSE]
      gtab <- table(as.character(d[[fac]]))
      groups <- intersect(groups, names(gtab)[gtab >= min_per_group])
      if (length(groups) < 2) next
      d <- d[as.character(d[[fac]]) %in% groups, , drop = FALSE]
      d$.grp <- factor(d[[fac]], levels = groups)
      if (!is.null(spec$reference) && spec$reference %in% groups)
        d$.grp <- relevel(d$.grp, ref = spec$reference)
      fit <- tryCatch(lm(.dv ~ .grp, data = d), error = function(e) NULL)
      if (is.null(fit)) next
      glev <- levels(d$.grp)
      src <- as.character(ed[[source_col]][1]); tgt <- as.character(ed[[target_col]][1])
      for (hyp in hyps) {
        # contrast/equivalence weights may reference levels dropped by min_per_group
        if (hyp$kind %in% c("contrast", "equivalence")) {
          need <- names(hyp$weights)[hyp$weights != 0]
          if (!all(need %in% glev)) next
        }
        res <- tryCatch(.adapt_cell(fit, hyp, glev, d), error = function(e) NULL)
        if (is.null(res) || nrow(res) == 0) next
        res$hypothesis <- hyp$name; res$kind <- hyp$kind
        res$role <- hyp$role; res$label <- hyp$label; res$test <- hyp$test
        res$band <- if (has_band) bn else NA_character_
        res$spatial <- eg; res$source <- src; res$target <- tgt
        rows[[length(rows) + 1]] <- res
      }
    }
  }
  df <- bind_rows(rows)
  if (nrow(df) == 0) return(df)

  # FDR per hypothesis across its band x edge family (declarative scope).
  out <- list()
  for (hn in unique(df$hypothesis)) {
    sub <- df[df$hypothesis == hn, , drop = FALSE]
    hyp <- Filter(function(h) identical(h$name, hn), hyps)[[1]]
    fdr <- .resolve_fdr(hyp, spec, default_method = fdr_method)
    out[[hn]] <- .apply_fdr(sub, fdr$method, fdr$scope)
  }
  df <- bind_rows(out)

  front <- c("hypothesis", "kind", "role", "band", "spatial", "source", "target")
  df[, c(front, setdiff(names(df), front)), drop = FALSE]
}

#' Run every declared hypothesis across directed edges and write
#' <prefix>_directed_edges_hypotheses.csv. Directed-edge analogue of
#' write_module_hypotheses (one model per ordered source->target edge, reused
#' across hypotheses). Honors a --hypothesis name filter.
write_module_directed_edges <- function(df, config, tbl_dir, prefix, dv_cols,
                                        source_col = "source_roi", target_col = "target_roi",
                                        band_col = "band", hypothesis = NULL,
                                        fit_scope = "shared", bands = NULL,
                                        min_per_group = 3) {
  spec <- parse_design_spec(config)
  if (length(spec$hypotheses) == 0) {
    message("  No hypotheses/contrasts declared — skipping ", prefix, " directed edges.")
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
  band_levels <- if (has_band) (bands %||% names(config$bands)) else NULL
  out <- list()
  for (dv in dv_cols) {
    res <- tryCatch(
      run_directed_edges(df, hyp_names, spec, dv_col = dv,
                         source_col = source_col, target_col = target_col,
                         band_col = if (has_band) band_col else NULL,
                         bands = band_levels, fit_scope = fit_scope,
                         min_per_group = min_per_group),
      error = function(e) { message("  directed edges/", dv, ": ", conditionMessage(e)); NULL })
    if (!is.null(res) && nrow(res) > 0) { res$dv <- dv; out[[dv]] <- res }
  }
  hyp_df <- bind_rows(out)
  if (nrow(hyp_df) > 0) {
    path <- file.path(tbl_dir, paste0(prefix, "_directed_edges_hypotheses.csv"))
    write_csv(hyp_df, path)
    n_sig <- sum(hyp_df$significant, na.rm = TRUE)
    message("  Saved: ", basename(path), " (", nrow(hyp_df), " rows, ",
            length(hyp_names), " hypotheses x ", length(dv_cols), " DV; ", n_sig, " sig edges)")
  }
  invisible(hyp_df)
}
