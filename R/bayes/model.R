# R/bayes/model.R — the engine's model: coding a long table, the formula and the prior table, generated from the spec
# (part of R/bayes/engine.R; docs/methods/BAYES_ENGINE.md section 3)
# ============================================================================
# Model (bands pooled; a band-less outcome such as the aperiodic exponent is one "band"):
#   z ~ 0 + lv_<band>_<group> ... [+ dx_<band>_<group> ...]  group level per band (every group, reference included)
#       + (0 + x_<band>_<group> ... || roi)               ROI deviation of each band's group effect (partial pooling)
#       + (1 | roi:band) + (0 + bd_<band> ... || subject), ROI level per band; subject level per band
#   sigma ~ 0 + bd_<band> ...,  family = gaussian()       (bd_<band> = 1 on rows of that band: numeric, so one band works)
# No ROI (spatial_col = NULL; bayes-sl L3, one value per subject, e.g. its power along a located field): the two ROI
# terms are dropped, and the subject term is kept only if some subject has more than one row in a band (with one row it
# is the residual under another name, identified by the prior alone). The readout then has one spatial cell, "all".
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
#
# Uncentred outcomes (center = FALSE; declared tables' `center: false`, 2026-11-10, bayes-sl SL learning effects): z
# is divided by the same SD but not centred, so 0 keeps its meaning (a difference score such as structured - random
# zITC) and a group's own level can be tested against 0: a hypothesis whose weights do not sum to 0 (e.g. {KO: 1}).
# The ROI level per band, (1 | roi:band), is then the ROI deviation of the level shared by every group: its SD prior is
# half-N(0, BAYES_SD_ROI_BAND_UNCENTRED), not the centred outcome's 3 (no anatomical baseline in a difference). A
# set-I entry on a single-group level hypothesis, lv_<band>_<group>, replaces that level's W prior.
# ============================================================================

suppressMessages({ library(brms); library(posterior) })
`%||%` <- function(a, b) if (is.null(a)) b else a

BAYES_SETS <- c("W", "I")
BAYES_ROPE_SD <- 0.1      # default ROPE for plain contrasts, between-subject SD units (the Conductor, 2026-10-14)
BAYES_HDI <- 0.95
BAYES_W_EFFECT_SD <- 1  # W: every pairwise group contrast a priori N(0, 1), Track A's weak default (the Conductor, 2026-10-14; was 0.5)
BAYES_W_LEVEL_SD <- BAYES_W_EFFECT_SD / sqrt(2)   # ... from independent group levels N(0, 1/sqrt(2)) (2026-10-19)
BAYES_CODING <- "group-levels-2026-10-19"   # fixed-effect parametrisation; a gate verdict attaches only to fits of the same
BAYES_SD_ROI_BAND <- 3  # ROI level per band, half-N(0, 3): FORGE source power ROI means spread 0.9-3.0 SD (the Conductor, 2026-10-15; was 0.5)
BAYES_SD_ROI_BAND_UNCENTRED <- 0.5  # ... with an uncentred outcome (a difference score; 2026-11-10): ROI deviations of
                                    # the shared level, like the subject term's half-N(0, 0.5)
BAYES_LOG_SIGMA <- log(0.5)  # residual SD prior median (z units) when ROI or subject terms take part of the spread
BAYES_LOG_SIGMA_NOROI <- log(1)  # ... with no ROI and no subject term: z's within-group SD is 1 by construction, so the
                                 # residual is all of it (bayes-sl a97d, 2026-11-09: log 0.5 put 1 at the prior's 92nd pct)
BAYES_RHAT_MAX <- 1.05  # convergence rule, on what is decided: estimands and scales (the Conductor, 2026-10-21; .estimand_conv)

.safe <- function(x) gsub("[^A-Za-z0-9]", "", x)

#' Code a long table for the model, for one prior set (the set decides the fixed-effect parametrisation, see the
#' header). `informative` is a named list x_<band>_<group> -> c(mean, sd) (group - reference) for set I; terms without
#' an entry stay W, and entries for terms not in this design are unused. Returns the data plus the term map.
#' spatial_col = NULL: no ROI (see the header). center = FALSE: z is scaled but not centred (see the header);
#' `informative` may then also hold lv_<band>_<group> entries (a group's own level).
bayes_prepare <- function(data, spec, dv_col, spatial_col = "roi", band_col = "band", set = "W", informative = NULL,
                          center = TRUE) {
  center <- !isFALSE(center)
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
  no_roi <- is.null(spatial_col)
  if (!no_roi && !spatial_col %in% names(d)) stop(sprintf("bayes: no spatial column '%s' (NULL for no ROI)", spatial_col))
  d$roi <- factor(if (no_roi) rep("all", nrow(d)) else as.character(d[[spatial_col]]))
  d$subject <- factor(as.character(d$subject))
  subject_re <- anyDuplicated(paste(d$subject, d$band)) > 0      # some subject has repeated rows within a band
  # standardise per band by the pooled within-group between-subject SD within roi x band (the Hedges g unit)
  scale_tab <- do.call(rbind, lapply(levels(d$band), function(b) {
    x <- d[d$band == b, ]
    cell <- interaction(x$roi, x$.grp, drop = TRUE)
    n <- tapply(x[[dv_col]], cell, length); v <- tapply(x[[dv_col]], cell, var)
    ok <- n > 1 & !is.na(v)
    data.frame(band = b, center = if (center) mean(x[[dv_col]]) else 0, scale = sqrt(sum((n[ok] - 1) * v[ok]) / sum(n[ok] - 1)))
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
    lvi <- if (identical(set, "I") && length(informative)) informative[sprintf("lv_%s_%s", b, .safe(levels))]
           else vector("list", length(levels))                # set-I level entries, NULL where none
    has_lv <- !vapply(lvi, is.null, logical(1))
    if (any(has_lv) && center) stop("bayes: set-I level entries (lv_*) need an uncentred outcome (center = FALSE)")
    if (any(has_lv & levels %in% anch))
      stop(sprintf("bayes: band %s: a group has both a level entry and a difference entry in set I", b))
    ms <- t(sapply(seq_along(levels), function(k) { g <- levels[k]
      if (g %in% anch) informative[[sprintf("x_%s_%s", b, .safe(g))]]
      else if (has_lv[k]) lvi[[k]] else c(0, BAYES_W_LEVEL_SD) }))
    fixed[[b]] <- data.frame(col = cols, band = b, group = levels, prior_mean = ms[, 1], prior_sd = ms[, 2],
                             row.names = NULL)
    for (p in cols) d[[p]] <- as.numeric(d$band == b) * M[as.character(d$.grp), p]
  }
  covs <- intersect(spec$covariates, names(d))
  for (cv in covs) d[[paste0("cov_", .safe(cv))]] <- d[[cv]] - mean(d[[cv]], na.rm = TRUE)
  list(data = d, terms = terms, fixed = do.call(rbind, unname(fixed)), lmap = lmap, set = set,
       levels = levels, ref = ref, bands = levels(d$band), band_labels = band_labels,
       covs = if (length(covs)) paste0("cov_", .safe(covs)) else character(0), scale = scale_tab,
       dv = dv_col, spatial = spatial_col, no_roi = no_roi, subject_re = subject_re, centred = center,
       sd_roi_band = if (center) BAYES_SD_ROI_BAND else BAYES_SD_ROI_BAND_UNCENTRED)
}

#' Prior mean of log sigma for prep's design (BAYES_LOG_SIGMA_NOROI when there is no ROI and no subject term).
bayes_log_sigma_mean <- function(prep)
  if (isTRUE(prep$no_roi) && isFALSE(prep$subject_re)) BAYES_LOG_SIGMA_NOROI else BAYES_LOG_SIGMA

#' SD prior of the ROI level per band: 3 for a centred outcome, BAYES_SD_ROI_BAND_UNCENTRED otherwise (preps made
#' before 2026-11-10 carry no sd_roi_band: centred).
.sd_roi_band <- function(prep) prep$sd_roi_band %||% BAYES_SD_ROI_BAND

#' TRUE when hypothesis weights w test a level (do not sum to 0); an error unless prep's outcome is uncentred.
.is_level <- function(w, prep, name) {
  lvl <- abs(sum(w)) > 1e-9
  if (lvl && !isFALSE(prep$centred))
    stop(sprintf(paste("hypothesis '%s': weights must sum to 0 for the Bayes adapter (a group level against 0 needs",
                       "an uncentred outcome: the declared table's center: false)"), name))
  lvl
}

bayes_formula <- function(prep) {
  fx <- paste(prep$fixed$col, collapse = " + ")
  x <- paste(prep$terms$col, collapse = " + ")
  bd <- paste(paste0("bd_", prep$bands), collapse = " + ")
  cv <- if (length(prep$covs)) paste(" +", paste(prep$covs, collapse = " + ")) else ""
  re <- c(if (!isTRUE(prep$no_roi)) c(paste0("(0 + ", x, " || roi)"), "(1 | roi:band)"),
          if (!isFALSE(prep$subject_re)) paste0("(0 + ", bd, " || subject)"))
  bf(as.formula(paste("z ~", paste(c(paste0("0 + ", fx, cv), re), collapse = " + "))),
     as.formula(paste0("sigma ~ 0 + ", bd)))
}

#' Prior table of prep's set, generated from the spec (docs/methods/BAYES_ENGINE.md section 5): group levels
#' N(0, BAYES_W_LEVEL_SD), set-I differences dx ~ their entry (derived by script from a logged literature search).
bayes_priors <- function(prep) {
  fx <- prep$fixed
  c(set_prior("normal(0, 1)", class = "b"),                            # covariates
    do.call(c, lapply(seq_len(nrow(fx)), function(k)
      set_prior(sprintf("normal(%g, %g)", fx$prior_mean[k], fx$prior_sd[k]), class = "b", coef = fx$col[k]))),
    if (!isTRUE(prep$no_roi)) c(set_prior("normal(0, 0.25)", class = "sd", group = "roi"),
                                set_prior(sprintf("normal(0, %g)", .sd_roi_band(prep)), class = "sd", group = "roi:band")),
    if (!isFALSE(prep$subject_re)) set_prior("normal(0, 0.5)", class = "sd", group = "subject"),
    set_prior(sprintf("normal(%g, 0.5)", bayes_log_sigma_mean(prep)), class = "b", dpar = "sigma"))
}
