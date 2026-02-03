# stats_utils.R — Statistical analysis for source-analytics
#
# Functions: run_band_contrasts()
#   - Welch's t-test (subject-level means)
#   - LMM via lme4/lmerTest: relative ~ group + (1|subject)
#   - Hedges' g via effectsize
#   - FDR correction via p.adjust(method = "BH")

library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(effectsize)

#' Run all contrasts across all bands
#'
#' @param band_df data.frame with columns: subject, group, roi, band, absolute, relative, dB
#' @param contrasts list of lists, each with name, group_a, group_b
#' @param bands named list of c(fmin, fmax) — used only for ordering
#' @return data.frame of statistics (one row per contrast x band)
run_band_contrasts <- function(band_df, contrasts, bands) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      bdata <- band_df %>% filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      # Subject-level means (average across ROIs to avoid pseudoreplication)
      subj_means <- bdata %>%
        group_by(subject, group) %>%
        summarise(relative = mean(relative, na.rm = TRUE), .groups = "drop")

      vals_a <- subj_means %>% filter(group == ga) %>% pull(relative)
      vals_b <- subj_means %>% filter(group == gb) %>% pull(relative)

      # --- Welch's t-test ---
      if (length(vals_a) >= 2 && length(vals_b) >= 2) {
        tt <- t.test(vals_a, vals_b, var.equal = FALSE)
        t_stat <- tt$statistic
        t_p <- tt$p.value
        t_df <- tt$parameter
      } else {
        t_stat <- NA; t_p <- NA; t_df <- NA
      }

      # --- Hedges' g ---
      if (length(vals_a) >= 2 && length(vals_b) >= 2) {
        es <- hedges_g(vals_a, vals_b)
        hg <- es$Hedges_g
      } else {
        hg <- NA
      }

      # --- LMM: relative ~ group + (1|subject) on ROI-level data ---
      lmm_z <- NA; lmm_p <- NA; lmm_converged <- TRUE
      tryCatch({
        bdata$group <- factor(bdata$group, levels = c(gb, ga))
        fit <- lmer(relative ~ group + (1 | subject), data = bdata)
        coefs <- summary(fit)$coefficients
        # Group effect is the non-intercept row
        group_row <- grep("group", rownames(coefs), ignore.case = TRUE)
        if (length(group_row) > 0) {
          lmm_z <- coefs[group_row[1], "t value"]
          lmm_p <- coefs[group_row[1], "Pr(>|t|)"]
        }
      }, warning = function(w) {
        # Singular fit or convergence warnings
        lmm_converged <<- FALSE
      }, error = function(e) {
        lmm_converged <<- FALSE
      })

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        band = band_name,
        group_a = ga,
        group_b = gb,
        n_a = length(vals_a),
        n_b = length(vals_b),
        group_a_mean = mean(vals_a, na.rm = TRUE),
        group_b_mean = mean(vals_b, na.rm = TRUE),
        group_a_sd = sd(vals_a, na.rm = TRUE),
        group_b_sd = sd(vals_b, na.rm = TRUE),
        t_stat = as.numeric(t_stat),
        t_df = as.numeric(t_df),
        p_value = as.numeric(t_p),
        hedges_g = as.numeric(hg),
        lmm_z = as.numeric(lmm_z),
        lmm_p = as.numeric(lmm_p),
        lmm_converged = lmm_converged,
        stringsAsFactors = FALSE
      )
    }
  }

  stats_df <- bind_rows(results)
  if (nrow(stats_df) == 0) return(stats_df)

  # FDR correction per contrast
  stats_df <- stats_df %>%
    group_by(contrast) %>%
    mutate(
      q_value = p.adjust(p_value, method = "BH"),
      significant = q_value < 0.05,
      lmm_q = p.adjust(lmm_p, method = "BH"),
      lmm_significant = lmm_q < 0.05
    ) %>%
    ungroup()

  return(stats_df)
}
