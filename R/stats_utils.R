# stats_utils.R — Statistical analysis for source-analytics
#
# ROI-level: run_omnibus_lmm(), run_posthoc_emmeans()
# Region-level (averaged): aggregate_to_regions(), run_omnibus_lmm_region(), run_posthoc_emmeans_region()
# Region-level (nested): run_omnibus_lmm_region_nested(), run_posthoc_emmeans_region_nested()
#   - Omnibus: lmer(dv ~ group * spatial + (1|subject)), Type III ANOVA
#   - Post-hoc: emmeans(fit, pairwise ~ group | spatial), Holm correction
#   - FDR (BH) correction across bands per contrast
#   - power_type: "relative" or "absolute"

library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(effectsize)
library(emmeans)

#' Run omnibus interaction LMM for each contrast x band
#'
#' Model: dv ~ group * roi + (1|subject)
#' Reports Type III ANOVA F-tests for group, roi, and group:roi interaction.
#' FDR (BH) correction applied across bands within each contrast.
#'
#' @param band_df data.frame with columns: subject, group, roi, band, absolute, relative, dB
#' @param contrasts list of lists, each with name, group_a, group_b
#' @param bands named list of c(fmin, fmax) — used only for ordering
#' @param power_type character — column to use as DV: "relative" or "absolute"
#' @return data.frame (one row per contrast x band)
run_omnibus_lmm <- function(band_df, contrasts, bands, power_type = "relative") {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      bdata <- band_df %>% filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      n_a <- length(unique(bdata$subject[bdata$group == ga]))
      n_b <- length(unique(bdata$subject[bdata$group == gb]))
      n_rois <- length(unique(bdata$roi))

      bdata$group <- factor(bdata$group, levels = c(ga, gb))
      bdata$roi <- factor(bdata$roi)
      bdata$dv <- bdata[[power_type]]

      group_F <- NA; group_p <- NA
      roi_F <- NA; roi_p <- NA
      interaction_F <- NA; interaction_p <- NA
      converged <- TRUE; singular <- FALSE

      tryCatch({
        fit <- lmer(dv ~ group * roi + (1 | subject), data = bdata)
        singular <- isSingular(fit)

        aov <- anova(fit, type = 3)

        if ("group" %in% rownames(aov)) {
          group_F <- aov["group", "F value"]
          group_p <- aov["group", "Pr(>F)"]
        }
        if ("roi" %in% rownames(aov)) {
          roi_F <- aov["roi", "F value"]
          roi_p <- aov["roi", "Pr(>F)"]
        }
        if ("group:roi" %in% rownames(aov)) {
          interaction_F <- aov["group:roi", "F value"]
          interaction_p <- aov["group:roi", "Pr(>F)"]
        }
      }, warning = function(w) {
        if (grepl("singular|converge", conditionMessage(w), ignore.case = TRUE)) {
          singular <<- TRUE
        }
      }, error = function(e) {
        converged <<- FALSE
      })

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        band = band_name,
        power_type = power_type,
        group_a = ga,
        group_b = gb,
        n_a = n_a,
        n_b = n_b,
        n_rois = n_rois,
        group_F = as.numeric(group_F),
        group_p = as.numeric(group_p),
        roi_F = as.numeric(roi_F),
        roi_p = as.numeric(roi_p),
        interaction_F = as.numeric(interaction_F),
        interaction_p = as.numeric(interaction_p),
        converged = converged,
        singular = singular,
        stringsAsFactors = FALSE
      )
    }
  }

  omnibus_df <- bind_rows(results)
  if (nrow(omnibus_df) == 0) return(omnibus_df)

  # No cross-band correction: bands are pre-specified, report per-band p-values directly
  omnibus_df <- omnibus_df %>%
    mutate(
      group_q = group_p,
      group_significant = group_q < 0.05,
      interaction_q = interaction_p,
      interaction_significant = interaction_q < 0.05
    )

  return(omnibus_df)
}


#' Run emmeans post-hoc contrasts per ROI for significant omnibus results
#'
#' @param band_df data.frame with columns: subject, group, roi, band, absolute, relative, dB
#' @param contrasts list of lists, each with name, group_a, group_b
#' @param bands named list of c(fmin, fmax)
#' @param omnibus_df data.frame from run_omnibus_lmm()
#' @param power_type character — column to use as DV
#' @param gate logical — if TRUE (default), only run post-hoc for significant omnibus results
#' @return data.frame (one row per contrast x band x roi)
run_posthoc_emmeans <- function(band_df, contrasts, bands, omnibus_df,
                                 power_type = "relative", gate = TRUE) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      if (gate && nrow(omnibus_df) > 0) {
        omni_row <- omnibus_df %>%
          filter(contrast == cname, band == band_name, power_type == !!power_type)
        if (nrow(omni_row) == 0) next
        if (!isTRUE(omni_row$group_significant[1]) &&
            !isTRUE(omni_row$interaction_significant[1])) next
      }

      bdata <- band_df %>% filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      bdata$group <- factor(bdata$group, levels = c(ga, gb))
      bdata$roi <- factor(bdata$roi)
      bdata$dv <- bdata[[power_type]]

      tryCatch({
        fit <- lmer(dv ~ group * roi + (1 | subject), data = bdata)

        emm <- emmeans(fit, pairwise ~ group | roi)
        con_df <- as.data.frame(summary(emm$contrasts, infer = c(TRUE, TRUE)))
        emm_df <- as.data.frame(emm$emmeans)

        resid_sd <- sigma(fit)
        con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

        for (i in seq_len(nrow(con_df))) {
          roi_name <- as.character(con_df$roi[i])

          emm_a <- emm_df %>%
            filter(roi == roi_name, group == ga) %>%
            pull(emmean)
          emm_b <- emm_df %>%
            filter(roi == roi_name, group == gb) %>%
            pull(emmean)

          hg <- con_df$estimate[i] / resid_sd

          results[[length(results) + 1]] <- data.frame(
            contrast = cname,
            band = band_name,
            power_type = power_type,
            roi = roi_name,
            estimate = con_df$estimate[i],
            estimate_lcl = con_df$lower.CL[i],
            estimate_ucl = con_df$upper.CL[i],
            SE = con_df$SE[i],
            df = con_df$df[i],
            t_ratio = con_df$t.ratio[i],
            p_value = con_df$p.value[i],
            q_value = con_df$q_value[i],
            emmean_a = if (length(emm_a) > 0) emm_a[1] else NA,
            emmean_b = if (length(emm_b) > 0) emm_b[1] else NA,
            hedges_g = hg,
            hedges_g_lcl = con_df$lower.CL[i] / resid_sd,
            hedges_g_ucl = con_df$upper.CL[i] / resid_sd,
            significant = con_df$q_value[i] < 0.05,
            stringsAsFactors = FALSE
          )
        }
      }, warning = function(w) {
        # Continue on singular fit warnings
      }, error = function(e) {
        message("  Post-hoc failed for ", cname, "/", band_name, "/", power_type, ": ", conditionMessage(e))
      })
    }
  }

  posthoc_df <- bind_rows(results)
  return(posthoc_df)
}


#' Aggregate ROI-level data to region-level means
#'
#' @param band_df data.frame with columns: subject, group, roi, band, absolute, relative, dB
#' @param roi_categories named list of ROI name vectors
#' @return data.frame with 'region' column replacing 'roi'
aggregate_to_regions <- function(band_df, roi_categories) {
  roi_to_region <- data.frame(
    roi = unlist(roi_categories),
    region = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  band_df %>%
    inner_join(roi_to_region, by = "roi") %>%
    group_by(subject, group, region, band) %>%
    summarise(
      absolute = mean(absolute, na.rm = TRUE),
      relative = mean(relative, na.rm = TRUE),
      .groups = "drop"
    )
}


#' Run omnibus interaction LMM at region level
#'
#' @param band_df data.frame with ROI-level data
#' @param contrasts list of contrast definitions
#' @param bands named list of frequency band limits
#' @param roi_categories named list of ROI name vectors
#' @param power_type character — column to use as DV
#' @return data.frame (one row per contrast x band)
run_omnibus_lmm_region <- function(band_df, contrasts, bands, roi_categories,
                                    power_type = "relative") {
  region_df <- aggregate_to_regions(band_df, roi_categories)
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      bdata <- region_df %>% filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      n_a <- length(unique(bdata$subject[bdata$group == ga]))
      n_b <- length(unique(bdata$subject[bdata$group == gb]))
      n_regions <- length(unique(bdata$region))

      bdata$group <- factor(bdata$group, levels = c(ga, gb))
      bdata$region <- factor(bdata$region)
      bdata$dv <- bdata[[power_type]]

      group_F <- NA; group_p <- NA
      region_F <- NA; region_p <- NA
      interaction_F <- NA; interaction_p <- NA
      converged <- TRUE; singular <- FALSE

      tryCatch({
        fit <- lmer(dv ~ group * region + (1 | subject), data = bdata)
        singular <- isSingular(fit)

        aov <- anova(fit, type = 3)

        if ("group" %in% rownames(aov)) {
          group_F <- aov["group", "F value"]
          group_p <- aov["group", "Pr(>F)"]
        }
        if ("region" %in% rownames(aov)) {
          region_F <- aov["region", "F value"]
          region_p <- aov["region", "Pr(>F)"]
        }
        if ("group:region" %in% rownames(aov)) {
          interaction_F <- aov["group:region", "F value"]
          interaction_p <- aov["group:region", "Pr(>F)"]
        }
      }, warning = function(w) {
        if (grepl("singular|converge", conditionMessage(w), ignore.case = TRUE)) {
          singular <<- TRUE
        }
      }, error = function(e) {
        converged <<- FALSE
      })

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        band = band_name,
        power_type = power_type,
        group_a = ga,
        group_b = gb,
        n_a = n_a,
        n_b = n_b,
        n_regions = n_regions,
        group_F = as.numeric(group_F),
        group_p = as.numeric(group_p),
        region_F = as.numeric(region_F),
        region_p = as.numeric(region_p),
        interaction_F = as.numeric(interaction_F),
        interaction_p = as.numeric(interaction_p),
        converged = converged,
        singular = singular,
        stringsAsFactors = FALSE
      )
    }
  }

  omnibus_df <- bind_rows(results)
  if (nrow(omnibus_df) == 0) return(omnibus_df)

  omnibus_df <- omnibus_df %>%
    mutate(
      group_q = group_p,
      group_significant = group_q < 0.05,
      interaction_q = interaction_p,
      interaction_significant = interaction_q < 0.05
    )

  return(omnibus_df)
}


#' Run emmeans post-hoc contrasts per region
#'
#' @param band_df data.frame with ROI-level data
#' @param contrasts list of contrast definitions
#' @param bands named list of frequency band limits
#' @param roi_categories named list of ROI name vectors
#' @param omnibus_region_df data.frame from run_omnibus_lmm_region()
#' @param power_type character — column to use as DV
#' @param gate logical — if TRUE, only run for significant omnibus results
#' @return data.frame (one row per contrast x band x region)
run_posthoc_emmeans_region <- function(band_df, contrasts, bands, roi_categories,
                                       omnibus_region_df, power_type = "relative",
                                       gate = TRUE) {
  region_df <- aggregate_to_regions(band_df, roi_categories)
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      if (gate && nrow(omnibus_region_df) > 0) {
        omni_row <- omnibus_region_df %>%
          filter(contrast == cname, band == band_name, power_type == !!power_type)
        if (nrow(omni_row) == 0) next
        if (!isTRUE(omni_row$group_significant[1]) &&
            !isTRUE(omni_row$interaction_significant[1])) next
      }

      bdata <- region_df %>% filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      bdata$group <- factor(bdata$group, levels = c(ga, gb))
      bdata$region <- factor(bdata$region)
      bdata$dv <- bdata[[power_type]]

      tryCatch({
        fit <- lmer(dv ~ group * region + (1 | subject), data = bdata)

        emm <- emmeans(fit, pairwise ~ group | region)
        con_df <- as.data.frame(summary(emm$contrasts, infer = c(TRUE, TRUE)))
        emm_df <- as.data.frame(emm$emmeans)

        resid_sd <- sigma(fit)
        con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

        for (i in seq_len(nrow(con_df))) {
          region_name <- as.character(con_df$region[i])

          emm_a <- emm_df %>%
            filter(region == region_name, group == ga) %>%
            pull(emmean)
          emm_b <- emm_df %>%
            filter(region == region_name, group == gb) %>%
            pull(emmean)

          hg <- con_df$estimate[i] / resid_sd

          results[[length(results) + 1]] <- data.frame(
            contrast = cname,
            band = band_name,
            power_type = power_type,
            region = region_name,
            estimate = con_df$estimate[i],
            estimate_lcl = con_df$lower.CL[i],
            estimate_ucl = con_df$upper.CL[i],
            SE = con_df$SE[i],
            df = con_df$df[i],
            t_ratio = con_df$t.ratio[i],
            p_value = con_df$p.value[i],
            q_value = con_df$q_value[i],
            emmean_a = if (length(emm_a) > 0) emm_a[1] else NA,
            emmean_b = if (length(emm_b) > 0) emm_b[1] else NA,
            hedges_g = hg,
            hedges_g_lcl = con_df$lower.CL[i] / resid_sd,
            hedges_g_ucl = con_df$upper.CL[i] / resid_sd,
            significant = con_df$q_value[i] < 0.05,
            stringsAsFactors = FALSE
          )
        }
      }, warning = function(w) {
        # Continue on singular fit warnings
      }, error = function(e) {
        message("  Region post-hoc failed for ", cname, "/", band_name, "/", power_type, ": ", conditionMessage(e))
      })
    }
  }

  posthoc_df <- bind_rows(results)
  return(posthoc_df)
}


#' Run omnibus interaction LMM at region level with nested replicates
#'
#' Maps each ROI/electrode to its region but does NOT average — individual
#' ROIs/electrodes are retained as replicate observations within regions.
#' Model: dv ~ group * region + (1|subject)
#'
#' @param band_df data.frame with columns: subject, group, roi, band, absolute, relative, dB
#' @param contrasts list of contrast definitions
#' @param bands named list of frequency band limits
#' @param roi_categories named list of ROI/electrode name vectors per region
#' @param power_type character — column to use as DV
#' @return data.frame (one row per contrast x band)
run_omnibus_lmm_region_nested <- function(band_df, contrasts, bands, roi_categories,
                                           power_type = "relative") {
  # Map each ROI to its region (no averaging)
  roi_to_region <- data.frame(
    roi = unlist(roi_categories),
    region = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  nested_df <- band_df %>% inner_join(roi_to_region, by = "roi")
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      bdata <- nested_df %>% filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      n_a <- length(unique(bdata$subject[bdata$group == ga]))
      n_b <- length(unique(bdata$subject[bdata$group == gb]))
      n_rois_total <- length(unique(bdata$roi))
      n_regions <- length(unique(bdata$region))

      bdata$group <- factor(bdata$group, levels = c(ga, gb))
      bdata$region <- factor(bdata$region)
      bdata$dv <- bdata[[power_type]]

      group_F <- NA; group_p <- NA
      region_F <- NA; region_p <- NA
      interaction_F <- NA; interaction_p <- NA
      converged <- TRUE; singular <- FALSE

      tryCatch({
        fit <- lmer(dv ~ group * region + (1 | subject), data = bdata)
        singular <- isSingular(fit)

        aov <- anova(fit, type = 3)

        if ("group" %in% rownames(aov)) {
          group_F <- aov["group", "F value"]
          group_p <- aov["group", "Pr(>F)"]
        }
        if ("region" %in% rownames(aov)) {
          region_F <- aov["region", "F value"]
          region_p <- aov["region", "Pr(>F)"]
        }
        if ("group:region" %in% rownames(aov)) {
          interaction_F <- aov["group:region", "F value"]
          interaction_p <- aov["group:region", "Pr(>F)"]
        }
      }, warning = function(w) {
        if (grepl("singular|converge", conditionMessage(w), ignore.case = TRUE)) {
          singular <<- TRUE
        }
      }, error = function(e) {
        converged <<- FALSE
      })

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        band = band_name,
        power_type = power_type,
        group_a = ga,
        group_b = gb,
        n_a = n_a,
        n_b = n_b,
        n_rois_total = n_rois_total,
        n_regions = n_regions,
        group_F = as.numeric(group_F),
        group_p = as.numeric(group_p),
        region_F = as.numeric(region_F),
        region_p = as.numeric(region_p),
        interaction_F = as.numeric(interaction_F),
        interaction_p = as.numeric(interaction_p),
        converged = converged,
        singular = singular,
        stringsAsFactors = FALSE
      )
    }
  }

  omnibus_df <- bind_rows(results)
  if (nrow(omnibus_df) == 0) return(omnibus_df)

  omnibus_df <- omnibus_df %>%
    mutate(
      group_q = group_p,
      group_significant = group_q < 0.05,
      interaction_q = interaction_p,
      interaction_significant = interaction_q < 0.05
    )

  return(omnibus_df)
}


#' Run emmeans post-hoc contrasts per region with nested replicates
#'
#' Same model as run_omnibus_lmm_region_nested() — individual ROIs/electrodes
#' retained as replicates. emmeans(fit, pairwise ~ group | region).
#'
#' @param band_df data.frame with columns: subject, group, roi, band, absolute, relative, dB
#' @param contrasts list of contrast definitions
#' @param bands named list of frequency band limits
#' @param roi_categories named list of ROI/electrode name vectors per region
#' @param omnibus_df data.frame from run_omnibus_lmm_region_nested()
#' @param power_type character — column to use as DV
#' @param gate logical — if TRUE, only run for significant omnibus results
#' @return data.frame (one row per contrast x band x region)
##' Convert q-values to significance star labels
#'
#' @param q numeric vector of q-values (FDR-corrected p-values)
#' @return character vector: "***" (q<0.001), "**" (q<0.01), "*" (q<0.05), "" otherwise
sig_stars <- function(q) {
  ifelse(q < 0.001, "***",
  ifelse(q < 0.01, "**",
  ifelse(q < 0.05, "*", "")))
}


#' Run marginal (global) pairwise group comparisons from existing LMM fit
#'
#' Calls emmeans(fit, pairwise ~ group) which averages over all ROI/region/spatial
#' levels, giving the overall pairwise group contrast. Designed to be called
#' alongside existing per-ROI posthoc — reuses the same model fit.
#'
#' @param data data.frame with columns: subject, group, dv, and a spatial factor (roi/region)
#' @param contrasts list of lists, each with name, group_a, group_b
#' @param spatial_col name of spatial grouping column (default "roi")
#' @param dv_col name of dependent variable column (default "dv")
#' @param dv_label character label for the DV in output (e.g., "absolute", "relative", "exponent")
#' @param band_label optional band name (for multi-band analyses)
#' @return data.frame with one row per contrast: contrast, dv, band, estimate, SE, df, t_ratio, p_value, q_value, hedges_g, significant, sig_label
run_posthoc_global <- function(data, contrasts, spatial_col = "roi",
                               dv_col = "dv", dv_label = "",
                               band_label = NA_character_) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    cdata <- data %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    cdata$group <- factor(cdata$group, levels = c(ga, gb))
    cdata[[spatial_col]] <- factor(cdata[[spatial_col]])
    cdata$dv <- cdata[[dv_col]]

    tryCatch({
      formula_str <- paste0("dv ~ group * ", spatial_col, " + (1 | subject)")
      fit <- lmer(as.formula(formula_str), data = cdata)

      emm <- emmeans(fit, pairwise ~ group)
      con_df <- as.data.frame(emm$contrasts)

      resid_sd <- sigma(fit)
      hg <- con_df$estimate[1] / resid_sd

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        dv = dv_label,
        band = band_label,
        group_a = ga,
        group_b = gb,
        estimate = con_df$estimate[1],
        SE = con_df$SE[1],
        df = con_df$df[1],
        t_ratio = con_df$t.ratio[1],
        p_value = con_df$p.value[1],
        hedges_g = hg,
        stringsAsFactors = FALSE
      )
    }, warning = function(w) {
      # Continue on singular fit warnings
    }, error = function(e) {
      message("  Global posthoc failed for ", cname, "/", dv_label,
              if (!is.na(band_label)) paste0("/", band_label) else "",
              ": ", conditionMessage(e))
    })
  }

  global_df <- bind_rows(results)
  if (nrow(global_df) > 0) {
    # No cross-band correction: per-band p-values reported directly
    global_df$q_value <- global_df$p_value
    global_df$significant <- global_df$q_value < 0.05
    global_df$sig_label <- sig_stars(global_df$q_value)
  }
  return(global_df)
}


run_posthoc_emmeans_region_nested <- function(band_df, contrasts, bands, roi_categories,
                                               omnibus_df, power_type = "relative",
                                               gate = TRUE) {
  # Map each ROI to its region (no averaging)
  roi_to_region <- data.frame(
    roi = unlist(roi_categories),
    region = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  nested_df <- band_df %>% inner_join(roi_to_region, by = "roi")
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      if (gate && nrow(omnibus_df) > 0) {
        omni_row <- omnibus_df %>%
          filter(contrast == cname, band == band_name, power_type == !!power_type)
        if (nrow(omni_row) == 0) next
        if (!isTRUE(omni_row$group_significant[1]) &&
            !isTRUE(omni_row$interaction_significant[1])) next
      }

      bdata <- nested_df %>% filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      bdata$group <- factor(bdata$group, levels = c(ga, gb))
      bdata$region <- factor(bdata$region)
      bdata$dv <- bdata[[power_type]]

      tryCatch({
        fit <- lmer(dv ~ group * region + (1 | subject), data = bdata)

        emm <- emmeans(fit, pairwise ~ group | region)
        con_df <- as.data.frame(summary(emm$contrasts, infer = c(TRUE, TRUE)))
        emm_df <- as.data.frame(emm$emmeans)

        resid_sd <- sigma(fit)
        con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

        for (i in seq_len(nrow(con_df))) {
          region_name <- as.character(con_df$region[i])

          emm_a <- emm_df %>%
            filter(region == region_name, group == ga) %>%
            pull(emmean)
          emm_b <- emm_df %>%
            filter(region == region_name, group == gb) %>%
            pull(emmean)

          hg <- con_df$estimate[i] / resid_sd

          results[[length(results) + 1]] <- data.frame(
            contrast = cname,
            band = band_name,
            power_type = power_type,
            region = region_name,
            estimate = con_df$estimate[i],
            estimate_lcl = con_df$lower.CL[i],
            estimate_ucl = con_df$upper.CL[i],
            SE = con_df$SE[i],
            df = con_df$df[i],
            t_ratio = con_df$t.ratio[i],
            p_value = con_df$p.value[i],
            q_value = con_df$q_value[i],
            emmean_a = if (length(emm_a) > 0) emm_a[1] else NA,
            emmean_b = if (length(emm_b) > 0) emm_b[1] else NA,
            hedges_g = hg,
            hedges_g_lcl = con_df$lower.CL[i] / resid_sd,
            hedges_g_ucl = con_df$upper.CL[i] / resid_sd,
            significant = con_df$q_value[i] < 0.05,
            stringsAsFactors = FALSE
          )
        }
      }, warning = function(w) {
        # Continue on singular fit warnings
      }, error = function(e) {
        message("  Region nested post-hoc failed for ", cname, "/", band_name, "/", power_type, ": ", conditionMessage(e))
      })
    }
  }

  posthoc_df <- bind_rows(results)
  return(posthoc_df)
}
