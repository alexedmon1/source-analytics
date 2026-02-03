# stats_utils.R — Statistical analysis for source-analytics
#
# ROI-level: run_omnibus_lmm(), run_posthoc_emmeans()
# Region-level: aggregate_to_regions(), run_omnibus_lmm_region(), run_posthoc_emmeans_region()
#   - Omnibus: lmer(dv ~ group * spatial + (1|subject)), Type III ANOVA
#   - Post-hoc: emmeans(fit, pairwise ~ group | spatial), Holm correction
#   - FDR (BH) correction across bands per contrast
#   - power_type: "relative", "absolute", or "dB"

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
#' @param power_type character — column to use as DV: "relative", "absolute", or "dB"
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

      bdata$group <- factor(bdata$group, levels = c(gb, ga))
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

  # FDR correction across bands within each contrast x power_type
  omnibus_df <- omnibus_df %>%
    group_by(contrast, power_type) %>%
    mutate(
      group_q = p.adjust(group_p, method = "BH"),
      group_significant = group_q < 0.05,
      interaction_q = p.adjust(interaction_p, method = "BH"),
      interaction_significant = interaction_q < 0.05
    ) %>%
    ungroup()

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

      bdata$group <- factor(bdata$group, levels = c(gb, ga))
      bdata$roi <- factor(bdata$roi)
      bdata$dv <- bdata[[power_type]]

      tryCatch({
        fit <- lmer(dv ~ group * roi + (1 | subject), data = bdata)

        emm <- emmeans(fit, pairwise ~ group | roi)
        con_df <- as.data.frame(emm$contrasts)
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
            SE = con_df$SE[i],
            df = con_df$df[i],
            t_ratio = con_df$t.ratio[i],
            p_value = con_df$p.value[i],
            q_value = con_df$q_value[i],
            emmean_a = if (length(emm_a) > 0) emm_a[1] else NA,
            emmean_b = if (length(emm_b) > 0) emm_b[1] else NA,
            hedges_g = hg,
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
      dB = mean(dB, na.rm = TRUE),
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

      bdata$group <- factor(bdata$group, levels = c(gb, ga))
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
    group_by(contrast, power_type) %>%
    mutate(
      group_q = p.adjust(group_p, method = "BH"),
      group_significant = group_q < 0.05,
      interaction_q = p.adjust(interaction_p, method = "BH"),
      interaction_significant = interaction_q < 0.05
    ) %>%
    ungroup()

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

      bdata$group <- factor(bdata$group, levels = c(gb, ga))
      bdata$region <- factor(bdata$region)
      bdata$dv <- bdata[[power_type]]

      tryCatch({
        fit <- lmer(dv ~ group * region + (1 | subject), data = bdata)

        emm <- emmeans(fit, pairwise ~ group | region)
        con_df <- as.data.frame(emm$contrasts)
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
            SE = con_df$SE[i],
            df = con_df$df[i],
            t_ratio = con_df$t.ratio[i],
            p_value = con_df$p.value[i],
            q_value = con_df$q_value[i],
            emmean_a = if (length(emm_a) > 0) emm_a[1] else NA,
            emmean_b = if (length(emm_b) > 0) emm_b[1] else NA,
            hedges_g = hg,
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
