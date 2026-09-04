#!/usr/bin/env Rscript
# roi_pac_analysis.R — Phase-Amplitude Coupling statistics, figures, and report
#
# Called by Python: Rscript R/roi_pac_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads pac_values.csv exported by Python.
# Two analysis tiers:
#   1. Global PAC: average z-scored MI across all ROIs per subject x freq_pair, Welch t-test per pair, BH FDR
#   2. Region-level PAC: map ROIs to regions via roi_categories, LMM per freq_pair, post-hoc emmeans per region

library(argparse)
library(yaml)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(patchwork)
library(forcats)

# Resolve script directory for sourcing helpers
script_dir <- if (exists("script.dir")) {
  script.dir
} else {
  tryCatch({
    args_all <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args_all, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("^--file=", "", file_arg)))
    } else {
      "R"
    }
  }, error = function(e) "R")
}

source(file.path(script_dir, "stats_utils.R"))
source(file.path(script_dir, "hypothesis.R"))

# --- Hypothesis-layer adapters --------------------------------------------
# PAC's "band" axis is freq_pair (e.g. Delta-Alpha), not the study bands. The
# hypothesis layer emits one tidy row per (band=freq_pair, spatial) cell; these
# helpers reshape its contrast/omnibus rows into the freq_pair-keyed frames the
# PAC figures and report consume.

#' Contrast-kind rows -> legacy freq_pair-keyed posthoc shape.
.pac_contrast_rows <- function(h) {
  if (is.null(h) || nrow(h) == 0) return(data.frame())
  cr <- h[h$kind == "contrast", , drop = FALSE]
  if (nrow(cr) == 0) return(data.frame())
  cr$freq_pair <- cr$band
  cr
}

#' Omnibus-kind rows (group F, partial omega^2) -> freq_pair-keyed shape.
.pac_omnibus_rows <- function(h) {
  if (is.null(h) || nrow(h) == 0) return(data.frame())
  om <- h[h$kind == "omnibus", , drop = FALSE]
  if (nrow(om) == 0) return(data.frame())
  om$freq_pair <- om$band
  om
}

#' Rebuild the legacy `contrasts` list (name/group_a/group_b) from the hypothesis
#' contrast rows, so figures and the diagnostic omnibus LMM keep iterating
#' contrasts now that config$contrasts is no longer populated.
.contrasts_from_hyp <- function(h) {
  cr <- .pac_contrast_rows(h)
  if (nrow(cr) == 0 || !all(c("group_a", "group_b") %in% names(cr))) return(list())
  uc <- unique(cr[, c("hypothesis", "group_a", "group_b")])
  uc <- uc[!is.na(uc$group_a) & !is.na(uc$group_b), , drop = FALSE]
  if (nrow(uc) == 0) return(list())
  lapply(seq_len(nrow(uc)), function(i)
    list(name = uc$hypothesis[i], group_a = uc$group_a[i], group_b = uc$group_b[i]))
}

# Define sig_stars locally if not sourced
if (!exists("sig_stars")) {
  sig_stars <- function(q) {
    ifelse(q < 0.001, "***", ifelse(q < 0.01, "**", ifelse(q < 0.05, "*", "")))
  }
}

# Conditionally load LMM packages (only needed for region-level analysis)
has_lme4 <- requireNamespace("lme4", quietly = TRUE) &&
            requireNamespace("lmerTest", quietly = TRUE) &&
            requireNamespace("emmeans", quietly = TRUE)

# --- Publication theme (matches other analysis scripts) ---
theme_pub <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92"),
      strip.text = element_text(face = "bold", size = base_size),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = base_size + 2)
    )
}

# ===========================================================================
# Global PAC analysis: Welch t-tests
# ===========================================================================

#' Compute global PAC per subject x freq_pair (mean z-score across all ROIs)
#' @param pac data.frame with columns: subject, group, roi, freq_pair, z_score, mi
#' @return data.frame with subject, group, freq_pair, mean_z_score, mean_mi
compute_global_pac <- function(pac) {
  pac %>%
    group_by(subject, group, freq_pair) %>%
    summarise(
      mean_z_score = mean(z_score, na.rm = TRUE),
      mean_mi = mean(mi, na.rm = TRUE),
      n_rois = n(),
      .groups = "drop"
    )
}

#' Run Welch t-tests for global PAC per contrast x freq_pair
#' BH FDR correction across freq_pairs within each contrast
#' @param global_df data.frame from compute_global_pac()
#' @param contrasts list of contrast definitions
#' @return data.frame with t-test results
run_global_ttests <- function(global_df, contrasts) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (fp in unique(global_df$freq_pair)) {
      fpdata <- global_df %>%
        filter(freq_pair == fp, group %in% c(ga, gb))
      if (nrow(fpdata) == 0) next

      vals_a <- fpdata %>% filter(group == ga) %>% pull(mean_z_score)
      vals_b <- fpdata %>% filter(group == gb) %>% pull(mean_z_score)

      n_a <- length(vals_a)
      n_b <- length(vals_b)

      t_stat <- NA; p_val <- NA; df_val <- NA
      mean_a <- mean(vals_a, na.rm = TRUE)
      mean_b <- mean(vals_b, na.rm = TRUE)
      sd_a <- sd(vals_a, na.rm = TRUE)
      sd_b <- sd(vals_b, na.rm = TRUE)

      tryCatch({
        tt <- t.test(vals_a, vals_b, var.equal = FALSE)
        t_stat <- tt$statistic
        p_val <- tt$p.value
        df_val <- tt$parameter
      }, error = function(e) {
        message("  t-test failed for ", cname, "/", fp, ": ", conditionMessage(e))
      })

      # Hedges' g
      pooled_sd <- sqrt(((n_a - 1) * sd_a^2 + (n_b - 1) * sd_b^2) / (n_a + n_b - 2))
      hedges_g <- if (!is.na(pooled_sd) && pooled_sd > 0) (mean_a - mean_b) / pooled_sd else NA

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        freq_pair = fp,
        group_a = ga,
        group_b = gb,
        n_a = n_a,
        n_b = n_b,
        mean_a = mean_a,
        mean_b = mean_b,
        sd_a = sd_a,
        sd_b = sd_b,
        t_stat = as.numeric(t_stat),
        df = as.numeric(df_val),
        p_value = as.numeric(p_val),
        hedges_g = hedges_g,
        stringsAsFactors = FALSE
      )
    }
  }

  result_df <- bind_rows(results)
  if (nrow(result_df) == 0) return(result_df)

  # No cross-pair correction: freq pairs are pre-specified, per-pair p-values reported directly
  result_df <- result_df %>%
    mutate(
      q_value = p_value,
      significant = q_value < 0.05
    )

  return(result_df)
}

# ===========================================================================
# Region-level PAC analysis: LMM
# ===========================================================================

#' Map ROIs to regions and average z-score within
#' @param pac data.frame with columns: subject, group, roi, freq_pair, z_score, mi
#' @param roi_categories named list of ROI name vectors
#' @return data.frame with region replacing roi
aggregate_to_regions <- function(pac, roi_categories) {
  roi_to_region <- data.frame(
    roi = unlist(roi_categories),
    region = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  pac %>%
    inner_join(roi_to_region, by = "roi") %>%
    group_by(subject, group, freq_pair, region) %>%
    summarise(
      z_score = mean(z_score, na.rm = TRUE),
      mi = mean(mi, na.rm = TRUE),
      n_rois = n(),
      .groups = "drop"
    )
}

#' Run omnibus LMM at region level per contrast x freq_pair
#' Model: z_score ~ group * region + (1|subject)
#' @param region_df data.frame from aggregate_to_regions()
#' @param contrasts list of contrast definitions
#' @return data.frame with omnibus results
run_omnibus_lmm_region <- function(region_df, contrasts) {
  if (!has_lme4) {
    message("  lme4/lmerTest not available -- skipping region-level LMM")
    return(data.frame())
  }
  library(lme4)
  library(lmerTest)

  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (fp in unique(region_df$freq_pair)) {
      fpdata <- region_df %>%
        filter(freq_pair == fp, group %in% c(ga, gb))
      if (nrow(fpdata) == 0) next

      n_a <- length(unique(fpdata$subject[fpdata$group == ga]))
      n_b <- length(unique(fpdata$subject[fpdata$group == gb]))
      n_regions <- length(unique(fpdata$region))

      fpdata$group <- factor(fpdata$group, levels = c(ga, gb))
      fpdata$region <- factor(fpdata$region)

      group_F <- NA; group_p <- NA
      region_F <- NA; region_p <- NA
      interaction_F <- NA; interaction_p <- NA
      converged <- TRUE; singular <- FALSE

      tryCatch({
        fit <- lmer(z_score ~ group * region + (1 | subject), data = fpdata)
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
        message("  LMM failed for ", cname, "/", fp, ": ", conditionMessage(e))
      })

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        freq_pair = fp,
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

  # No cross-pair correction: freq pairs are pre-specified, per-pair p-values reported directly
  omnibus_df <- omnibus_df %>%
    mutate(
      group_q = group_p,
      group_significant = group_q < 0.05,
      interaction_q = interaction_p,
      interaction_significant = interaction_q < 0.05
    )

  return(omnibus_df)
}

#' Run emmeans post-hoc contrasts per region, gated on significant omnibus
#' @param region_df data.frame from aggregate_to_regions()
#' @param contrasts list of contrast definitions
#' @param omnibus_df data.frame from run_omnibus_lmm_region()
#' @param gate logical: if TRUE, only run for significant omnibus results
#' @return data.frame with post-hoc results
run_posthoc_emmeans_region <- function(region_df, contrasts, omnibus_df, gate = TRUE) {
  if (!has_lme4) return(data.frame())
  library(lme4)
  library(lmerTest)
  library(emmeans)

  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (fp in unique(region_df$freq_pair)) {
      if (gate && nrow(omnibus_df) > 0) {
        omni_row <- omnibus_df %>%
          filter(contrast == cname, freq_pair == fp)
        if (nrow(omni_row) == 0) next
        if (!isTRUE(omni_row$group_significant[1]) &&
            !isTRUE(omni_row$interaction_significant[1])) next
      }

      fpdata <- region_df %>%
        filter(freq_pair == fp, group %in% c(ga, gb))
      if (nrow(fpdata) == 0) next

      fpdata$group <- factor(fpdata$group, levels = c(ga, gb))
      fpdata$region <- factor(fpdata$region)

      tryCatch({
        fit <- lmer(z_score ~ group * region + (1 | subject), data = fpdata)

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
            freq_pair = fp,
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
        message("  Post-hoc failed for ", cname, "/", fp, ": ", conditionMessage(e))
      })
    }
  }

  posthoc_df <- bind_rows(results)
  return(posthoc_df)
}

# ===========================================================================
# Figures
# ===========================================================================

#' Bar chart of global z-scored MI by freq_pair x group
#' @param global_df from compute_global_pac()
#' @param group_colors, group_labels, group_order from config
#' @param output_dir figures/ directory
plot_global_pac_bar <- function(global_df, group_colors, group_labels,
                                group_order, output_dir, sig_df = NULL) {
  plot_data <- global_df %>%
    filter(group %in% group_order) %>%
    mutate(
      group_label = group_labels[group],
      group_label = factor(group_label, levels = group_labels[group_order])
    )

  summary_data <- plot_data %>%
    group_by(group_label, freq_pair) %>%
    summarise(
      mean_val = mean(mean_z_score, na.rm = TRUE),
      sem_val = sd(mean_z_score, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  p <- ggplot(plot_data, aes(x = freq_pair, y = mean_z_score, fill = group_label)) +
    geom_boxplot(width = 0.6, alpha = 0.7, position = position_dodge(0.8),
                 outlier.shape = NA) +
    geom_jitter(aes(color = group_label),
                position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.1),
                size = 1.5, alpha = 0.5, show.legend = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    labs(x = "Frequency Pair (phase-amplitude)", y = "Global z-scored MI",
         title = "Phase-Amplitude Coupling by Frequency Pair and Group") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  # Add asterisk annotations for significant freq_pair comparisons
  if (!is.null(sig_df) && nrow(sig_df) > 0) {
    sig_hits <- sig_df %>% filter(significant == TRUE)

    if (nrow(sig_hits) > 0) {
      # Compute y_max per freq_pair for positioning
      y_maxes <- plot_data %>%
        group_by(freq_pair) %>%
        summarise(y_max = max(mean_z_score, na.rm = TRUE), .groups = "drop")

      annot_data <- sig_hits %>%
        inner_join(y_maxes, by = "freq_pair") %>%
        mutate(
          y_pos = y_max * 1.05,
          sig_label_star = sig_stars(q_value)
        ) %>%
        filter(sig_label_star != "")

      if (nrow(annot_data) > 0) {
        p <- p + geom_text(
          data = annot_data,
          aes(x = freq_pair, y = y_pos, label = sig_label_star),
          inherit.aes = FALSE,
          fontface = "bold", size = 5, color = "black"
        )
      }
    }
  }

  n_pairs <- length(unique(summary_data$freq_pair))
  ggsave(file.path(output_dir, "roi_pac_global_bar.png"), p,
         width = max(8, 1.5 * n_pairs), height = 5, dpi = 300)
  message("  Saved: roi_pac_global_bar.png")
}

#' Comodulogram heatmap: phase band vs amp band, mean z-score
#' One per group + difference panel per contrast
#' @param pac data.frame with subject, group, phase_band, amp_band, z_score
#' @param contrasts, group_colors, group_labels, group_order from config
#' @param output_dir figures/ directory
plot_comodulogram <- function(pac, contrasts, group_colors, group_labels,
                              group_order, output_dir) {
  # Compute group-mean z-score per phase_band x amp_band
  group_means <- pac %>%
    filter(group %in% group_order) %>%
    group_by(group, phase_band, amp_band) %>%
    summarise(mean_z = mean(z_score, na.rm = TRUE), .groups = "drop") %>%
    mutate(group_label = group_labels[group])

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    cdata <- group_means %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    # Compute difference (group_a - group_b)
    wide <- cdata %>%
      select(group, phase_band, amp_band, mean_z) %>%
      pivot_wider(names_from = group, values_from = mean_z)

    if (ga %in% names(wide) && gb %in% names(wide)) {
      diff_data <- wide %>%
        mutate(
          mean_z = .data[[ga]] - .data[[gb]],
          group_label = paste0(group_labels[ga], " - ", group_labels[gb])
        ) %>%
        select(phase_band, amp_band, mean_z, group_label)
    } else {
      diff_data <- data.frame()
    }

    # Combine group panels + difference panel
    plot_data <- cdata %>%
      select(phase_band, amp_band, mean_z, group_label)

    if (nrow(diff_data) > 0) {
      plot_data <- bind_rows(plot_data, diff_data)
    }

    # Order panels: group_a, group_b, difference
    panel_levels <- c(group_labels[ga], group_labels[gb])
    if (nrow(diff_data) > 0) {
      panel_levels <- c(panel_levels, paste0(group_labels[ga], " - ", group_labels[gb]))
    }
    plot_data$group_label <- factor(plot_data$group_label, levels = panel_levels)

    p <- ggplot(plot_data, aes(x = phase_band, y = amp_band, fill = mean_z)) +
      geom_tile(color = "white", linewidth = 0.5) +
      scale_fill_gradient2(
        low = "#2166AC", mid = "white", high = "#B2182B",
        midpoint = 0, name = "Mean z-score"
      ) +
      geom_text(aes(label = sprintf("%.1f", mean_z)), size = 3) +
      facet_wrap(~ group_label, nrow = 1) +
      labs(x = "Phase Band", y = "Amplitude Band",
           title = paste0("PAC Comodulogram: ", cname)) +
      theme_pub() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        aspect.ratio = 1
      )

    fname <- paste0("roi_pac_comodulogram_", cname, ".png")
    n_panels <- length(panel_levels)
    ggsave(file.path(output_dir, fname), p,
           width = 4 * n_panels + 1, height = 5, dpi = 300)
    message("  Saved: ", fname)
  }
}

#' Significance heatmap (region x freq_pair) for PAC analysis
#'
#' Heatmap with regions on the y-axis and frequency pairs on the x-axis.
#' Fill = Hedges' g, asterisks on significant cells.
#' Uses PRGn diverging palette (PAC-specific).
#'
#' @param posthoc_df data.frame from run_posthoc_emmeans_region()
#' @param output_dir figures/ directory
plot_pac_significance_heatmap <- function(posthoc_df, output_dir) {
  if (nrow(posthoc_df) == 0) {
    message("  Skipping PAC significance heatmap: no post-hoc results")
    return(invisible(NULL))
  }

  for (cname in unique(posthoc_df$hypothesis)) {
    pdata <- posthoc_df %>%
      filter(hypothesis == cname) %>%
      mutate(
        sig_label = ifelse(significant, "*", ""),
        region = fct_reorder(region, effect_size, .fun = function(x) mean(abs(x), na.rm = TRUE))
      )

    if (nrow(pdata) == 0) next

    # Symmetric color scale centered at 0
    max_abs_g <- max(abs(pdata$effect_size), na.rm = TRUE)
    clim <- ceiling(max_abs_g * 10) / 10

    n_regions <- length(unique(pdata$region))
    n_pairs <- length(unique(pdata$freq_pair))

    p <- ggplot(pdata, aes(x = freq_pair, y = region, fill = effect_size)) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(aes(label = sig_label), size = 5, color = "black", fontface = "bold") +
      geom_text(aes(label = sprintf("%.2f", effect_size)), size = 3, vjust = -0.5) +
      scale_fill_gradient2(
        low = "#1B7837", mid = "white", high = "#762A83",
        midpoint = 0, limits = c(-clim, clim),
        name = "Hedges' g"
      ) +
      labs(x = "Frequency Pair", y = NULL,
           title = paste0("Region x Freq Pair Significance: ", cname),
           subtitle = "* = significant after Holm correction") +
      theme_pub() +
      theme(
        axis.text.y = element_text(size = 9),
        axis.text.x = element_text(angle = 45, hjust = 1)
      )

    fname <- paste0("roi_pac_significance_heatmap_", cname, ".png")
    ggsave(file.path(output_dir, fname), p,
           width = max(8, n_pairs * 1.2 + 2), height = max(5, n_regions * 0.5 + 2),
           dpi = 300, limitsize = FALSE)
    message("  Saved: ", fname)
  }
}

# ===========================================================================
# Report
# ===========================================================================

#' Write ANALYSIS_SUMMARY.md for PAC
write_pac_summary <- function(global_df, global_ttest_df,
                              omnibus_region_df, posthoc_region_df,
                              config, n_subjects, sfreq,
                              fig_dir, output_path) {
  lines <- character()
  add <- function(...) lines <<- c(lines, paste0(...))

  has_region <- nrow(omnibus_region_df) > 0

  # Header
  add("# Phase-Amplitude Coupling Analysis \u2014 ", config$name)
  add("")
  add("**Generated:** ", format(Sys.time(), "%Y-%m-%d %H:%M"))
  add("")

  # Methods
  add("## Methods")
  add("")
  group_str <- paste(
    sapply(names(n_subjects), function(g) paste0(config$groups[[g]], " (n=", n_subjects[g], ")")),
    collapse = ", "
  )
  band_str <- paste(
    sapply(names(config$bands), function(b) {
      lims <- config$bands[[b]]
      paste0(b, ": ", lims[1], "-", lims[2], " Hz")
    }),
    collapse = ", "
  )

  add("**Analysis:** Phase-Amplitude Coupling (Modulation Index, Tort et al., 2010)")
  add("")
  add("**Groups:** ", group_str)
  add("")
  add("**Sampling Rate:** ", sfreq, " Hz")
  add("")
  add("**Frequency Bands:** ", band_str)
  add("")
  add("**Method:** Bandpass filter (Butterworth, zero-phase) -> Hilbert transform -> ",
      "phase binning (18 bins, 20\u00b0 each) -> KL divergence from uniform / log(N)")
  add("")
  add("**Surrogate z-scoring:** 200 surrogates via circular time-shifts of amplitude envelope (\u22651 sec shift). ",
      "z-score = (observed MI - mean(surrogates)) / std(surrogates)")
  add("")
  add("**DV:** z-scored Modulation Index (normalizes for spectral differences across subjects)")
  add("")
  add("**Timeseries:** Signed (phase-preserving) ROI source timeseries")
  add("")

  stats_lines <- paste0(
    "**Statistics:** Two analysis tiers. ",
    "(1) **Global:** Mean z-scored MI across all ROIs per subject x freq_pair. ",
    "Welch t-test per freq_pair, BH FDR correction across pairs within each contrast. "
  )
  if (has_region) {
    n_reg <- omnibus_region_df$n_regions[1]
    stats_lines <- paste0(stats_lines,
      "(2) **Region level:** ROIs mapped to ", n_reg, " regions via roi_categories, ",
      "averaged within. LMM: z_score ~ group * region + (1|subject). ",
      "Type III ANOVA with Satterthwaite df. ",
      "FDR (BH) across freq_pairs. ",
      "Post-hoc: emmeans pairwise group contrasts per region, Holm correction. ",
      "Hedges' g = emmean difference / residual SD."
    )
  }
  add(stats_lines)
  add("")

  # --- Global PAC contrasts (hypothesis layer, marginal over ROIs) ---
  add("## Global PAC Contrasts")
  add("")
  add("*Marginal group contrast over ROIs (group×roi LMM, emmeans ~ group). ",
      "Estimate = group difference in z-scored MI; Hedges' g = estimate / residual SD.*")
  add("")
  if (nrow(global_ttest_df) > 0) {
    add("| Contrast | Freq Pair | Estimate | t | df | p | q | g | Sig |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (i in seq_len(nrow(global_ttest_df))) {
      row <- global_ttest_df[i, ]
      sig_str <- if (isTRUE(row$significant)) "**Yes**" else "No"
      add(sprintf("| %s | %s | %.3f | %.2f | %.1f | %.4f | %.4f | %.2f | %s |",
                  row$hypothesis, row$freq_pair,
                  ifelse(is.na(row$estimate), 0, row$estimate),
                  ifelse(is.na(row$stat), 0, row$stat),
                  ifelse(is.na(row$df), 0, row$df),
                  ifelse(is.na(row$p_value), 1, row$p_value),
                  ifelse(is.na(row$q_value), 1, row$q_value),
                  ifelse(is.na(row$effect_size), 0, row$effect_size),
                  sig_str))
    }
    add("")
  } else {
    add("*No global contrast results computed.*")
    add("")
  }

  # --- Region LMM results ---
  if (has_region) {
    add("## Region-Level LMM Results")
    add("")
    add("| Contrast | Freq Pair | n_a | n_b | n_regions | group_F | group_q | Sig | interaction_F | interaction_q | Int Sig |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (i in seq_len(nrow(omnibus_region_df))) {
      row <- omnibus_region_df[i, ]
      grp_sig <- if (isTRUE(row$group_significant)) "**Yes**" else "No"
      int_sig <- if (isTRUE(row$interaction_significant)) "**Yes**" else "No"
      add(sprintf("| %s | %s | %d | %d | %d | %.2f | %.4f | %s | %.2f | %.4f | %s |",
                  row$contrast, row$freq_pair, row$n_a, row$n_b,
                  row$n_regions,
                  ifelse(is.na(row$group_F), 0, row$group_F),
                  ifelse(is.na(row$group_q), 1, row$group_q), grp_sig,
                  ifelse(is.na(row$interaction_F), 0, row$interaction_F),
                  ifelse(is.na(row$interaction_q), 1, row$interaction_q), int_sig))
    }
    add("")

    # Post-hoc
    add("### Region Post-Hoc Contrasts")
    add("")
    if (nrow(posthoc_region_df) > 0) {
      sig_ph <- posthoc_region_df %>% filter(significant == TRUE)
      if (nrow(sig_ph) > 0) {
        add("Significant region group differences (Holm-corrected q < 0.05):")
        add("")
        for (fp in unique(sig_ph$freq_pair)) {
          fp_sig <- sig_ph %>% filter(freq_pair == fp)
          add("#### ", fp)
          add("")
          add("| Region | Estimate | SE | t | q | Hedges' g |")
          add("| --- | --- | --- | --- | --- | --- |")
          for (i in seq_len(nrow(fp_sig))) {
            row <- fp_sig[i, ]
            add(sprintf("| %s | %.3f | %.3f | %.2f | %.4f | %.2f |",
                        row$region, row$estimate, row$SE, row$stat,
                        row$q_value, row$effect_size))
          }
          add("")
        }
      } else {
        add("No individual regions reached significance after Holm correction.")
        add("")
      }
      add("**Total regions tested:** ", length(unique(posthoc_region_df$region)),
          " across ", length(unique(posthoc_region_df$freq_pair)), " freq pair(s)")
      add("")
    } else {
      add("*Region post-hoc not performed (no significant omnibus effects).*")
      add("")
    }
  }

  # Key findings
  add("## Key Findings")
  add("")
  any_sig <- FALSE

  if (nrow(global_ttest_df) > 0) {
    sig_global <- global_ttest_df %>% filter(significant == TRUE)
    if (nrow(sig_global) > 0) {
      any_sig <- TRUE
      for (i in seq_len(nrow(sig_global))) {
        row <- sig_global[i, ]
        add(sprintf("- **%s** [%s, global]: t=%.2f, q=%.4f, g=%.2f (estimate=%.3f)",
                    row$freq_pair, row$hypothesis,
                    row$stat, row$q_value, row$effect_size, row$estimate))
      }
    }
  }

  if (has_region) {
    for (i in seq_len(nrow(omnibus_region_df))) {
      row <- omnibus_region_df[i, ]
      findings <- character()
      if (isTRUE(row$group_significant))
        findings <- c(findings, sprintf("group main effect (F=%.2f, q=%.4f)", row$group_F, row$group_q))
      if (isTRUE(row$interaction_significant))
        findings <- c(findings, sprintf("group x region interaction (F=%.2f, q=%.4f)", row$interaction_F, row$interaction_q))
      if (length(findings) > 0) {
        any_sig <- TRUE
        add(sprintf("- **%s** [%s, region level]: %s", row$freq_pair,
                    row$contrast, paste(findings, collapse = "; ")))
      }
    }
  }

  if (!any_sig) {
    add("- No frequency pairs reached significance after FDR correction at either analysis level.")
  }
  add("")

  # Figure references
  fig_files <- sort(list.files(fig_dir, pattern = "\\.png$"))
  if (length(fig_files) > 0) {
    add("## Figures")
    add("")
    for (ff in fig_files) {
      caption <- gsub("_", " ", tools::file_path_sans_ext(ff))
      caption <- tools::toTitleCase(caption)
      add(sprintf("![%s](figures/%s)", caption, ff))
      add("")
    }
  }

  writeLines(lines, output_path)
  message("  Report written: ", output_path)
}

# ===========================================================================
# Main
# ===========================================================================

parser <- ArgumentParser(description = "PAC statistical analysis (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing pac_values.csv")
parser$add_argument("--config", required = TRUE,
                    help = "Path to study YAML config")
parser$add_argument("--output-dir", required = TRUE,
                    help = "Root output directory for this analysis")
parser$add_argument("--fig-dir", default = NULL,
                    help = "Directory for figures (default: output-dir/figures)")
parser$add_argument("--tbl-dir", default = NULL,
                    help = "Directory for tables (default: output-dir/tables)")
parser$add_argument("--figures-only", action = "store_true", default = FALSE,
                    help = "Skip statistics; regenerate figures from existing data/tables")
parser$add_argument("--no-figures", action = "store_true", default = FALSE,
                    help = "Skip all figure generation (stats/tables only)")
parser$add_argument("--roi-categories", default = NULL,
                    help = "Path to roi_categories.yaml (atlas ROI groupings)")
parser$add_argument("--hypothesis", default = NULL,
                    help = "Comma-separated declared hypothesis name(s) to run (default: all)")
args <- parser$parse_args()

data_dir <- args$data_dir
config_path <- args$config
output_dir <- args$output_dir
figures_only <- args$figures_only
no_figures <- args$no_figures

if (no_figures) {
  ggsave <- function(...) invisible(NULL)
}

# Create output subdirs
fig_dir <- if (!is.null(args$fig_dir)) args$fig_dir else file.path(output_dir, "figures")
tbl_dir <- if (!is.null(args$tbl_dir)) args$tbl_dir else file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load data ---
message("Loading data...")
pac <- read_csv(file.path(data_dir, "pac_values.csv"), show_col_types = FALSE)
message("  pac_values.csv: ", nrow(pac), " rows")

# --- Load config ---
config <- read_yaml(config_path)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))
message("Freq pairs: ", paste(unique(pac$freq_pair), collapse = ", "))

# Load roi_categories from atlas file if provided
if (!is.null(args$roi_categories) && file.exists(args$roi_categories)) {
  config$roi_categories <- read_yaml(args$roi_categories)
  message("Loaded roi_categories from: ", args$roi_categories,
          " (", length(config$roi_categories), " regions)")
}

# ===========================================================================
# 1. Global PAC analysis (summary always needed for figures)
# ===========================================================================
global_df <- compute_global_pac(pac)
message("  Global PAC computed: ", nrow(global_df), " subject x freq_pair rows")

if (!figures_only) {
  # ===========================================================================
  # Declarative hypothesis layer (sole per-contrast inference path).
  #
  # PAC's band axis is freq_pair, so we pass the explicit freq_pair levels as the
  # `bands` override. The hypothesis layer is the sole inference engine; the
  # legacy run_omnibus_lmm_region is kept ONLY as a diagnostic (it provides the
  # group x region interaction-F the hypothesis layer does not express).
  # ===========================================================================
  freq_pairs <- unique(as.character(pac$freq_pair))

  # --- Global level: marginal group contrast over ROIs (group*roi fit) ---
  message("\n=== Global PAC (hypothesis layer, marginal over ROIs) ===")
  hyp_global <- write_module_hypotheses(
    pac, config, tbl_dir, prefix = "roi_pac_global",
    dv_cols = "z_score", spatial_col = "roi", band_col = "freq_pair",
    bands = freq_pairs, hypothesis = args$hypothesis, marginal = TRUE)

  global_ttest_df <- .pac_contrast_rows(hyp_global)
  if (nrow(global_ttest_df) > 0) {
    for (i in seq_len(nrow(global_ttest_df))) {
      row <- global_ttest_df[i, ]
      sig_str <- if (isTRUE(row$significant)) " ***" else ""
      message(sprintf("  %s | %s: t=%.2f, q=%.4f%s",
                      row$hypothesis, row$freq_pair,
                      ifelse(is.na(row$stat), 0, row$stat),
                      ifelse(is.na(row$q_value), 1, row$q_value), sig_str))
    }
  }

  # Rebuild the legacy `contrasts` list from the hypothesis rows so the
  # comodulogram figure and the diagnostic omnibus can keep iterating contrasts.
  pac_contrasts <- .contrasts_from_hyp(hyp_global)

  # ===========================================================================
  # Region-level analysis (if roi_categories defined)
  # ===========================================================================
  omnibus_region_df <- data.frame()
  posthoc_region_df <- data.frame()

  if (length(config$roi_categories) > 0 && has_lme4) {
    message("\n=== Region-Level PAC (hypothesis layer) ===")

    region_df <- aggregate_to_regions(pac, config$roi_categories)
    n_regions <- length(unique(region_df$region))
    message("  Aggregated to ", n_regions, " regions")

    hyp_region <- write_module_hypotheses(
      region_df, config, tbl_dir, prefix = "roi_pac_region",
      dv_cols = "z_score", spatial_col = "region", band_col = "freq_pair",
      bands = freq_pairs, hypothesis = args$hypothesis)

    posthoc_region_df <- .pac_contrast_rows(hyp_region)
    if (nrow(posthoc_region_df) > 0) posthoc_region_df$region <- posthoc_region_df$spatial

    # Diagnostic omnibus LMM (group x region interaction-F) — NOT a hypothesis.
    omnibus_region_df <- run_omnibus_lmm_region(region_df, pac_contrasts)
    if (nrow(omnibus_region_df) > 0) {
      message("\n  === Region Omnibus (diagnostic) ===")
      for (i in seq_len(nrow(omnibus_region_df))) {
        row <- omnibus_region_df[i, ]
        grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
        int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
        message(sprintf("  %s | %s: group F=%.2f q=%.4f%s | interaction F=%.2f q=%.4f%s",
                        row$contrast, row$freq_pair,
                        row$group_F, row$group_q, grp_sig,
                        row$interaction_F, row$interaction_q, int_sig))
      }
    }

    if (nrow(posthoc_region_df) > 0) {
      sig_count <- sum(posthoc_region_df$significant, na.rm = TRUE)
      message("  ", nrow(posthoc_region_df), " region contrasts, ", sig_count, " significant")
    }
  } else if (length(config$roi_categories) == 0) {
    message("\n  No roi_categories in config -- skipping region-level analysis")
  } else {
    message("\n  lme4/lmerTest not available -- skipping region-level LMM analysis")
  }

  # ===========================================================================
  # Export legacy-named tables (rebuilt from the hypothesis rows; the native
  # roi_pac_global_hypotheses.csv / roi_pac_region_hypotheses.csv are written by
  # write_module_hypotheses). Figures/report consume these unchanged.
  # ===========================================================================
  message("\nExporting tables...")

  if (nrow(global_ttest_df) > 0) {
    write_csv(global_ttest_df, file.path(tbl_dir, "roi_pac_global.csv"))
    message("  Saved: tables/roi_pac_global.csv (", nrow(global_ttest_df),
            " rows, hypothesis-derived marginal)")
  }
  if (nrow(omnibus_region_df) > 0) {
    write_csv(omnibus_region_df, file.path(tbl_dir, "roi_pac_omnibus_region.csv"))
    message("  Saved: tables/roi_pac_omnibus_region.csv (diagnostic)")
  }
  if (nrow(posthoc_region_df) > 0) {
    write_csv(posthoc_region_df, file.path(tbl_dir, "roi_pac_posthoc_region.csv"))
    message("  Saved: tables/roi_pac_posthoc_region.csv (", nrow(posthoc_region_df),
            " rows, hypothesis-derived)")
  }

} else {
  # --figures-only: load existing tables from disk
  message("\n=== Figures-only mode: loading existing tables from ", tbl_dir, " ===")

  global_ttest_path <- file.path(tbl_dir, "roi_pac_global.csv")
  global_ttest_df <- if (file.exists(global_ttest_path)) {
    message("  Loading: roi_pac_global.csv")
    read_csv(global_ttest_path, show_col_types = FALSE)
  } else {
    message("  Not found: roi_pac_global.csv -- significance annotations will be omitted")
    data.frame()
  }

  omnibus_path <- file.path(tbl_dir, "roi_pac_omnibus_region.csv")
  omnibus_region_df <- if (file.exists(omnibus_path)) {
    message("  Loading: roi_pac_omnibus_region.csv")
    read_csv(omnibus_path, show_col_types = FALSE)
  } else {
    message("  Not found: roi_pac_omnibus_region.csv")
    data.frame()
  }

  posthoc_path <- file.path(tbl_dir, "roi_pac_posthoc_region.csv")
  posthoc_region_df <- if (file.exists(posthoc_path)) {
    message("  Loading: roi_pac_posthoc_region.csv")
    read_csv(posthoc_path, show_col_types = FALSE)
  } else {
    message("  Not found: roi_pac_posthoc_region.csv")
    data.frame()
  }

  # Rebuild the contrasts list from the loaded global table (carries the
  # contrast/group_a/group_b columns) for the comodulogram figure.
  pac_contrasts <- .contrasts_from_hyp(global_ttest_df)
}

# ===========================================================================
# Figures
# ===========================================================================
message("\nGenerating figures...")

# Global PAC bar chart (with significance annotations from t-test results)
plot_global_pac_bar(global_df, group_colors, group_labels, group_order, fig_dir,
                     sig_df = if (nrow(global_ttest_df) > 0) global_ttest_df else NULL)

# Comodulogram heatmaps (contrasts derived from the hypothesis rows, since
# config$contrasts is no longer populated post design-spec migration)
plot_comodulogram(pac, pac_contrasts, group_colors, group_labels, group_order, fig_dir)

# Region significance heatmaps (if post-hoc was performed)
if (nrow(posthoc_region_df) > 0) {
  plot_pac_significance_heatmap(posthoc_region_df, fig_dir)
}

# ===========================================================================
# Summary report (skip in figures-only mode)
# ===========================================================================
if (!figures_only) {
  message("\nWriting summary...")

  n_subjects <- pac %>%
    dplyr::distinct(subject, group) %>%
    dplyr::count(group) %>%
    { setNames(.$n, .$group) }

  sfreq <- if (!is.null(config$sfreq)) config$sfreq else 500

  write_pac_summary(
    global_df, global_ttest_df,
    omnibus_region_df, posthoc_region_df,
    config, n_subjects, sfreq,
    fig_dir, file.path(output_dir, "ANALYSIS_SUMMARY.md")
  )
} else {
  message("\nSkipping summary report (--figures-only)")
}

message("\nDone. Output: ", output_dir)
