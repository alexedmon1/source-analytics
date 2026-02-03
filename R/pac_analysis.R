#!/usr/bin/env Rscript
# pac_analysis.R — Phase-Amplitude Coupling statistics, figures, and report
#
# Called by Python: Rscript R/pac_analysis.R --data-dir ... --config ... --output-dir ...
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

# Conditionally load LMM packages (only needed for region-level analysis)
has_lme4 <- requireNamespace("lme4", quietly = TRUE) &&
            requireNamespace("lmerTest", quietly = TRUE) &&
            requireNamespace("emmeans", quietly = TRUE)

# --- Publication theme (matches other analysis scripts) ---
theme_pub <- function(base_size = 11) {
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

  # BH FDR across freq_pairs within each contrast
  result_df <- result_df %>%
    group_by(contrast) %>%
    mutate(
      q_value = p.adjust(p_value, method = "BH"),
      significant = q_value < 0.05
    ) %>%
    ungroup()

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

      fpdata$group <- factor(fpdata$group, levels = c(gb, ga))
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

  # BH FDR across freq_pairs within each contrast
  omnibus_df <- omnibus_df %>%
    group_by(contrast) %>%
    mutate(
      group_q = p.adjust(group_p, method = "BH"),
      group_significant = group_q < 0.05,
      interaction_q = p.adjust(interaction_p, method = "BH"),
      interaction_significant = interaction_q < 0.05
    ) %>%
    ungroup()

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

      fpdata$group <- factor(fpdata$group, levels = c(gb, ga))
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
                                group_order, output_dir) {
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

  p <- ggplot(summary_data, aes(x = freq_pair, y = mean_val, fill = group_label)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
    geom_errorbar(
      aes(ymin = mean_val - sem_val, ymax = mean_val + sem_val),
      position = position_dodge(width = 0.8), width = 0.3
    ) +
    geom_point(
      data = plot_data,
      aes(x = freq_pair, y = mean_z_score, fill = group_label),
      position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.1),
      size = 1, alpha = 0.4, shape = 21, color = "grey30", show.legend = FALSE
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    scale_fill_manual(values = color_vals, name = NULL) +
    labs(x = "Frequency Pair (phase-amplitude)", y = "Global z-scored MI",
         title = "Phase-Amplitude Coupling by Frequency Pair and Group") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  n_pairs <- length(unique(summary_data$freq_pair))
  ggsave(file.path(output_dir, "pac_global_bar.png"), p,
         width = max(8, 1.5 * n_pairs), height = 5, dpi = 300)
  message("  Saved: pac_global_bar.png")
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

    fname <- paste0("pac_comodulogram_", cname, ".png")
    n_panels <- length(panel_levels)
    ggsave(file.path(output_dir, fname), p,
           width = 4 * n_panels + 1, height = 5, dpi = 300)
    message("  Saved: ", fname)
  }
}

#' Forest plot for region-level post-hoc results
#' @param posthoc_df data.frame from run_posthoc_emmeans_region()
#' @param output_dir figures/ directory
plot_region_forest <- function(posthoc_df, output_dir) {
  if (nrow(posthoc_df) == 0) {
    message("  Skipping region forest plot: no post-hoc results")
    return(invisible(NULL))
  }

  for (cname in unique(posthoc_df$contrast)) {
    pdata <- posthoc_df %>%
      filter(contrast == cname) %>%
      mutate(
        region = fct_reorder(region, estimate),
        sig_label = ifelse(significant, "*", "")
      )

    if (nrow(pdata) == 0) next

    n_pairs <- length(unique(pdata$freq_pair))

    p <- ggplot(pdata, aes(x = estimate, y = region)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
      geom_errorbar(aes(xmin = estimate - 1.96 * SE, xmax = estimate + 1.96 * SE),
                    width = 0.3, color = "grey40", orientation = "y") +
      geom_point(aes(color = significant), size = 2.5) +
      scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "#E74C3C"),
                         labels = c("n.s.", "p < .05"), name = NULL) +
      facet_wrap(~ freq_pair, scales = "free_x") +
      labs(x = "Group Difference (emmean z-score)", y = NULL,
           title = paste0("PAC Region Contrasts: ", cname)) +
      theme_pub() +
      theme(
        axis.text.y = element_text(size = 8),
        strip.text = element_text(size = 10)
      )

    n_regions <- length(unique(pdata$region))
    fname <- paste0("pac_region_forest_", cname, ".png")
    ggsave(file.path(output_dir, fname), p,
           width = max(10, 4 * n_pairs), height = max(6, n_regions * 0.4 + 2),
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

  # --- Global t-test results ---
  add("## Global PAC T-Tests")
  add("")
  if (nrow(global_ttest_df) > 0) {
    add("| Contrast | Freq Pair | n_a | n_b | mean_a | mean_b | t | df | p | q | g | Sig |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (i in seq_len(nrow(global_ttest_df))) {
      row <- global_ttest_df[i, ]
      sig_str <- if (isTRUE(row$significant)) "**Yes**" else "No"
      add(sprintf("| %s | %s | %d | %d | %.3f | %.3f | %.2f | %.1f | %.4f | %.4f | %.2f | %s |",
                  row$contrast, row$freq_pair, row$n_a, row$n_b,
                  row$mean_a, row$mean_b,
                  ifelse(is.na(row$t_stat), 0, row$t_stat),
                  ifelse(is.na(row$df), 0, row$df),
                  ifelse(is.na(row$p_value), 1, row$p_value),
                  ifelse(is.na(row$q_value), 1, row$q_value),
                  ifelse(is.na(row$hedges_g), 0, row$hedges_g),
                  sig_str))
    }
    add("")
  } else {
    add("*No global t-test results computed.*")
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
                        row$region, row$estimate, row$SE, row$t_ratio,
                        row$q_value, row$hedges_g))
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
        add(sprintf("- **%s** [%s, global]: t=%.2f, q=%.4f, g=%.2f (mean_a=%.3f, mean_b=%.3f)",
                    row$freq_pair, row$contrast,
                    row$t_stat, row$q_value, row$hedges_g, row$mean_a, row$mean_b))
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
args <- parser$parse_args()

data_dir <- args$data_dir
config_path <- args$config
output_dir <- args$output_dir

# Create output subdirs
fig_dir <- file.path(output_dir, "figures")
tbl_dir <- file.path(output_dir, "tables")
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

# ===========================================================================
# 1. Global PAC analysis
# ===========================================================================
message("\n=== Global PAC Analysis ===")

global_df <- compute_global_pac(pac)
message("  Global PAC computed: ", nrow(global_df), " subject x freq_pair rows")

global_ttest_df <- run_global_ttests(global_df, config$contrasts)
if (nrow(global_ttest_df) > 0) {
  message("\n  === Global T-Test Results ===")
  for (i in seq_len(nrow(global_ttest_df))) {
    row <- global_ttest_df[i, ]
    sig_str <- if (isTRUE(row$significant)) " ***" else ""
    message(sprintf("  %s | %s: t=%.2f, q=%.4f%s",
                    row$contrast, row$freq_pair,
                    ifelse(is.na(row$t_stat), 0, row$t_stat),
                    ifelse(is.na(row$q_value), 1, row$q_value), sig_str))
  }
}

# ===========================================================================
# 2. Region-level analysis (if roi_categories defined)
# ===========================================================================
omnibus_region_df <- data.frame()
posthoc_region_df <- data.frame()

if (length(config$roi_categories) > 0 && has_lme4) {
  message("\n=== Region-Level PAC Analysis ===")

  region_df <- aggregate_to_regions(pac, config$roi_categories)
  n_regions <- length(unique(region_df$region))
  message("  Aggregated to ", n_regions, " regions")

  omnibus_region_df <- run_omnibus_lmm_region(region_df, config$contrasts)

  if (nrow(omnibus_region_df) > 0) {
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

  posthoc_region_df <- run_posthoc_emmeans_region(region_df, config$contrasts, omnibus_region_df)

  if (nrow(posthoc_region_df) > 0) {
    sig_count <- sum(posthoc_region_df$significant, na.rm = TRUE)
    message("  ", nrow(posthoc_region_df), " region contrasts, ", sig_count, " significant")
  } else {
    message("  No post-hoc tests (no significant omnibus effects)")
  }
} else if (length(config$roi_categories) == 0) {
  message("\n  No roi_categories in config -- skipping region-level analysis")
} else {
  message("\n  lme4/lmerTest not available -- skipping region-level LMM analysis")
}

# ===========================================================================
# Export tables
# ===========================================================================
message("\nExporting tables...")

if (nrow(global_ttest_df) > 0) {
  write_csv(global_ttest_df, file.path(tbl_dir, "pac_global.csv"))
  message("  Saved: tables/pac_global.csv")
}
if (nrow(omnibus_region_df) > 0) {
  write_csv(omnibus_region_df, file.path(tbl_dir, "pac_omnibus_region.csv"))
  message("  Saved: tables/pac_omnibus_region.csv")
}
if (nrow(posthoc_region_df) > 0) {
  write_csv(posthoc_region_df, file.path(tbl_dir, "pac_posthoc_region.csv"))
  message("  Saved: tables/pac_posthoc_region.csv")
}

# ===========================================================================
# Figures
# ===========================================================================
message("\nGenerating figures...")

# Global PAC bar chart
plot_global_pac_bar(global_df, group_colors, group_labels, group_order, fig_dir)

# Comodulogram heatmaps
plot_comodulogram(pac, config$contrasts, group_colors, group_labels, group_order, fig_dir)

# Region forest plots (if post-hoc was performed)
if (nrow(posthoc_region_df) > 0) {
  plot_region_forest(posthoc_region_df, fig_dir)
}

# ===========================================================================
# Summary report
# ===========================================================================
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

message("\nDone. Output: ", output_dir)
