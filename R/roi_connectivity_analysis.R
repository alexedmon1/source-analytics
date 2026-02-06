#!/usr/bin/env Rscript
# roi_connectivity_analysis.R — ROI functional connectivity statistics, figures, and report
#
# Called by Python: Rscript R/connectivity_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads roi_connectivity_edges.csv exported by Python.
# Two analysis tiers:
#   1. Global connectivity: average all edges per subject x band, Welch t-test per band, BH FDR
#   2. Region-pair level: map edges to region pairs via roi_categories, LMM per band, post-hoc emmeans

library(argparse)
library(yaml)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(patchwork)
library(forcats)

# Conditionally load LMM packages (only needed for region-pair analysis)
has_lme4 <- requireNamespace("lme4", quietly = TRUE) &&
            requireNamespace("lmerTest", quietly = TRUE) &&
            requireNamespace("emmeans", quietly = TRUE)

# --- Publication theme (matches plot_psd.R) ---
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
# Global connectivity analysis: Welch t-tests
# ===========================================================================

#' Compute global connectivity per subject x band (mean of all edges)
#' @param edges data.frame with columns: subject, group, band, roi1, roi2, coherence, imag_coherence
#' @return data.frame with subject, group, band, mean_coherence, mean_imag_coherence
compute_global_connectivity <- function(edges) {
  edges %>%
    group_by(subject, group, band) %>%
    summarise(
      mean_coherence = mean(coherence, na.rm = TRUE),
      mean_imag_coherence = mean(imag_coherence, na.rm = TRUE),
      n_edges = n(),
      .groups = "drop"
    )
}

#' Run Welch t-tests for global connectivity per contrast x band x metric
#' BH FDR correction across bands within each contrast x metric
#' @param global_df data.frame from compute_global_connectivity()
#' @param contrasts list of contrast definitions
#' @param bands named list of band limits
#' @return data.frame with t-test results
run_global_ttests <- function(global_df, contrasts, bands) {
  metrics <- c("mean_coherence", "mean_imag_coherence")
  metric_labels <- c("coherence", "imag_coherence")
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (mi in seq_along(metrics)) {
      metric_col <- metrics[mi]
      metric_name <- metric_labels[mi]

      for (band_name in names(bands)) {
        bdata <- global_df %>%
          filter(band == band_name, group %in% c(ga, gb))
        if (nrow(bdata) == 0) next

        vals_a <- bdata %>% filter(group == ga) %>% pull(!!sym(metric_col))
        vals_b <- bdata %>% filter(group == gb) %>% pull(!!sym(metric_col))

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
          message("  t-test failed for ", cname, "/", band_name, "/", metric_name, ": ", conditionMessage(e))
        })

        # Hedges' g (pooled SD approximation)
        pooled_sd <- sqrt(((n_a - 1) * sd_a^2 + (n_b - 1) * sd_b^2) / (n_a + n_b - 2))
        hedges_g <- if (!is.na(pooled_sd) && pooled_sd > 0) (mean_a - mean_b) / pooled_sd else NA

        results[[length(results) + 1]] <- data.frame(
          contrast = cname,
          metric = metric_name,
          band = band_name,
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
  }

  result_df <- bind_rows(results)
  if (nrow(result_df) == 0) return(result_df)

  # BH FDR across bands within each contrast x metric
  result_df <- result_df %>%
    group_by(contrast, metric) %>%
    mutate(
      q_value = p.adjust(p_value, method = "BH"),
      significant = q_value < 0.05
    ) %>%
    ungroup()

  return(result_df)
}

# ===========================================================================
# Region-pair connectivity analysis: LMM
# ===========================================================================

#' Map edges to region pairs, average within
#' @param edges data.frame with columns: subject, group, band, roi1, roi2, coherence, imag_coherence
#' @param roi_categories named list of ROI name vectors
#' @return data.frame with region_pair replacing roi1/roi2
aggregate_edges_to_region_pairs <- function(edges, roi_categories) {
  # Build ROI -> region mapping
  roi_to_region <- data.frame(
    roi = unlist(roi_categories),
    region = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  # Map roi1 and roi2 to regions
  edges_mapped <- edges %>%
    inner_join(roi_to_region, by = c("roi1" = "roi")) %>%
    rename(region1 = region) %>%
    inner_join(roi_to_region, by = c("roi2" = "roi")) %>%
    rename(region2 = region)

  # Create canonical pair name (sorted alphabetically)
  edges_mapped <- edges_mapped %>%
    mutate(
      region_pair = ifelse(region1 <= region2,
                           paste(region1, region2, sep = " - "),
                           paste(region2, region1, sep = " - "))
    )

  # Average edge values within each subject x band x region_pair
  edges_mapped %>%
    group_by(subject, group, band, region_pair) %>%
    summarise(
      coherence = mean(coherence, na.rm = TRUE),
      imag_coherence = mean(imag_coherence, na.rm = TRUE),
      n_edges = n(),
      .groups = "drop"
    )
}

#' Run omnibus LMM at region-pair level per contrast x band x metric
#' Model: dv ~ group * region_pair + (1|subject)
#' @param region_pair_df data.frame from aggregate_edges_to_region_pairs()
#' @param contrasts list of contrast definitions
#' @param bands named list of band limits
#' @param metric character: "coherence" or "imag_coherence"
#' @return data.frame with omnibus results
run_omnibus_lmm_region_pair <- function(region_pair_df, contrasts, bands, metric = "coherence") {
  if (!has_lme4) {
    message("  lme4/lmerTest not available -- skipping region-pair LMM")
    return(data.frame())
  }
  library(lme4)
  library(lmerTest)

  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      bdata <- region_pair_df %>%
        filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      n_a <- length(unique(bdata$subject[bdata$group == ga]))
      n_b <- length(unique(bdata$subject[bdata$group == gb]))
      n_pairs <- length(unique(bdata$region_pair))

      bdata$group <- factor(bdata$group, levels = c(gb, ga))
      bdata$region_pair <- factor(bdata$region_pair)
      bdata$dv <- bdata[[metric]]

      group_F <- NA; group_p <- NA
      region_pair_F <- NA; region_pair_p <- NA
      interaction_F <- NA; interaction_p <- NA
      converged <- TRUE; singular <- FALSE

      tryCatch({
        fit <- lmer(dv ~ group * region_pair + (1 | subject), data = bdata)
        singular <- isSingular(fit)

        aov <- anova(fit, type = 3)

        if ("group" %in% rownames(aov)) {
          group_F <- aov["group", "F value"]
          group_p <- aov["group", "Pr(>F)"]
        }
        if ("region_pair" %in% rownames(aov)) {
          region_pair_F <- aov["region_pair", "F value"]
          region_pair_p <- aov["region_pair", "Pr(>F)"]
        }
        if ("group:region_pair" %in% rownames(aov)) {
          interaction_F <- aov["group:region_pair", "F value"]
          interaction_p <- aov["group:region_pair", "Pr(>F)"]
        }
      }, warning = function(w) {
        if (grepl("singular|converge", conditionMessage(w), ignore.case = TRUE)) {
          singular <<- TRUE
        }
      }, error = function(e) {
        converged <<- FALSE
        message("  LMM failed for ", cname, "/", band_name, "/", metric, ": ", conditionMessage(e))
      })

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        metric = metric,
        band = band_name,
        group_a = ga,
        group_b = gb,
        n_a = n_a,
        n_b = n_b,
        n_region_pairs = n_pairs,
        group_F = as.numeric(group_F),
        group_p = as.numeric(group_p),
        region_pair_F = as.numeric(region_pair_F),
        region_pair_p = as.numeric(region_pair_p),
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

  # BH FDR across bands within each contrast x metric
  omnibus_df <- omnibus_df %>%
    group_by(contrast, metric) %>%
    mutate(
      group_q = p.adjust(group_p, method = "BH"),
      group_significant = group_q < 0.05,
      interaction_q = p.adjust(interaction_p, method = "BH"),
      interaction_significant = interaction_q < 0.05
    ) %>%
    ungroup()

  return(omnibus_df)
}

#' Run emmeans post-hoc contrasts per region pair
#' @param region_pair_df data.frame from aggregate_edges_to_region_pairs()
#' @param contrasts list of contrast definitions
#' @param bands named list of band limits
#' @param omnibus_df data.frame from run_omnibus_lmm_region_pair()
#' @param metric character: "coherence" or "imag_coherence"
#' @param gate logical: if TRUE, only run for significant omnibus results
#' @return data.frame with post-hoc results
run_posthoc_emmeans_region_pair <- function(region_pair_df, contrasts, bands,
                                            omnibus_df, metric = "coherence",
                                            gate = TRUE) {
  if (!has_lme4) return(data.frame())
  library(lme4)
  library(lmerTest)
  library(emmeans)

  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      if (gate && nrow(omnibus_df) > 0) {
        omni_row <- omnibus_df %>%
          filter(contrast == cname, band == band_name, metric == !!metric)
        if (nrow(omni_row) == 0) next
        if (!isTRUE(omni_row$group_significant[1]) &&
            !isTRUE(omni_row$interaction_significant[1])) next
      }

      bdata <- region_pair_df %>%
        filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      bdata$group <- factor(bdata$group, levels = c(gb, ga))
      bdata$region_pair <- factor(bdata$region_pair)
      bdata$dv <- bdata[[metric]]

      tryCatch({
        fit <- lmer(dv ~ group * region_pair + (1 | subject), data = bdata)

        emm <- emmeans(fit, pairwise ~ group | region_pair)
        con_df <- as.data.frame(emm$contrasts)
        emm_df <- as.data.frame(emm$emmeans)

        resid_sd <- sigma(fit)
        con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

        for (i in seq_len(nrow(con_df))) {
          pair_name <- as.character(con_df$region_pair[i])

          emm_a <- emm_df %>%
            filter(region_pair == pair_name, group == ga) %>%
            pull(emmean)
          emm_b <- emm_df %>%
            filter(region_pair == pair_name, group == gb) %>%
            pull(emmean)

          hg <- con_df$estimate[i] / resid_sd

          results[[length(results) + 1]] <- data.frame(
            contrast = cname,
            metric = metric,
            band = band_name,
            region_pair = pair_name,
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
        message("  Post-hoc failed for ", cname, "/", band_name, "/", metric, ": ", conditionMessage(e))
      })
    }
  }

  posthoc_df <- bind_rows(results)
  return(posthoc_df)
}

# ===========================================================================
# Figures
# ===========================================================================

#' Plot group-mean connectivity matrices (heatmaps) per band
#' @param edges data.frame with subject, group, band, roi1, roi2, coherence, imag_coherence
#' @param metric_col column name: "coherence" or "imag_coherence"
#' @param group_colors, group_labels, group_order from config
#' @param output_dir figures/ directory
plot_connectivity_matrices <- function(edges, metric_col, group_colors,
                                       group_labels, group_order, output_dir) {
  # Compute group-mean per edge per band
  group_means <- edges %>%
    filter(group %in% group_order) %>%
    group_by(group, band, roi1, roi2) %>%
    summarise(value = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop") %>%
    mutate(group_label = group_labels[group])

  bands_present <- unique(group_means$band)

  for (bname in bands_present) {
    bdata <- group_means %>% filter(band == bname)

    # Build symmetric matrix data (add reverse direction)
    bdata_sym <- bind_rows(
      bdata %>% select(group, group_label, band, roi1, roi2, value),
      bdata %>% select(group, group_label, band, roi1 = roi2, roi2 = roi1, value)
    )

    # Determine ROI order (alphabetical)
    all_rois <- sort(unique(c(bdata_sym$roi1, bdata_sym$roi2)))

    bdata_sym <- bdata_sym %>%
      mutate(
        roi1 = factor(roi1, levels = all_rois),
        roi2 = factor(roi2, levels = rev(all_rois)),
        group_label = factor(group_label, levels = group_labels[group_order])
      )

    p <- ggplot(bdata_sym, aes(x = roi1, y = roi2, fill = value)) +
      geom_tile() +
      scale_fill_viridis_c(option = "inferno", name = tools::toTitleCase(gsub("_", " ", metric_col))) +
      facet_wrap(~ group_label, nrow = 1) +
      labs(x = NULL, y = NULL,
           title = paste0(tools::toTitleCase(gsub("_", " ", metric_col)), " - ", bname)) +
      theme_pub(base_size = 9) +
      theme(
        axis.text.x = element_text(angle = 90, hjust = 1, size = 4),
        axis.text.y = element_text(size = 4),
        aspect.ratio = 1
      )

    n_groups <- length(unique(bdata_sym$group_label))
    fname <- paste0("roi_connectivity_matrix_", metric_col, "_", bname, ".png")
    ggsave(file.path(output_dir, fname), p,
           width = 7 * n_groups, height = 7, dpi = 200, limitsize = FALSE)
    message("  Saved: ", fname)
  }
}

#' Bar chart of global connectivity by band x group
#' @param global_df from compute_global_connectivity()
#' @param group_colors, group_labels, group_order from config
#' @param output_dir figures/ directory
plot_global_connectivity_bar <- function(global_df, group_colors, group_labels,
                                         group_order, output_dir) {
  # Pivot to long format for both metrics
  plot_data <- global_df %>%
    filter(group %in% group_order) %>%
    pivot_longer(
      cols = c(mean_coherence, mean_imag_coherence),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      metric = case_when(
        metric == "mean_coherence" ~ "Coherence",
        metric == "mean_imag_coherence" ~ "Imag. Coherence"
      ),
      group_label = group_labels[group],
      group_label = factor(group_label, levels = group_labels[group_order])
    )

  # Group-level summary
  summary_data <- plot_data %>%
    group_by(group_label, band, metric) %>%
    summarise(
      mean_val = mean(value, na.rm = TRUE),
      sem_val = sd(value, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  p <- ggplot(summary_data, aes(x = band, y = mean_val, fill = group_label)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
    geom_errorbar(
      aes(ymin = mean_val - sem_val, ymax = mean_val + sem_val),
      position = position_dodge(width = 0.8), width = 0.3
    ) +
    geom_point(
      data = plot_data,
      aes(x = band, y = value, fill = group_label),
      position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.1),
      size = 1, alpha = 0.4, shape = 21, color = "grey30", show.legend = FALSE
    ) +
    scale_fill_manual(values = color_vals, name = NULL) +
    facet_wrap(~ metric, scales = "free_y") +
    labs(x = "Frequency Band", y = "Global Connectivity (mean of all edges)",
         title = "Global Functional Connectivity by Band and Group") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  n_bands <- length(unique(summary_data$band))
  ggsave(file.path(output_dir, "roi_connectivity_global_bar.png"), p,
         width = max(8, 2 * n_bands), height = 5, dpi = 300)
  message("  Saved: roi_connectivity_global_bar.png")
}

#' Forest plot for region-pair post-hoc results
#' @param posthoc_df data.frame from run_posthoc_emmeans_region_pair()
#' @param output_dir figures/ directory
plot_region_pair_forest <- function(posthoc_df, output_dir) {
  if (nrow(posthoc_df) == 0) {
    message("  Skipping region-pair forest plot: no post-hoc results")
    return(invisible(NULL))
  }

  for (metric_name in unique(posthoc_df$metric)) {
    for (cname in unique(posthoc_df$contrast)) {
      pdata <- posthoc_df %>%
        filter(contrast == cname, metric == metric_name) %>%
        mutate(
          region_pair = fct_reorder(region_pair, estimate),
          sig_label = ifelse(significant, "*", "")
        )

      if (nrow(pdata) == 0) next

      n_bands <- length(unique(pdata$band))

      p <- ggplot(pdata, aes(x = estimate, y = region_pair)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbar(aes(xmin = estimate - 1.96 * SE, xmax = estimate + 1.96 * SE),
                      width = 0.3, color = "grey40", orientation = "y") +
        geom_point(aes(color = significant), size = 2.5) +
        scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "#E74C3C"),
                           labels = c("n.s.", "p < .05"), name = NULL) +
        facet_wrap(~ band, scales = "free_x") +
        labs(x = "Group Difference (emmean)", y = NULL,
             title = paste0("Region-Pair Contrasts: ", cname,
                            " (", gsub("_", " ", metric_name), ")")) +
        theme_pub() +
        theme(
          axis.text.y = element_text(size = 8),
          strip.text = element_text(size = 10)
        )

      n_pairs <- length(unique(pdata$region_pair))
      fname <- paste0("roi_connectivity_region_pair_forest_", metric_name, "_", cname, ".png")
      ggsave(file.path(output_dir, fname), p,
             width = max(10, 4 * n_bands), height = max(6, n_pairs * 0.35 + 2),
             dpi = 300, limitsize = FALSE)
      message("  Saved: ", fname)
    }
  }
}

# ===========================================================================
# Report
# ===========================================================================

#' Write ANALYSIS_SUMMARY.md for connectivity
write_connectivity_summary <- function(global_df, global_ttest_df,
                                        omnibus_region_pair_df,
                                        posthoc_region_pair_df,
                                        config, n_subjects, sfreq,
                                        fig_dir, output_path) {
  lines <- character()
  add <- function(...) lines <<- c(lines, paste0(...))

  has_region_pair <- nrow(omnibus_region_pair_df) > 0

  # Header
  add("# ROI Connectivity Analysis \u2014 ", config$name)
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

  add("**Analysis:** Functional Connectivity (Coherence & Imaginary Coherence)")
  add("")
  add("**Groups:** ", group_str)
  add("")
  add("**Sampling Rate:** ", sfreq, " Hz")
  add("")
  add("**Frequency Bands:** ", band_str)
  add("")
  add("**Metrics:** Magnitude-squared coherence (MSC) and absolute imaginary coherence (|iCoh|)")
  add("")
  add("**Timeseries:** Signed (phase-preserving) ROI source timeseries used for all computations.")
  add("")
  add("**CSD Method:** Welch cross-spectral density (2-second Hann windows, 50% overlap)")
  add("")

  n_rois <- length(unique(c(global_df$band)))  # placeholder
  edge_info <- global_df %>% slice(1)
  n_edges_str <- if (nrow(edge_info) > 0) as.character(edge_info$n_edges[1]) else "unknown"

  stats_lines <- paste0(
    "**Statistics:** Two analysis tiers. ",
    "(1) **Global:** Mean connectivity across all ", n_edges_str, " edges per subject x band. ",
    "Welch t-test per band, BH FDR correction across bands within each contrast x metric. "
  )
  if (has_region_pair) {
    n_rp <- omnibus_region_pair_df$n_region_pairs[1]
    stats_lines <- paste0(stats_lines,
      "(2) **Region-pair level:** Edges mapped to ", n_rp, " region pairs via roi_categories, ",
      "averaged within. LMM: dv ~ group * region_pair + (1|subject). ",
      "Type III ANOVA with Satterthwaite df. ",
      "FDR (BH) across bands. ",
      "Post-hoc: emmeans pairwise group contrasts per region pair, Holm correction. ",
      "Hedges' g = emmean difference / residual SD."
    )
  }
  add(stats_lines)
  add("")

  # --- Global t-test results ---
  add("## Global Connectivity T-Tests")
  add("")
  if (nrow(global_ttest_df) > 0) {
    add("| Contrast | Metric | Band | n_a | n_b | mean_a | mean_b | t | df | p | q | g | Sig |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (i in seq_len(nrow(global_ttest_df))) {
      row <- global_ttest_df[i, ]
      sig_str <- if (isTRUE(row$significant)) "**Yes**" else "No"
      add(sprintf("| %s | %s | %s | %d | %d | %.4f | %.4f | %.2f | %.1f | %.4f | %.4f | %.2f | %s |",
                  row$contrast, row$metric, row$band, row$n_a, row$n_b,
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

  # --- Region-pair LMM results ---
  if (has_region_pair) {
    add("## Region-Pair LMM Results")
    add("")
    add("| Contrast | Metric | Band | n_a | n_b | n_pairs | group_F | group_q | Sig | interaction_F | interaction_q | Int Sig |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (i in seq_len(nrow(omnibus_region_pair_df))) {
      row <- omnibus_region_pair_df[i, ]
      grp_sig <- if (isTRUE(row$group_significant)) "**Yes**" else "No"
      int_sig <- if (isTRUE(row$interaction_significant)) "**Yes**" else "No"
      add(sprintf("| %s | %s | %s | %d | %d | %d | %.2f | %.4f | %s | %.2f | %.4f | %s |",
                  row$contrast, row$metric, row$band, row$n_a, row$n_b,
                  row$n_region_pairs,
                  ifelse(is.na(row$group_F), 0, row$group_F),
                  ifelse(is.na(row$group_q), 1, row$group_q), grp_sig,
                  ifelse(is.na(row$interaction_F), 0, row$interaction_F),
                  ifelse(is.na(row$interaction_q), 1, row$interaction_q), int_sig))
    }
    add("")

    # Post-hoc
    add("### Region-Pair Post-Hoc Contrasts")
    add("")
    if (nrow(posthoc_region_pair_df) > 0) {
      sig_ph <- posthoc_region_pair_df %>% filter(significant == TRUE)
      if (nrow(sig_ph) > 0) {
        add("Significant region-pair group differences (Holm-corrected q < 0.05):")
        add("")
        for (m in unique(sig_ph$metric)) {
          for (bname in unique(sig_ph$band[sig_ph$metric == m])) {
            band_sig <- sig_ph %>% filter(metric == m, band == bname)
            add("#### ", bname, " (", gsub("_", " ", m), ")")
            add("")
            add("| Region Pair | Estimate | SE | t | q | Hedges' g |")
            add("| --- | --- | --- | --- | --- | --- |")
            for (i in seq_len(nrow(band_sig))) {
              row <- band_sig[i, ]
              add(sprintf("| %s | %.4f | %.4f | %.2f | %.4f | %.2f |",
                          row$region_pair, row$estimate, row$SE, row$t_ratio,
                          row$q_value, row$hedges_g))
            }
            add("")
          }
        }
      } else {
        add("No individual region pairs reached significance after Holm correction.")
        add("")
      }
      add("**Total region pairs tested:** ", length(unique(posthoc_region_pair_df$region_pair)),
          " across ", length(unique(posthoc_region_pair_df$band)), " band(s)")
      add("")
    } else {
      add("*Region-pair post-hoc not performed (no significant omnibus effects).*")
      add("")
    }
  }

  # Key findings
  add("## Key Findings")
  add("")
  any_sig <- FALSE

  # Global findings
  if (nrow(global_ttest_df) > 0) {
    sig_global <- global_ttest_df %>% filter(significant == TRUE)
    if (nrow(sig_global) > 0) {
      any_sig <- TRUE
      for (i in seq_len(nrow(sig_global))) {
        row <- sig_global[i, ]
        add(sprintf("- **%s %s** [%s, global]: t=%.2f, q=%.4f, g=%.2f (mean_a=%.4f, mean_b=%.4f)",
                    row$band, row$metric, row$contrast,
                    row$t_stat, row$q_value, row$hedges_g, row$mean_a, row$mean_b))
      }
    }
  }

  # Region-pair findings
  if (has_region_pair) {
    for (i in seq_len(nrow(omnibus_region_pair_df))) {
      row <- omnibus_region_pair_df[i, ]
      findings <- character()
      if (isTRUE(row$group_significant))
        findings <- c(findings, sprintf("group main effect (F=%.2f, q=%.4f)", row$group_F, row$group_q))
      if (isTRUE(row$interaction_significant))
        findings <- c(findings, sprintf("group x region_pair interaction (F=%.2f, q=%.4f)", row$interaction_F, row$interaction_q))
      if (length(findings) > 0) {
        any_sig <- TRUE
        add(sprintf("- **%s %s** [%s, region-pair level]: %s", row$band, row$metric,
                    row$contrast, paste(findings, collapse = "; ")))
      }
    }
  }

  if (!any_sig) {
    add("- No bands reached significance after FDR correction at either analysis level.")
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

parser <- ArgumentParser(description = "Connectivity statistical analysis (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing roi_connectivity_edges.csv")
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
edges <- read_csv(file.path(data_dir, "roi_connectivity_edges.csv"), show_col_types = FALSE)
message("  roi_connectivity_edges.csv: ", nrow(edges), " rows")

# --- Load config ---
config <- read_yaml(config_path)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))
message("Bands: ", paste(names(config$bands), collapse = ", "))

# ===========================================================================
# 1. Global connectivity analysis
# ===========================================================================
message("\n=== Global Connectivity Analysis ===")

global_df <- compute_global_connectivity(edges)
message("  Global connectivity computed: ", nrow(global_df), " subject x band rows")

global_ttest_df <- run_global_ttests(global_df, config$contrasts, config$bands)
if (nrow(global_ttest_df) > 0) {
  message("\n  === Global T-Test Results ===")
  for (i in seq_len(nrow(global_ttest_df))) {
    row <- global_ttest_df[i, ]
    sig_str <- if (isTRUE(row$significant)) " ***" else ""
    message(sprintf("  %s | %s | %s: t=%.2f, q=%.4f%s",
                    row$contrast, row$metric, row$band,
                    ifelse(is.na(row$t_stat), 0, row$t_stat),
                    ifelse(is.na(row$q_value), 1, row$q_value), sig_str))
  }
}

# ===========================================================================
# 2. Region-pair analysis (if roi_categories defined)
# ===========================================================================
omnibus_region_pair_df <- data.frame()
posthoc_region_pair_df <- data.frame()

if (length(config$roi_categories) > 0 && has_lme4) {
  message("\n=== Region-Pair Connectivity Analysis ===")

  region_pair_df <- aggregate_edges_to_region_pairs(edges, config$roi_categories)
  n_region_pairs <- length(unique(region_pair_df$region_pair))
  message("  Aggregated to ", n_region_pairs, " region pairs")

  all_omnibus <- list()
  all_posthoc <- list()

  for (metric in c("coherence", "imag_coherence")) {
    message("\n  --- Metric: ", metric, " ---")

    omnibus <- run_omnibus_lmm_region_pair(region_pair_df, config$contrasts,
                                            config$bands, metric = metric)
    all_omnibus[[metric]] <- omnibus

    if (nrow(omnibus) > 0) {
      for (i in seq_len(nrow(omnibus))) {
        row <- omnibus[i, ]
        grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
        int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
        message(sprintf("  %s | %s: group F=%.2f q=%.4f%s | interaction F=%.2f q=%.4f%s",
                        row$contrast, row$band,
                        row$group_F, row$group_q, grp_sig,
                        row$interaction_F, row$interaction_q, int_sig))
      }
    }

    posthoc <- run_posthoc_emmeans_region_pair(region_pair_df, config$contrasts,
                                               config$bands, omnibus, metric = metric)
    all_posthoc[[metric]] <- posthoc

    if (nrow(posthoc) > 0) {
      sig_count <- sum(posthoc$significant, na.rm = TRUE)
      message("  ", nrow(posthoc), " region-pair contrasts, ", sig_count, " significant")
    } else {
      message("  No post-hoc tests (no significant omnibus effects)")
    }
  }

  omnibus_region_pair_df <- bind_rows(all_omnibus)
  posthoc_region_pair_df <- bind_rows(all_posthoc)
} else if (length(config$roi_categories) == 0) {
  message("\n  No roi_categories in config -- skipping region-pair analysis")
} else {
  message("\n  lme4/lmerTest not available -- skipping region-pair LMM analysis")
}

# ===========================================================================
# Export tables
# ===========================================================================
message("\nExporting tables...")

if (nrow(global_ttest_df) > 0) {
  write_csv(global_ttest_df, file.path(tbl_dir, "roi_connectivity_global.csv"))
  message("  Saved: tables/roi_connectivity_global.csv")
}
if (nrow(omnibus_region_pair_df) > 0) {
  write_csv(omnibus_region_pair_df, file.path(tbl_dir, "roi_connectivity_omnibus_region_pair.csv"))
  message("  Saved: tables/roi_connectivity_omnibus_region_pair.csv")
}
if (nrow(posthoc_region_pair_df) > 0) {
  write_csv(posthoc_region_pair_df, file.path(tbl_dir, "roi_connectivity_posthoc_region_pair.csv"))
  message("  Saved: tables/roi_connectivity_posthoc_region_pair.csv")
}

# ===========================================================================
# Figures
# ===========================================================================
message("\nGenerating figures...")

# Connectivity matrices (per metric)
for (mc in c("coherence", "imag_coherence")) {
  plot_connectivity_matrices(edges, mc, group_colors, group_labels, group_order, fig_dir)
}

# Global connectivity bar chart
plot_global_connectivity_bar(global_df, group_colors, group_labels, group_order, fig_dir)

# Region-pair forest plots (if post-hoc was performed)
if (nrow(posthoc_region_pair_df) > 0) {
  plot_region_pair_forest(posthoc_region_pair_df, fig_dir)
}

# ===========================================================================
# Summary report
# ===========================================================================
message("\nWriting summary...")

n_subjects <- edges %>%
  dplyr::distinct(subject, group) %>%
  dplyr::count(group) %>%
  { setNames(.$n, .$group) }

sfreq <- if (!is.null(config$sfreq)) config$sfreq else 500

write_connectivity_summary(
  global_df, global_ttest_df,
  omnibus_region_pair_df, posthoc_region_pair_df,
  config, n_subjects, sfreq,
  fig_dir, file.path(output_dir, "ANALYSIS_SUMMARY.md")
)

message("\nDone. Output: ", output_dir)
