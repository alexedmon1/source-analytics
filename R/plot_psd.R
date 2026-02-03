# plot_psd.R — ggplot2 visualizations for PSD analysis
#
# Functions:
#   plot_psd_by_region()   — PSD curves faceted by ROI category
#   plot_band_power_box()  — Band power boxplots by group
#   plot_regional_heatmap() — Region x band heatmap per group

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(patchwork)
library(forcats)

# Publication theme
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

#' Plot PSD curves by region category
#'
#' @param psd_df data.frame with columns: subject, group, roi, freq_hz, psd
#' @param roi_categories named list of ROI name vectors
#' @param group_colors named character vector of hex colors
#' @param group_labels named character vector of display labels
#' @param group_order character vector of group IDs in plot order
#' @param output_dir path to figures/ directory
#' @param fmax maximum frequency to display
plot_psd_by_region <- function(psd_df, roi_categories, group_colors,
                                group_labels, group_order, output_dir,
                                fmax = 110) {

  # Map ROIs to categories
  roi_to_cat <- data.frame(
    roi = unlist(roi_categories),
    category = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  plot_data <- psd_df %>%
    inner_join(roi_to_cat, by = "roi") %>%
    filter(freq_hz <= fmax, group %in% group_order) %>%
    # Average across ROIs within category per subject
    group_by(subject, group, category, freq_hz) %>%
    summarise(psd = mean(psd, na.rm = TRUE), .groups = "drop") %>%
    # Group-level mean and SEM
    group_by(group, category, freq_hz) %>%
    summarise(
      mean_psd = mean(psd, na.rm = TRUE),
      sem_psd = sd(psd, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      group = factor(group, levels = group_order),
      group_label = group_labels[as.character(group)]
    )

  if (nrow(plot_data) == 0) return(invisible(NULL))

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  p <- ggplot(plot_data, aes(x = freq_hz, y = mean_psd, color = group_label, fill = group_label)) +
    geom_ribbon(aes(ymin = mean_psd - sem_psd, ymax = mean_psd + sem_psd), alpha = 0.2, color = NA) +
    geom_line(linewidth = 0.8) +
    scale_y_log10(labels = label_scientific()) +
    scale_color_manual(values = color_vals, name = NULL) +
    scale_fill_manual(values = color_vals, name = NULL) +
    facet_wrap(~ category, scales = "free_y") +
    labs(x = "Frequency (Hz)", y = "PSD (log scale)",
         title = "Power Spectral Density by Region") +
    theme_pub()

  ggsave(file.path(output_dir, "psd_by_region.png"), p,
         width = 12, height = 8, dpi = 300)
  message("  Saved: psd_by_region.png")
}


#' Band power boxplots by group
#'
#' @param band_df data.frame with columns: subject, group, roi, band, relative, absolute, dB
#' @param group_colors, group_labels, group_order — study config
#' @param output_dir path to figures/ directory
#' @param power_type one of "relative", "absolute", "dB"
plot_band_power_box <- function(band_df, group_colors, group_labels,
                                 group_order, output_dir,
                                 power_type = "relative") {

  # Subject-level means (average across ROIs)
  subj_means <- band_df %>%
    filter(group %in% group_order) %>%
    group_by(subject, group, band) %>%
    summarise(value = mean(.data[[power_type]], na.rm = TRUE), .groups = "drop") %>%
    mutate(
      group = factor(group, levels = group_order),
      group_label = group_labels[as.character(group)]
    )

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  # Preserve band order from data
  band_order <- unique(band_df$band)
  subj_means$band <- factor(subj_means$band, levels = band_order)

  p <- ggplot(subj_means, aes(x = group_label, y = value, fill = group_label)) +
    geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
    geom_jitter(width = 0.15, size = 1.5, alpha = 0.6,
                aes(color = group_label), show.legend = FALSE) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    facet_wrap(~ band, scales = "free_y", nrow = 1) +
    labs(x = NULL, y = paste0(tools::toTitleCase(power_type), " Power"),
         title = paste0("Band Power (", tools::toTitleCase(power_type), ") by Group")) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")

  fname <- paste0("band_power_", power_type, ".png")
  ggsave(file.path(output_dir, fname), p,
         width = 3.5 * length(band_order), height = 5, dpi = 300)
  message("  Saved: ", fname)
}


#' Regional power heatmap
#'
#' @param band_df data.frame with columns: subject, group, roi, band, relative
#' @param roi_categories named list of ROI name vectors
#' @param group_colors, group_labels, group_order — study config
#' @param output_dir path to figures/ directory
#' @param power_type one of "relative", "absolute", "dB"
plot_regional_heatmap <- function(band_df, roi_categories, group_colors,
                                   group_labels, group_order, output_dir,
                                   power_type = "relative") {

  roi_to_cat <- data.frame(
    roi = unlist(roi_categories),
    category = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  for (grp in group_order) {
    gdata <- band_df %>%
      filter(group == grp) %>%
      inner_join(roi_to_cat, by = "roi") %>%
      group_by(category, band) %>%
      summarise(value = mean(.data[[power_type]], na.rm = TRUE), .groups = "drop")

    if (nrow(gdata) == 0) next

    # Preserve ordering
    gdata$category <- factor(gdata$category, levels = rev(names(roi_categories)))
    band_order <- unique(band_df$band)
    gdata$band <- factor(gdata$band, levels = band_order)

    label <- group_labels[grp]

    p <- ggplot(gdata, aes(x = band, y = category, fill = value)) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(aes(label = sprintf("%.4f", value)), size = 3) +
      scale_fill_viridis_c(option = "inferno", name = tools::toTitleCase(power_type)) +
      labs(x = "Band", y = "Region",
           title = paste0(label, " \u2014 ", tools::toTitleCase(power_type), " Power")) +
      theme_pub() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))

    fname <- paste0("heatmap_", power_type, "_", grp, ".png")
    ggsave(file.path(output_dir, fname), p,
           width = 8, height = max(4, length(roi_categories) * 0.7 + 1), dpi = 300)
    message("  Saved: ", fname)
  }
}


#' Forest plot of group contrast per ROI (emmeans post-hoc)
#'
#' Dot-and-whisker plot showing the estimated group difference at each ROI,
#' faceted by frequency band. Significant ROIs highlighted.
#'
#' @param posthoc_df data.frame from run_posthoc_emmeans()
#' @param output_dir path to figures/ directory
plot_roi_forest <- function(posthoc_df, output_dir) {
  if (nrow(posthoc_df) == 0) {
    message("  Skipping forest plot: no post-hoc results")
    return(invisible(NULL))
  }

  power_types <- unique(posthoc_df$power_type)
  if (length(power_types) == 0) power_types <- "relative"

  for (ptype in power_types) {
    for (cname in unique(posthoc_df$contrast)) {
      pdata <- posthoc_df %>%
        filter(contrast == cname, power_type == ptype) %>%
        mutate(
          roi = fct_reorder(roi, estimate),
          sig_label = ifelse(significant, "*", "")
        )

      if (nrow(pdata) == 0) next

      n_bands <- length(unique(pdata$band))
      n_rois <- length(unique(pdata$roi))

      p <- ggplot(pdata, aes(x = estimate, y = roi)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbar(aes(xmin = estimate - 1.96 * SE, xmax = estimate + 1.96 * SE),
                      width = 0.3, color = "grey40", orientation = "y") +
        geom_point(aes(color = significant), size = 2) +
        scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "#E74C3C"),
                           labels = c("n.s.", "p < .05"), name = NULL) +
        facet_wrap(~ band, scales = "free_x") +
        labs(x = "Group Difference (emmean)", y = NULL,
             title = paste0("ROI-Level Group Contrasts: ", cname, " (", ptype, ")")) +
        theme_pub() +
        theme(
          axis.text.y = element_text(size = 7),
          strip.text = element_text(size = 10)
        )

      fname <- paste0("roi_forest_plot_", cname, "_", ptype, ".png")
      ggsave(file.path(output_dir, fname), p,
             width = max(10, 4 * n_bands), height = max(8, n_rois * 0.22 + 2),
             dpi = 300, limitsize = FALSE)
      message("  Saved: ", fname)
    }
  }
}


#' Significance heatmap (ROI x band)
#'
#' Heatmap with ROIs on the y-axis and frequency bands on the x-axis.
#' Fill = Hedges' g, asterisks on significant cells.
#'
#' @param posthoc_df data.frame from run_posthoc_emmeans()
#' @param output_dir path to figures/ directory
plot_significance_heatmap <- function(posthoc_df, output_dir) {
  if (nrow(posthoc_df) == 0) {
    message("  Skipping significance heatmap: no post-hoc results")
    return(invisible(NULL))
  }

  power_types <- unique(posthoc_df$power_type)
  if (length(power_types) == 0) power_types <- "relative"

  for (ptype in power_types) {
    for (cname in unique(posthoc_df$contrast)) {
      pdata <- posthoc_df %>%
        filter(contrast == cname, power_type == ptype) %>%
        mutate(
          sig_label = ifelse(significant, "*", ""),
          roi = fct_reorder(roi, hedges_g, .fun = function(x) mean(abs(x), na.rm = TRUE))
        )

      if (nrow(pdata) == 0) next

      # Symmetric color scale centered at 0
      max_abs_g <- max(abs(pdata$hedges_g), na.rm = TRUE)
      clim <- ceiling(max_abs_g * 10) / 10  # Round up to nearest 0.1

      n_rois <- length(unique(pdata$roi))

      p <- ggplot(pdata, aes(x = band, y = roi, fill = hedges_g)) +
        geom_tile(color = "white", linewidth = 0.5) +
        geom_text(aes(label = sig_label), size = 5, color = "black", fontface = "bold") +
        scale_fill_gradient2(
          low = "#2166AC", mid = "white", high = "#B2182B",
          midpoint = 0, limits = c(-clim, clim),
          name = "Hedges' g"
        ) +
        labs(x = "Frequency Band", y = NULL,
             title = paste0("ROI x Band Significance: ", cname, " (", ptype, ")"),
             subtitle = "* = significant after Holm correction") +
        theme_pub() +
        theme(
          axis.text.y = element_text(size = 7),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )

      fname <- paste0("roi_significance_heatmap_", cname, "_", ptype, ".png")
      ggsave(file.path(output_dir, fname), p,
             width = 8, height = max(6, n_rois * 0.22 + 2),
             dpi = 300, limitsize = FALSE)
      message("  Saved: ", fname)
    }
  }
}


#' Forest plot of group contrast per region (emmeans post-hoc)
#'
#' @param posthoc_region_df data.frame from run_posthoc_emmeans_region()
#' @param output_dir path to figures/ directory
plot_region_forest <- function(posthoc_region_df, output_dir) {
  if (nrow(posthoc_region_df) == 0) {
    message("  Skipping region forest plot: no post-hoc results")
    return(invisible(NULL))
  }

  power_types <- unique(posthoc_region_df$power_type)
  if (length(power_types) == 0) power_types <- "relative"

  for (ptype in power_types) {
    for (cname in unique(posthoc_region_df$contrast)) {
      pdata <- posthoc_region_df %>%
        filter(contrast == cname, power_type == ptype) %>%
        mutate(
          region = fct_reorder(region, estimate),
          sig_label = ifelse(significant, "*", "")
        )

      if (nrow(pdata) == 0) next

      n_bands <- length(unique(pdata$band))

      p <- ggplot(pdata, aes(x = estimate, y = region)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbar(aes(xmin = estimate - 1.96 * SE, xmax = estimate + 1.96 * SE),
                      width = 0.3, color = "grey40", orientation = "y") +
        geom_point(aes(color = significant), size = 3) +
        scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "#E74C3C"),
                           labels = c("n.s.", "p < .05"), name = NULL) +
        facet_wrap(~ band, scales = "free_x") +
        labs(x = "Group Difference (emmean)", y = NULL,
             title = paste0("Region-Level Group Contrasts: ", cname, " (", ptype, ")")) +
        theme_pub() +
        theme(strip.text = element_text(size = 10))

      fname <- paste0("region_forest_plot_", cname, "_", ptype, ".png")
      ggsave(file.path(output_dir, fname), p,
             width = max(10, 4 * n_bands), height = 6,
             dpi = 300, limitsize = FALSE)
      message("  Saved: ", fname)
    }
  }
}


#' Significance heatmap (region x band)
#'
#' @param posthoc_region_df data.frame from run_posthoc_emmeans_region()
#' @param output_dir path to figures/ directory
plot_region_significance_heatmap <- function(posthoc_region_df, output_dir) {
  if (nrow(posthoc_region_df) == 0) {
    message("  Skipping region significance heatmap: no post-hoc results")
    return(invisible(NULL))
  }

  power_types <- unique(posthoc_region_df$power_type)
  if (length(power_types) == 0) power_types <- "relative"

  for (ptype in power_types) {
    for (cname in unique(posthoc_region_df$contrast)) {
      pdata <- posthoc_region_df %>%
        filter(contrast == cname, power_type == ptype) %>%
        mutate(
          sig_label = ifelse(significant, "*", ""),
          region = fct_reorder(region, hedges_g, .fun = function(x) mean(abs(x), na.rm = TRUE))
        )

      if (nrow(pdata) == 0) next

      max_abs_g <- max(abs(pdata$hedges_g), na.rm = TRUE)
      clim <- ceiling(max_abs_g * 10) / 10

      p <- ggplot(pdata, aes(x = band, y = region, fill = hedges_g)) +
        geom_tile(color = "white", linewidth = 0.5) +
        geom_text(aes(label = sig_label), size = 6, color = "black", fontface = "bold") +
        geom_text(aes(label = sprintf("%.2f", hedges_g)), size = 3, vjust = -0.5) +
        scale_fill_gradient2(
          low = "#2166AC", mid = "white", high = "#B2182B",
          midpoint = 0, limits = c(-clim, clim),
          name = "Hedges' g"
        ) +
        labs(x = "Frequency Band", y = NULL,
             title = paste0("Region x Band Significance: ", cname, " (", ptype, ")"),
             subtitle = "* = significant after Holm correction") +
        theme_pub() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))

      fname <- paste0("region_significance_heatmap_", cname, "_", ptype, ".png")
      ggsave(file.path(output_dir, fname), p,
             width = 8, height = 5, dpi = 300)
      message("  Saved: ", fname)
    }
  }
}
