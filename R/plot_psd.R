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
