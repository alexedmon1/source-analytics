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
library(ggsignif)

# Publication theme
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
                                group_linetypes = NULL,
                                fmax = 80, notch_lo = 55, notch_hi = 65) {

  notch_width <- notch_hi - notch_lo

  # Map ROIs to categories
  roi_to_cat <- data.frame(
    roi = unlist(roi_categories),
    category = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  plot_data <- psd_df %>%
    inner_join(roi_to_cat, by = "roi") %>%
    filter(freq_hz <= fmax, group %in% group_order,
           !(freq_hz >= notch_lo & freq_hz <= notch_hi)) %>%
    group_by(subject, group, category, freq_hz) %>%
    summarise(psd = mean(psd, na.rm = TRUE), .groups = "drop") %>%
    group_by(group, category, freq_hz) %>%
    summarise(
      mean_psd = mean(psd, na.rm = TRUE),
      sem_psd = sd(psd, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    ) %>%
    mutate(
      group = factor(group, levels = group_order),
      group_label = factor(group_labels[as.character(group)], levels = group_labels[group_order]),
      # Shift frequencies above notch leftward to close the gap
      freq_plot = ifelse(freq_hz > notch_hi, freq_hz - notch_width, freq_hz),
      segment = ifelse(freq_hz < notch_lo, "low", "high")
    )

  if (nrow(plot_data) == 0) return(invisible(NULL))

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  # Per-group line patterns (fall back to all solid). Kept simple so overlaid
  # curves stay legible; the two primary groups are solid by config.
  if (is.null(group_linetypes) || length(group_linetypes) == 0) {
    linetype_vals <- setNames(rep("solid", length(group_order)), group_labels[group_order])
  } else {
    linetype_vals <- group_linetypes[group_order]
    linetype_vals[is.na(linetype_vals)] <- "solid"
    names(linetype_vals) <- group_labels[group_order]
  }

  # Custom x-axis breaks and labels (show true Hz values)
  breaks_low <- seq(0, notch_lo, by = 10)
  breaks_high_real <- seq(notch_hi + 5, fmax, by = 10)
  breaks_high_plot <- breaks_high_real - notch_width
  all_breaks <- c(breaks_low, breaks_high_plot)
  all_labels <- c(as.character(breaks_low), as.character(breaks_high_real))

  # Position of the break marker (in plot coords)
  break_x <- notch_lo

  # Frequency band definitions (in real Hz) and their plot-space boundaries
  to_plot_x <- function(hz) ifelse(hz > notch_hi, hz - notch_width, pmin(hz, notch_lo))
  fmax_plot <- to_plot_x(fmax)

  bands <- data.frame(
    label = c("\u03b4", "\u03b8", "\u03b1", "\u03b2", "\u03b3L", "\u03b3H"),
    xmin  = c(  1,   4,  10,  13,  30, to_plot_x(65)),
    xmax  = c(  4,  10,  13,  30,  55, to_plot_x(80)),
    stringsAsFactors = FALSE
  )
  bands$xmax <- pmin(bands$xmax, fmax_plot)
  bands <- bands[bands$xmin < bands$xmax, ]
  bands$xmid <- (bands$xmin + bands$xmax) / 2

  p <- ggplot(plot_data, aes(x = freq_plot, y = mean_psd,
                              color = group_label, fill = group_label))

  # Band Greek labels at top of each panel
  for (i in seq_len(nrow(bands))) {
    p <- p + annotate("text",
                       x = bands$xmid[i], y = Inf,
                       label = bands$label[i],
                       vjust = 1.5, size = 3.5,
                       color = "grey40", fontface = "bold.italic")
  }

  p <- p +
    geom_ribbon(aes(ymin = mean_psd - sem_psd, ymax = mean_psd + sem_psd,
                    group = interaction(group_label, segment)),
                alpha = 0.2, color = NA) +
    geom_line(aes(group = interaction(group_label, segment), linetype = group_label),
              linewidth = 0.8) +
    geom_vline(xintercept = break_x, linetype = "dashed", color = "grey50", linewidth = 0.4) +
    scale_x_continuous(breaks = all_breaks, labels = all_labels,
                       expand = expansion(mult = c(0.02, 0.02))) +
    scale_y_log10(labels = label_scientific(),
                  expand = expansion(mult = c(0.05, 0.15))) +
    scale_color_manual(values = color_vals, name = NULL) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_linetype_manual(values = linetype_vals, name = NULL) +
    facet_wrap(~ category, scales = "free_y") +
    labs(x = "Frequency (Hz)", y = "PSD (log scale)",
         title = "Power Spectral Density by Region",
         caption = "Dashed line: 55\u201365 Hz notch excluded") +
    theme_pub() +
    theme(plot.caption = element_text(hjust = 0.5, size = 10, color = "grey40"))

  ggsave(file.path(output_dir, "psd_by_region.png"), p,
         width = 12, height = 8, dpi = 300)
  message("  Saved: psd_by_region.png")
}


#' Band power boxplots by group
#'
#' @param band_df data.frame with columns: subject, group, roi, band, relative, absolute, dB
#' @param group_colors, group_labels, group_order — study config
#' @param output_dir path to figures/ directory
#' @param power_type one of "relative" or "absolute"
plot_band_power_box <- function(band_df, group_colors, group_labels,
                                 group_order, output_dir,
                                 power_type = "relative", sig_df = NULL) {

  # Subject-level means (average across ROIs)
  subj_means <- band_df %>%
    filter(group %in% group_order) %>%
    group_by(subject, group, band) %>%
    summarise(value = mean(.data[[power_type]], na.rm = TRUE), .groups = "drop") %>%
    mutate(
      group = factor(group, levels = group_order),
      group_label = factor(group_labels[as.character(group)], levels = group_labels[group_order])
    )

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  # Preserve band order from data
  band_order <- order_bands(band_df$band)
  subj_means$band <- factor(subj_means$band, levels = band_order)

  show_jitter <- power_type == "absolute"
  p <- ggplot(subj_means, aes(x = group_label, y = value, fill = group_label)) +
    geom_boxplot(width = 0.5, outlier.shape = if (show_jitter) NA else 16,
                 alpha = 0.7) +
    { if (show_jitter) geom_jitter(width = 0.15, size = 1.5, alpha = 0.6,
                aes(color = group_label), show.legend = FALSE) } +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    facet_wrap(~ band, scales = "free_y", nrow = 2) +
    labs(x = NULL,
         y = if (power_type == "absolute") "Power density (dB/Hz)" else paste0(tools::toTitleCase(power_type), " Power"),
         title = if (power_type == "absolute") "Band Power Density (dB/Hz) by Group" else paste0("Band Power (", tools::toTitleCase(power_type), ") by Group")) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")

  # Add significance brackets per facet
  if (!is.null(sig_df) && nrow(sig_df) > 0) {
    sig_hits <- sig_df %>% filter(significant == TRUE)

    if (nrow(sig_hits) > 0) {
      # Compute y_max per band for positioning
      y_ranges <- subj_means %>%
        group_by(band) %>%
        summarise(y_max = max(value, na.rm = TRUE),
                  y_range = diff(range(value, na.rm = TRUE)),
                  .groups = "drop")

      for (b in unique(sig_hits$band)) {
        b_sig <- sig_hits %>% filter(band == b)
        b_range <- y_ranges %>% filter(band == b)
        if (nrow(b_range) == 0) next

        y_step <- b_range$y_range[1] * 0.08

        for (i in seq_len(nrow(b_sig))) {
          row <- b_sig[i, ]
          label_a <- group_labels[row$group_a]
          label_b <- group_labels[row$group_b]
          y_pos <- b_range$y_max[1] + y_step * i

          p <- p + geom_signif(
            data = subj_means %>% filter(band == b),
            comparisons = list(c(label_a, label_b)),
            annotations = row$sig_label,
            y_position = y_pos,
            tip_length = 0.02,
            textsize = 5,
            color = "black"
          )
        }
      }
    }
  }

  n_cols <- ceiling(length(band_order) / 2)
  fname <- paste0("band_power_", power_type, ".png")
  ggsave(file.path(output_dir, fname), p,
         width = 3.5 * n_cols, height = 9, dpi = 300)
  message("  Saved: ", fname)
}


#' Regional power heatmap
#'
#' @param band_df data.frame with columns: subject, group, roi, band, relative
#' @param roi_categories named list of ROI name vectors
#' @param group_colors, group_labels, group_order — study config
#' @param output_dir path to figures/ directory
#' @param power_type one of "relative" or "absolute"
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
    band_order <- order_bands(band_df$band)
    gdata$band <- factor(gdata$band, levels = band_order)

    label <- group_labels[grp]

    p <- ggplot(gdata, aes(x = band, y = category, fill = value)) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(aes(label = sprintf("%.4f", value)), size = 5) +
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

  power_types <- unique(posthoc_df$dv)
  if (length(power_types) == 0) power_types <- "relative"

  for (ptype in power_types) {
    for (cname in unique(posthoc_df$hypothesis)) {
      pdata <- posthoc_df %>%
        filter(hypothesis == cname, dv == ptype) %>%
        mutate(
          sig_label = ifelse(significant, "*", ""),
          roi = fct_reorder(spatial, effect_size, .fun = function(x) mean(abs(x), na.rm = TRUE))
        )

      if (nrow(pdata) == 0) next

      # Symmetric color scale centered at 0
      max_abs_g <- max(abs(pdata$effect_size), na.rm = TRUE)
      clim <- ceiling(max_abs_g * 10) / 10  # Round up to nearest 0.1

      n_rois <- length(unique(pdata$roi))

      p <- ggplot(pdata, aes(x = band, y = roi, fill = effect_size)) +
        geom_tile(color = "white", linewidth = 0.5) +
        geom_text(aes(label = sig_label), size = 7, color = "black", fontface = "bold") +
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
          axis.text.y = element_text(size = 9),
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


#' Significance heatmap (region x band)
#'
#' @param posthoc_region_df data.frame from run_posthoc_emmeans_region()
#' @param output_dir path to figures/ directory
plot_region_significance_heatmap <- function(posthoc_region_df, output_dir) {
  if (nrow(posthoc_region_df) == 0) {
    message("  Skipping region significance heatmap: no post-hoc results")
    return(invisible(NULL))
  }

  power_types <- unique(posthoc_region_df$dv)
  if (length(power_types) == 0) power_types <- "relative"

  for (ptype in power_types) {
    for (cname in unique(posthoc_region_df$hypothesis)) {
      pdata <- posthoc_region_df %>%
        filter(hypothesis == cname, dv == ptype) %>%
        mutate(
          sig_label = ifelse(significant, "*", ""),
          region = fct_reorder(region, effect_size, .fun = function(x) mean(abs(x), na.rm = TRUE))
        )

      if (nrow(pdata) == 0) next

      max_abs_g <- max(abs(pdata$effect_size), na.rm = TRUE)
      clim <- ceiling(max_abs_g * 10) / 10

      p <- ggplot(pdata, aes(x = band, y = region, fill = effect_size)) +
        geom_tile(color = "white", linewidth = 0.5) +
        # One centered label per cell: Hedges' g with a trailing * when
        # significant (the two separate layers + vjust=-0.5 shifted the text up).
        geom_text(aes(label = paste0(sprintf("%.2f", effect_size), sig_label),
                      fontface = ifelse(significant, "bold", "plain")),
                  size = 5, color = "black") +
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


#' Band power by region for a single band, with significance markers
#'
#' @param band_df data.frame with columns: subject, group, roi, band, relative, absolute
#' @param roi_categories named list of ROI name vectors
#' @param group_colors, group_labels, group_order — study config
#' @param output_dir path to figures/ directory
#' @param target_band character: band name to plot
#' @param power_type one of "relative" or "absolute"
#' @param posthoc_region_df data.frame from run_posthoc_emmeans_region() (optional)
#' @param contrast character: which contrast to show significance for (optional)
plot_band_by_region <- function(band_df, roi_categories, group_colors,
                                 group_labels, group_order, output_dir,
                                 target_band, power_type = "absolute",
                                 posthoc_region_df = NULL, contrast = NULL) {

  roi_to_cat <- data.frame(
    roi = unlist(roi_categories),
    category = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  # Subject-level region means
  subj_region <- band_df %>%
    filter(band == target_band, group %in% group_order) %>%
    inner_join(roi_to_cat, by = "roi") %>%
    group_by(subject, group, category) %>%
    summarise(value = mean(.data[[power_type]], na.rm = TRUE), .groups = "drop") %>%
    mutate(
      group = factor(group, levels = group_order),
      group_label = factor(group_labels[as.character(group)], levels = group_labels[group_order]),
      category = factor(category, levels = names(roi_categories))
    )

  if (nrow(subj_region) == 0) return(invisible(NULL))

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  y_label <- if (power_type == "absolute") "Power density (dB/Hz)" else "Relative Power"
  band_label <- gsub("_", " ", tools::toTitleCase(target_band))

  p <- ggplot(subj_region, aes(x = category, y = value, fill = group_label)) +
    geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.7,
                 position = position_dodge(width = 0.7)) +
    geom_jitter(aes(color = group_label),
                position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.7),
                size = 1.5, alpha = 0.6, show.legend = FALSE) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    labs(x = NULL, y = y_label,
         title = paste0(band_label, " Power by Region")) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  # Add significance markers for specific contrast
  if (!is.null(posthoc_region_df) && nrow(posthoc_region_df) > 0 && !is.null(contrast)) {
    sig_regions <- posthoc_region_df %>%
      filter(band == target_band, dv == !!power_type,
             hypothesis == !!contrast, significant == TRUE)

    if (nrow(sig_regions) > 0) {
      # Get y positions for brackets
      y_ranges <- subj_region %>%
        group_by(category) %>%
        summarise(y_max = max(value, na.rm = TRUE),
                  y_range = diff(range(value, na.rm = TRUE)),
                  .groups = "drop")

      # Extract group labels for the contrast
      parts <- strsplit(contrast, "_vs_")[[1]]
      label_a <- group_labels[parts[1]]
      label_b <- group_labels[parts[2]]

      for (i in seq_len(nrow(sig_regions))) {
        row <- sig_regions[i, ]
        reg_range <- y_ranges %>% filter(category == row$region)
        if (nrow(reg_range) == 0) next

        y_pos <- reg_range$y_max[1] + reg_range$y_range[1] * 0.12

        sig_label <- if ("sig_label" %in% names(sig_regions) && nchar(row$sig_label) > 0) {
          row$sig_label
        } else {
          "*"
        }

        p <- p + annotate("text", x = row$region, y = y_pos,
                           label = sig_label, size = 7, fontface = "bold")
      }
    }
  }

  fname <- paste0("band_by_region_", target_band, "_", power_type, ".png")
  ggsave(file.path(output_dir, fname), p,
         width = 12, height = 6, dpi = 300)
  message("  Saved: ", fname)
}
