#!/usr/bin/env Rscript
# electrode_analysis.R — Electrode-level PSD statistical analysis
#
# Called by Python: Rscript R/electrode_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads electrode_band_power.csv, runs omnibus LMM + emmeans post-hoc
# for electrode-level data (group * channel), generates figures and summary.

library(argparse)
library(yaml)
library(readr)

# Resolve script directory for sourcing helpers
script_dir <- if (exists("script.dir")) {
  script.dir
} else {
  tryCatch({
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("^--file=", "", file_arg)))
    } else {
      "R"
    }
  }, error = function(e) "R")
}

source(file.path(script_dir, "stats_utils.R"))
source(file.path(script_dir, "plot_psd.R"))
source(file.path(script_dir, "report.R"))

# --- Argument parsing ---
parser <- ArgumentParser(description = "Electrode-level PSD analysis (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing electrode_band_power.csv")
parser$add_argument("--config", required = TRUE,
                    help = "Path to study YAML config")
parser$add_argument("--output-dir", required = TRUE,
                    help = "Root output directory for this analysis")
args <- parser$parse_args()

data_dir <- args$data_dir
config_path <- args$config
output_dir <- args$output_dir

fig_dir <- file.path(output_dir, "figures")
tbl_dir <- file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load data ---
message("Loading electrode data...")
band_df <- read_csv(file.path(data_dir, "electrode_band_power.csv"), show_col_types = FALSE)
message("  electrode_band_power.csv: ", nrow(band_df), " rows")

# Rename 'channel' to 'roi' so we can reuse stats_utils functions
band_df <- band_df %>% dplyr::rename(roi = channel)

# --- Load config ---
config <- read_yaml(config_path)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))
message("Bands: ", paste(names(config$bands), collapse = ", "))
message("Channels: ", length(unique(band_df$roi)))

# --- Run LMMs for each power type ---
power_types <- c("relative", "dB")

all_omnibus <- list()
all_posthoc <- list()
all_omnibus_region_nested <- list()
all_posthoc_region_nested <- list()

# Check for electrode_categories in config
electrode_categories <- config$electrode_categories

for (ptype in power_types) {
  message("\n=== Power type: ", ptype, " ===")

  # Omnibus LMM: dv ~ group * channel + (1|subject)
  message("Running electrode-level omnibus LMM (group * channel)...")
  omnibus <- run_omnibus_lmm(band_df, config$contrasts, config$bands, power_type = ptype)
  all_omnibus[[ptype]] <- omnibus

  if (nrow(omnibus) > 0) {
    message("\n  === Electrode Omnibus (", ptype, ") ===")
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

  # Post-hoc: emmeans per channel
  message("Running channel-level post-hoc emmeans...")
  posthoc <- run_posthoc_emmeans(band_df, config$contrasts, config$bands, omnibus,
                                  power_type = ptype)
  all_posthoc[[ptype]] <- posthoc

  if (nrow(posthoc) > 0) {
    sig_count <- sum(posthoc$significant, na.rm = TRUE)
    message("  ", nrow(posthoc), " channel contrasts, ", sig_count, " significant")
  } else {
    message("  No post-hoc tests (no significant omnibus effects)")
  }

  # --- Region-level nested (electrodes as replicates within regions) ---
  if (length(electrode_categories) > 0) {
    message("Running region-level nested omnibus LMM (group * region, electrodes as replicates)...")
    omnibus_reg_nested <- run_omnibus_lmm_region_nested(band_df, config$contrasts, config$bands,
                                                         electrode_categories, power_type = ptype)
    all_omnibus_region_nested[[ptype]] <- omnibus_reg_nested

    if (nrow(omnibus_reg_nested) > 0) {
      message("\n  === Region-Level Nested Omnibus (", ptype, ") ===")
      for (i in seq_len(nrow(omnibus_reg_nested))) {
        row <- omnibus_reg_nested[i, ]
        grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
        int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
        message(sprintf("  %s | %s: group F=%.2f q=%.4f%s | interaction F=%.2f q=%.4f%s",
                        row$contrast, row$band,
                        row$group_F, row$group_q, grp_sig,
                        row$interaction_F, row$interaction_q, int_sig))
      }
    }

    message("Running region-level nested post-hoc emmeans...")
    posthoc_reg_nested <- run_posthoc_emmeans_region_nested(band_df, config$contrasts, config$bands,
                                                             electrode_categories, omnibus_reg_nested,
                                                             power_type = ptype)
    all_posthoc_region_nested[[ptype]] <- posthoc_reg_nested

    if (nrow(posthoc_reg_nested) > 0) {
      sig_count <- sum(posthoc_reg_nested$significant, na.rm = TRUE)
      message("  ", nrow(posthoc_reg_nested), " nested region contrasts, ", sig_count, " significant")
    } else {
      message("  No nested region post-hoc tests (no significant omnibus effects)")
    }
  }
}

# --- Combine results ---
omnibus_df <- bind_rows(all_omnibus)
posthoc_df <- bind_rows(all_posthoc)
omnibus_region_nested_df <- bind_rows(all_omnibus_region_nested)
posthoc_region_nested_df <- bind_rows(all_posthoc_region_nested)

# --- Export tables ---
message("\nExporting tables...")
if (nrow(omnibus_df) > 0) {
  write_csv(omnibus_df, file.path(tbl_dir, "electrode_omnibus.csv"))
  message("  Saved: tables/electrode_omnibus.csv")
}
if (nrow(posthoc_df) > 0) {
  write_csv(posthoc_df, file.path(tbl_dir, "electrode_posthoc.csv"))
  message("  Saved: tables/electrode_posthoc.csv")
}
if (nrow(omnibus_region_nested_df) > 0) {
  write_csv(omnibus_region_nested_df, file.path(tbl_dir, "electrode_omnibus_region_nested.csv"))
  message("  Saved: tables/electrode_omnibus_region_nested.csv")
}
if (nrow(posthoc_region_nested_df) > 0) {
  write_csv(posthoc_region_nested_df, file.path(tbl_dir, "electrode_posthoc_region_nested.csv"))
  message("  Saved: tables/electrode_posthoc_region_nested.csv")
}

# --- Figures ---
message("\nGenerating figures...")

# Band power boxplots (electrode-level means per subject)
for (ptype in c("relative", "absolute", "dB")) {
  # Subject-level means (average across channels)
  subj_means <- band_df %>%
    dplyr::filter(group %in% group_order) %>%
    dplyr::group_by(subject, group, band) %>%
    dplyr::summarise(value = mean(.data[[ptype]], na.rm = TRUE), .groups = "drop") %>%
    dplyr::mutate(
      group = factor(group, levels = group_order),
      group_label = group_labels[as.character(group)]
    )

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  band_order <- unique(band_df$band)
  subj_means$band <- factor(subj_means$band, levels = band_order)

  p <- ggplot(subj_means, aes(x = group_label, y = value, fill = group_label)) +
    geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
    geom_jitter(width = 0.15, size = 1.5, alpha = 0.6,
                aes(color = group_label), show.legend = FALSE) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    facet_wrap(~ band, scales = "free_y", nrow = 1) +
    labs(x = NULL, y = paste0(tools::toTitleCase(ptype), " Power"),
         title = paste0("Electrode Band Power (", tools::toTitleCase(ptype), ") by Group")) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")

  fname <- paste0("electrode_band_power_", ptype, ".png")
  ggsave(file.path(fig_dir, fname), p,
         width = 3.5 * length(band_order), height = 5, dpi = 300)
  message("  Saved: ", fname)
}

# Channel-level forest plots (post-hoc)
if (nrow(posthoc_df) > 0) {
  for (ptype in unique(posthoc_df$power_type)) {
    for (cname in unique(posthoc_df$contrast)) {
      pdata <- posthoc_df %>%
        dplyr::filter(contrast == cname, power_type == ptype) %>%
        dplyr::mutate(
          roi = forcats::fct_reorder(roi, estimate),
          sig_label = ifelse(significant, "*", "")
        )

      if (nrow(pdata) == 0) next

      n_bands <- length(unique(pdata$band))

      p <- ggplot(pdata, aes(x = estimate, y = roi)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbar(aes(xmin = estimate - 1.96 * SE, xmax = estimate + 1.96 * SE),
                      width = 0.3, color = "grey40", orientation = "y") +
        geom_point(aes(color = significant), size = 2.5) +
        scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "#E74C3C"),
                           labels = c("n.s.", "p < .05"), name = NULL) +
        facet_wrap(~ band, scales = "free_x") +
        labs(x = "Group Difference (emmean)", y = "Channel",
             title = paste0("Channel-Level Group Contrasts: ", cname, " (", ptype, ")")) +
        theme_pub()

      fname <- paste0("electrode_channel_forest_", cname, "_", ptype, ".png")
      ggsave(file.path(fig_dir, fname), p,
             width = max(10, 4 * n_bands), height = 8, dpi = 300, limitsize = FALSE)
      message("  Saved: ", fname)
    }
  }
}

# --- Summary report ---
message("\nWriting summary...")

n_subjects <- band_df %>%
  dplyr::distinct(subject, group) %>%
  dplyr::count(group) %>%
  { setNames(.$n, .$group) }

sfreq <- if (!is.null(config$sfreq)) config$sfreq else 500

write_summary(omnibus_df, posthoc_df, config, n_subjects, sfreq,
              fig_dir, file.path(output_dir, "ANALYSIS_SUMMARY.md"),
              omnibus_region_nested_df = omnibus_region_nested_df,
              posthoc_region_nested_df = posthoc_region_nested_df)

message("\nDone. Output: ", output_dir)
