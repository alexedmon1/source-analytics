#!/usr/bin/env Rscript
# psd_analysis.R — Main R entry point for PSD statistical analysis
#
# Called by Python: Rscript R/psd_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads CSVs exported by Python (band_power.csv, psd_curves.csv),
# runs omnibus LMM + emmeans post-hoc for both relative and absolute power,
# generates ggplot2 figures, writes summary.

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
parser <- ArgumentParser(description = "PSD statistical analysis (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing band_power.csv and psd_curves.csv")
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
band_df <- read_csv(file.path(data_dir, "band_power.csv"), show_col_types = FALSE)
message("  band_power.csv: ", nrow(band_df), " rows")

psd_file <- file.path(data_dir, "psd_curves.csv")
has_psd_curves <- file.exists(psd_file)
if (has_psd_curves) {
  psd_df <- read_csv(psd_file, show_col_types = FALSE)
  message("  psd_curves.csv: ", nrow(psd_df), " rows")
}

# --- Load config ---
config <- read_yaml(config_path)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))
message("Bands: ", paste(names(config$bands), collapse = ", "))

# --- Run LMMs for each power type ---
power_types <- c("relative", "dB")

all_omnibus <- list()
all_posthoc <- list()
all_omnibus_region <- list()
all_posthoc_region <- list()

for (ptype in power_types) {
  message("\n=== Power type: ", ptype, " ===")

  # --- ROI-level omnibus ---
  message("\nRunning ROI-level omnibus LMM (group * roi)...")
  omnibus <- run_omnibus_lmm(band_df, config$contrasts, config$bands, power_type = ptype)
  all_omnibus[[ptype]] <- omnibus

  if (nrow(omnibus) > 0) {
    message("\n  === ROI-Level Omnibus (", ptype, ") ===")
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

  # --- ROI-level post-hoc ---
  message("Running ROI-level post-hoc emmeans...")
  posthoc <- run_posthoc_emmeans(band_df, config$contrasts, config$bands, omnibus,
                                  power_type = ptype)
  all_posthoc[[ptype]] <- posthoc

  if (nrow(posthoc) > 0) {
    sig_count <- sum(posthoc$significant, na.rm = TRUE)
    message("  ", nrow(posthoc), " ROI contrasts, ", sig_count, " significant")
  } else {
    message("  No post-hoc tests (no significant omnibus effects)")
  }

  # --- Region-level (if roi_categories defined) ---
  if (length(config$roi_categories) > 0) {
    message("Running region-level omnibus LMM (group * region)...")
    omnibus_reg <- run_omnibus_lmm_region(band_df, config$contrasts, config$bands,
                                           config$roi_categories, power_type = ptype)
    all_omnibus_region[[ptype]] <- omnibus_reg

    if (nrow(omnibus_reg) > 0) {
      message("\n  === Region-Level Omnibus (", ptype, ") ===")
      for (i in seq_len(nrow(omnibus_reg))) {
        row <- omnibus_reg[i, ]
        grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
        int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
        message(sprintf("  %s | %s: group F=%.2f q=%.4f%s | interaction F=%.2f q=%.4f%s",
                        row$contrast, row$band,
                        row$group_F, row$group_q, grp_sig,
                        row$interaction_F, row$interaction_q, int_sig))
      }
    }

    message("Running region-level post-hoc emmeans...")
    posthoc_reg <- run_posthoc_emmeans_region(band_df, config$contrasts, config$bands,
                                               config$roi_categories, omnibus_reg,
                                               power_type = ptype)
    all_posthoc_region[[ptype]] <- posthoc_reg

    if (nrow(posthoc_reg) > 0) {
      sig_count <- sum(posthoc_reg$significant, na.rm = TRUE)
      message("  ", nrow(posthoc_reg), " region contrasts, ", sig_count, " significant")
    } else {
      message("  No region post-hoc tests (no significant omnibus effects)")
    }
  }
}

# --- Combine results across power types ---
omnibus_df <- bind_rows(all_omnibus)
posthoc_df <- bind_rows(all_posthoc)
omnibus_region_df <- bind_rows(all_omnibus_region)
posthoc_region_df <- bind_rows(all_posthoc_region)

# --- Export tables ---
message("\nExporting tables...")
if (nrow(omnibus_df) > 0) {
  write_csv(omnibus_df, file.path(tbl_dir, "psd_omnibus.csv"))
  message("  Saved: tables/psd_omnibus.csv")
}
if (nrow(posthoc_df) > 0) {
  write_csv(posthoc_df, file.path(tbl_dir, "psd_posthoc_roi.csv"))
  message("  Saved: tables/psd_posthoc_roi.csv")
}
if (nrow(omnibus_region_df) > 0) {
  write_csv(omnibus_region_df, file.path(tbl_dir, "psd_omnibus_region.csv"))
  message("  Saved: tables/psd_omnibus_region.csv")
}
if (nrow(posthoc_region_df) > 0) {
  write_csv(posthoc_region_df, file.path(tbl_dir, "psd_posthoc_region.csv"))
  message("  Saved: tables/psd_posthoc_region.csv")
}

# --- Figures ---
message("\nGenerating figures...")

# Band power boxplots (3 power types)
for (ptype in c("relative", "absolute", "dB")) {
  plot_band_power_box(band_df, group_colors, group_labels, group_order,
                      fig_dir, power_type = ptype)
}

# Regional heatmaps
if (length(config$roi_categories) > 0) {
  plot_regional_heatmap(band_df, config$roi_categories, group_colors,
                        group_labels, group_order, fig_dir)
}

# PSD curves by region
if (has_psd_curves && length(config$roi_categories) > 0) {
  plot_psd_by_region(psd_df, config$roi_categories, group_colors,
                     group_labels, group_order, fig_dir)
}

# Post-hoc figures (ROI-level and region-level)
if (nrow(posthoc_df) > 0) {
  plot_roi_forest(posthoc_df, fig_dir)
  plot_significance_heatmap(posthoc_df, fig_dir)
}
if (nrow(posthoc_region_df) > 0) {
  plot_region_forest(posthoc_region_df, fig_dir)
  plot_region_significance_heatmap(posthoc_region_df, fig_dir)
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
              omnibus_region_df = omnibus_region_df,
              posthoc_region_df = posthoc_region_df)

message("\nDone. Output: ", output_dir)
