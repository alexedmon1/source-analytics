#!/usr/bin/env Rscript
# psd_analysis.R — Main R entry point for PSD statistical analysis
#
# Called by Python: Rscript R/psd_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads CSVs exported by Python (band_power.csv, psd_curves.csv),
# runs LMM statistics, generates ggplot2 figures, writes summary.

library(argparse)
library(yaml)
library(readr)

# Resolve script directory for sourcing helpers
script_dir <- if (exists("script.dir")) {
  script.dir
} else {
  tryCatch({
    # When called via Rscript
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

# --- Statistics ---
message("\nRunning statistics...")
stats_df <- run_band_contrasts(band_df, config$contrasts, config$bands)

if (nrow(stats_df) > 0) {
  write_csv(stats_df, file.path(tbl_dir, "psd_statistics.csv"))
  message("  Saved: tables/psd_statistics.csv")

  # Print summary
  message("\n  === Results ===")
  for (i in seq_len(nrow(stats_df))) {
    row <- stats_df[i, ]
    sig_marker <- if (!is.na(row$significant) && row$significant) " ***" else ""
    message(sprintf("  %s | %s: t=%.2f, p=%.4f, q=%.4f, g=%.2f, LMM z=%.2f, LMM p=%.4f%s",
                    row$contrast, row$band,
                    row$t_stat, row$p_value, row$q_value, row$hedges_g,
                    row$lmm_z, row$lmm_p, sig_marker))
  }
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

# PSD curves by region (if psd_curves.csv available)
if (has_psd_curves && length(config$roi_categories) > 0) {
  plot_psd_by_region(psd_df, config$roi_categories, group_colors,
                     group_labels, group_order, fig_dir)
}

# --- Summary report ---
message("\nWriting summary...")

# Count subjects per group
n_subjects <- band_df %>%
  dplyr::distinct(subject, group) %>%
  dplyr::count(group) %>%
  { setNames(.$n, .$group) }

# Get sfreq from config (set by Python) or default
sfreq <- if (!is.null(config$sfreq)) config$sfreq else 500

write_summary(stats_df, config, n_subjects, sfreq,
              fig_dir, file.path(output_dir, "ANALYSIS_SUMMARY.md"))

message("\nDone. Output: ", output_dir)
