#!/usr/bin/env Rscript
# electrode_comparison_analysis.R — Electrode vs Source comparison report
#
# Called by Python:
#   Rscript R/electrode_comparison_analysis.R \
#     --data-dir ... --tables-dir ... --config ... --output-dir ...
#
# Reads pre-computed CSVs from Python, generates formatted tables and
# ANALYSIS_SUMMARY.md with manuscript-ready methods and interpretation.

library(argparse)
library(yaml)
library(readr)
library(dplyr)
library(tidyr)

# --- Argument parsing ---
parser <- ArgumentParser(description = "Electrode vs Source comparison report (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing comparison_data.csv")
parser$add_argument("--tables-dir", required = TRUE,
                    help = "Directory containing comparison_stats.csv, regional_effect_sizes.csv")
parser$add_argument("--config", required = TRUE,
                    help = "Path to study YAML config")
parser$add_argument("--output-dir", required = TRUE,
                    help = "Root output directory for this analysis")
args <- parser$parse_args()

data_dir <- args$data_dir
tables_dir <- args$tables_dir
config_path <- args$config
output_dir <- args$output_dir

# --- Load data ---
message("Loading comparison data...")

config <- read_yaml(config_path)
group_labels <- unlist(config$groups)

stats_df <- read_csv(file.path(tables_dir, "comparison_stats.csv"), show_col_types = FALSE)
message("  comparison_stats.csv: ", nrow(stats_df), " rows")

comp_df <- read_csv(file.path(data_dir, "comparison_data.csv"), show_col_types = FALSE)
message("  comparison_data.csv: ", nrow(comp_df), " rows")

has_regional <- file.exists(file.path(tables_dir, "regional_effect_sizes.csv"))
if (has_regional) {
  regional_df <- read_csv(file.path(tables_dir, "regional_effect_sizes.csv"), show_col_types = FALSE)
  message("  regional_effect_sizes.csv: ", nrow(regional_df), " rows")
}

# --- Build summary report ---
lines <- character()
add <- function(...) lines <<- c(lines, paste0(...))

add("# Electrode vs Source Comparison \u2014 ", config$name)
add("")
add("**Generated:** ", format(Sys.time(), "%Y-%m-%d %H:%M"))
add("")

# Methods
add("## Methods")
add("")
n_subjects <- length(unique(comp_df$subject))
group_str <- paste(
  sapply(names(config$groups), function(g) {
    n <- sum(comp_df$group == g & comp_df$band == comp_df$band[1])
    paste0(config$groups[[g]], " (n=", n, ")")
  }),
  collapse = ", "
)
band_str <- paste(
  sapply(names(config$bands), function(b) {
    lims <- config$bands[[b]]
    paste0(b, ": ", lims[1], "-", lims[2], " Hz")
  }),
  collapse = ", "
)

add("This analysis compares group-level effect sizes between electrode-level ",
    "(scalp channel) and source-localized (ROI-level) power spectral density. ",
    "For each frequency band, subject-level power is computed by averaging ",
    "across all channels (electrode) or all ROIs (source). ",
    "Hedges' g (bias-corrected Cohen's d) quantifies the group difference. ",
    "Pearson correlations assess correspondence between electrode and source ",
    "power across subjects. Regional specificity is evaluated by comparing ",
    "per-region source effect sizes against the global electrode effect.")
add("")
add("**Groups:** ", group_str)
add("")
add("**Frequency Bands:** ", band_str)
add("")
add("**Matched Subjects:** ", n_subjects)
add("")

# Correlation table
add("## Electrode-Source Correlations")
add("")
add("| Band | Power Type | r | p | n |")
add("| --- | --- | --- | --- | --- |")
for (i in seq_len(nrow(stats_df))) {
  row <- stats_df[i, ]
  add(sprintf("| %s | %s | %.3f | %.4f | %d |",
              row$band, row$power_type,
              row$correlation_r, row$correlation_p, row$n_subjects))
}
add("")

# Effect size comparison table
add("## Effect Size Comparison (Hedges' g)")
add("")
add("| Band | Power Type | Electrode g [95% CI] | Source g [95% CI] |")
add("| --- | --- | --- | --- |")
for (i in seq_len(nrow(stats_df))) {
  row <- stats_df[i, ]
  elec_str <- sprintf("%.2f [%.2f, %.2f]",
                       row$electrode_hedges_g, row$electrode_ci_lo, row$electrode_ci_hi)
  src_str <- sprintf("%.2f [%.2f, %.2f]",
                      row$source_hedges_g, row$source_ci_lo, row$source_ci_hi)
  add(sprintf("| %s | %s | %s | %s |", row$band, row$power_type, elec_str, src_str))
}
add("")

# Regional specificity
if (has_regional) {
  add("## Regional Specificity")
  add("")
  add("Regions where source-level |Hedges' g| exceeds electrode-level |Hedges' g|:")
  add("")

  for (ptype in unique(regional_df$power_type)) {
    add("### ", ptype)
    add("")
    pt_df <- regional_df %>% filter(power_type == ptype)

    for (bname in unique(pt_df$band)) {
      bdata <- pt_df %>%
        filter(band == bname) %>%
        arrange(desc(abs(region_hedges_g)))

      exceeding <- bdata %>% filter(exceeds_electrode == TRUE)

      if (nrow(exceeding) > 0) {
        add("**", bname, "** (electrode g = ",
            sprintf("%.2f", bdata$electrode_hedges_g[1]), "):")
        add("")
        add("| Region | Source g [95% CI] | Advantage |")
        add("| --- | --- | --- |")
        for (j in seq_len(nrow(exceeding))) {
          row <- exceeding[j, ]
          advantage <- abs(row$region_hedges_g) - abs(row$electrode_hedges_g)
          add(sprintf("| %s | %.2f [%.2f, %.2f] | +%.2f |",
                      row$region, row$region_hedges_g,
                      row$region_ci_lo, row$region_ci_hi,
                      advantage))
        }
        add("")
      }
    }
  }
}

# Key findings
add("## Key Findings")
add("")

# Identify bands with strong correlations
strong_corr <- stats_df %>% filter(abs(correlation_r) > 0.5, correlation_p < 0.05)
if (nrow(strong_corr) > 0) {
  add("### Strong Electrode-Source Correlations")
  add("")
  for (i in seq_len(nrow(strong_corr))) {
    row <- strong_corr[i, ]
    add(sprintf("- **%s** (%s): r = %.2f (p = %.4f)",
                row$band, row$power_type, row$correlation_r, row$correlation_p))
  }
  add("")
}

# Identify bands where source exceeds electrode
for (ptype in unique(stats_df$power_type)) {
  pt_df <- stats_df %>% filter(power_type == ptype)
  source_advantage <- pt_df %>%
    filter(abs(source_hedges_g) > abs(electrode_hedges_g))

  if (nrow(source_advantage) > 0) {
    add("### Bands Where Source Sensitivity Exceeds Electrode (", ptype, ")")
    add("")
    for (i in seq_len(nrow(source_advantage))) {
      row <- source_advantage[i, ]
      add(sprintf("- **%s**: electrode g = %.2f, source g = %.2f (advantage: %.2f)",
                  row$band, row$electrode_hedges_g, row$source_hedges_g,
                  abs(row$source_hedges_g) - abs(row$electrode_hedges_g)))
    }
    add("")
  }
}

# Figure references
fig_dir <- file.path(output_dir, "figures")
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

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
message("  Report written: ", file.path(output_dir, "ANALYSIS_SUMMARY.md"))
message("\nDone. Output: ", output_dir)
