#!/usr/bin/env Rscript
# wholebrain_analysis.R — Summary report generation for whole-brain analysis
#
# Called by Python: Rscript R/wholebrain_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads pre-computed CSV tables from Python (voxelwise_stats.csv, cluster_results.csv,
# wholebrain_values.csv, wholebrain_features.csv) and generates formatted summary
# tables + ANALYSIS_SUMMARY.md.
#
# Statistics and figures are handled in Python (cluster permutation + glass brain),
# so this script focuses on report generation.

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

# Source report utilities if available
report_file <- file.path(script_dir, "report.R")
if (file.exists(report_file)) {
  source(report_file)
}

# --- Argument parsing ---
parser <- ArgumentParser(description = "Whole-brain analysis report (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing wholebrain CSVs")
parser$add_argument("--config", required = TRUE,
                    help = "Path to study YAML config")
parser$add_argument("--output-dir", required = TRUE,
                    help = "Root output directory")
args <- parser$parse_args()

data_dir <- args$data_dir
config_path <- args$config
output_dir <- args$output_dir

tbl_dir <- file.path(output_dir, "tables")
fig_dir <- file.path(output_dir, "figures")
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load config ---
config <- read_yaml(config_path)
group_labels <- unlist(config$groups)
group_order <- config$group_order
wb_config <- config$wholebrain

message("Study: ", config$name)
message("Whole-brain report generation")

# --- Load data ---
cluster_file <- file.path(tbl_dir, "cluster_results.csv")
voxelwise_file <- file.path(tbl_dir, "voxelwise_stats.csv")
values_file <- file.path(data_dir, "wholebrain_values.csv")
features_file <- file.path(data_dir, "wholebrain_features.csv")

has_clusters <- file.exists(cluster_file)
has_voxelwise <- file.exists(voxelwise_file)
has_values <- file.exists(values_file)
has_features <- file.exists(features_file)

if (has_clusters) {
  cluster_df <- read_csv(cluster_file, show_col_types = FALSE)
  message("  cluster_results.csv: ", nrow(cluster_df), " rows")
}

if (has_voxelwise) {
  voxelwise_df <- read_csv(voxelwise_file, show_col_types = FALSE)
  message("  voxelwise_stats.csv: ", nrow(voxelwise_df), " rows")
}

if (has_values) {
  values_df <- read_csv(values_file, show_col_types = FALSE)
  message("  wholebrain_values.csv: ", nrow(values_df), " rows")
  n_subjects <- length(unique(values_df$subject))
  n_vertices <- length(unique(values_df$vertex_idx))
  n_bands <- length(unique(values_df$band))
  message("  Subjects: ", n_subjects, ", Vertices: ", n_vertices, ", Bands: ", n_bands)
}

if (has_features) {
  features_df <- read_csv(features_file, show_col_types = FALSE)
  message("  wholebrain_features.csv: ", nrow(features_df), " rows")
}

# --- Compute effect size summaries ---
effect_summary <- NULL
if (has_voxelwise) {
  effect_summary <- voxelwise_df %>%
    dplyr::group_by(contrast, band, metric) %>%
    dplyr::summarise(
      mean_t = mean(t, na.rm = TRUE),
      max_abs_t = max(abs(t), na.rm = TRUE),
      mean_hedges_g = mean(hedges_g, na.rm = TRUE),
      max_abs_hedges_g = max(abs(hedges_g), na.rm = TRUE),
      n_nominal_sig = sum(p < 0.05, na.rm = TRUE),
      .groups = "drop"
    )

  write_csv(effect_summary, file.path(tbl_dir, "effect_size_summary.csv"))
  message("Exported effect_size_summary.csv")
}

# --- Generate ANALYSIS_SUMMARY.md ---
n_perms <- ifelse(is.null(wb_config$n_permutations), 1000, wb_config$n_permutations)
cluster_thresh <- ifelse(is.null(wb_config$cluster_threshold), 2.0, wb_config$cluster_threshold)
adj_dist <- ifelse(is.null(wb_config$adjacency_distance_mm), 5.0, wb_config$adjacency_distance_mm)

md_lines <- c(
  "# Whole-Brain Analysis Summary",
  "",
  paste0("**Study**: ", config$name),
  paste0("**Analysis**: Vertex-level spectral analysis with cluster permutation testing"),
  paste0("**Date**: ", Sys.Date()),
  ""
)

if (has_values) {
  md_lines <- c(md_lines,
    "## Data Overview",
    "",
    paste0("- **Subjects**: ", n_subjects),
    paste0("- **Vertices**: ", n_vertices),
    paste0("- **Frequency bands**: ", n_bands, " (", paste(unique(values_df$band), collapse = ", "), ")"),
    paste0("- **Groups**: ", paste(sapply(group_order, function(g) {
      n <- sum(values_df$group == g) / (n_vertices * n_bands)
      paste0(group_labels[g], " (n=", n, ")")
    }), collapse = ", ")),
    ""
  )
}

md_lines <- c(md_lines,
  "## Methods",
  "",
  "Power spectral density (PSD) was computed for each source vertex using Welch's",
  "method (2-second Hann windows, 50% overlap). The following metrics were extracted:",
  "",
  "- **Relative band power**: Band power / total power per vertex",
  "- **Absolute band power (dB)**: 10 * log10(band power)",
  "- **fALFF**: High-gamma (65-100 Hz) / total (1-100 Hz) power ratio",
  "- **Spectral slope**: 1/f exponent via log-log regression (2-50 Hz)",
  "- **Peak alpha frequency**: Argmax in 6-13 Hz range",
  "",
  "Group differences were tested using independent-samples Welch's t-tests at each",
  paste0("vertex, with cluster-based permutation correction (", n_perms, " permutations,"),
  paste0("t-threshold = ", cluster_thresh, ", adjacency distance = ", adj_dist, " mm;"),
  "Maris & Oostenveld, 2007).",
  ""
)

if (has_clusters) {
  sig_clusters <- cluster_df[cluster_df$p_corrected < 0.05, ]

  md_lines <- c(md_lines,
    "## Results",
    ""
  )

  if (nrow(sig_clusters) > 0) {
    md_lines <- c(md_lines,
      paste0("### Significant Clusters (", nrow(sig_clusters), " at p < 0.05)"),
      "",
      "| Contrast | Band/Metric | Type | Vertices | Peak t | Cluster stat | p_corrected |",
      "|----------|-------------|------|----------|--------|--------------|-------------|"
    )
    for (i in seq_len(nrow(sig_clusters))) {
      row <- sig_clusters[i, ]
      md_lines <- c(md_lines, sprintf(
        "| %s | %s | %s | %d | %.2f | %.2f | %.4f |",
        row$contrast, row$band, row$metric,
        row$n_vertices, row$peak_t, row$cluster_stat, row$p_corrected
      ))
    }
    md_lines <- c(md_lines, "")
  } else {
    md_lines <- c(md_lines,
      "No significant clusters at p < 0.05.",
      ""
    )
  }

  md_lines <- c(md_lines,
    paste0("Total clusters tested: ", nrow(cluster_df)),
    ""
  )
}

if (!is.null(effect_summary)) {
  md_lines <- c(md_lines,
    "### Effect Size Summary (across all vertices)",
    "",
    "| Contrast | Band | Metric | Mean |t| | Max |t| | Mean |g| | Max |g| | N sig (uncorr) |",
    "|----------|------|--------|---------|---------|---------|---------|----------------|"
  )
  for (i in seq_len(nrow(effect_summary))) {
    row <- effect_summary[i, ]
    md_lines <- c(md_lines, sprintf(
      "| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %d |",
      row$contrast, row$band, row$metric,
      abs(row$mean_t), row$max_abs_t,
      abs(row$mean_hedges_g), row$max_abs_hedges_g,
      row$n_nominal_sig
    ))
  }
  md_lines <- c(md_lines, "")
}

md_lines <- c(md_lines,
  "## Output Files",
  "",
  "- `data/wholebrain_values.csv` -- per-subject per-vertex band power (long format)",
  "- `data/wholebrain_features.csv` -- per-subject per-vertex fALFF, spectral slope, peak alpha",
  "- `data/source_coords.csv` -- vertex coordinates in mm",
  "- `tables/voxelwise_stats.csv` -- per-vertex t-statistics, p-values, Hedges' g",
  "- `tables/cluster_results.csv` -- cluster summaries with permutation-corrected p-values",
  "- `tables/effect_size_summary.csv` -- aggregated effect sizes across vertices",
  "- `figures/wholebrain_*.png` -- glass brain visualizations per band/metric",
  "- `figures/wholebrain_summary.png` -- summary figure across all metrics",
  ""
)

summary_path <- file.path(output_dir, "ANALYSIS_SUMMARY.md")
writeLines(md_lines, summary_path)
message("Wrote ", summary_path)

message("Done.")
