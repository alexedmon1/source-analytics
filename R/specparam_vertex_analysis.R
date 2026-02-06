#!/usr/bin/env Rscript
# Specparam Vertex Analysis Report Generator
# Reads vertex-level spectral parameterization results, generates ANALYSIS_SUMMARY.md

suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
})

option_list <- list(
  make_option("--data-dir", type = "character", help = "Path to data/ directory"),
  make_option("--config",   type = "character", help = "Path to study_config.yaml"),
  make_option("--output-dir", type = "character", help = "Path to output directory")
)
opts <- parse_args(OptionParser(option_list = option_list))

data_dir    <- opts[["data-dir"]]
config_path <- opts[["config"]]
output_dir  <- opts[["output-dir"]]

config <- read_yaml(config_path)

# --- Load data ----------------------------------------------------------------
param_path <- file.path(data_dir, "specparam_vertex.csv")
if (!file.exists(param_path)) {
  cat("No specparam_vertex.csv found.\n")
  quit(status = 0)
}

params <- read.csv(param_path, stringsAsFactors = FALSE)

sp_cfg <- config$specparam_vertex %||% list()
freq_range <- sp_cfg$freq_range %||% c(1, 100)
max_peaks  <- sp_cfg$max_n_peaks %||% 6

# --- Summaries ----------------------------------------------------------------
n_subjects <- length(unique(params$subject))
n_vertices <- length(unique(params$vertex_idx))
groups <- unique(params$group)

# Per-group summary of aperiodic parameters
group_summary <- do.call(rbind, lapply(groups, function(g) {
  sub <- params[params$group == g, ]
  data.frame(
    Group = g,
    Mean_Exponent = round(mean(sub$exponent, na.rm = TRUE), 3),
    SD_Exponent = round(sd(sub$exponent, na.rm = TRUE), 3),
    Mean_Offset = round(mean(sub$offset, na.rm = TRUE), 3),
    SD_Offset = round(sd(sub$offset, na.rm = TRUE), 3),
    Mean_R2 = round(mean(sub$r_squared, na.rm = TRUE), 3),
    Gamma_Peak_Rate = round(mean(sub$has_gamma_peak, na.rm = TRUE), 3),
    stringsAsFactors = FALSE
  )
}))

# Method distribution
method_table <- table(params$method)

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
lines <- c(
  "# Spectral Parameterization (Vertex-Level) Summary",
  "",
  sprintf("**Study**: %s", config$name),
  "**Analysis**: Vertex-level spectral parameterization (aperiodic + peaks)",
  sprintf("**Frequency range**: %d-%d Hz", freq_range[1], freq_range[2]),
  sprintf("**Max peaks**: %d", max_peaks),
  sprintf("**Subjects**: %d (%s)", n_subjects, paste(groups, collapse = ", ")),
  sprintf("**Vertices**: %d", n_vertices),
  ""
)

# Epoch info
wb_cfg <- config$wholebrain %||% list()
epoch_cfg <- wb_cfg$epoch_sampling
if (!is.null(epoch_cfg) && isTRUE(epoch_cfg$enabled)) {
  lines <- c(lines,
    sprintf("**Epoch sampling**: %d epochs of %.1fs",
            epoch_cfg$n_epochs, epoch_cfg$epoch_duration_sec),
    ""
  )
}

lines <- c(lines,
  "## Methods",
  "",
  "Spectral parameterization (specparam/FOOOF) was applied to the PSD at each vertex.",
  "The aperiodic component (1/f slope and offset) and oscillatory peaks were extracted.",
  "Group differences in aperiodic parameters were tested with cluster-based permutation.",
  "Gamma peak presence rates were compared with per-vertex chi-squared tests.",
  "",
  "## Fitting Methods Used",
  "",
  sprintf("- specparam: %d fits", method_table["specparam"] %||% 0),
  sprintf("- linreg: %d fits", method_table["linreg"] %||% 0),
  sprintf("- failed: %d fits", method_table["failed"] %||% 0),
  "",
  "## Group Summary",
  "",
  "| Group | Mean Exp | SD Exp | Mean Offset | SD Offset | Mean R\u00b2 | Gamma Rate |",
  "|-------|----------|--------|-------------|-----------|---------|------------|"
)

for (i in seq_len(nrow(group_summary))) {
  r <- group_summary[i, ]
  lines <- c(lines, sprintf(
    "| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |",
    r$Group, r$Mean_Exponent, r$SD_Exponent,
    r$Mean_Offset, r$SD_Offset, r$Mean_R2, r$Gamma_Peak_Rate
  ))
}

# Cluster stats
stats_path <- file.path(output_dir, "tables", "specparam_vertex_stats.csv")
if (file.exists(stats_path)) {
  stats <- read.csv(stats_path, stringsAsFactors = FALSE)
  lines <- c(lines, "", "## Cluster Permutation Results", "")

  for (param in unique(stats$parameter)) {
    sub <- stats[stats$parameter == param, ]
    cluster_ids <- unique(sub$cluster_id[sub$cluster_id > 0])
    lines <- c(lines, sprintf("- **%s**: %d clusters identified", param, length(cluster_ids)))
  }
  lines <- c(lines, "")
}

# Chi-squared results
chi2_path <- file.path(output_dir, "tables", "gamma_peak_chi2.csv")
if (file.exists(chi2_path)) {
  chi2 <- read.csv(chi2_path, stringsAsFactors = FALSE)
  n_sig <- sum(chi2$p < 0.05)
  lines <- c(lines,
    "## Gamma Peak Presence (Chi-squared)",
    "",
    sprintf("- %d/%d vertices with significant group difference (uncorrected p<0.05)",
            n_sig, nrow(chi2)),
    ""
  )
}

lines <- c(lines,
  "## Output Files",
  "",
  "- `data/specparam_vertex.csv` — per-subject per-vertex specparam fit parameters",
  "- `tables/specparam_vertex_stats.csv` — cluster permutation statistics",
  "- `tables/gamma_peak_chi2.csv` — gamma peak presence chi-squared tests",
  "- `figures/specparam_*.png` — aperiodic parameter glass brain maps",
  "- `figures/gamma_peak_presence.png` — gamma peak prevalence map",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
