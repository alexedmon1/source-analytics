#!/usr/bin/env Rscript
# TFCE Analysis Report Generator
# Reads pre-computed TFCE stats from Python and generates ANALYSIS_SUMMARY.md

suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
})

# --- CLI args ----------------------------------------------------------------
option_list <- list(
  make_option("--data-dir", type = "character", help = "Path to data/ directory"),
  make_option("--config",   type = "character", help = "Path to study_config.yaml"),
  make_option("--output-dir", type = "character", help = "Path to output directory")
)
opts <- parse_args(OptionParser(option_list = option_list))

data_dir   <- opts[["data-dir"]]
config_path <- opts[["config"]]
output_dir  <- opts[["output-dir"]]

# --- Load data ----------------------------------------------------------------
config <- read_yaml(config_path)

tfce_stats_path <- file.path(output_dir, "tables", "tfce_stats.csv")
if (!file.exists(tfce_stats_path)) {
  cat("No tfce_stats.csv found — nothing to report.\n")
  quit(status = 0)
}

tfce_stats <- read.csv(tfce_stats_path, stringsAsFactors = FALSE)

# --- TFCE config --------------------------------------------------------------
tfce_cfg <- config$tfce %||% list()
n_perm <- tfce_cfg$n_permutations %||% 1000
E_val  <- tfce_cfg$E %||% 0.5
H_val  <- tfce_cfg$H %||% 2.0
dh_val <- tfce_cfg$dh %||% 0.1

wb_cfg <- config$wholebrain %||% list()
adj_dist <- wb_cfg$adjacency_distance_mm %||% 5.0

# --- Compute summaries -------------------------------------------------------
n_vertices <- length(unique(tfce_stats$vertex_idx))
n_bands    <- length(unique(tfce_stats$band))
bands      <- unique(tfce_stats$band)
metrics    <- unique(tfce_stats$metric)

sig_stats <- tfce_stats[tfce_stats$p_corrected < 0.05, ]

# Per-band-metric summary
band_summary <- do.call(rbind, lapply(bands, function(b) {
  do.call(rbind, lapply(metrics, function(m) {
    sub <- tfce_stats[tfce_stats$band == b & tfce_stats$metric == m, ]
    sig_sub <- sub[sub$p_corrected < 0.05, ]
    data.frame(
      Band = b,
      Metric = m,
      N_Significant = nrow(sig_sub),
      N_Total = nrow(sub),
      Mean_TFCE = round(mean(abs(sub$tfce_score)), 3),
      Max_TFCE = round(max(abs(sub$tfce_score)), 3),
      Mean_Hedges_g = round(mean(abs(sub$hedges_g)), 3),
      Max_Hedges_g = round(max(abs(sub$hedges_g)), 3),
      stringsAsFactors = FALSE
    )
  }))
}))

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
lines <- c(
  "# TFCE Analysis Summary",
  "",
  sprintf("**Study**: %s", config$name),
  "**Analysis**: Threshold-Free Cluster Enhancement (TFCE)",
  sprintf("**Parameters**: E=%.1f, H=%.1f, dh=%.1f", E_val, H_val, dh_val),
  sprintf("**Permutations**: %d", n_perm),
  sprintf("**Adjacency distance**: %.1f mm", adj_dist),
  sprintf("**Vertices**: %d", n_vertices),
  "",
  "## Methods",
  "",
  "TFCE (Smith & Nichols, 2009) was applied to vertex-level band power maps.",
  "TFCE integrates cluster extent and height across all possible thresholds,",
  "eliminating the need for an arbitrary cluster-forming threshold.",
  "Statistical significance was assessed via permutation testing of max|TFCE|.",
  ""
)

# Epoch sampling info
epoch_cfg <- wb_cfg$epoch_sampling
if (!is.null(epoch_cfg) && isTRUE(epoch_cfg$enabled)) {
  lines <- c(lines,
    sprintf("**Epoch sampling**: %d epochs of %.1fs (seed=%d)",
            epoch_cfg$n_epochs, epoch_cfg$epoch_duration_sec, epoch_cfg$seed),
    ""
  )
}

lines <- c(lines,
  "## Results",
  "",
  sprintf("Total significant vertices (p < 0.05): **%d** across all bands/metrics",
          nrow(sig_stats)),
  "",
  "### Per-Band Summary",
  "",
  "| Band | Metric | Sig. Vertices | Total | Mean |TFCE| | Max |TFCE| | Mean |g| | Max |g| |",
  "|------|--------|---------------|-------|-------------|------------|----------|---------|"
)

for (i in seq_len(nrow(band_summary))) {
  r <- band_summary[i, ]
  lines <- c(lines, sprintf(
    "| %s | %s | %d | %d | %.3f | %.3f | %.3f | %.3f |",
    r$Band, r$Metric, r$N_Significant, r$N_Total,
    r$Mean_TFCE, r$Max_TFCE, r$Mean_Hedges_g, r$Max_Hedges_g
  ))
}

lines <- c(lines,
  "",
  "## Output Files",
  "",
  "- `data/tfce_band_power.csv` — per-subject per-vertex band power",
  "- `data/source_coords.csv` — vertex coordinates (mm)",
  "- `tables/tfce_stats.csv` — per-vertex TFCE scores and corrected p-values",
  "- `figures/tfce_scores_*.png` — TFCE score glass brain maps",
  "- `figures/tfce_significant_*.png` — significant vertex maps",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
