#!/usr/bin/env Rscript
# Vertex Connectivity Analysis Report Generator
# Reads pre-computed FCD data and statistics, generates ANALYSIS_SUMMARY.md

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
fcd_path <- file.path(data_dir, "vertex_fcd.csv")
stats_path <- file.path(output_dir, "tables", "vertex_connectivity_stats.csv")

if (!file.exists(fcd_path)) {
  cat("No vertex_fcd.csv found.\n")
  quit(status = 0)
}

fcd <- read.csv(fcd_path, stringsAsFactors = FALSE)

# --- Config -------------------------------------------------------------------
vc_cfg <- config$vertex_connectivity %||% list()
metric <- vc_cfg$metric %||% "imag_coherence"
fcd_threshold <- vc_cfg$fcd_threshold %||% 0.05
n_perm <- vc_cfg$n_permutations %||% 1000

wb_cfg <- config$wholebrain %||% list()
adj_dist <- wb_cfg$adjacency_distance_mm %||% 5.0

# --- Summaries ----------------------------------------------------------------
n_subjects <- length(unique(fcd$subject))
n_vertices <- length(unique(fcd$vertex_idx))
bands <- unique(fcd$band)
groups <- unique(fcd$group)

# Per-band FCD summary
band_summary <- do.call(rbind, lapply(bands, function(b) {
  sub <- fcd[fcd$band == b, ]
  do.call(rbind, lapply(groups, function(g) {
    g_sub <- sub[sub$group == g, ]
    data.frame(
      Band = b,
      Group = g,
      Mean_FCD = round(mean(g_sub$fcd), 4),
      SD_FCD = round(sd(g_sub$fcd), 4),
      stringsAsFactors = FALSE
    )
  }))
}))

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
lines <- c(
  "# Vertex Connectivity Analysis Summary",
  "",
  sprintf("**Study**: %s", config$name),
  sprintf("**Analysis**: All-to-all vertex connectivity (%s) + FCD", metric),
  sprintf("**FCD threshold**: %.3f", fcd_threshold),
  sprintf("**Permutations**: %d", n_perm),
  sprintf("**Subjects**: %d (%s)", n_subjects, paste(groups, collapse = ", ")),
  sprintf("**Vertices**: %d", n_vertices),
  ""
)

# Epoch info
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
  sprintf("Imaginary coherence was computed between all %d vertex pairs ", n_vertices * (n_vertices - 1) / 2),
  sprintf("using cross-spectral density (Welch's method). FCD was derived "),
  sprintf("by counting the fraction of connections > %.3f per vertex. ", fcd_threshold),
  "Group differences in FCD were tested with cluster-based permutation.",
  "",
  "## FCD Summary by Group",
  "",
  "| Band | Group | Mean FCD | SD FCD |",
  "|------|-------|----------|--------|"
)

for (i in seq_len(nrow(band_summary))) {
  r <- band_summary[i, ]
  lines <- c(lines, sprintf("| %s | %s | %.4f | %.4f |",
                            r$Band, r$Group, r$Mean_FCD, r$SD_FCD))
}

# Statistics if available
if (file.exists(stats_path)) {
  stats <- read.csv(stats_path, stringsAsFactors = FALSE)
  lines <- c(lines, "", "## Cluster Statistics", "")

  for (b in bands) {
    sub <- stats[stats$band == b, ]
    n_clust <- length(unique(sub$cluster_id[sub$cluster_id > 0]))
    lines <- c(lines, sprintf("- **%s**: %d clusters identified", b, n_clust))
  }
  lines <- c(lines, "")
}

lines <- c(lines,
  "## Output Files",
  "",
  "- `data/vertex_fcd.csv` — FCD per subject per vertex per band",
  "- `data/vertex_connectivity_matrices.pkl` — full connectivity matrices",
  "- `tables/vertex_connectivity_stats.csv` — cluster statistics",
  "- `figures/fcd_*.png` — FCD glass brain maps",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
