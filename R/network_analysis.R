#!/usr/bin/env Rscript
# Network Analysis Report Generator
# Reads graph metrics and NBS results, generates ANALYSIS_SUMMARY.md

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
global_path <- file.path(data_dir, "network_global_metrics.csv")
nodal_path  <- file.path(data_dir, "network_nodal_metrics.csv")

if (!file.exists(global_path)) {
  cat("No network_global_metrics.csv found.\n")
  quit(status = 0)
}

global_metrics <- read.csv(global_path, stringsAsFactors = FALSE)

# --- Config -------------------------------------------------------------------
net_cfg <- config$network %||% list()
threshold_method <- net_cfg$threshold_method %||% "proportional"
threshold_value  <- net_cfg$threshold_value %||% 0.1
nbs_threshold    <- net_cfg$nbs_threshold %||% 3.0
nbs_perm         <- net_cfg$nbs_permutations %||% 5000

# --- Summaries ----------------------------------------------------------------
n_subjects <- length(unique(global_metrics$subject))
bands <- unique(global_metrics$band)
groups <- unique(global_metrics$group)

# Group-level global metric summaries
global_summary <- do.call(rbind, lapply(bands, function(b) {
  do.call(rbind, lapply(groups, function(g) {
    sub <- global_metrics[global_metrics$band == b & global_metrics$group == g, ]
    data.frame(
      Band = b,
      Group = g,
      Mean_Efficiency = round(mean(sub$global_efficiency), 4),
      Mean_Modularity = round(mean(sub$modularity), 4),
      Mean_SW = round(mean(sub$small_worldness), 3),
      Mean_Edges = round(mean(sub$n_edges), 1),
      stringsAsFactors = FALSE
    )
  }))
}))

# T-tests on global metrics
global_tests <- do.call(rbind, lapply(bands, function(b) {
  sub <- global_metrics[global_metrics$band == b, ]
  if (length(groups) < 2) return(NULL)
  g1 <- sub[sub$group == groups[1], ]
  g2 <- sub[sub$group == groups[2], ]

  do.call(rbind, lapply(c("global_efficiency", "modularity", "small_worldness"), function(m) {
    tryCatch({
      tt <- t.test(g1[[m]], g2[[m]])
      data.frame(
        Band = b,
        Metric = m,
        t = round(tt$statistic, 3),
        p = round(tt$p.value, 4),
        stringsAsFactors = FALSE
      )
    }, error = function(e) NULL)
  }))
}))

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
lines <- c(
  "# Network Analysis Summary",
  "",
  sprintf("**Study**: %s", config$name),
  "**Analysis**: Graph-theoretic metrics + Network-Based Statistic (NBS)",
  sprintf("**Threshold**: %s (%.2f)", threshold_method, threshold_value),
  sprintf("**NBS threshold**: t = %.1f", nbs_threshold),
  sprintf("**NBS permutations**: %d", nbs_perm),
  sprintf("**Subjects**: %d (%s)", n_subjects, paste(groups, collapse = ", ")),
  "",
  "## Methods",
  "",
  "Graph metrics (degree, clustering, betweenness, efficiency, modularity,",
  "small-worldness) were computed from thresholded imaginary coherence matrices.",
  "Group differences in nodal metrics: cluster-based permutation testing.",
  "Subnetwork identification: Network-Based Statistic (Zalesky et al., 2010).",
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
  "## Global Metrics Summary",
  "",
  "| Band | Group | Efficiency | Modularity | Small-World | Edges |",
  "|------|-------|------------|------------|-------------|-------|"
)

for (i in seq_len(nrow(global_summary))) {
  r <- global_summary[i, ]
  lines <- c(lines, sprintf(
    "| %s | %s | %.4f | %.4f | %.3f | %.1f |",
    r$Band, r$Group, r$Mean_Efficiency, r$Mean_Modularity, r$Mean_SW, r$Mean_Edges
  ))
}

# Global metric t-tests
if (!is.null(global_tests) && nrow(global_tests) > 0) {
  lines <- c(lines,
    "",
    "## Global Metric Group Comparisons",
    "",
    "| Band | Metric | t | p |",
    "|------|--------|---|---|"
  )
  for (i in seq_len(nrow(global_tests))) {
    r <- global_tests[i, ]
    lines <- c(lines, sprintf("| %s | %s | %.3f | %.4f |", r$Band, r$Metric, r$t, r$p))
  }
}

# NBS results
nbs_path <- file.path(output_dir, "tables", "nbs_results.csv")
if (file.exists(nbs_path)) {
  nbs <- read.csv(nbs_path, stringsAsFactors = FALSE)
  sig_nbs <- nbs[nbs$p_corrected < 0.05, ]
  lines <- c(lines, "", "## NBS Results", "")
  if (nrow(sig_nbs) > 0) {
    lines <- c(lines, sprintf("**%d significant subnetworks** (p < 0.05):", nrow(sig_nbs)))
    for (i in seq_len(nrow(sig_nbs))) {
      r <- sig_nbs[i, ]
      lines <- c(lines, sprintf("- %s component %d: %d edges, p = %.4f",
                                r$key, r$component, r$n_edges, r$p_corrected))
    }
  } else {
    lines <- c(lines, "No significant NBS subnetworks at p < 0.05.")
  }
}

lines <- c(lines,
  "",
  "## Output Files",
  "",
  "- `data/network_nodal_metrics.csv` — per-vertex graph metrics",
  "- `data/network_global_metrics.csv` — global graph metrics",
  "- `tables/network_stats.csv` — cluster permutation on nodal metrics",
  "- `tables/nbs_results.csv` — NBS subnetwork results",
  "- `figures/network_*.png` — nodal metric glass brains",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
