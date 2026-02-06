#!/usr/bin/env Rscript
# MVPA Analysis Report Generator
# Reads MVPA classification results and generates ANALYSIS_SUMMARY.md

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
results_path <- file.path(output_dir, "tables", "mvpa_results.csv")
if (!file.exists(results_path)) {
  cat("No mvpa_results.csv found.\n")
  quit(status = 0)
}

results <- read.csv(results_path, stringsAsFactors = FALSE)

mvpa_cfg <- config$mvpa %||% list()
classifier <- mvpa_cfg$classifier %||% "svm_linear"
cv_method  <- mvpa_cfg$cv_method %||% "loocv"
n_perm     <- mvpa_cfg$n_permutations %||% 1000

# --- Features info -----------------------------------------------------------
features_path <- file.path(data_dir, "mvpa_features.csv")
n_subjects <- 0
n_features <- 0
if (file.exists(features_path)) {
  feats <- read.csv(features_path, stringsAsFactors = FALSE)
  n_subjects <- length(unique(feats$subject))
  n_features <- length(unique(feats$vertex_idx))
}

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
lines <- c(
  "# MVPA Analysis Summary",
  "",
  sprintf("**Study**: %s", config$name),
  "**Analysis**: Multivariate Pattern Analysis (MVPA)",
  sprintf("**Classifier**: %s", classifier),
  sprintf("**CV method**: %s", cv_method),
  sprintf("**Permutations**: %d", n_perm),
  sprintf("**Subjects**: %d", n_subjects),
  sprintf("**Features (vertices)**: %d", n_features),
  "",
  "## Methods",
  "",
  "Linear SVM with LOOCV was used to classify groups based on whole-brain",
  "spatial patterns of relative band power. Statistical significance was",
  "assessed via permutation testing (shuffled group labels).",
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
  "## Classification Results",
  "",
  "| Band | Accuracy | p-value | Sensitivity | Specificity | AUC | 95% CI |",
  "|------|----------|---------|-------------|-------------|-----|--------|"
)

for (i in seq_len(nrow(results))) {
  r <- results[i, ]
  lines <- c(lines, sprintf(
    "| %s | %.1f%% | %.4f | %.1f%% | %.1f%% | %.3f | [%.1f%%, %.1f%%] |",
    r$band, r$accuracy * 100, r$p_value,
    r$sensitivity * 100, r$specificity * 100, r$auc,
    r$ci_lower * 100, r$ci_upper * 100
  ))
}

# Highlight significant bands
sig_bands <- results[results$p_value < 0.05, ]
if (nrow(sig_bands) > 0) {
  lines <- c(lines, "",
    sprintf("**Significant bands (p < 0.05)**: %s",
            paste(sig_bands$band, collapse = ", ")))
} else {
  lines <- c(lines, "", "No bands reached significance at p < 0.05.")
}

lines <- c(lines,
  "",
  "## Output Files",
  "",
  "- `data/mvpa_features.csv` — feature matrix",
  "- `tables/mvpa_results.csv` — classification accuracy per band",
  "- `figures/mvpa_importance_*.png` — feature importance glass brains",
  "- `figures/mvpa_null_*.png` — permutation null distributions",
  "- `figures/mvpa_confusion_*.png` — confusion matrices",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
