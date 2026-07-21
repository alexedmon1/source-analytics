#!/usr/bin/env Rscript
# vertex_signature_analysis.R — Report Generator
# Reads MVPA classification results and generates ANALYSIS_SUMMARY.md

suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
})

option_list <- list(
  make_option("--data-dir", type = "character", help = "Path to data/ directory"),
  make_option("--config",   type = "character", help = "Path to study_config.yaml"),
  make_option("--output-dir", type = "character", help = "Path to output directory"),
  make_option("--no-figures", action = "store_true", default = FALSE,
              help = "Skip all figure generation")
)
opts <- parse_args(OptionParser(option_list = option_list))

no_figures <- isTRUE(opts[["no-figures"]])

if (no_figures) {
  ggsave <- function(...) invisible(NULL)
}

data_dir    <- opts[["data-dir"]]
config_path <- opts[["config"]]
output_dir  <- opts[["output-dir"]]

config <- read_yaml(config_path)

# --- Load data ----------------------------------------------------------------
results_path <- file.path(output_dir, "tables", "vertex_signature_results.csv")
if (!file.exists(results_path)) {
  cat("No vertex_signature_results.csv found.\n")
  quit(status = 0)
}

results <- read.csv(results_path, stringsAsFactors = FALSE)

sig_cfg <- config$vertex_signature %||% list()
classifiers <- sig_cfg$classifiers %||% list(sig_cfg$classifier %||% "svm_linear")
classifiers <- unlist(classifiers)
cv_method   <- sig_cfg$cv_method %||% "loocv"
n_perm      <- sig_cfg$n_permutations %||% 1000

# --- Features info -----------------------------------------------------------
features_path <- file.path(data_dir, "vertex_signature_features.csv")
n_subjects <- 0
n_features <- 0
if (file.exists(features_path)) {
  feats <- read.csv(features_path, stringsAsFactors = FALSE)
  n_subjects <- length(unique(feats$subject))
  n_features <- length(unique(feats$vertex_idx))
}

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
lines <- c(
  "# Neural Signature Analysis Summary",
  "",
  sprintf("**Study**: %s", config$name),
  "**Analysis**: Whole-brain vertex-level neural signature (classification)",
  sprintf("**Classifiers**: %s", paste(classifiers, collapse = ", ")),
  sprintf("**CV method**: %s", cv_method),
  sprintf("**Permutations**: %d", n_perm),
  sprintf("**Subjects**: %d", n_subjects),
  sprintf("**Features (vertices)**: %d", n_features),
  "",
  "## Methods",
  "",
  "Each classifier, with LOOCV, was trained to distinguish groups from whole-brain",
  "spatial patterns of relative band power. Statistical significance was assessed",
  "via permutation testing (shuffled group labels). Linear models report per-vertex",
  "feature importance; non-linear models report accuracy only.",
  ""
)

# Epoch info
wb_cfg <- config$vertex %||% list()
epoch_cfg <- wb_cfg$epoch_sampling
if (!is.null(epoch_cfg) && isTRUE(epoch_cfg$enabled)) {
  lines <- c(lines,
    sprintf("**Epoch sampling**: %d epochs of %.1fs",
            epoch_cfg$n_epochs, epoch_cfg$epoch_duration_sec),
    ""
  )
}

has_model <- "model" %in% names(results)
lines <- c(lines,
  "## Classification Results",
  "",
  "| Model | Band | Accuracy | p-value | Sensitivity | Specificity | AUC | 95% CI |",
  "|-------|------|----------|---------|-------------|-------------|-----|--------|"
)

for (i in seq_len(nrow(results))) {
  r <- results[i, ]
  model <- if (has_model) r$model else "—"
  lines <- c(lines, sprintf(
    "| %s | %s | %.1f%% | %.4f | %.1f%% | %.1f%% | %.3f | [%.1f%%, %.1f%%] |",
    model, r$band, r$accuracy * 100, r$p_value,
    r$sensitivity * 100, r$specificity * 100, r$auc,
    r$ci_lower * 100, r$ci_upper * 100
  ))
}

# Highlight significant model x band cells
sig <- results[results$p_value < 0.05, ]
if (nrow(sig) > 0) {
  lab <- if (has_model) paste(sig$model, sig$band) else sig$band
  lines <- c(lines, "",
    sprintf("**Significant (p < 0.05)**: %s", paste(lab, collapse = ", ")))
} else {
  lines <- c(lines, "", "No model reached significance at p < 0.05.")
}

lines <- c(lines,
  "",
  "## Output Files",
  "",
  "- `data/vertex_signature_features.csv` — feature matrix",
  "- `tables/vertex_signature_results.csv` — classification accuracy per band",
  "- `figures/vertex_signature_importance_*.png` — feature importance glass brains",
  "- `figures/vertex_signature_null_*.png` — permutation null distributions",
  "- `figures/vertex_signature_confusion_*.png` — confusion matrices",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
