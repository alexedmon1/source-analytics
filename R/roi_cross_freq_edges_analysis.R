#!/usr/bin/env Rscript
# roi_cross_freq_edges_analysis.R — AAC / PPC (edge-level cross-frequency) statistics
#
# Called by Python (roi_cross_freq) after roi_pac_analysis.R:
#   Rscript R/roi_cross_freq_edges_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads aac_edges.csv and/or ppc_edges.csv exported by Python (one row per
# subject x freq_pair x roi_x x roi_y; the roi_x slow band is paired with the
# roi_y fast band, so the matrix is ASYMMETRIC and roi_x == roi_y is the local,
# within-ROI coupling). The declared design:/hypotheses: are tested at three
# tiers, mirroring roi_directed:
#   1. Global   — per-subject mean over all ROI x ROI cells, one cell per freq_pair
#                 (directed-edge adapter with a single synthetic "(global)" edge)
#   2. Edges    — mass-univariate per ordered roi_x -> roi_y cell
#                 (directed-edge adapter; FDR across the edge family)
#   3. Region   — emmeans-tabular over directed region pairs (roi_categories)
# The band axis is freq_pair (slow-fast), not the study bands, so the explicit
# freq_pair level vector is passed as `bands` (same pattern as roi_pac_analysis.R).
#
# Output tables (per metric m in {aac, ppc}):
#   roi_cross_freq_<m>_global_hypotheses.csv
#   roi_cross_freq_<m>_directed_edges_hypotheses.csv
#   roi_cross_freq_<m>_region_hypotheses.csv      (roi_categories only)
# Figure: roi_cross_freq_<m>_global_bar.png
# Report: appends an AAC/PPC section to ANALYSIS_SUMMARY.md (written by the PAC
# script), or creates it when the PAC script did not run.

library(argparse)
library(yaml)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

script_dir <- if (exists("script.dir")) {
  script.dir
} else {
  tryCatch({
    args_all <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args_all, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("^--file=", "", file_arg)))
    } else {
      "R"
    }
  }, error = function(e) "R")
}

source(file.path(script_dir, "stats_utils.R"))
source(file.path(script_dir, "hypothesis.R"))

has_lme4 <- requireNamespace("lme4", quietly = TRUE) &&
            requireNamespace("lmerTest", quietly = TRUE) &&
            requireNamespace("emmeans", quietly = TRUE)

EDGE_METRICS <- c("aac", "ppc")
METRIC_TITLE <- c(aac = "Amplitude-Amplitude Coupling (AAC)",
                  ppc = "n:m Phase-Phase Coupling (PPC)")

theme_pub <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92"),
      strip.text = element_text(face = "bold", size = base_size),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = base_size + 2)
    )
}

# --- Helpers ---------------------------------------------------------------

#' Per-subject mean over all ROI x ROI cells, per freq_pair, per DV.
compute_global_edges <- function(edges, dv_cols) {
  edges %>%
    group_by(subject, group, freq_pair) %>%
    summarise(across(all_of(dv_cols), ~ mean(.x, na.rm = TRUE), .names = "mean_{.col}"),
              n_cells = n(), .groups = "drop")
}

#' Map roi_x -> roi_y cells onto DIRECTED region pairs (slow-region -> fast-region)
#' and average within subject x freq_pair x region_pair.
aggregate_cells_to_region_pairs <- function(edges, roi_categories, dv_cols) {
  roi_to_region <- data.frame(
    roi = unlist(roi_categories),
    region = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )
  edges %>%
    inner_join(roi_to_region, by = c("roi_x" = "roi")) %>%
    rename(region_x = region) %>%
    inner_join(roi_to_region, by = c("roi_y" = "roi")) %>%
    rename(region_y = region) %>%
    mutate(region_pair = paste(region_x, "->", region_y)) %>%
    group_by(subject, group, freq_pair, region_pair) %>%
    summarise(across(all_of(dv_cols), ~ mean(.x, na.rm = TRUE)),
              n_cells = n(), .groups = "drop")
}

plot_global_bar <- function(global_df, dv, metric, group_colors, group_labels,
                            group_order, output_dir) {
  mean_col <- paste0("mean_", dv)
  if (!mean_col %in% names(global_df)) return(invisible(NULL))
  plot_data <- global_df %>%
    filter(group %in% group_order) %>%
    mutate(value = .data[[mean_col]],
           group_label = factor(group_labels[group], levels = group_labels[group_order]))
  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  p <- ggplot(plot_data, aes(x = freq_pair, y = value, fill = group_label)) +
    geom_boxplot(width = 0.6, alpha = 0.7, position = position_dodge(0.8),
                 outlier.shape = NA) +
    geom_jitter(aes(color = group_label),
                position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.1),
                size = 1.5, alpha = 0.5, show.legend = FALSE) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    labs(x = "Frequency pair (slow-fast)",
         y = paste0("Global ", toupper(dv), " (mean of all ROI x ROI cells)"),
         title = paste0("Global ", METRIC_TITLE[metric], " by Frequency Pair and Group")) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  n_pairs <- length(unique(plot_data$freq_pair))
  fname <- paste0("roi_cross_freq_", metric, "_global_bar",
                  if (dv == metric) "" else paste0("_", sub(paste0("^", metric, "_"), "", dv)),
                  ".png")
  ggsave(file.path(output_dir, fname), p, width = max(8, 2 * n_pairs), height = 5, dpi = 300)
  message("  Saved: ", fname)
}

# --- Argument parsing --------------------------------------------------------

parser <- ArgumentParser(description = "AAC / PPC edge-level cross-frequency statistics (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing aac_edges.csv / ppc_edges.csv")
parser$add_argument("--config", required = TRUE, help = "Path to study YAML config")
parser$add_argument("--output-dir", required = TRUE,
                    help = "Root output directory for this analysis")
parser$add_argument("--fig-dir", default = NULL,
                    help = "Directory for figures (default: output-dir/figures)")
parser$add_argument("--tbl-dir", default = NULL,
                    help = "Directory for tables (default: output-dir/tables)")
parser$add_argument("--figures-only", action = "store_true", default = FALSE,
                    help = "Skip statistics; regenerate figures from existing data/tables")
parser$add_argument("--no-figures", action = "store_true", default = FALSE,
                    help = "Skip all figure generation (stats/tables only)")
parser$add_argument("--roi-categories", default = NULL,
                    help = "Path to roi_categories.yaml (atlas ROI groupings)")
parser$add_argument("--hypothesis", default = NULL,
                    help = "Comma-separated declared hypothesis name(s) to run (default: all)")
parser$add_argument("--metric", default = NULL,
                    help = "Comma-separated subset of aac,ppc (default: every edge CSV present)")
args <- parser$parse_args()

data_dir <- args$data_dir
output_dir <- args$output_dir
figures_only <- args$figures_only
no_figures <- args$no_figures
if (no_figures) ggsave <- function(...) invisible(NULL)

fig_dir <- if (!is.null(args$fig_dir)) args$fig_dir else file.path(output_dir, "figures")
tbl_dir <- if (!is.null(args$tbl_dir)) args$tbl_dir else file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

config <- read_yaml(args$config)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order
message("Study: ", config$name)

if (!is.null(args$roi_categories) && file.exists(args$roi_categories)) {
  rc <- read_yaml(args$roi_categories)
  if (length(rc) == 1 && identical(names(rc), "roi_categories")) rc <- rc[["roi_categories"]]
  config$roi_categories <- rc
  message("Loaded roi_categories from: ", args$roi_categories,
          " (", length(config$roi_categories), " regions)")
}

metrics <- if (!is.null(args$metric)) trimws(strsplit(args$metric, ",")[[1]]) else EDGE_METRICS
metrics <- intersect(EDGE_METRICS, metrics)
metrics <- metrics[file.exists(file.path(data_dir, paste0(metrics, "_edges.csv")))]
if (length(metrics) == 0) {
  message("No aac_edges.csv / ppc_edges.csv in ", data_dir, " — nothing to do.")
  quit(status = 0)
}

spec <- tryCatch(parse_design_spec(config), error = function(e) NULL)
if (!figures_only && (is.null(spec) || length(spec$hypotheses) == 0))
  stop("No design:/hypotheses: declared in config — nothing to test.")

report_lines <- character()
add <- function(...) report_lines <<- c(report_lines, paste0(...))

for (metric in metrics) {
  message("\n=== ", METRIC_TITLE[metric], " ===")
  edges <- read_csv(file.path(data_dir, paste0(metric, "_edges.csv")), show_col_types = FALSE)
  message("  ", metric, "_edges.csv: ", nrow(edges), " rows")
  dv_cols <- intersect(c(metric, paste0(metric, "_z")), names(edges))
  if (length(dv_cols) == 0) {
    message("  no '", metric, "' column — skipping")
    next
  }
  freq_pairs <- unique(as.character(edges$freq_pair))
  prefix <- paste0("roi_cross_freq_", metric)
  global_df <- compute_global_edges(edges, dv_cols)

  hyp_global <- data.frame(); hyp_edges <- data.frame(); hyp_region <- data.frame()
  if (!figures_only) {
    # 1. Global: one synthetic edge, lm(mean_dv ~ group) per freq_pair.
    message("  --- Global (hypothesis layer) ---")
    global_edges <- global_df
    global_edges$roi_x <- "(global)"; global_edges$roi_y <- "(global)"
    hyp_global <- bind_rows(lapply(dv_cols, function(dv) {
      h <- run_directed_edges(global_edges, names(spec$hypotheses), spec,
                              dv_col = paste0("mean_", dv), source_col = "roi_x",
                              target_col = "roi_y", band_col = "freq_pair",
                              bands = freq_pairs)
      if (nrow(h) > 0) h$dv <- dv
      h
    }))
    if (!is.null(args$hypothesis) && nrow(hyp_global) > 0)
      hyp_global <- hyp_global[hyp_global$hypothesis %in%
                               trimws(strsplit(args$hypothesis, ",")[[1]]), , drop = FALSE]
    if (nrow(hyp_global) > 0) {
      write_csv(hyp_global, file.path(tbl_dir, paste0(prefix, "_global_hypotheses.csv")))
      message("  Saved: ", prefix, "_global_hypotheses.csv (", nrow(hyp_global), " rows)")
    }

    # 2. Cells: mass-univariate per ordered roi_x -> roi_y.
    message("  --- ROI x ROI cells (hypothesis layer, mass-univariate) ---")
    hyp_edges <- write_module_directed_edges(
      edges, config, tbl_dir, prefix = prefix, dv_cols = dv_cols,
      source_col = "roi_x", target_col = "roi_y", band_col = "freq_pair",
      bands = freq_pairs, hypothesis = args$hypothesis)
    if (is.null(hyp_edges)) hyp_edges <- data.frame()

    # 3. Region pairs (emmeans-tabular, group * region_pair).
    if (length(config$roi_categories) > 0 && has_lme4) {
      message("  --- Directed region pairs (hypothesis layer) ---")
      region_df <- aggregate_cells_to_region_pairs(edges, config$roi_categories, dv_cols)
      message("  Aggregated to ", length(unique(region_df$region_pair)), " directed region pairs")
      hyp_region <- write_module_hypotheses(
        region_df, config, tbl_dir, prefix = paste0(prefix, "_region"),
        dv_cols = dv_cols, spatial_col = "region_pair", band_col = "freq_pair",
        bands = freq_pairs, hypothesis = args$hypothesis)
      if (is.null(hyp_region)) hyp_region <- data.frame()
    } else if (length(config$roi_categories) == 0) {
      message("  No roi_categories in config -- skipping region-pair tier")
    } else {
      message("  lme4/lmerTest/emmeans not available -- skipping region-pair tier")
    }
  }

  # Figures
  for (dv in dv_cols)
    plot_global_bar(global_df, dv, metric, group_colors, group_labels, group_order, fig_dir)

  # Report section
  if (!figures_only) {
    add("## ", METRIC_TITLE[metric], " — edge-level cross-frequency")
    add("")
    add("Cells are ROI×ROI per slow–fast frequency pair (roi_x carries the slow band, ",
        "roi_y the fast band; the matrix is asymmetric and roi_x = roi_y is the local ",
        "within-ROI coupling). DV(s): ", paste(dv_cols, collapse = ", "), ".")
    add("")
    add("**Statistics (declarative hypothesis layer):** (1) global — per-subject mean over all ",
        "cells, lm(mean ~ group) per frequency pair; (2) cells — mass-univariate contrast per ",
        "ordered roi_x→roi_y cell, FDR across the cell family per frequency pair",
        if (nrow(hyp_region) > 0) "; (3) directed region pairs — emmeans contrast over dv ~ group * region_pair + (1|subject)" else "",
        ".")
    add("")
    gc <- if (nrow(hyp_global) > 0) hyp_global[hyp_global$kind == "contrast", , drop = FALSE] else data.frame()
    if (nrow(gc) > 0) {
      add("| Hypothesis | DV | Freq pair | estimate | t | q | Hedges' g | Sig |")
      add("| --- | --- | --- | --- | --- | --- | --- | --- |")
      for (i in seq_len(nrow(gc))) {
        row <- gc[i, ]
        add(sprintf("| %s | %s | %s | %.4f | %.2f | %.4f | %.2f | %s |",
                    row$label %||% row$hypothesis, row$dv, row$band,
                    ifelse(is.na(row$estimate), 0, row$estimate),
                    ifelse(is.na(row$stat), 0, row$stat),
                    ifelse(is.na(row$q_value), 1, row$q_value),
                    ifelse(is.na(row$effect_size), 0, row$effect_size),
                    if (isTRUE(row$significant)) "**Yes**" else "No"))
      }
      add("")
    } else {
      add("*No global contrast rows.*")
      add("")
    }
    n_sig_cells <- if (nrow(hyp_edges) > 0) sum(hyp_edges$significant, na.rm = TRUE) else 0
    add("**Cells tested:** ", if (nrow(hyp_edges) > 0) length(unique(hyp_edges$spatial)) else 0,
        " per frequency pair; **significant after FDR:** ", n_sig_cells, ".")
    add("")
    add("Tables: `", prefix, "_global_hypotheses.csv`, `", prefix, "_directed_edges_hypotheses.csv`",
        if (nrow(hyp_region) > 0) paste0(", `", prefix, "_region_hypotheses.csv`") else "", ".")
    add("")
  }
}

if (!figures_only && length(report_lines) > 0) {
  summary_path <- file.path(output_dir, "ANALYSIS_SUMMARY.md")
  if (file.exists(summary_path)) {
    existing <- readLines(summary_path, warn = FALSE)
    # Replace a previous AAC/PPC block (idempotent re-runs) before appending.
    marker <- "<!-- roi_cross_freq edges -->"
    cut <- which(existing == marker)
    if (length(cut) > 0) existing <- existing[seq_len(cut[1] - 1)]
    writeLines(c(existing, marker, "", report_lines), summary_path)
    message("  Report section appended: ", summary_path)
  } else {
    writeLines(c(paste0("# ROI Cross-Frequency (AAC / PPC) — ", config$name), "",
                 paste0("**Generated:** ", format(Sys.time(), "%Y-%m-%d %H:%M")), "",
                 "<!-- roi_cross_freq edges -->", "", report_lines), summary_path)
    message("  Report written: ", summary_path)
  }
}

message("\nDone. Output: ", output_dir)
