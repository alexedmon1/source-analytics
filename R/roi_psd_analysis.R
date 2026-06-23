#!/usr/bin/env Rscript
# roi_psd_analysis.R — Main R entry point for PSD statistical analysis
#
# Called by Python: Rscript R/roi_psd_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads CSVs exported by Python (band_power.csv, psd_curves.csv),
# runs omnibus LMM + emmeans post-hoc for both relative and absolute power,
# generates ggplot2 figures, writes summary.

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

source(file.path(script_dir, "stats_utils.R"))
source(file.path(script_dir, "hypothesis.R"))
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
                    help = "Run only the named hypothesis(es) (comma-separated) from the design spec; default = all")
args <- parser$parse_args()

data_dir <- args$data_dir
config_path <- args$config
output_dir <- args$output_dir
figures_only <- args$figures_only
no_figures <- args$no_figures

if (no_figures) {
  ggsave <- function(...) invisible(NULL)
}

# Create output subdirs
fig_dir <- if (!is.null(args$fig_dir)) args$fig_dir else file.path(output_dir, "figures")
tbl_dir <- if (!is.null(args$tbl_dir)) args$tbl_dir else file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load data ---
message("Loading data...")
band_df <- read_csv(file.path(data_dir, "band_power.csv"), show_col_types = FALSE)
message("  band_power.csv: ", nrow(band_df), " rows")

# Exclude Corpus Callosum (white matter) ROIs from all analyses
cc_rois <- c("Corpus_Callosum_Genu_L", "Corpus_Callosum_Genu_R",
             "Corpus_Callosum_Body_L", "Corpus_Callosum_Body_R",
             "Corpus_Callosum_Splenium_L", "Corpus_Callosum_Splenium_R")
n_before <- nrow(band_df)
band_df <- band_df %>% filter(!roi %in% cc_rois)
message("  Excluded ", length(cc_rois), " CC ROIs: ", n_before, " -> ", nrow(band_df), " rows")

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

# Load roi_categories from atlas file if provided
if (!is.null(args$roi_categories) && file.exists(args$roi_categories)) {
  config$roi_categories <- read_yaml(args$roi_categories)
  message("Loaded roi_categories from: ", args$roi_categories,
          " (", length(config$roi_categories), " regions)")
}

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))
message("Bands: ", paste(names(config$bands), collapse = ", "))

if (!figures_only) {
  # --- Run LMMs for each power type ---
  power_types <- c("relative", "absolute")

  all_omnibus <- list()
  all_omnibus_region <- list()
  all_posthoc_region <- list()

  for (ptype in power_types) {
    message("\n=== Power type: ", ptype, " ===")

    # --- ROI-level omnibus (DIAGNOSTIC, not a hypothesis) ---
    # The group*roi omnibus + interaction F are retained as a model-fit
    # diagnostic for the report. Scientific per-contrast inference is handled
    # by the declarative hypothesis layer below (run_posthoc_emmeans retired).
    message("\nRunning ROI-level omnibus LMM (group * roi) [diagnostic]...")
    omnibus <- run_omnibus_lmm(band_df, config$contrasts, config$bands, power_type = ptype)
    all_omnibus[[ptype]] <- omnibus

    if (nrow(omnibus) > 0) {
      message("\n  === ROI-Level Omnibus (", ptype, ") [diagnostic] ===")
      for (i in seq_len(nrow(omnibus))) {
        row <- omnibus[i, ]
        grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
        int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
        message(sprintf("  %s | %s: group F=%.2f q=%.4f%s | interaction F=%.2f q=%.4f%s",
                        row$contrast, row$band,
                        row$group_F, row$group_q, grp_sig,
                        row$interaction_F, row$interaction_q, int_sig))
      }
    }

    # ROI-level per-contrast post-hoc now comes from the hypothesis layer
    # (see the write_module_hypotheses call after the loop). No legacy call here.

    # --- Region-level (if roi_categories defined) ---
    if (length(config$roi_categories) > 0) {
      message("Running region-level omnibus LMM (group * region)...")
      omnibus_reg <- run_omnibus_lmm_region(band_df, config$contrasts, config$bands,
                                             config$roi_categories, power_type = ptype)
      all_omnibus_region[[ptype]] <- omnibus_reg

      if (nrow(omnibus_reg) > 0) {
        message("\n  === Region-Level Omnibus (", ptype, ") ===")
        for (i in seq_len(nrow(omnibus_reg))) {
          row <- omnibus_reg[i, ]
          grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
          int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
          message(sprintf("  %s | %s: group F=%.2f q=%.4f%s | interaction F=%.2f q=%.4f%s",
                          row$contrast, row$band,
                          row$group_F, row$group_q, grp_sig,
                          row$interaction_F, row$interaction_q, int_sig))
        }
      }

      message("Running region-level post-hoc emmeans...")
      posthoc_reg <- run_posthoc_emmeans_region(band_df, config$contrasts, config$bands,
                                                 config$roi_categories, omnibus_reg,
                                                 power_type = ptype)
      all_posthoc_region[[ptype]] <- posthoc_reg

      if (nrow(posthoc_reg) > 0) {
        sig_count <- sum(posthoc_reg$significant, na.rm = TRUE)
        message("  ", nrow(posthoc_reg), " region contrasts, ", sig_count, " significant")
      } else {
        message("  No region post-hoc tests (no significant omnibus effects)")
      }

    }
  }

  # --- Declarative hypotheses (hypothesis layer) — ROI-level per-contrast ---
  # This is now the SOLE per-contrast inference engine for the ROI level. It
  # writes roi_psd_hypotheses.csv (native) and returns the tidy rows; the
  # legacy roi_psd_posthoc_roi.csv is rebuilt from those rows (legacy column
  # aliases via .add_legacy_aliases) so figures/report consume it unchanged.
  message("\nRunning declarative hypotheses (hypothesis layer) — ROI level...")
  hyp_roi <- write_module_hypotheses(band_df, config, tbl_dir, prefix = "roi_psd",
                                     dv_cols = power_types, spatial_col = "roi",
                                     band_col = "band", hypothesis = args$hypothesis)

  # --- Combine results across power types ---
  omnibus_df <- bind_rows(all_omnibus)
  # ROI post-hoc = the contrast-kind hypothesis rows (legacy schema via aliases).
  posthoc_df <- if (!is.null(hyp_roi) && nrow(hyp_roi) > 0)
    hyp_roi[hyp_roi$kind == "contrast", , drop = FALSE] else data.frame()
  omnibus_region_df <- bind_rows(all_omnibus_region)
  posthoc_region_df <- bind_rows(all_posthoc_region)
  # --- Export tables ---
  message("\nExporting tables...")
  if (nrow(omnibus_df) > 0) {
    write_csv(omnibus_df, file.path(tbl_dir, "roi_psd_omnibus.csv"))
    message("  Saved: tables/roi_psd_omnibus.csv (diagnostic)")
  }
  if (nrow(posthoc_df) > 0) {
    write_csv(posthoc_df, file.path(tbl_dir, "roi_psd_posthoc_roi.csv"))
    message("  Saved: tables/roi_psd_posthoc_roi.csv (", nrow(posthoc_df),
            " rows, hypothesis-derived)")
  }
  if (nrow(omnibus_region_df) > 0) {
    write_csv(omnibus_region_df, file.path(tbl_dir, "roi_psd_omnibus_region.csv"))
    message("  Saved: tables/roi_psd_omnibus_region.csv")
  }
  if (nrow(posthoc_region_df) > 0) {
    write_csv(posthoc_region_df, file.path(tbl_dir, "roi_psd_posthoc_region.csv"))
    message("  Saved: tables/roi_psd_posthoc_region.csv")
  }

  # --- Global posthoc (marginal group comparisons averaged over ROIs) ---
  message("\nComputing global posthoc (marginal group comparisons)...")
  global_posthoc_list <- list()
  for (ptype in power_types) {
    for (band_name in names(config$bands)) {
      gp_data <- band_df %>%
        filter(band == band_name,
               group %in% unlist(lapply(config$contrasts, function(c) c(c$group_a, c$group_b)))) %>%
        mutate(dv = .data[[ptype]])

      if (nrow(gp_data) == 0) next

      gp <- run_posthoc_global(gp_data, config$contrasts, spatial_col = "roi",
                                dv_col = "dv", dv_label = ptype,
                                band_label = band_name)
      if (nrow(gp) > 0) global_posthoc_list[[paste(ptype, band_name)]] <- gp
    }
  }
  global_posthoc_df <- bind_rows(global_posthoc_list)

  if (nrow(global_posthoc_df) > 0) {
    # No cross-band correction: per-band p-values reported directly
    global_posthoc_df <- global_posthoc_df %>%
      mutate(
        q_value = p_value,
        significant = q_value < 0.05,
        sig_label = sig_stars(q_value)
      )

    write_csv(global_posthoc_df, file.path(tbl_dir, "roi_psd_posthoc_global.csv"))
    message("  Saved: tables/roi_psd_posthoc_global.csv")
    sig_global <- global_posthoc_df %>% filter(significant == TRUE)
    message("  ", nrow(global_posthoc_df), " global contrasts, ", nrow(sig_global), " significant")
  }

} else {
  message("Figures-only mode: loading existing tables...")
  omnibus_df <- tryCatch(read_csv(file.path(tbl_dir, "roi_psd_omnibus.csv"), show_col_types = FALSE), error = function(e) data.frame())
  posthoc_df <- tryCatch(read_csv(file.path(tbl_dir, "roi_psd_posthoc_roi.csv"), show_col_types = FALSE), error = function(e) data.frame())
  omnibus_region_df <- tryCatch(read_csv(file.path(tbl_dir, "roi_psd_omnibus_region.csv"), show_col_types = FALSE), error = function(e) data.frame())
  posthoc_region_df <- tryCatch(read_csv(file.path(tbl_dir, "roi_psd_posthoc_region.csv"), show_col_types = FALSE), error = function(e) data.frame())
  global_posthoc_df <- tryCatch(read_csv(file.path(tbl_dir, "roi_psd_posthoc_global.csv"), show_col_types = FALSE), error = function(e) data.frame())
}

# --- Figures ---
message("\nGenerating figures...")

# Band power boxplots (with significance brackets)
for (ptype in c("relative", "absolute")) {
  ptype_sig <- if (nrow(global_posthoc_df) > 0) {
    global_posthoc_df %>% filter(dv == ptype)
  } else NULL

  plot_band_power_box(band_df, group_colors, group_labels, group_order,
                      fig_dir, power_type = ptype, sig_df = ptype_sig)
}

# Regional heatmaps
if (length(config$roi_categories) > 0) {
  plot_regional_heatmap(band_df, config$roi_categories, group_colors,
                        group_labels, group_order, fig_dir)
}

# PSD curves by region
if (has_psd_curves && length(config$roi_categories) > 0) {
  plot_psd_by_region(psd_df, config$roi_categories, group_colors,
                     group_labels, group_order, fig_dir)
}

# Post-hoc figures (ROI-level and region-level significance heatmaps)
if (nrow(posthoc_df) > 0) {
  plot_significance_heatmap(posthoc_df, fig_dir)
}
if (nrow(posthoc_region_df) > 0) {
  plot_region_significance_heatmap(posthoc_region_df, fig_dir)
}

# Band-by-region figure for significant bands
if (length(config$roi_categories) > 0) {
  # Combine region posthoc from both figures_only and normal path
  region_ph <- if (exists("posthoc_region_df") && nrow(posthoc_region_df) > 0) {
    posthoc_region_df
  } else {
    tryCatch(read_csv(file.path(tbl_dir, "roi_psd_posthoc_region.csv"), show_col_types = FALSE),
             error = function(e) data.frame())
  }

  if (nrow(region_ph) > 0) {
    sig_bands <- region_ph %>%
      filter(significant == TRUE) %>%
      distinct(band, power_type, contrast)

    for (i in seq_len(nrow(sig_bands))) {
      b <- sig_bands$band[i]
      pt <- sig_bands$power_type[i]
      ctr <- sig_bands$contrast[i]
      message("  Plotting band-by-region: ", b, " (", pt, ", ", ctr, ")")
      plot_band_by_region(band_df, config$roi_categories, group_colors,
                          group_labels, group_order, fig_dir,
                          target_band = b, power_type = pt,
                          posthoc_region_df = region_ph, contrast = ctr)
    }
  }
}

if (!figures_only) {
  # --- Summary report ---
  message("\nWriting summary...")

  n_subjects <- band_df %>%
    dplyr::distinct(subject, group) %>%
    dplyr::count(group) %>%
    { setNames(.$n, .$group) }

  sfreq <- if (!is.null(config$sfreq)) config$sfreq else 500

  write_summary(omnibus_df, posthoc_df, config, n_subjects, sfreq,
                fig_dir, file.path(output_dir, "ANALYSIS_SUMMARY.md"),
                omnibus_region_df = omnibus_region_df,
                posthoc_region_df = posthoc_region_df)
}

message("\nDone. Output: ", output_dir)
