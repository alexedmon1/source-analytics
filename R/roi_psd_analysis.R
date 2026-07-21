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
group_linetypes <- unlist(config$group_linetypes)
group_labels <- unlist(config$groups)
group_order <- config$group_order

# Legacy pairwise contrast list for the kept run_omnibus_lmm* DIAGNOSTIC. Derived
# from the design spec because config$contrasts is no longer populated post
# design-spec migration; falls back to config$contrasts if hypothesis.R is absent.
diag_contrasts <- tryCatch(contrasts_from_spec(parse_design_spec(config)),
                           error = function(e) config$contrasts)

# --- Profile-narrowed contrast set for the DIAGNOSTIC omnibus ---
# The declarative hypothesis layer already honors --hypothesis, but the omnibus
# builds diag_contrasts from the full design spec. Under --profile external the
# omnibus must be narrowed to the same set, or dropped contrasts (dose/route)
# leak back into roi_psd_omnibus*.csv. Filter by the same --hypothesis names.
if (!is.null(args$hypothesis) && length(diag_contrasts) > 0) {
  want <- trimws(strsplit(args$hypothesis, ",")[[1]])
  diag_contrasts <- Filter(function(ct) ct$name %in% want, diag_contrasts)
  message("Omnibus contrasts narrowed to --hypothesis set: ",
          paste(vapply(diag_contrasts, function(ct) ct$name, character(1)), collapse = ", "))
}

# --- Dependent-variable (DV) set ---
# `dvs:` in a profile block selects which power columns to analyse; default
# (exploratory) is the built-in absolute/relative pair. delta_ref (R2) is present
# only when the external profile promoted a `delta_reference` window into config.
power_types <- if (!is.null(config$dvs)) as.character(unlist(config$dvs)) else c("relative", "absolute")
missing_dv <- setdiff(power_types, names(band_df))
if (length(missing_dv) > 0) {
  message("  Requested DV(s) absent from band_power.csv, skipping: ",
          paste(missing_dv, collapse = ", "))
  power_types <- intersect(power_types, names(band_df))
}

# --- Delta-referenced power (R2): drop the reference band(s) from testing ---
# In the delta-ref DV the anchor band (Delta) is ~constant by construction, so it
# is excluded from delta_ref stats/plots. NA-out those cells (wide format: the DVs
# share rows, so we can't drop rows) — the omnibus + hypothesis fits skip all-NA
# (band x dv) families, and the boxplots filter NA before plotting.
if ("delta_ref" %in% power_types && !is.null(config$delta_reference$exclude_bands)) {
  excl <- as.character(unlist(config$delta_reference$exclude_bands))
  band_df$delta_ref[band_df$band %in% excl] <- NA_real_
  message("  delta_ref: excluded reference band(s) from testing: ", paste(excl, collapse = ", "))
}

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
  # --- Run LMMs for each power type (power_types resolved from config$dvs above) ---
  message("DVs: ", paste(power_types, collapse = ", "))

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
    omnibus <- run_omnibus_lmm(band_df, diag_contrasts, config$bands, power_type = ptype)
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

    # --- Region-level omnibus (DIAGNOSTIC, not a hypothesis) ---
    if (length(config$roi_categories) > 0) {
      message("Running region-level omnibus LMM (group * region) [diagnostic]...")
      omnibus_reg <- run_omnibus_lmm_region(band_df, diag_contrasts, config$bands,
                                             config$roi_categories, power_type = ptype)
      all_omnibus_region[[ptype]] <- omnibus_reg

      if (nrow(omnibus_reg) > 0) {
        message("\n  === Region-Level Omnibus (", ptype, ") [diagnostic] ===")
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
      # Region-level per-contrast post-hoc now comes from the hypothesis layer
      # (run_posthoc_emmeans_region retired). See after the loop.
    }
  }

  # --- Declarative hypotheses (hypothesis layer) — ROI-level per-contrast ---
  # This is now the SOLE per-contrast inference engine for the ROI level. It
  # writes roi_psd_hypotheses.csv (native) and returns the tidy rows; the
  # legacy roi_psd_posthoc_roi.csv is rebuilt from those rows (legacy column
  # aliases via .add_legacy_aliases) so figures/report consume it unchanged.
  # The hypothesis layer is now the SOLE per-contrast inference engine at all
  # three spatial granularities (roi / region / global). Each is a separate fit;
  # the legacy posthoc CSVs are rebuilt from the contrast-kind rows (legacy
  # schema via .add_legacy_aliases) so figures/report consume them unchanged.
  message("\nRunning declarative hypotheses (hypothesis layer) — ROI level...")
  hyp_roi <- write_module_hypotheses(band_df, config, tbl_dir, prefix = "roi_psd",
                                     dv_cols = power_types, spatial_col = "roi",
                                     band_col = "band", hypothesis = args$hypothesis)

  # Region level: aggregate ROIs to region means, then group*region fit.
  hyp_region <- NULL
  if (length(config$roi_categories) > 0) {
    message("Running declarative hypotheses (hypothesis layer) — region level...")
    region_df <- aggregate_to_regions(band_df, config$roi_categories)
    hyp_region <- write_module_hypotheses(region_df, config, tbl_dir,
                                          prefix = "roi_psd_region",
                                          dv_cols = power_types, spatial_col = "region",
                                          band_col = "band", hypothesis = args$hypothesis)
  }

  # Global level: marginal group contrast over ROIs (group*roi fit, emmeans ~ group).
  message("Running declarative hypotheses (hypothesis layer) — global (marginal)...")
  hyp_global <- write_module_hypotheses(band_df, config, tbl_dir,
                                        prefix = "roi_psd_global",
                                        dv_cols = power_types, spatial_col = "roi",
                                        band_col = "band", hypothesis = args$hypothesis,
                                        marginal = TRUE)

  .contrast_rows <- function(h) if (!is.null(h) && nrow(h) > 0)
    h[h$kind == "contrast", , drop = FALSE] else data.frame()

  # --- Combine results across power types ---
  omnibus_df <- bind_rows(all_omnibus)
  omnibus_region_df <- bind_rows(all_omnibus_region)
  posthoc_df <- .contrast_rows(hyp_roi)
  posthoc_region_df <- .contrast_rows(hyp_region)
  if (nrow(posthoc_region_df) > 0) posthoc_region_df$region <- posthoc_region_df$spatial
  global_posthoc_df <- .contrast_rows(hyp_global)

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
    message("  Saved: tables/roi_psd_omnibus_region.csv (diagnostic)")
  }
  if (nrow(posthoc_region_df) > 0) {
    write_csv(posthoc_region_df, file.path(tbl_dir, "roi_psd_posthoc_region.csv"))
    message("  Saved: tables/roi_psd_posthoc_region.csv (", nrow(posthoc_region_df),
            " rows, hypothesis-derived)")
  }

  if (nrow(global_posthoc_df) > 0) {
    global_posthoc_df$sig_label <- sig_stars(global_posthoc_df$q_value)
    write_csv(global_posthoc_df, file.path(tbl_dir, "roi_psd_posthoc_global.csv"))
    message("  Saved: tables/roi_psd_posthoc_global.csv (", nrow(global_posthoc_df),
            " rows, hypothesis-derived marginal)")
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

# Band power boxplots (with significance brackets). power_types comes from
# config$dvs (default relative/absolute; + delta_ref under the external profile).
for (ptype in power_types) {
  ptype_sig <- if (nrow(global_posthoc_df) > 0) {
    global_posthoc_df %>% filter(dv == ptype)
  } else NULL

  # delta_ref NA-outs the excluded reference band(s); drop those rows so the
  # boxplot doesn't render an empty facet for them.
  bdf_p <- band_df %>% filter(!is.na(.data[[ptype]]))

  plot_band_power_box(bdf_p, group_colors, group_labels, group_order,
                      fig_dir, power_type = ptype, sig_df = ptype_sig)
}

# Regional power heatmaps dropped (§6): the plain group x band x ROI grids are
# superseded by the MS1-style effect-size brain mosaics rendered in Python
# (_render_brain_mosaics -> plot_effect_size_mosaic, auto-sliced, study q_value).

# PSD curves by region — the 1/f spectra. Rendered in BOTH absolute (raw log-PSD,
# the broadband level) and delta-referenced (each spectrum ÷ its own 1–1.5 Hz
# anchor, dB) views so a broadband/offset shift (curves translate in absolute,
# collapse in delta-ref) is visually separable from a genuine spectral-shape change.
if (has_psd_curves && length(config$roi_categories) > 0) {
  plot_psd_by_region(psd_df, config$roi_categories, group_colors,
                     group_labels, group_order, fig_dir,
                     group_linetypes = group_linetypes, mode = "absolute")
  dref <- config$delta_reference
  if ("delta_ref" %in% power_types && !is.null(dref)) {
    plot_psd_by_region(psd_df, config$roi_categories, group_colors,
                       group_labels, group_order, fig_dir,
                       group_linetypes = group_linetypes, mode = "delta_ref",
                       dref_fmin = dref$fmin %||% 1.0, dref_fmax = dref$fmax %||% 1.5)
  }
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
      distinct(band, dv, hypothesis)

    for (i in seq_len(nrow(sig_bands))) {
      b <- sig_bands$band[i]
      pt <- sig_bands$dv[i]
      ctr <- sig_bands$hypothesis[i]
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
