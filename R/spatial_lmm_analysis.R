#!/usr/bin/env Rscript
# Spatial LMM Analysis — primary computation module
# Fits nlme::gls with exponential spatial correlation per contrast x band x metric,
# compares to non-spatial, generates variograms, exports residuals, writes ANALYSIS_SUMMARY.md
#
# Multi-group design: iterates over contrasts from config (subsetting to 2 groups
# per contrast), consistent with stats_utils.R pattern.

suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(nlme)
})

option_list <- list(
  make_option("--data-dir", type = "character", help = "Path to data/ directory"),
  make_option("--config",   type = "character", help = "Path to study_config.yaml"),
  make_option("--output-dir", type = "character", help = "Path to output directory"),
  make_option("--fig-dir", type = "character", default = NULL,
              help = "Directory for figures (default: output-dir/figures)"),
  make_option("--tbl-dir", type = "character", default = NULL,
              help = "Directory for tables (default: output-dir/tables)")
)
opts <- parse_args(OptionParser(option_list = option_list))

data_dir    <- opts[["data-dir"]]
config_path <- opts[["config"]]
output_dir  <- opts[["output-dir"]]

config <- read_yaml(config_path)

# --- Load data ----------------------------------------------------------------
data_path <- file.path(data_dir, "spatial_lmm_data.csv")
if (!file.exists(data_path)) {
  cat("No spatial_lmm_data.csv found.\n")
  quit(status = 0)
}

dat <- read.csv(data_path, stringsAsFactors = FALSE)

slmm_cfg <- config$spatial_lmm %||% list()
corr_struct <- slmm_cfg$correlation_structure %||% "exponential"
range_mm    <- slmm_cfg$spatial_range_mm %||% 3.0

bands <- unique(dat$band)
metrics <- c("relative", "absolute")
contrasts <- config$contrasts
n_subjects_total <- length(unique(dat$subject))
n_vertices <- length(unique(dat$vertex_idx))

cat(sprintf("Spatial LMM: %d subjects total, %d bands, %d metrics, %d vertices per subject\n",
            n_subjects_total, length(bands), length(metrics), n_vertices))
cat(sprintf("Contrasts: %d\n", length(contrasts)))

# --- Fit models per contrast x band x metric ---------------------------------
fig_dir <- if (!is.null(opts[["fig-dir"]])) opts[["fig-dir"]] else file.path(output_dir, "figures")
tbl_dir <- if (!is.null(opts[["tbl-dir"]])) opts[["tbl-dir"]] else file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

model_results <- list()
all_residuals <- data.frame()
result_idx <- 0

for (contrast in contrasts) {
  cname <- contrast$name
  ga <- contrast$group_a
  gb <- contrast$group_b

  cat(sprintf("\n========== Contrast: %s (%s vs %s) ==========\n", cname, ga, gb))

  # Subset to the two groups for this contrast
  cdat <- dat[dat$group %in% c(ga, gb), ]
  cdat$group <- factor(cdat$group, levels = c(ga, gb))

  n_a <- length(unique(cdat$subject[cdat$group == ga]))
  n_b <- length(unique(cdat$subject[cdat$group == gb]))
  cat(sprintf("  n(%s)=%d, n(%s)=%d\n", ga, n_a, gb, n_b))

  for (band in bands) {
    for (metric in metrics) {
      result_idx <- result_idx + 1
      cat(sprintf("\n--- %s: %s [%s] ---\n", cname, band, metric))

      band_dat <- cdat[cdat$band == band, ]

      if (nrow(band_dat) < 10) {
        cat(sprintf("  Skipping: too few rows (%d)\n", nrow(band_dat)))
        next
      }

      if (!(metric %in% names(band_dat))) {
        cat(sprintf("  Skipping: column '%s' not found\n", metric))
        next
      }

      band_dat$response <- band_dat[[metric]]

      # Non-spatial model (baseline)
      aic_nonspatial <- NA
      bic_nonspatial <- NA
      fit_nonspatial <- NULL

      tryCatch({
        fit_nonspatial <- gls(response ~ group, data = band_dat)
        aic_nonspatial <- AIC(fit_nonspatial)
        bic_nonspatial <- BIC(fit_nonspatial)
        cat(sprintf("  Non-spatial GLS: AIC=%.1f, BIC=%.1f\n", aic_nonspatial, bic_nonspatial))
      }, error = function(e) {
        cat(sprintf("  Non-spatial GLS failed: %s\n", e$message))
      })

      # Spatial model with exponential correlation
      fit_spatial <- NULL
      convergence <- "failed"
      aic_spatial <- NA
      bic_spatial <- NA

      tryCatch({
        fit_spatial <- gls(
          response ~ group,
          data = band_dat,
          correlation = corExp(value = range_mm, form = ~ x + y + z | subject, nugget = TRUE),
          control = glsControl(maxIter = 200, msMaxIter = 200, tolerance = 1e-4)
        )
        aic_spatial <- AIC(fit_spatial)
        bic_spatial <- BIC(fit_spatial)
        convergence <- "converged"
        cat(sprintf("  Spatial GLS (corExp): AIC=%.1f, BIC=%.1f\n", aic_spatial, bic_spatial))
      }, error = function(e) {
        cat(sprintf("  Spatial GLS (corExp) failed: %s\n", e$message))

        # Fallback: without nugget
        tryCatch({
          fit_spatial <<- gls(
            response ~ group,
            data = band_dat,
            correlation = corExp(value = range_mm, form = ~ x + y + z | subject),
            control = glsControl(maxIter = 100, msMaxIter = 100)
          )
          aic_spatial <<- AIC(fit_spatial)
          bic_spatial <<- BIC(fit_spatial)
          convergence <<- "converged (no nugget)"
          cat(sprintf("  Fallback spatial GLS: AIC=%.1f, BIC=%.1f\n", aic_spatial, bic_spatial))
        }, error = function(e2) {
          cat(sprintf("  Fallback also failed: %s\n", e2$message))

          # Final fallback: GAM with spatial smooth
          tryCatch({
            library(mgcv)
            fit_gam <- gam(response ~ group + s(x, y, z, bs = "tp", k = 20),
                           data = band_dat)
            aic_spatial <<- AIC(fit_gam)
            bic_spatial <<- BIC(fit_gam)
            fit_spatial <<- fit_gam
            convergence <<- "gam_fallback"
            cat(sprintf("  GAM fallback: AIC=%.1f\n", aic_spatial))
          }, error = function(e3) {
            cat(sprintf("  GAM fallback also failed: %s\n", e3$message))
          })
        })
      })

      # Extract group effect — with 2-group subset, row 2 is the single group contrast
      coef_val <- NA
      se_val <- NA
      t_val <- NA
      p_val <- NA
      estimated_range <- NA

      if (!is.null(fit_spatial)) {
        tryCatch({
          s <- summary(fit_spatial)

          if (convergence != "gam_fallback") {
            tbl <- s$tTable
            # With 2 groups, row 2 = group effect (ga vs gb, ga is reference)
            if (nrow(tbl) >= 2) {
              coef_val <- tbl[2, "Value"]
              se_val <- tbl[2, "Std.Error"]
              t_val <- tbl[2, "t-value"]
              p_val <- tbl[2, "p-value"]
            }

            # Extract estimated spatial range
            tryCatch({
              cs <- coef(fit_spatial$modelStruct$corStruct, unconstrained = FALSE)
              estimated_range <- cs["range"]
            }, error = function(e) {})

          } else {
            # GAM summary
            ptbl <- s$p.table
            if (nrow(ptbl) >= 2) {
              coef_val <- ptbl[2, "Estimate"]
              se_val <- ptbl[2, "Std. Error"]
              t_val <- ptbl[2, "t value"]
              p_val <- ptbl[2, "Pr(>|t|)"]
            }
          }
        }, error = function(e) {
          cat(sprintf("  Summary extraction failed: %s\n", e$message))
        })

        # Extract residuals
        tryCatch({
          resids <- data.frame(
            contrast = cname,
            subject = band_dat$subject,
            vertex_idx = band_dat$vertex_idx,
            band = band,
            metric = metric,
            residual = residuals(fit_spatial),
            stringsAsFactors = FALSE
          )
          all_residuals <- rbind(all_residuals, resids)
        }, error = function(e) {})
      }

      model_results[[result_idx]] <- data.frame(
        contrast = cname,
        group_a = ga,
        group_b = gb,
        n_a = n_a,
        n_b = n_b,
        band = band,
        metric = metric,
        convergence = convergence,
        aic_spatial = aic_spatial,
        bic_spatial = bic_spatial,
        aic_nonspatial = aic_nonspatial,
        bic_nonspatial = bic_nonspatial,
        aic_improvement = ifelse(!is.na(aic_nonspatial) & !is.na(aic_spatial),
                                 aic_nonspatial - aic_spatial, NA),
        coefficient = coef_val,
        std_error = se_val,
        t_value = t_val,
        p_value = p_val,
        estimated_range_mm = estimated_range,
        stringsAsFactors = FALSE
      )

      cat(sprintf("  Group effect (%s vs %s): coef=%.4f, SE=%.4f, t=%.3f, p=%.4f\n",
                  gb, ga, coef_val, se_val, t_val, p_val))

      # Variogram plot (only for relative metric to avoid figure clutter)
      if (metric == "relative" && !is.null(fit_spatial) && convergence != "gam_fallback") {
        tryCatch({
          safe_band <- gsub(" ", "_", tolower(band))
          safe_contrast <- gsub(" ", "_", cname)
          png(file.path(fig_dir, sprintf("variogram_%s_%s.png", safe_contrast, safe_band)),
              width = 800, height = 500)
          plot(Variogram(fit_spatial, form = ~ x + y + z | subject, maxDist = 8),
               main = sprintf("Variogram — %s (relative) [%s]", band, cname))
          dev.off()
          cat(sprintf("  Saved variogram_%s_%s.png\n", safe_contrast, safe_band))
        }, error = function(e) {
          cat(sprintf("  Variogram plot failed: %s\n", e$message))
          tryCatch(dev.off(), error = function(e2) {})
        })
      }
    }
  }
}

# --- Compile results and apply FDR correction --------------------------------
results_df <- do.call(rbind, model_results)

# FDR correction across bands within each contrast x metric
if (nrow(results_df) > 0) {
  results_df <- do.call(rbind, lapply(split(results_df,
    interaction(results_df$contrast, results_df$metric, drop = TRUE)), function(sub) {
    sub$q_value <- p.adjust(sub$p_value, method = "BH")
    sub$significant <- !is.na(sub$q_value) & sub$q_value < 0.05
    sub
  }))
  rownames(results_df) <- NULL
}

write.csv(results_df, file.path(tbl_dir, "spatial_lmm_results.csv"), row.names = FALSE)
cat(sprintf("\nExported spatial_lmm_results.csv (%d rows)\n", nrow(results_df)))

if (nrow(all_residuals) > 0) {
  write.csv(all_residuals, file.path(tbl_dir, "spatial_residuals.csv"), row.names = FALSE)
  cat(sprintf("Exported spatial_residuals.csv (%d rows)\n", nrow(all_residuals)))
}

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
groups_all <- unique(dat$group)
lines <- c(
  "# Spatial LMM Analysis Summary",
  "",
  sprintf("**Study**: %s", config$name),
  "**Analysis**: Spatial Linear Mixed Effects Model (per-contrast)",
  sprintf("**Correlation structure**: %s", corr_struct),
  sprintf("**Initial spatial range**: %.1f mm", range_mm),
  sprintf("**Subjects**: %d total (%s)", n_subjects_total, paste(groups_all, collapse = ", ")),
  sprintf("**Vertices**: %d (dorsal, z >= 0)", n_vertices),
  sprintf("**Contrasts**: %d", length(contrasts)),
  ""
)

for (contrast in contrasts) {
  n_a <- length(unique(dat$subject[dat$group == contrast$group_a]))
  n_b <- length(unique(dat$subject[dat$group == contrast$group_b]))
  lines <- c(lines, sprintf("- **%s**: %s (n=%d) vs %s (n=%d)",
                             contrast$name, contrast$group_a, n_a, contrast$group_b, n_b))
}

lines <- c(lines,
  "",
  "## Methods",
  "",
  "Spatial generalized least squares (nlme::gls) was used to model vertex-level",
  "band power as a function of group, with an exponential spatial correlation",
  "structure (`corExp(form = ~x+y+z | subject)`). For multi-group designs,",
  "separate models are fit per contrast (subsetting to the two groups in each",
  "comparison), consistent with the pairwise approach used across all analyses.",
  "",
  "Models were compared to non-spatial GLS via AIC/BIC. FDR (BH) correction",
  "was applied across bands within each contrast x metric. Fallback to GAM with",
  "thin-plate spatial smooth (`s(x,y,z, bs=\"tp\")`) was used if GLS failed.",
  ""
)

lines <- c(lines,
  "## Model Results",
  "",
  "| Contrast | Band | Metric | Convergence | AIC Improvement | Coef | SE | t | p | q |",
  "|----------|------|--------|-------------|-----------------|------|----|---|---|---|"
)

for (i in seq_len(nrow(results_df))) {
  r <- results_df[i, ]
  lines <- c(lines, sprintf(
    "| %s | %s | %s | %s | %.1f | %.4f | %.4f | %.3f | %.4f | %.4f |",
    r$contrast, r$band, r$metric, r$convergence,
    ifelse(is.na(r$aic_improvement), NA, r$aic_improvement),
    ifelse(is.na(r$coefficient), NA, r$coefficient),
    ifelse(is.na(r$std_error), NA, r$std_error),
    ifelse(is.na(r$t_value), NA, r$t_value),
    ifelse(is.na(r$p_value), NA, r$p_value),
    ifelse(is.na(r$q_value), NA, r$q_value)
  ))
}

# Significant results
sig_results <- results_df[!is.na(results_df$q_value) & results_df$q_value < 0.05, ]
if (nrow(sig_results) > 0) {
  lines <- c(lines, "",
    sprintf("**Significant results (q < 0.05): %d**", nrow(sig_results)), "")
  for (i in seq_len(nrow(sig_results))) {
    r <- sig_results[i, ]
    lines <- c(lines, sprintf("- **%s** %s [%s]: t=%.3f, p=%.4f, q=%.4f",
                               r$band, r$metric, r$contrast, r$t_value, r$p_value, r$q_value))
  }
} else {
  lines <- c(lines, "", "No results reached significance at q < 0.05 (FDR-corrected).")
}

# Spatial range estimates (relative metric only, for clarity)
range_results <- results_df[!is.na(results_df$estimated_range_mm) & results_df$metric == "relative", ]
if (nrow(range_results) > 0) {
  lines <- c(lines, "", "## Estimated Spatial Ranges (relative metric)", "")
  for (i in seq_len(nrow(range_results))) {
    r <- range_results[i, ]
    lines <- c(lines, sprintf("- **%s** [%s]: %.2f mm", r$band, r$contrast, r$estimated_range_mm))
  }
}

lines <- c(lines,
  "",
  "## Output Files",
  "",
  "- `data/spatial_lmm_data.csv` — per-subject per-vertex band power with coordinates",
  "- `tables/spatial_lmm_results.csv` — GLS model results per contrast x band x metric",
  "- `tables/spatial_residuals.csv` — spatial model residuals",
  "- `figures/variogram_*.png` — empirical vs fitted variograms",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
