#!/usr/bin/env Rscript
# Spatial LMM Analysis — primary computation module
# Fits nlme::gls with exponential spatial correlation, compares to non-spatial,
# generates variograms, exports residuals, writes ANALYSIS_SUMMARY.md

suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(nlme)
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
data_path <- file.path(data_dir, "spatial_lmm_data.csv")
if (!file.exists(data_path)) {
  cat("No spatial_lmm_data.csv found.\n")
  quit(status = 0)
}

dat <- read.csv(data_path, stringsAsFactors = FALSE)
dat$group <- factor(dat$group)

slmm_cfg <- config$spatial_lmm %||% list()
corr_struct <- slmm_cfg$correlation_structure %||% "exponential"
range_mm    <- slmm_cfg$spatial_range_mm %||% 3.0

bands <- unique(dat$band)
metrics <- c("relative", "dB", "absolute")
groups <- levels(dat$group)
n_subjects <- length(unique(dat$subject))

cat(sprintf("Spatial LMM: %d subjects, %d bands, %d metrics, %d vertices per subject\n",
            n_subjects, length(bands), length(metrics), length(unique(dat$vertex_idx))))

# --- Fit models per band x metric --------------------------------------------
fig_dir <- file.path(output_dir, "figures")
tbl_dir <- file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

model_results <- list()
all_residuals <- data.frame()
result_idx <- 0

for (band in bands) {
  for (metric in metrics) {
    result_idx <- result_idx + 1
    result_key <- paste(band, metric, sep = "_")
    cat(sprintf("\n--- Fitting spatial GLS for %s [%s] ---\n", band, metric))
    band_dat <- dat[dat$band == band, ]

    # Ensure enough data
    if (nrow(band_dat) < 10) {
      cat(sprintf("  Skipping %s [%s]: too few rows (%d)\n", band, metric, nrow(band_dat)))
      next
    }

    # Check metric column exists
    if (!(metric %in% names(band_dat))) {
      cat(sprintf("  Skipping %s [%s]: column not found\n", band, metric))
      next
    }

    # Create formula-friendly response column
    band_dat$response <- band_dat[[metric]]

    # Non-spatial model (baseline)
    tryCatch({
      fit_nonspatial <- gls(response ~ group, data = band_dat)
      aic_nonspatial <- AIC(fit_nonspatial)
      bic_nonspatial <- BIC(fit_nonspatial)
      cat(sprintf("  Non-spatial GLS: AIC=%.1f, BIC=%.1f\n", aic_nonspatial, bic_nonspatial))
    }, error = function(e) {
      cat(sprintf("  Non-spatial GLS failed: %s\n", e$message))
      aic_nonspatial <<- NA
      bic_nonspatial <<- NA
      fit_nonspatial <<- NULL
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

      # Fallback: simpler correlation structure
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

    # Extract group effect
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
          # Group effect is the second row (first non-intercept)
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

    model_results[[result_key]] <- data.frame(
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

    cat(sprintf("  Group effect: coef=%.4f, SE=%.4f, t=%.3f, p=%.4f\n",
                coef_val, se_val, t_val, p_val))

    # Variogram plot (only for relative to avoid figure clutter)
    if (metric == "relative" && !is.null(fit_spatial) && convergence != "gam_fallback") {
      tryCatch({
        safe_band <- gsub(" ", "_", tolower(band))
        png(file.path(fig_dir, sprintf("variogram_%s.png", safe_band)),
            width = 800, height = 500)
        plot(Variogram(fit_spatial, form = ~ x + y + z | subject, maxDist = 8),
             main = sprintf("Variogram — %s (relative)", band))
        dev.off()
        cat(sprintf("  Saved variogram_%s.png\n", safe_band))
      }, error = function(e) {
        cat(sprintf("  Variogram plot failed: %s\n", e$message))
      })
    }
  }
}

# --- Export results -----------------------------------------------------------
results_df <- do.call(rbind, model_results)
write.csv(results_df, file.path(tbl_dir, "spatial_lmm_results.csv"), row.names = FALSE)
cat(sprintf("\nExported spatial_lmm_results.csv (%d rows)\n", nrow(results_df)))

if (nrow(all_residuals) > 0) {
  write.csv(all_residuals, file.path(tbl_dir, "spatial_residuals.csv"), row.names = FALSE)
  cat(sprintf("Exported spatial_residuals.csv (%d rows)\n", nrow(all_residuals)))
}

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
lines <- c(
  "# Spatial LMM Analysis Summary",
  "",
  sprintf("**Study**: %s", config$name),
  "**Analysis**: Spatial Linear Mixed Effects Model",
  sprintf("**Correlation structure**: %s", corr_struct),
  sprintf("**Initial spatial range**: %.1f mm", range_mm),
  sprintf("**Subjects**: %d (%s)", n_subjects, paste(groups, collapse = ", ")),
  sprintf("**Vertices**: %d", length(unique(dat$vertex_idx))),
  "",
  "## Methods",
  "",
  "Spatial generalized least squares (nlme::gls) was used to model vertex-level",
  "relative band power as a function of group, with an exponential spatial",
  "correlation structure (`corExp(form = ~x+y+z | subject)`). This accounts",
  "for spatial autocorrelation and provides a single omnibus test per band,",
  "avoiding the multiple comparison problem of vertex-wise testing.",
  "",
  "Models were compared to non-spatial GLS via AIC/BIC. Fallback to GAM with",
  "thin-plate spatial smooth (`s(x,y,z, bs=\"tp\")`) was used if GLS failed.",
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
  "## Model Results",
  "",
  "| Band | Metric | Convergence | AIC (spatial) | AIC (non-spatial) | AIC Improvement | Coef | SE | t | p |",
  "|------|--------|-------------|---------------|-------------------|-----------------|------|----|---|---|"
)

for (i in seq_len(nrow(results_df))) {
  r <- results_df[i, ]
  lines <- c(lines, sprintf(
    "| %s | %s | %s | %.1f | %.1f | %.1f | %.4f | %.4f | %.3f | %.4f |",
    r$band, r$metric, r$convergence,
    ifelse(is.na(r$aic_spatial), NA, r$aic_spatial),
    ifelse(is.na(r$aic_nonspatial), NA, r$aic_nonspatial),
    ifelse(is.na(r$aic_improvement), NA, r$aic_improvement),
    ifelse(is.na(r$coefficient), NA, r$coefficient),
    ifelse(is.na(r$std_error), NA, r$std_error),
    ifelse(is.na(r$t_value), NA, r$t_value),
    ifelse(is.na(r$p_value), NA, r$p_value)
  ))
}

# Significant bands
sig_bands <- results_df[!is.na(results_df$p_value) & results_df$p_value < 0.05, ]
if (nrow(sig_bands) > 0) {
  lines <- c(lines, "",
    sprintf("**Significant bands (p < 0.05)**: %s",
            paste(sig_bands$band, collapse = ", ")))
} else {
  lines <- c(lines, "", "No bands reached significance at p < 0.05.")
}

# Spatial range estimates
range_bands <- results_df[!is.na(results_df$estimated_range_mm), ]
if (nrow(range_bands) > 0) {
  lines <- c(lines, "", "## Estimated Spatial Ranges", "")
  for (i in seq_len(nrow(range_bands))) {
    r <- range_bands[i, ]
    lines <- c(lines, sprintf("- **%s**: %.2f mm", r$band, r$estimated_range_mm))
  }
}

lines <- c(lines,
  "",
  "## Output Files",
  "",
  "- `data/spatial_lmm_data.csv` — per-subject per-vertex band power with coordinates",
  "- `tables/spatial_lmm_results.csv` — GLS model results per band",
  "- `tables/spatial_residuals.csv` — spatial model residuals",
  "- `figures/variogram_*.png` — empirical vs fitted variograms",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
