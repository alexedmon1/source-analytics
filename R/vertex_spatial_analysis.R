#!/usr/bin/env Rscript
# vertex_spatial_analysis.R — primary computation module
# Fits nlme::lme with random subject intercept + exponential spatial correlation
# per contrast x band x metric. Compares to non-spatial LME via AIC/BIC,
# generates variograms, exports residuals, writes ANALYSIS_SUMMARY.md
#
# Multi-group design: iterates over contrasts from config (subsetting to 2 groups
# per contrast), consistent with stats_utils.R pattern.

suppressPackageStartupMessages({
  library(optparse)
  library(yaml)
  library(nlme)
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(emmeans)
})

option_list <- list(
  make_option("--data-dir", type = "character", help = "Path to data/ directory"),
  make_option("--config",   type = "character", help = "Path to study_config.yaml"),
  make_option("--output-dir", type = "character", help = "Path to output directory"),
  make_option("--fig-dir", type = "character", default = NULL,
              help = "Directory for figures (default: output-dir/figures)"),
  make_option("--tbl-dir", type = "character", default = NULL,
              help = "Directory for tables (default: output-dir/tables)"),
  make_option("--no-figures", action = "store_true", default = FALSE,
              help = "Skip all figure generation")
)
opts <- parse_args(OptionParser(option_list = option_list))

no_figures <- isTRUE(opts[["no-figures"]])

data_dir    <- opts[["data-dir"]]
config_path <- opts[["config"]]
output_dir  <- opts[["output-dir"]]

config <- read_yaml(config_path)

# --- Load data ----------------------------------------------------------------
data_path <- file.path(data_dir, "vertex_spatial_data.csv")
if (!file.exists(data_path)) {
  cat("No vertex_spatial_data.csv found.\n")
  quit(status = 0)
}

dat <- read.csv(data_path, stringsAsFactors = FALSE)

slmm_cfg <- config$vertex_spatial %||% config$spatial_lmm %||% list()
stat_method <- slmm_cfg$stat_method %||% "gls"
corr_struct <- slmm_cfg$correlation_structure %||% "exponential"
range_mm    <- slmm_cfg$spatial_range_mm %||% 3.0

cat(sprintf("stat_method: %s\n", stat_method))

bands <- unique(dat$band)
metrics <- c("relative", "absolute")
contrasts <- config$contrasts
n_subjects_total <- length(unique(dat$subject))
n_vertices <- length(unique(dat$vertex_idx))

cat(sprintf("Vertex spatial: %d subjects total, %d bands, %d metrics, %d vertices per subject\n",
            n_subjects_total, length(bands), length(metrics), n_vertices))
cat(sprintf("Contrasts: %d\n", length(contrasts)))

# --- Fit models per contrast x band x metric ---------------------------------
fig_dir <- if (!is.null(opts[["fig-dir"]])) opts[["fig-dir"]] else file.path(output_dir, "figures")
tbl_dir <- if (!is.null(opts[["tbl-dir"]])) opts[["tbl-dir"]] else file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# RETIRED (design-spec migration, 2026-06). vertex_spatial fit a per-contrast
# GLS spatial-covariance model (corExp + nugget) as a robustness check on the
# vertex group difference, iterating `config$contrasts`. The contrasts:/
# hypothesis_testing: blocks were replaced by the declarative design:/hypotheses:
# spec, so `config$contrasts` is now NULL and this module has no contrasts to fit.
# It is retired rather than migrated: spatially-resolved vertex inference is
# delivered by vertex_cluster (cluster-based permutation glass-brain maps) and
# vertex_nbs (network-based statistic); the spatial-covariance robustness table
# was never a manuscript result. We emit empty result/residual frames + a note so
# downstream consumers find a well-formed (empty) output, then exit cleanly.
.retire_note <- paste0(
  "vertex_spatial is RETIRED (design-spec migration). The per-contrast GLS ",
  "spatial-covariance robustness model iterated config$contrasts, which the ",
  "declarative design:/hypotheses: spec no longer populates. Spatially-resolved ",
  "vertex inference is provided by vertex_cluster (cluster-permutation glass-brain ",
  "maps) and vertex_nbs (network-based statistic).")
write.csv(data.frame(), file.path(tbl_dir, "vertex_spatial_results.csv"), row.names = FALSE)
write.csv(data.frame(), file.path(tbl_dir, "vertex_spatial_residuals.csv"), row.names = FALSE)
writeLines(c("# Vertex Spatial Analysis — RETIRED", "", .retire_note),
           file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("\n", .retire_note, "\n", sep = "")
quit(status = 0)

# ---- Dead code below (pre-retirement GLS machinery; left for reference) ------
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

      coef_val <- NA; se_val <- NA; t_val <- NA; p_val <- NA
      f_val <- NA; df1 <- NA; df2 <- NA; interaction_p <- NA
      estimated_range <- NA
      aic_spatial <- NA; bic_spatial <- NA
      aic_nonspatial <- NA; bic_nonspatial <- NA
      convergence <- "failed"
      fit_model <- NULL

      # ── GLS: spatial correlation, no random effect (replicates antwerp) ──────
      if (stat_method == "gls") {

        # Baseline non-spatial GLS for AIC comparison
        tryCatch({
          fit_base <- gls(response ~ group, data = band_dat,
                          control = glsControl(opt = "optim"))
          aic_nonspatial <- AIC(fit_base)
          bic_nonspatial <- BIC(fit_base)
          cat(sprintf("  Non-spatial GLS: AIC=%.1f\n", aic_nonspatial))
        }, error = function(e) cat(sprintf("  Non-spatial GLS failed: %s\n", e$message)))

        tryCatch({
          fit_model <<- gls(
            response ~ group,
            correlation = corExp(value = range_mm, form = ~ x + y + z | subject, nugget = TRUE),
            data = band_dat,
            control = glsControl(opt = "optim", maxIter = 200, msMaxIter = 200, tolerance = 1e-4)
          )
          aic_spatial <<- AIC(fit_model)
          bic_spatial <<- BIC(fit_model)
          convergence <<- "converged"
          cat(sprintf("  GLS (corExp+nugget): AIC=%.1f\n", aic_spatial))
        }, error = function(e) {
          cat(sprintf("  GLS (corExp+nugget) failed: %s\n", e$message))
          tryCatch({
            fit_model <<- gls(
              response ~ group,
              correlation = corExp(value = range_mm, form = ~ x + y + z | subject),
              data = band_dat,
              control = glsControl(opt = "optim", maxIter = 200)
            )
            aic_spatial <<- AIC(fit_model)
            bic_spatial <<- BIC(fit_model)
            convergence <<- "converged (no nugget)"
            cat(sprintf("  GLS fallback (no nugget): AIC=%.1f\n", aic_spatial))
          }, error = function(e2) cat(sprintf("  GLS fallback failed: %s\n", e2$message)))
        })

        if (!is.null(fit_model)) {
          tryCatch({
            s <- summary(fit_model)
            tbl <- s$tTable
            if (nrow(tbl) >= 2) {
              coef_val <- tbl[2, "Value"]
              se_val   <- tbl[2, "Std.Error"]
              t_val    <- tbl[2, "t-value"]
              p_val    <- tbl[2, "p-value"]
            }
            cs <- coef(fit_model$modelStruct$corStruct, unconstrained = FALSE)
            estimated_range <- if ("range" %in% names(cs)) cs["range"] else cs[1]
          }, error = function(e) cat(sprintf("  GLS summary failed: %s\n", e$message)))

          if (!no_figures && metric == "relative" && convergence != "failed") {
            tryCatch({
              safe_band <- gsub(" ", "_", tolower(band))
              safe_cname <- gsub(" ", "_", cname)
              png(file.path(fig_dir, sprintf("variogram_%s_%s.png", safe_cname, safe_band)),
                  width = 800, height = 500)
              plot(Variogram(fit_model, form = ~ x + y + z | subject, maxDist = 8),
                   main = sprintf("Variogram — %s (relative) [%s]", band, cname))
              dev.off()
            }, error = function(e) {
              cat(sprintf("  Variogram failed: %s\n", e$message))
              tryCatch(dev.off(), error = function(e2) {})
            })
          }
        }

      # ── Spatial LME: random subject intercept + spatial correlation ───────────
      } else if (stat_method == "spatial_lme") {

        tryCatch({
          fit_base <- lme(response ~ group, random = ~ 1 | subject, data = band_dat,
                          control = lmeControl(opt = "optim"))
          aic_nonspatial <<- AIC(fit_base)
          bic_nonspatial <<- BIC(fit_base)
          cat(sprintf("  Non-spatial LME: AIC=%.1f\n", aic_nonspatial))
        }, error = function(e) cat(sprintf("  Non-spatial LME failed: %s\n", e$message)))

        tryCatch({
          fit_model <<- lme(
            response ~ group,
            random = ~ 1 | subject,
            correlation = corExp(value = range_mm, form = ~ x + y + z, nugget = TRUE),
            data = band_dat,
            control = lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200, tolerance = 1e-4)
          )
          aic_spatial <<- AIC(fit_model)
          bic_spatial <<- BIC(fit_model)
          convergence <<- "converged"
          cat(sprintf("  Spatial LME (corExp): AIC=%.1f\n", aic_spatial))
        }, error = function(e) {
          cat(sprintf("  Spatial LME failed: %s\n", e$message))
          tryCatch({
            fit_model <<- lme(
              response ~ group,
              random = ~ 1 | subject,
              correlation = corExp(value = range_mm, form = ~ x + y + z),
              data = band_dat,
              control = lmeControl(opt = "optim", maxIter = 100, msMaxIter = 100)
            )
            aic_spatial <<- AIC(fit_model)
            bic_spatial <<- BIC(fit_model)
            convergence <<- "converged (no nugget)"
            cat(sprintf("  Spatial LME fallback: AIC=%.1f\n", aic_spatial))
          }, error = function(e2) cat(sprintf("  Spatial LME fallback failed: %s\n", e2$message)))
        })

        if (!is.null(fit_model)) {
          tryCatch({
            s <- summary(fit_model)
            tbl <- s$tTable
            if (nrow(tbl) >= 2) {
              coef_val <- tbl[2, "Value"]
              se_val   <- tbl[2, "Std.Error"]
              t_val    <- tbl[2, "t-value"]
              p_val    <- tbl[2, "p-value"]
            }
            cs <- coef(fit_model$modelStruct$corStruct, unconstrained = FALSE)
            estimated_range <- if ("range" %in% names(cs)) cs["range"] else cs[1]
          }, error = function(e) cat(sprintf("  Spatial LME summary failed: %s\n", e$message)))
        }

      # ── LMM: omnibus group × vertex LMM, consistent with ROI framework ────────
      } else if (stat_method == "lmm") {

        # Treat vertex_idx as a factor so we get group*vertex interaction
        band_dat$vertex_f <- factor(band_dat$vertex_idx)
        convergence <- "failed"

        tryCatch({
          fit_model <<- lmer(
            response ~ group * vertex_f + (1 | subject),
            data = band_dat,
            REML = FALSE
          )
          convergence <<- "converged"
          aic_spatial <<- AIC(fit_model)
          bic_spatial <<- BIC(fit_model)

          # Baseline: no vertex interaction
          fit_base <- lmer(response ~ group + vertex_f + (1 | subject),
                           data = band_dat, REML = FALSE)
          aic_nonspatial <<- AIC(fit_base)
          bic_nonspatial <<- BIC(fit_base)

          cat(sprintf("  LMM converged: AIC=%.1f (interaction) vs %.1f (additive)\n",
                      aic_spatial, aic_nonspatial))
        }, error = function(e) cat(sprintf("  LMM failed: %s\n", e$message)))

        if (!is.null(fit_model) && convergence == "converged") {
          tryCatch({
            # Type III ANOVA for group main effect and group:vertex interaction
            an <- anova(fit_model, type = "III")

            # Group main effect
            if ("group" %in% rownames(an)) {
              f_val    <<- an["group", "F value"]
              df1      <<- an["group", "NumDF"]
              df2      <<- an["group", "DenDF"]
              p_val    <<- an["group", "Pr(>F)"]
            }
            # Group × vertex interaction
            int_row <- grep("group:vertex_f|vertex_f:group", rownames(an), value = TRUE)[1]
            if (!is.na(int_row)) {
              interaction_p <<- an[int_row, "Pr(>F)"]
            }

            # Marginal group contrast (emmeans), averaged over vertices
            em <- emmeans(fit_model, ~ group)
            ct <- contrast(em, method = "pairwise")
            ct_df <- as.data.frame(ct)
            row1 <- ct_df[1, ]
            coef_val <<- row1$estimate
            se_val   <<- row1$SE
            t_val    <<- row1$t.ratio
            # Use interaction p if group main p not more informative
            if (!is.na(interaction_p)) p_val <<- interaction_p

            cat(sprintf("  LMM group F=%.3f p=%.4f; interaction p=%.4f\n",
                        ifelse(is.na(f_val), NA, f_val), p_val,
                        ifelse(is.na(interaction_p), NA, interaction_p)))
          }, error = function(e) cat(sprintf("  LMM summary failed: %s\n", e$message)))
        }
      }

      # Extract residuals for spatial methods
      if (!is.null(fit_model) && stat_method %in% c("gls", "spatial_lme")) {
        tryCatch({
          resids <- data.frame(
            contrast = cname, subject = band_dat$subject,
            vertex_idx = band_dat$vertex_idx, band = band, metric = metric,
            residual = residuals(fit_model), stringsAsFactors = FALSE
          )
          all_residuals <- rbind(all_residuals, resids)
        }, error = function(e) {})
      }

      cat(sprintf("  Result: coef=%.4f, t=%.3f, p=%.4f\n",
                  ifelse(is.na(coef_val), NA, coef_val),
                  ifelse(is.na(t_val), NA, t_val),
                  ifelse(is.na(p_val), NA, p_val)))

      model_results[[result_idx]] <- data.frame(
        contrast = cname, group_a = ga, group_b = gb,
        n_a = n_a, n_b = n_b,
        band = band, metric = metric,
        stat_method = stat_method, convergence = convergence,
        aic_spatial = aic_spatial, bic_spatial = bic_spatial,
        aic_nonspatial = aic_nonspatial, bic_nonspatial = bic_nonspatial,
        aic_improvement = ifelse(!is.na(aic_nonspatial) & !is.na(aic_spatial),
                                 aic_nonspatial - aic_spatial, NA),
        coefficient = coef_val, std_error = se_val,
        t_value = t_val, p_value = p_val,
        interaction_p = interaction_p,
        estimated_range_mm = estimated_range,
        stringsAsFactors = FALSE
      )
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

write.csv(results_df, file.path(tbl_dir, "vertex_spatial_results.csv"), row.names = FALSE)
cat(sprintf("\nExported vertex_spatial_results.csv (%d rows)\n", nrow(results_df)))

if (nrow(all_residuals) > 0) {
  write.csv(all_residuals, file.path(tbl_dir, "vertex_spatial_residuals.csv"), row.names = FALSE)
  cat(sprintf("Exported vertex_spatial_residuals.csv (%d rows)\n", nrow(all_residuals)))
}

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
groups_all <- unique(dat$group)
lines <- c(
  "# Vertex Spatial Analysis Summary",
  "",
  sprintf("**Study**: %s", config$name),
  sprintf("**Analysis**: Vertex Spatial (%s)", stat_method),
  sprintf("**stat_method**: %s", stat_method),
  if (stat_method %in% c("gls", "spatial_lme")) sprintf("**Correlation structure**: %s", corr_struct) else NULL,
  if (stat_method %in% c("gls", "spatial_lme")) sprintf("**Initial spatial range**: %.1f mm", range_mm) else NULL,
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
  if (stat_method == "gls") paste(
    "Spatial GLS (nlme::gls) was used to model vertex-level band power as a function",
    "of group, with an exponential spatial correlation structure",
    "(`corExp(form = ~x+y+z|subject, nugget=TRUE)`). The `|subject` grouping allows",
    "a separate correlation matrix per subject, effectively controlling for",
    "between-subject variance without a separate random effect. This approach",
    "replicates the antwerp manuscript analysis. Models were compared to a",
    "non-spatial GLS (identity correlation) via AIC/BIC."
  ) else if (stat_method == "spatial_lme") paste(
    "Spatial linear mixed effects models (nlme::lme) were used with a random",
    "subject intercept (`random = ~1|subject`) and an exponential spatial",
    "correlation structure (`corExp(form = ~x+y+z, nugget=TRUE)`). The random",
    "subject intercept partitions between-subject variance, while the spatial",
    "correlation structure accounts for autocorrelation between nearby vertices."
  ) else paste(
    "Omnibus LMM (lmerTest::lmer) treating vertex location as a fixed categorical",
    "factor: `power ~ group * vertex + (1|subject)`. Tests both the group main",
    "effect (averaged over vertices) and the group x vertex interaction",
    "(spatially heterogeneous group differences). FDR (BH) correction applied",
    "across bands within each contrast x metric. Consistent with the ROI-level",
    "analysis framework."
  ),
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
  "- `data/vertex_spatial_data.csv` — per-subject per-vertex band power with coordinates",
  "- `tables/vertex_spatial_results.csv` — GLS model results per contrast x band x metric",
  "- `tables/vertex_spatial_residuals.csv` — spatial model residuals",
  "- `figures/variogram_*.png` — empirical vs fitted variograms",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
