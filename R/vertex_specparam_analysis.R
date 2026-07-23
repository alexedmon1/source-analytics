#!/usr/bin/env Rscript
# vertex_specparam_analysis.R — Report Generator
# Reads vertex-level spectral parameterization results, generates ANALYSIS_SUMMARY.md

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
param_path <- file.path(data_dir, "vertex_specparam.csv")
if (!file.exists(param_path)) {
  cat("No vertex_specparam.csv found.\n")
  quit(status = 0)
}

params <- read.csv(param_path, stringsAsFactors = FALSE)

sp_cfg <- config$vertex_specparam %||% list()
# Keep in step with spectral/aperiodic.py::DEFAULT_FREQ_RANGE (12-45 Hz); see
# docs/APERIODIC_FIT_WINDOW.md for why. This is only the label fallback — the
# actual fit happens in Python, which stamps fit_fmin/fit_fmax into the params.
freq_range <- sp_cfg$freq_range %||% c(12, 45)
if (all(c("fit_fmin", "fit_fmax") %in% names(params))) {
  freq_range <- c(params$fit_fmin[1], params$fit_fmax[1])  # authoritative
}
max_peaks  <- sp_cfg$max_n_peaks %||% 6

# Peak-detection window. When it differs from the aperiodic window the run used
# a two-fit design: narrow window for the exponent, wider one for the peaks, so
# the narrow window's borders can be checked against where the peaks actually
# are instead of asserted. Python stamps peak_fmin/peak_fmax into the params.
peak_range <- freq_range
if (all(c("peak_fmin", "peak_fmax") %in% names(params))) {
  peak_range <- c(params$peak_fmin[1], params$peak_fmax[1])
}
two_fit <- !isTRUE(all.equal(as.numeric(peak_range), as.numeric(freq_range)))

# --- Summaries ----------------------------------------------------------------
n_subjects <- length(unique(params$subject))
n_vertices <- length(unique(params$vertex_idx))
groups <- unique(params$group)

# Detect per-band peak columns dynamically
peak_cols <- grep("^has_.*_peak$", names(params), value = TRUE)
band_labels <- sub("^has_(.*)_peak$", "\\1", peak_cols)

# Per-group summary of aperiodic parameters + per-band peak rates
group_summary <- do.call(rbind, lapply(groups, function(g) {
  sub <- params[params$group == g, ]
  base <- data.frame(
    Group = g,
    Mean_Exponent = round(mean(sub$exponent, na.rm = TRUE), 3),
    SD_Exponent = round(sd(sub$exponent, na.rm = TRUE), 3),
    Mean_Offset = round(mean(sub$offset, na.rm = TRUE), 3),
    SD_Offset = round(sd(sub$offset, na.rm = TRUE), 3),
    Mean_R2 = round(mean(sub$r_squared, na.rm = TRUE), 3),
    stringsAsFactors = FALSE
  )
  for (col in peak_cols) {
    label <- paste0(sub("^has_(.*)_peak$", "\\1", col), "_Peak_Rate")
    base[[label]] <- round(mean(sub[[col]], na.rm = TRUE), 3)
  }
  base
}))

# Method distribution
method_table <- table(params$method)

# --- Write ANALYSIS_SUMMARY.md -----------------------------------------------
lines <- c(
  "# Spectral Parameterization (Vertex-Level) Summary",
  "",
  sprintf("**Study**: %s", config$name),
  "**Analysis**: Vertex-level spectral parameterization (aperiodic + peaks)",
  sprintf("**Aperiodic fit range**: %g-%g Hz", freq_range[1], freq_range[2]),
  sprintf("**Peak detection range**: %g-%g Hz%s",
          peak_range[1], peak_range[2],
          if (two_fit) " (separate wider fit)" else " (same fit)"),
  sprintf("**Max peaks**: %d", max_peaks),
  sprintf("**Subjects**: %d (%s)", n_subjects, paste(groups, collapse = ", ")),
  sprintf("**Vertices**: %d", n_vertices),
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

lines <- c(lines,
  "## Methods",
  "",
  "Spectral parameterization (specparam/FOOOF) was applied to the PSD at each vertex.",
  "The aperiodic component (1/f slope and offset) and oscillatory peaks were extracted.",
  "Group differences in aperiodic parameters were tested with cluster-based permutation.",
  "Per-band peak presence rates were compared with per-vertex chi-squared tests.",
  ""
)

if (two_fit) {
  lines <- c(lines,
    sprintf(paste0(
      "Aperiodic parameters come from a %g-%g Hz fit; peaks come from a ",
      "separate %g-%g Hz fit. The narrow window is what makes the exponent ",
      "unbiased, but it can only find peaks inside itself, so it cannot be ",
      "used to check its own borders. Peak columns are emitted only for bands ",
      "the peak window can reach: a band outside it would otherwise report a ",
      "0%% detection rate that is structural rather than measured."),
      freq_range[1], freq_range[2], peak_range[1], peak_range[2]),
    "")
}

lines <- c(lines,
  "## Fitting Methods Used",
  "",
  sprintf("- specparam: %d fits", method_table["specparam"] %||% 0),
  sprintf("- linreg: %d fits", method_table["linreg"] %||% 0),
  sprintf("- failed: %d fits", method_table["failed"] %||% 0),
  "",
  "## Group Summary",
  ""
)

# Build dynamic table header with per-band peak rate columns
rate_cols <- grep("_Peak_Rate$", names(group_summary), value = TRUE)
rate_headers <- sub("_Peak_Rate$", "", rate_cols)
header_line <- paste0("| Group | Mean Exp | SD Exp | Mean Offset | SD Offset | Mean R\u00b2 |",
                       paste0(" ", rate_headers, " Rate |", collapse = ""))
sep_line <- paste0("|-------|----------|--------|-------------|-----------|---------|",
                    paste0(rep("------|", length(rate_cols)), collapse = ""))
lines <- c(lines, header_line, sep_line)

for (i in seq_len(nrow(group_summary))) {
  r <- group_summary[i, ]
  base_fmt <- sprintf("| %s | %.3f | %.3f | %.3f | %.3f | %.3f |",
                       r$Group, r$Mean_Exponent, r$SD_Exponent,
                       r$Mean_Offset, r$SD_Offset, r$Mean_R2)
  rate_vals <- paste0(sprintf(" %.3f |", unlist(r[rate_cols])), collapse = "")
  lines <- c(lines, paste0(base_fmt, rate_vals))
}

# Fit-window diagnostic — does the aperiodic window satisfy Gerster's border
# rule on THIS data? Reported before the results it underwrites.
diag_path <- file.path(output_dir, "tables", "fit_window_diagnostic.csv")
if (file.exists(diag_path)) {
  diag <- read.csv(diag_path, stringsAsFactors = FALSE)
  all_row <- diag[diag$band == "ALL", ]
  lines <- c(lines, "", "## Fit-Window Diagnostic", "",
    "Gerster et al. (2022): oscillations crossing the fit borders must be",
    "avoided, since a peak on a border inflates exponent error. A peak counts",
    "as crossing when its support (centre frequency +/- half the specparam",
    "bandwidth) straddles a border.",
    "")
  if (nrow(all_row) == 1) {
    verdict <- if (all_row$frac_crossing < 0.05) {
      "SATISFIED - the borders sit in spectral gaps on this data."
    } else if (all_row$frac_crossing < 0.15) {
      "MARGINAL - a minority of peaks touch a border; exponents carry some added error."
    } else {
      "VIOLATED - peaks sit on the borders; the window needs revisiting for this dataset."
    }
    lines <- c(lines,
      sprintf("**%d peaks** detected over %g-%g Hz. **%d (%.1f%%)** cross an aperiodic border (%g / %g Hz).",
              all_row$n_peaks, all_row$peak_fmin, all_row$peak_fmax,
              all_row$n_cross_fmin + all_row$n_cross_fmax,
              100 * all_row$frac_crossing,
              all_row$aperiodic_fmin, all_row$aperiodic_fmax),
      "",
      sprintf("**Verdict: %s**", verdict),
      "")
  }
  band_rows <- diag[diag$band != "ALL", ]
  if (nrow(band_rows) > 0) {
    lines <- c(lines,
      "| Band | Range (Hz) | Reachable | Censored | Peaks | Median CF | Crossing |",
      "|------|-----------|-----------|----------|-------|-----------|----------|")
    for (i in seq_len(nrow(band_rows))) {
      r <- band_rows[i, ]
      lines <- c(lines, sprintf("| %s | %g-%g | %s | %s | %d | %s | %.1f%% |",
        r$band, r$band_lo, r$band_hi,
        if (isTRUE(as.logical(r$reachable))) "yes" else "**no**",
        if (isTRUE(as.logical(r$censored))) "**yes**" else "no",
        r$n_peaks,
        if (is.na(r$cf_median)) "--" else sprintf("%.1f", r$cf_median),
        100 * r$frac_crossing))
    }
    lines <- c(lines, "",
      "Unreachable bands emit no peak columns at all - their absence is a",
      "property of the window, not a measurement. Censored bands extend past",
      "the peak window, so their detection rates are a lower bound.",
      "")
  }
}

# Cluster stats
stats_path <- file.path(output_dir, "tables", "vertex_specparam_stats.csv")
if (file.exists(stats_path)) {
  stats <- read.csv(stats_path, stringsAsFactors = FALSE)
  lines <- c(lines, "", "## Cluster Permutation Results", "")

  for (param in unique(stats$parameter)) {
    sub <- stats[stats$parameter == param, ]
    cluster_ids <- unique(sub$cluster_id[sub$cluster_id > 0])
    lines <- c(lines, sprintf("- **%s**: %d clusters identified", param, length(cluster_ids)))
  }
  lines <- c(lines, "")
}

# Per-band chi-squared results
chi2_path <- file.path(output_dir, "tables", "band_peak_chi2.csv")
if (!file.exists(chi2_path)) {
  chi2_path <- file.path(output_dir, "tables", "gamma_peak_chi2.csv")
}
if (file.exists(chi2_path)) {
  chi2 <- read.csv(chi2_path, stringsAsFactors = FALSE)
  lines <- c(lines, "## Peak Presence by Band (Chi-squared)", "")
  if ("band" %in% names(chi2)) {
    for (band in unique(chi2$band)) {
      sub <- chi2[chi2$band == band, ]
      n_sig <- sum(sub$p < 0.05)
      lines <- c(lines, sprintf("- **%s**: %d/%d vertices significant (uncorrected p<0.05)",
                                band, n_sig, nrow(sub)))
    }
  } else {
    n_sig <- sum(chi2$p < 0.05)
    lines <- c(lines, sprintf("- %d/%d vertices significant (uncorrected p<0.05)",
                              n_sig, nrow(chi2)))
  }
  lines <- c(lines, "")
}

lines <- c(lines,
  "## Output Files",
  "",
  "- `data/vertex_specparam.csv` — per-subject per-vertex specparam fit parameters",
  "- `data/peak_inventory.csv` — every peak found by the peak-window fit (long format)",
  "- `tables/vertex_specparam_stats.csv` — cluster permutation statistics",
  "- `tables/fit_window_diagnostic.csv` — aperiodic border check + per-band reachability",
  "- `tables/band_peak_chi2.csv` — per-band peak presence chi-squared tests",
  "- `figures/fit_window_diagnostic.png` — peak centre frequencies vs the fit borders",
  "- `figures/specparam_*.png` — aperiodic parameter glass brain maps",
  "- `figures/{band}_peak_presence.png` — per-band peak prevalence maps",
  ""
)

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
cat("Wrote ANALYSIS_SUMMARY.md\n")
