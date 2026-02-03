# report.R — Markdown summary report writer

library(dplyr)

#' Write ANALYSIS_SUMMARY.md
#'
#' @param stats_df data.frame from run_band_contrasts()
#' @param config parsed YAML study config
#' @param n_subjects named integer vector (group -> count)
#' @param sfreq sampling frequency
#' @param fig_dir path to figures/ directory
#' @param output_path path to write ANALYSIS_SUMMARY.md
write_summary <- function(stats_df, config, n_subjects, sfreq,
                           fig_dir, output_path) {

  lines <- character()
  add <- function(...) lines <<- c(lines, paste0(...))

  # Header
  add("# PSD Analysis \u2014 ", config$name)
  add("")
  add("**Generated:** ", format(Sys.time(), "%Y-%m-%d %H:%M"))
  add("")

  # Methods
  add("## Methods")
  add("")
  group_str <- paste(
    sapply(names(n_subjects), function(g) paste0(config$groups[[g]], " (n=", n_subjects[g], ")")),
    collapse = ", "
  )
  band_str <- paste(
    sapply(names(config$bands), function(b) {
      lims <- config$bands[[b]]
      paste0(b, ": ", lims[1], "-", lims[2], " Hz")
    }),
    collapse = ", "
  )

  add("**Analysis:** Power Spectral Density")
  add("")
  add("**Groups:** ", group_str)
  add("")
  add("**Sampling Rate:** ", sfreq, " Hz")
  add("")
  add("**Frequency Bands:** ", band_str)
  add("")
  add("**PSD Method:** Welch's method (2-second Hann windows, 50% overlap)")
  add("")
  add("**Statistics:** Independent samples Welch's t-test and linear mixed models ",
      "(LMM: relative_power ~ group + (1|subject), lme4/lmerTest). ",
      "Effect sizes reported as Hedges' g. ",
      "Multiple comparison correction via Benjamini-Hochberg FDR.")
  add("")

  # Statistics table
  add("## Band Power Statistics")
  add("")
  if (nrow(stats_df) > 0) {
    display_cols <- c("contrast", "band", "t_stat", "p_value", "q_value",
                      "hedges_g", "lmm_z", "lmm_p", "lmm_q",
                      "group_a_mean", "group_b_mean", "significant")
    display_cols <- intersect(display_cols, names(stats_df))
    tbl <- stats_df[, display_cols]

    # Format numerics
    for (col in names(tbl)) {
      if (is.numeric(tbl[[col]])) {
        tbl[[col]] <- sprintf("%.4f", tbl[[col]])
      }
      if (is.logical(tbl[[col]])) {
        tbl[[col]] <- ifelse(tbl[[col]], "**Yes**", "No")
      }
    }

    # Markdown table
    header <- paste("|", paste(names(tbl), collapse = " | "), "|")
    sep <- paste("|", paste(rep("---", ncol(tbl)), collapse = " | "), "|")
    rows <- apply(tbl, 1, function(r) paste("|", paste(r, collapse = " | "), "|"))
    add(header)
    add(sep)
    for (r in rows) add(r)
    add("")
  } else {
    add("*No statistics computed.*")
    add("")
  }

  # Key findings
  add("## Key Findings")
  add("")
  if (nrow(stats_df) > 0 && any(stats_df$significant, na.rm = TRUE)) {
    sig_rows <- stats_df %>% filter(significant == TRUE)
    for (i in seq_len(nrow(sig_rows))) {
      row <- sig_rows[i, ]
      dir <- if (!is.na(row$hedges_g) && row$hedges_g > 0) ">" else "<"
      add(sprintf("- **%s**: %s %s %s (t=%.2f, q=%.4f, g=%.2f)",
                  row$band, row$group_a, dir, row$group_b,
                  row$t_stat, row$q_value, row$hedges_g))
    }
  } else {
    add("- No bands reached significance after FDR correction.")
  }
  add("")

  # Figure references
  fig_files <- sort(list.files(fig_dir, pattern = "\\.png$"))
  if (length(fig_files) > 0) {
    add("## Figures")
    add("")
    for (ff in fig_files) {
      caption <- gsub("_", " ", tools::file_path_sans_ext(ff))
      caption <- tools::toTitleCase(caption)
      add(sprintf("![%s](figures/%s)", caption, ff))
      add("")
    }
  }

  writeLines(lines, output_path)
  message("  Report written: ", output_path)
}
