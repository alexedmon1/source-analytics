# report.R — Markdown summary report writer

library(dplyr)

#' Format an omnibus table as markdown
#'
#' @param df data.frame with omnibus results
#' @return character vector of markdown lines
format_omnibus_table <- function(df) {
  lines <- character()
  add <- function(...) lines <<- c(lines, paste0(...))

  display_cols <- intersect(
    c("contrast", "band", "power_type", "n_a", "n_b",
      "n_rois", "n_regions",
      "group_F", "group_p", "group_q", "group_significant",
      "interaction_F", "interaction_p", "interaction_q",
      "interaction_significant", "singular"),
    names(df)
  )
  tbl <- df[, display_cols]

  for (col in names(tbl)) {
    if (is.numeric(tbl[[col]])) tbl[[col]] <- sprintf("%.4f", tbl[[col]])
    if (is.logical(tbl[[col]])) tbl[[col]] <- ifelse(tbl[[col]], "**Yes**", "No")
  }

  header <- paste("|", paste(names(tbl), collapse = " | "), "|")
  sep <- paste("|", paste(rep("---", ncol(tbl)), collapse = " | "), "|")
  rows <- apply(tbl, 1, function(r) paste("|", paste(r, collapse = " | "), "|"))
  add(header)
  add(sep)
  for (r in rows) add(r)
  return(lines)
}


#' Format post-hoc results as markdown
#'
#' @param sig_df data.frame of significant post-hoc results
#' @param spatial_col character — "roi" or "region"
#' @return character vector of markdown lines
format_posthoc_section <- function(sig_df, spatial_col = "roi") {
  lines <- character()
  add <- function(...) lines <<- c(lines, paste0(...))

  label <- if (spatial_col == "roi") "ROI" else "Region"

  if (nrow(sig_df) > 0) {
    add("Significant ", tolower(label), "-level group differences (Holm-corrected q < 0.05):")
    add("")

    for (pt in unique(sig_df$power_type)) {
      for (bname in unique(sig_df$band[sig_df$power_type == pt])) {
        band_sig <- sig_df %>% filter(power_type == pt, band == bname)
        add("#### ", bname, " (", pt, ")")
        add("")
        add("| ", label, " | Estimate | SE | t | q | Hedges' g |")
        add("| --- | --- | --- | --- | --- | --- |")
        for (i in seq_len(nrow(band_sig))) {
          row <- band_sig[i, ]
          add(sprintf("| %s | %.4f | %.4f | %.2f | %.4f | %.2f |",
                      row[[spatial_col]], row$estimate, row$SE, row$t_ratio,
                      row$q_value, row$hedges_g))
        }
        add("")
      }
    }
  } else {
    add("No individual ", tolower(label), "s reached significance after Holm correction.")
    add("")
  }

  return(lines)
}


#' Write ANALYSIS_SUMMARY.md
#'
#' @param omnibus_df data.frame from run_omnibus_lmm() (may contain multiple power_types)
#' @param posthoc_df data.frame from run_posthoc_emmeans()
#' @param config parsed YAML study config
#' @param n_subjects named integer vector (group -> count)
#' @param sfreq sampling frequency
#' @param fig_dir path to figures/ directory
#' @param output_path path to write ANALYSIS_SUMMARY.md
#' @param omnibus_region_df optional data.frame from run_omnibus_lmm_region()
#' @param posthoc_region_df optional data.frame from run_posthoc_emmeans_region()
write_summary <- function(omnibus_df, posthoc_df, config, n_subjects, sfreq,
                           fig_dir, output_path,
                           omnibus_region_df = data.frame(),
                           posthoc_region_df = data.frame()) {

  lines <- character()
  add <- function(...) lines <<- c(lines, paste0(...))
  add_lines <- function(x) lines <<- c(lines, x)

  has_region <- nrow(omnibus_region_df) > 0
  power_types_tested <- unique(omnibus_df$power_type)

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
  add("**Power Types Tested:** ", paste(power_types_tested, collapse = ", "))
  add("")
  add("**PSD Method:** Welch's method (2-second Hann windows, 50% overlap)")
  add("")

  stats_lines <- paste0(
    "**Statistics:** Two spatial levels of analysis are performed for each power type. ",
    "(1) **ROI-level:** Omnibus LMM (dv ~ group * roi + (1|subject)). ")
  if (has_region) {
    n_reg <- omnibus_region_df$n_regions[1]
    stats_lines <- paste0(stats_lines,
      "(2) **Region-level:** ROIs averaged within ", n_reg, " anatomical regions, ",
      "then dv ~ group * region + (1|subject). ")
  }
  stats_lines <- paste0(stats_lines,
    "Type III ANOVA with Satterthwaite df (lme4/lmerTest). ",
    "FDR (BH) correction across bands within each contrast x power type. ",
    "Post-hoc: emmeans pairwise group contrasts, gated on significant omnibus effects. ",
    "Holm correction within each band. ",
    "Hedges' g = emmean difference / residual SD.")
  add(stats_lines)
  add("")

  # --- Region-level results (primary) ---
  if (has_region) {
    add("## Region-Level LMM Results")
    add("")
    add("ROIs averaged within anatomical regions for increased statistical power.")
    add("")
    add_lines(format_omnibus_table(omnibus_region_df))
    add("")

    add("### Region-Level Post-Hoc Contrasts")
    add("")
    if (nrow(posthoc_region_df) > 0) {
      sig_region <- posthoc_region_df %>% filter(significant == TRUE)
      add_lines(format_posthoc_section(sig_region, "region"))
      add("**Total regions tested:** ", length(unique(posthoc_region_df$region)),
          " across ", length(unique(posthoc_region_df$band)), " band(s)")
      add("")
      add("**Significant regions:** ", nrow(sig_region))
      add("")
    } else {
      add("*Region post-hoc not performed (no significant omnibus effects).*")
      add("")
    }
  }

  # --- ROI-level results ---
  add("## ROI-Level LMM Results")
  add("")
  if (nrow(omnibus_df) > 0) {
    add_lines(format_omnibus_table(omnibus_df))
    add("")
  } else {
    add("*No ROI-level omnibus statistics computed.*")
    add("")
  }

  add("### ROI-Level Post-Hoc Contrasts")
  add("")
  if (nrow(posthoc_df) > 0) {
    sig_posthoc <- posthoc_df %>% filter(significant == TRUE)
    add_lines(format_posthoc_section(sig_posthoc, "roi"))
    add("**Total ROIs tested:** ", length(unique(posthoc_df$roi)),
        " across ", length(unique(posthoc_df$band)), " band(s)")
    add("")
    add("**Significant ROIs:** ", nrow(sig_posthoc))
    add("")
  } else {
    add("*ROI post-hoc not performed (no significant omnibus effects).*")
    add("")
  }

  # Key findings
  add("## Key Findings")
  add("")
  any_sig <- FALSE

  # Collect all significant omnibus rows
  sig_sources <- list()

  if (has_region) {
    for (i in seq_len(nrow(omnibus_region_df))) {
      row <- omnibus_region_df[i, ]
      findings <- character()
      if (isTRUE(row$group_significant))
        findings <- c(findings, sprintf("group main effect (F=%.2f, q=%.4f)", row$group_F, row$group_q))
      if (isTRUE(row$interaction_significant))
        findings <- c(findings, sprintf("group x region interaction (F=%.2f, q=%.4f)", row$interaction_F, row$interaction_q))
      if (length(findings) > 0) {
        any_sig <- TRUE
        add(sprintf("- **%s %s** [%s, region-level]: %s", row$band, row$power_type, row$contrast,
                    paste(findings, collapse = "; ")))
        if (nrow(posthoc_region_df) > 0) {
          band_regions <- posthoc_region_df %>%
            filter(contrast == row$contrast, band == row$band,
                   power_type == row$power_type, significant == TRUE)
          if (nrow(band_regions) > 0) {
            reg_strs <- sprintf("%s (g=%.2f)", band_regions$region, band_regions$hedges_g)
            add("  - Significant regions: ", paste(reg_strs, collapse = ", "))
          }
        }
      }
    }
  }

  if (nrow(omnibus_df) > 0) {
    for (i in seq_len(nrow(omnibus_df))) {
      row <- omnibus_df[i, ]
      findings <- character()
      if (isTRUE(row$group_significant))
        findings <- c(findings, sprintf("group main effect (F=%.2f, q=%.4f)", row$group_F, row$group_q))
      if (isTRUE(row$interaction_significant))
        findings <- c(findings, sprintf("group x ROI interaction (F=%.2f, q=%.4f)", row$interaction_F, row$interaction_q))
      if (length(findings) > 0) {
        any_sig <- TRUE
        add(sprintf("- **%s %s** [%s, ROI-level]: %s", row$band, row$power_type, row$contrast,
                    paste(findings, collapse = "; ")))
        if (nrow(posthoc_df) > 0) {
          band_rois <- posthoc_df %>%
            filter(contrast == row$contrast, band == row$band,
                   power_type == row$power_type, significant == TRUE)
          if (nrow(band_rois) > 0) {
            roi_strs <- sprintf("%s (g=%.2f)", band_rois$roi, band_rois$hedges_g)
            add("  - Significant ROIs: ", paste(roi_strs, collapse = ", "))
          }
        }
      }
    }
  }

  if (!any_sig) {
    add("- No bands reached significance after FDR correction at either analysis level.")
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
