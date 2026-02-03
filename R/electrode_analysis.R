#!/usr/bin/env Rscript
# electrode_analysis.R — Electrode-level PSD statistical analysis
#
# Called by Python: Rscript R/electrode_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads electrode_band_power.csv, runs omnibus LMM + emmeans post-hoc
# for electrode-level data (group * channel), generates figures and summary.

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
source(file.path(script_dir, "plot_psd.R"))

# --- Argument parsing ---
parser <- ArgumentParser(description = "Electrode-level PSD analysis (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing electrode_band_power.csv")
parser$add_argument("--config", required = TRUE,
                    help = "Path to study YAML config")
parser$add_argument("--output-dir", required = TRUE,
                    help = "Root output directory for this analysis")
args <- parser$parse_args()

data_dir <- args$data_dir
config_path <- args$config
output_dir <- args$output_dir

fig_dir <- file.path(output_dir, "figures")
tbl_dir <- file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load data ---
message("Loading electrode data...")
band_df <- read_csv(file.path(data_dir, "electrode_band_power.csv"), show_col_types = FALSE)
message("  electrode_band_power.csv: ", nrow(band_df), " rows")

# Rename 'channel' to 'roi' so we can reuse stats_utils functions
band_df <- band_df %>% dplyr::rename(roi = channel)

# --- Load config ---
config <- read_yaml(config_path)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))
message("Bands: ", paste(names(config$bands), collapse = ", "))
message("Channels: ", length(unique(band_df$roi)))

# --- Run LMMs for each power type ---
power_types <- c("relative", "dB")

all_omnibus <- list()
all_posthoc <- list()

for (ptype in power_types) {
  message("\n=== Power type: ", ptype, " ===")

  # Omnibus LMM: dv ~ group * channel + (1|subject)
  message("Running electrode-level omnibus LMM (group * channel)...")
  omnibus <- run_omnibus_lmm(band_df, config$contrasts, config$bands, power_type = ptype)
  all_omnibus[[ptype]] <- omnibus

  if (nrow(omnibus) > 0) {
    message("\n  === Electrode Omnibus (", ptype, ") ===")
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

  # Post-hoc: emmeans per channel
  message("Running channel-level post-hoc emmeans...")
  posthoc <- run_posthoc_emmeans(band_df, config$contrasts, config$bands, omnibus,
                                  power_type = ptype)
  all_posthoc[[ptype]] <- posthoc

  if (nrow(posthoc) > 0) {
    sig_count <- sum(posthoc$significant, na.rm = TRUE)
    message("  ", nrow(posthoc), " channel contrasts, ", sig_count, " significant")
  } else {
    message("  No post-hoc tests (no significant omnibus effects)")
  }
}

# --- Combine results ---
omnibus_df <- bind_rows(all_omnibus)
posthoc_df <- bind_rows(all_posthoc)

# --- Export tables ---
message("\nExporting tables...")
if (nrow(omnibus_df) > 0) {
  write_csv(omnibus_df, file.path(tbl_dir, "electrode_omnibus.csv"))
  message("  Saved: tables/electrode_omnibus.csv")
}
if (nrow(posthoc_df) > 0) {
  write_csv(posthoc_df, file.path(tbl_dir, "electrode_posthoc.csv"))
  message("  Saved: tables/electrode_posthoc.csv")
}

# --- Figures ---
message("\nGenerating figures...")

# Band power boxplots (electrode-level means per subject)
for (ptype in c("relative", "absolute", "dB")) {
  # Subject-level means (average across channels)
  subj_means <- band_df %>%
    dplyr::filter(group %in% group_order) %>%
    dplyr::group_by(subject, group, band) %>%
    dplyr::summarise(value = mean(.data[[ptype]], na.rm = TRUE), .groups = "drop") %>%
    dplyr::mutate(
      group = factor(group, levels = group_order),
      group_label = group_labels[as.character(group)]
    )

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  band_order <- unique(band_df$band)
  subj_means$band <- factor(subj_means$band, levels = band_order)

  p <- ggplot(subj_means, aes(x = group_label, y = value, fill = group_label)) +
    geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
    geom_jitter(width = 0.15, size = 1.5, alpha = 0.6,
                aes(color = group_label), show.legend = FALSE) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    facet_wrap(~ band, scales = "free_y", nrow = 1) +
    labs(x = NULL, y = paste0(tools::toTitleCase(ptype), " Power"),
         title = paste0("Electrode Band Power (", tools::toTitleCase(ptype), ") by Group")) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")

  fname <- paste0("electrode_band_power_", ptype, ".png")
  ggsave(file.path(fig_dir, fname), p,
         width = 3.5 * length(band_order), height = 5, dpi = 300)
  message("  Saved: ", fname)
}

# Channel-level forest plots (post-hoc)
if (nrow(posthoc_df) > 0) {
  for (ptype in unique(posthoc_df$power_type)) {
    for (cname in unique(posthoc_df$contrast)) {
      pdata <- posthoc_df %>%
        dplyr::filter(contrast == cname, power_type == ptype) %>%
        dplyr::mutate(
          roi = forcats::fct_reorder(roi, estimate),
          sig_label = ifelse(significant, "*", "")
        )

      if (nrow(pdata) == 0) next

      n_bands <- length(unique(pdata$band))

      p <- ggplot(pdata, aes(x = estimate, y = roi)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbar(aes(xmin = estimate - 1.96 * SE, xmax = estimate + 1.96 * SE),
                      width = 0.3, color = "grey40", orientation = "y") +
        geom_point(aes(color = significant), size = 2.5) +
        scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "#E74C3C"),
                           labels = c("n.s.", "p < .05"), name = NULL) +
        facet_wrap(~ band, scales = "free_x") +
        labs(x = "Group Difference (emmean)", y = "Channel",
             title = paste0("Channel-Level Group Contrasts: ", cname, " (", ptype, ")")) +
        theme_pub()

      fname <- paste0("electrode_channel_forest_", cname, "_", ptype, ".png")
      ggsave(file.path(fig_dir, fname), p,
             width = max(10, 4 * n_bands), height = 8, dpi = 300, limitsize = FALSE)
      message("  Saved: ", fname)
    }
  }
}

# --- Summary report ---
message("\nWriting summary...")

n_subjects <- band_df %>%
  dplyr::distinct(subject, group) %>%
  dplyr::count(group) %>%
  { setNames(.$n, .$group) }

n_channels <- length(unique(band_df$roi))
sfreq <- if (!is.null(config$sfreq)) config$sfreq else 500

lines <- character()
add <- function(...) lines <<- c(lines, paste0(...))

add("# Electrode-Level PSD Analysis \u2014 ", config$name)
add("")
add("**Generated:** ", format(Sys.time(), "%Y-%m-%d %H:%M"))
add("")

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

add("**Analysis:** Electrode-Level Power Spectral Density")
add("")
add("**Groups:** ", group_str)
add("")
add("**Channels:** ", n_channels, " scalp electrodes")
add("")
add("**Sampling Rate:** ", sfreq, " Hz")
add("")
add("**Frequency Bands:** ", band_str)
add("")
add("**PSD Method:** Welch's method (2-second Hann windows, 50% overlap)")
add("")
add("**Statistics:** Omnibus LMM: dv ~ group * channel + (1|subject), Type III ANOVA ",
    "with Satterthwaite df. FDR (BH) correction across bands. ",
    "Post-hoc: emmeans pairwise group contrasts per channel, Holm correction. ",
    "Hedges' g = emmean difference / residual SD.")
add("")

# Omnibus results
add("## Omnibus LMM Results")
add("")
if (nrow(omnibus_df) > 0) {
  display_cols <- intersect(
    c("contrast", "band", "power_type", "n_a", "n_b", "n_rois",
      "group_F", "group_p", "group_q", "group_significant",
      "interaction_F", "interaction_p", "interaction_q", "interaction_significant"),
    names(omnibus_df)
  )
  tbl <- omnibus_df[, display_cols]
  for (col in names(tbl)) {
    if (is.numeric(tbl[[col]])) tbl[[col]] <- sprintf("%.4f", tbl[[col]])
    if (is.logical(tbl[[col]])) tbl[[col]] <- ifelse(tbl[[col]], "**Yes**", "No")
  }
  # Rename n_rois -> n_channels for clarity
  names(tbl)[names(tbl) == "n_rois"] <- "n_channels"

  header <- paste("|", paste(names(tbl), collapse = " | "), "|")
  sep <- paste("|", paste(rep("---", ncol(tbl)), collapse = " | "), "|")
  rows <- apply(tbl, 1, function(r) paste("|", paste(r, collapse = " | "), "|"))
  add(header)
  add(sep)
  for (r in rows) add(r)
  add("")
}

# Post-hoc results
add("## Channel-Level Post-Hoc Contrasts")
add("")
if (nrow(posthoc_df) > 0) {
  sig_posthoc <- posthoc_df %>% dplyr::filter(significant == TRUE)
  if (nrow(sig_posthoc) > 0) {
    add("Significant channel-level group differences (Holm-corrected q < 0.05):")
    add("")
    for (pt in unique(sig_posthoc$power_type)) {
      for (bname in unique(sig_posthoc$band[sig_posthoc$power_type == pt])) {
        band_sig <- sig_posthoc %>% dplyr::filter(power_type == pt, band == bname)
        add("#### ", bname, " (", pt, ")")
        add("")
        add("| Channel | Estimate | SE | t | q | Hedges' g |")
        add("| --- | --- | --- | --- | --- | --- |")
        for (i in seq_len(nrow(band_sig))) {
          row <- band_sig[i, ]
          add(sprintf("| %s | %.4f | %.4f | %.2f | %.4f | %.2f |",
                      row$roi, row$estimate, row$SE, row$t_ratio,
                      row$q_value, row$hedges_g))
        }
        add("")
      }
    }
  } else {
    add("No individual channels reached significance after Holm correction.")
    add("")
  }
} else {
  add("*Post-hoc not performed (no significant omnibus effects).*")
  add("")
}

# Key findings
add("## Key Findings")
add("")
any_sig <- FALSE
if (nrow(omnibus_df) > 0) {
  for (i in seq_len(nrow(omnibus_df))) {
    row <- omnibus_df[i, ]
    findings <- character()
    if (isTRUE(row$group_significant))
      findings <- c(findings, sprintf("group main effect (F=%.2f, q=%.4f)", row$group_F, row$group_q))
    if (isTRUE(row$interaction_significant))
      findings <- c(findings, sprintf("group x channel interaction (F=%.2f, q=%.4f)", row$interaction_F, row$interaction_q))
    if (length(findings) > 0) {
      any_sig <- TRUE
      add(sprintf("- **%s %s** [%s]: %s", row$band, row$power_type, row$contrast,
                  paste(findings, collapse = "; ")))
    }
  }
}
if (!any_sig) {
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

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
message("  Report written: ", file.path(output_dir, "ANALYSIS_SUMMARY.md"))
message("\nDone. Output: ", output_dir)
