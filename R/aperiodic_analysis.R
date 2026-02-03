#!/usr/bin/env Rscript
# aperiodic_analysis.R — Main R entry point for aperiodic (1/f) statistical analysis
#
# Called by Python: Rscript R/aperiodic_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads aperiodic_params.csv exported by Python,
# runs omnibus LMM + emmeans post-hoc for exponent and offset DVs,
# generates ggplot2 figures, writes ANALYSIS_SUMMARY.md.

library(argparse)
library(yaml)
library(readr)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(scales)
library(forcats)

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

# Reuse plot theme from plot_psd.R if available, otherwise define inline
tryCatch(source(file.path(script_dir, "plot_psd.R")), error = function(e) NULL)

if (!exists("theme_pub")) {
  theme_pub <- function(base_size = 11) {
    theme_minimal(base_size = base_size) +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "grey92"),
        strip.text = element_text(face = "bold", size = base_size),
        legend.position = "bottom",
        plot.title = element_text(face = "bold", size = base_size + 2)
      )
  }
}

# --- Argument parsing ---
parser <- ArgumentParser(description = "Aperiodic (1/f) statistical analysis (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing aperiodic_params.csv")
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
message("Loading data...")
ap_df <- read_csv(file.path(data_dir, "aperiodic_params.csv"), show_col_types = FALSE)
message("  aperiodic_params.csv: ", nrow(ap_df), " rows")

# --- Load config ---
config <- read_yaml(config_path)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))

# ============================================================
# Aperiodic-specific LMM functions (no band dimension)
# ============================================================

#' Run omnibus LMM for aperiodic DV
#'
#' Model: dv ~ group * roi + (1|subject)
#' No FDR across bands (single DV per call).
run_omnibus_lmm_aperiodic <- function(ap_df, contrasts, dv_name) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    cdata <- ap_df %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    n_a <- length(unique(cdata$subject[cdata$group == ga]))
    n_b <- length(unique(cdata$subject[cdata$group == gb]))
    n_rois <- length(unique(cdata$roi))

    cdata$group <- factor(cdata$group, levels = c(gb, ga))
    cdata$roi <- factor(cdata$roi)
    cdata$dv <- cdata[[dv_name]]

    group_F <- NA; group_p <- NA
    roi_F <- NA; roi_p <- NA
    interaction_F <- NA; interaction_p <- NA
    converged <- TRUE; singular <- FALSE

    tryCatch({
      fit <- lmer(dv ~ group * roi + (1 | subject), data = cdata)
      singular <- isSingular(fit)

      aov <- anova(fit, type = 3)

      if ("group" %in% rownames(aov)) {
        group_F <- aov["group", "F value"]
        group_p <- aov["group", "Pr(>F)"]
      }
      if ("roi" %in% rownames(aov)) {
        roi_F <- aov["roi", "F value"]
        roi_p <- aov["roi", "Pr(>F)"]
      }
      if ("group:roi" %in% rownames(aov)) {
        interaction_F <- aov["group:roi", "F value"]
        interaction_p <- aov["group:roi", "Pr(>F)"]
      }
    }, warning = function(w) {
      if (grepl("singular|converge", conditionMessage(w), ignore.case = TRUE)) {
        singular <<- TRUE
      }
    }, error = function(e) {
      converged <<- FALSE
      message("  Omnibus LMM failed for ", cname, "/", dv_name, ": ", conditionMessage(e))
    })

    results[[length(results) + 1]] <- data.frame(
      contrast = cname,
      dv = dv_name,
      group_a = ga,
      group_b = gb,
      n_a = n_a,
      n_b = n_b,
      n_rois = n_rois,
      group_F = as.numeric(group_F),
      group_p = as.numeric(group_p),
      roi_F = as.numeric(roi_F),
      roi_p = as.numeric(roi_p),
      interaction_F = as.numeric(interaction_F),
      interaction_p = as.numeric(interaction_p),
      converged = converged,
      singular = singular,
      stringsAsFactors = FALSE
    )
  }

  omnibus_df <- bind_rows(results)
  if (nrow(omnibus_df) > 0) {
    # No FDR needed (single DV, no band dimension) — use raw p as q
    omnibus_df$group_q <- omnibus_df$group_p
    omnibus_df$group_significant <- omnibus_df$group_q < 0.05
    omnibus_df$interaction_q <- omnibus_df$interaction_p
    omnibus_df$interaction_significant <- omnibus_df$interaction_q < 0.05
  }
  return(omnibus_df)
}


#' Run emmeans post-hoc per ROI for aperiodic DV
run_posthoc_emmeans_aperiodic <- function(ap_df, contrasts, dv_name, omnibus_df, gate = TRUE) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    if (gate && nrow(omnibus_df) > 0) {
      omni_row <- omnibus_df %>%
        filter(contrast == cname, dv == dv_name)
      if (nrow(omni_row) == 0) next
      if (!isTRUE(omni_row$group_significant[1]) &&
          !isTRUE(omni_row$interaction_significant[1])) next
    }

    cdata <- ap_df %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    cdata$group <- factor(cdata$group, levels = c(gb, ga))
    cdata$roi <- factor(cdata$roi)
    cdata$dv <- cdata[[dv_name]]

    tryCatch({
      fit <- lmer(dv ~ group * roi + (1 | subject), data = cdata)

      emm <- emmeans(fit, pairwise ~ group | roi)
      con_df <- as.data.frame(emm$contrasts)
      emm_df <- as.data.frame(emm$emmeans)

      resid_sd <- sigma(fit)
      con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

      for (i in seq_len(nrow(con_df))) {
        roi_name <- as.character(con_df$roi[i])

        emm_a <- emm_df %>% filter(roi == roi_name, group == ga) %>% pull(emmean)
        emm_b <- emm_df %>% filter(roi == roi_name, group == gb) %>% pull(emmean)
        hg <- con_df$estimate[i] / resid_sd

        results[[length(results) + 1]] <- data.frame(
          contrast = cname,
          dv = dv_name,
          roi = roi_name,
          estimate = con_df$estimate[i],
          SE = con_df$SE[i],
          df = con_df$df[i],
          t_ratio = con_df$t.ratio[i],
          p_value = con_df$p.value[i],
          q_value = con_df$q_value[i],
          emmean_a = if (length(emm_a) > 0) emm_a[1] else NA,
          emmean_b = if (length(emm_b) > 0) emm_b[1] else NA,
          hedges_g = hg,
          significant = con_df$q_value[i] < 0.05,
          stringsAsFactors = FALSE
        )
      }
    }, warning = function(w) {
      # continue on singular fit
    }, error = function(e) {
      message("  Post-hoc failed for ", cname, "/", dv_name, ": ", conditionMessage(e))
    })
  }

  bind_rows(results)
}


#' Aggregate ROI-level aperiodic data to region-level means
aggregate_to_regions_aperiodic <- function(ap_df, roi_categories) {
  roi_to_region <- data.frame(
    roi = unlist(roi_categories),
    region = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  ap_df %>%
    inner_join(roi_to_region, by = "roi") %>%
    group_by(subject, group, region) %>%
    summarise(
      exponent = mean(exponent, na.rm = TRUE),
      offset = mean(offset, na.rm = TRUE),
      r_squared = mean(r_squared, na.rm = TRUE),
      n_peaks = mean(n_peaks, na.rm = TRUE),
      .groups = "drop"
    )
}


#' Run omnibus LMM at region level for aperiodic DV
run_omnibus_lmm_region_aperiodic <- function(ap_df, contrasts, roi_categories, dv_name) {
  region_df <- aggregate_to_regions_aperiodic(ap_df, roi_categories)
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    cdata <- region_df %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    n_a <- length(unique(cdata$subject[cdata$group == ga]))
    n_b <- length(unique(cdata$subject[cdata$group == gb]))
    n_regions <- length(unique(cdata$region))

    cdata$group <- factor(cdata$group, levels = c(gb, ga))
    cdata$region <- factor(cdata$region)
    cdata$dv <- cdata[[dv_name]]

    group_F <- NA; group_p <- NA
    region_F <- NA; region_p <- NA
    interaction_F <- NA; interaction_p <- NA
    converged <- TRUE; singular <- FALSE

    tryCatch({
      fit <- lmer(dv ~ group * region + (1 | subject), data = cdata)
      singular <- isSingular(fit)

      aov <- anova(fit, type = 3)

      if ("group" %in% rownames(aov)) {
        group_F <- aov["group", "F value"]
        group_p <- aov["group", "Pr(>F)"]
      }
      if ("region" %in% rownames(aov)) {
        region_F <- aov["region", "F value"]
        region_p <- aov["region", "Pr(>F)"]
      }
      if ("group:region" %in% rownames(aov)) {
        interaction_F <- aov["group:region", "F value"]
        interaction_p <- aov["group:region", "Pr(>F)"]
      }
    }, warning = function(w) {
      if (grepl("singular|converge", conditionMessage(w), ignore.case = TRUE)) {
        singular <<- TRUE
      }
    }, error = function(e) {
      converged <<- FALSE
    })

    results[[length(results) + 1]] <- data.frame(
      contrast = cname,
      dv = dv_name,
      group_a = ga,
      group_b = gb,
      n_a = n_a,
      n_b = n_b,
      n_regions = n_regions,
      group_F = as.numeric(group_F),
      group_p = as.numeric(group_p),
      region_F = as.numeric(region_F),
      region_p = as.numeric(region_p),
      interaction_F = as.numeric(interaction_F),
      interaction_p = as.numeric(interaction_p),
      converged = converged,
      singular = singular,
      stringsAsFactors = FALSE
    )
  }

  omnibus_df <- bind_rows(results)
  if (nrow(omnibus_df) > 0) {
    omnibus_df$group_q <- omnibus_df$group_p
    omnibus_df$group_significant <- omnibus_df$group_q < 0.05
    omnibus_df$interaction_q <- omnibus_df$interaction_p
    omnibus_df$interaction_significant <- omnibus_df$interaction_q < 0.05
  }
  return(omnibus_df)
}


#' Run emmeans post-hoc per region for aperiodic DV
run_posthoc_emmeans_region_aperiodic <- function(ap_df, contrasts, roi_categories,
                                                  omnibus_region_df, dv_name, gate = TRUE) {
  region_df <- aggregate_to_regions_aperiodic(ap_df, roi_categories)
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    if (gate && nrow(omnibus_region_df) > 0) {
      omni_row <- omnibus_region_df %>%
        filter(contrast == cname, dv == dv_name)
      if (nrow(omni_row) == 0) next
      if (!isTRUE(omni_row$group_significant[1]) &&
          !isTRUE(omni_row$interaction_significant[1])) next
    }

    cdata <- region_df %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    cdata$group <- factor(cdata$group, levels = c(gb, ga))
    cdata$region <- factor(cdata$region)
    cdata$dv <- cdata[[dv_name]]

    tryCatch({
      fit <- lmer(dv ~ group * region + (1 | subject), data = cdata)

      emm <- emmeans(fit, pairwise ~ group | region)
      con_df <- as.data.frame(emm$contrasts)
      emm_df <- as.data.frame(emm$emmeans)

      resid_sd <- sigma(fit)
      con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

      for (i in seq_len(nrow(con_df))) {
        region_name <- as.character(con_df$region[i])

        emm_a <- emm_df %>% filter(region == region_name, group == ga) %>% pull(emmean)
        emm_b <- emm_df %>% filter(region == region_name, group == gb) %>% pull(emmean)
        hg <- con_df$estimate[i] / resid_sd

        results[[length(results) + 1]] <- data.frame(
          contrast = cname,
          dv = dv_name,
          region = region_name,
          estimate = con_df$estimate[i],
          SE = con_df$SE[i],
          df = con_df$df[i],
          t_ratio = con_df$t.ratio[i],
          p_value = con_df$p.value[i],
          q_value = con_df$q_value[i],
          emmean_a = if (length(emm_a) > 0) emm_a[1] else NA,
          emmean_b = if (length(emm_b) > 0) emm_b[1] else NA,
          hedges_g = hg,
          significant = con_df$q_value[i] < 0.05,
          stringsAsFactors = FALSE
        )
      }
    }, warning = function(w) {
      # continue
    }, error = function(e) {
      message("  Region post-hoc failed for ", cname, "/", dv_name, ": ", conditionMessage(e))
    })
  }

  bind_rows(results)
}


# ============================================================
# Plotting functions
# ============================================================

plot_aperiodic_boxplot <- function(ap_df, dv_name, group_colors, group_labels,
                                    group_order, output_dir) {
  # Subject-level means (average across ROIs)
  subj_means <- ap_df %>%
    filter(group %in% group_order) %>%
    group_by(subject, group) %>%
    summarise(value = mean(.data[[dv_name]], na.rm = TRUE), .groups = "drop") %>%
    mutate(
      group = factor(group, levels = group_order),
      group_label = group_labels[as.character(group)]
    )

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  dv_label <- switch(dv_name,
    exponent = "Aperiodic Exponent",
    offset = "Aperiodic Offset (log power)",
    dv_name
  )

  p <- ggplot(subj_means, aes(x = group_label, y = value, fill = group_label)) +
    geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.7) +
    geom_jitter(width = 0.15, size = 2, alpha = 0.6,
                aes(color = group_label), show.legend = FALSE) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    labs(x = NULL, y = dv_label,
         title = paste0(dv_label, " by Group")) +
    theme_pub() +
    theme(legend.position = "none")

  fname <- paste0("aperiodic_", dv_name, "_boxplot.png")
  ggsave(file.path(output_dir, fname), p, width = 5, height = 5, dpi = 300)
  message("  Saved: ", fname)
}


plot_aperiodic_by_region <- function(ap_df, roi_categories, group_colors,
                                      group_labels, group_order, output_dir) {
  if (length(roi_categories) == 0) return(invisible(NULL))

  roi_to_cat <- data.frame(
    roi = unlist(roi_categories),
    category = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  # Region-level subject means
  region_data <- ap_df %>%
    filter(group %in% group_order) %>%
    inner_join(roi_to_cat, by = "roi") %>%
    group_by(subject, group, category) %>%
    summarise(exponent = mean(exponent, na.rm = TRUE), .groups = "drop") %>%
    mutate(
      group = factor(group, levels = group_order),
      group_label = group_labels[as.character(group)]
    )

  if (nrow(region_data) == 0) return(invisible(NULL))

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  p <- ggplot(region_data, aes(x = category, y = exponent, fill = group_label)) +
    geom_boxplot(position = position_dodge(0.8), width = 0.6,
                 outlier.shape = NA, alpha = 0.7) +
    geom_point(position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8),
               size = 1.2, alpha = 0.5, aes(color = group_label), show.legend = FALSE) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    labs(x = "Brain Region", y = "Aperiodic Exponent",
         title = "Aperiodic Exponent by Region and Group") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  ggsave(file.path(output_dir, "aperiodic_by_region.png"), p,
         width = max(7, length(roi_categories) * 1.5 + 2), height = 6, dpi = 300)
  message("  Saved: aperiodic_by_region.png")
}


plot_aperiodic_roi_forest <- function(posthoc_df, output_dir) {
  if (nrow(posthoc_df) == 0) return(invisible(NULL))

  for (dv_name in unique(posthoc_df$dv)) {
    for (cname in unique(posthoc_df$contrast)) {
      pdata <- posthoc_df %>%
        filter(contrast == cname, dv == dv_name) %>%
        mutate(
          roi = fct_reorder(roi, estimate),
          sig_label = ifelse(significant, "*", "")
        )

      if (nrow(pdata) == 0) next

      n_rois <- length(unique(pdata$roi))

      dv_label <- switch(dv_name,
        exponent = "Exponent",
        offset = "Offset",
        dv_name
      )

      p <- ggplot(pdata, aes(x = estimate, y = roi)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbar(aes(xmin = estimate - 1.96 * SE, xmax = estimate + 1.96 * SE),
                      width = 0.3, color = "grey40", orientation = "y") +
        geom_point(aes(color = significant), size = 2) +
        scale_color_manual(values = c("FALSE" = "grey60", "TRUE" = "#E74C3C"),
                           labels = c("n.s.", "p < .05"), name = NULL) +
        labs(x = "Group Difference (emmean)", y = NULL,
             title = paste0("ROI-Level Group Contrasts: ", cname, " (", dv_label, ")")) +
        theme_pub() +
        theme(axis.text.y = element_text(size = 7))

      fname <- paste0("roi_forest_plot_", dv_name, "_", cname, ".png")
      ggsave(file.path(output_dir, fname), p,
             width = 10, height = max(8, n_rois * 0.22 + 2),
             dpi = 300, limitsize = FALSE)
      message("  Saved: ", fname)
    }
  }
}


# ============================================================
# Report writer
# ============================================================

write_aperiodic_summary <- function(omnibus_df, posthoc_df, config, n_subjects, sfreq,
                                     fig_dir, output_path, fitting_method,
                                     omnibus_region_df = data.frame(),
                                     posthoc_region_df = data.frame()) {
  lines <- character()
  add <- function(...) lines <<- c(lines, paste0(...))

  has_region <- nrow(omnibus_region_df) > 0

  # Header
  add("# Aperiodic (1/f) Analysis \u2014 ", config$name)
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

  add("**Analysis:** Aperiodic (1/f) Spectral Decomposition")
  add("")
  add("**Groups:** ", group_str)
  add("")
  add("**Sampling Rate:** ", sfreq, " Hz")
  add("")
  add("**Fitting Method:** ", fitting_method)
  add("")
  add("**Fitting Range:** 2\u201350 Hz")
  add("")
  add("**Dependent Variables:** Aperiodic exponent (E/I balance proxy), aperiodic offset (broadband power)")
  add("")

  stats_lines <- paste0(
    "**Statistics:** Omnibus LMM (dv ~ group * roi + (1|subject)), Type III ANOVA ",
    "with Satterthwaite df (lme4/lmerTest). ")
  if (has_region) {
    stats_lines <- paste0(stats_lines,
      "Region-level analysis: ROIs averaged within anatomical regions, ",
      "then dv ~ group * region + (1|subject). ")
  }
  stats_lines <- paste0(stats_lines,
    "Post-hoc: emmeans pairwise group contrasts per ROI/region, gated on significant ",
    "omnibus effects, Holm correction. Hedges' g = emmean difference / residual SD. ",
    "No FDR correction across bands (single DV per model).")
  add(stats_lines)
  add("")

  # --- Region-level results ---
  if (has_region) {
    add("## Region-Level LMM Results")
    add("")
    add("ROIs averaged within anatomical regions.")
    add("")

    # Format omnibus table
    display_cols <- intersect(
      c("contrast", "dv", "n_a", "n_b", "n_regions",
        "group_F", "group_p", "group_q", "group_significant",
        "interaction_F", "interaction_p", "interaction_q",
        "interaction_significant", "singular"),
      names(omnibus_region_df)
    )
    tbl <- omnibus_region_df[, display_cols]
    for (col in names(tbl)) {
      if (is.numeric(tbl[[col]])) tbl[[col]] <- sprintf("%.4f", tbl[[col]])
      if (is.logical(tbl[[col]])) tbl[[col]] <- ifelse(tbl[[col]], "**Yes**", "No")
    }
    add("| ", paste(names(tbl), collapse = " | "), " |")
    add("| ", paste(rep("---", ncol(tbl)), collapse = " | "), " |")
    for (i in seq_len(nrow(tbl))) {
      add("| ", paste(tbl[i, ], collapse = " | "), " |")
    }
    add("")

    if (nrow(posthoc_region_df) > 0) {
      sig_region <- posthoc_region_df %>% filter(significant == TRUE)
      add("### Region-Level Post-Hoc")
      add("")
      if (nrow(sig_region) > 0) {
        add("Significant region-level group differences (Holm-corrected q < 0.05):")
        add("")
        add("| DV | Region | Estimate | SE | t | q | Hedges' g |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for (i in seq_len(nrow(sig_region))) {
          row <- sig_region[i, ]
          add(sprintf("| %s | %s | %.4f | %.4f | %.2f | %.4f | %.2f |",
                      row$dv, row$region, row$estimate, row$SE, row$t_ratio,
                      row$q_value, row$hedges_g))
        }
        add("")
      } else {
        add("No individual regions reached significance after Holm correction.")
        add("")
      }
    } else {
      add("*Region post-hoc not performed (no significant omnibus effects).*")
      add("")
    }
  }

  # --- ROI-level results ---
  add("## ROI-Level LMM Results")
  add("")
  if (nrow(omnibus_df) > 0) {
    display_cols <- intersect(
      c("contrast", "dv", "n_a", "n_b", "n_rois",
        "group_F", "group_p", "group_q", "group_significant",
        "interaction_F", "interaction_p", "interaction_q",
        "interaction_significant", "singular"),
      names(omnibus_df)
    )
    tbl <- omnibus_df[, display_cols]
    for (col in names(tbl)) {
      if (is.numeric(tbl[[col]])) tbl[[col]] <- sprintf("%.4f", tbl[[col]])
      if (is.logical(tbl[[col]])) tbl[[col]] <- ifelse(tbl[[col]], "**Yes**", "No")
    }
    add("| ", paste(names(tbl), collapse = " | "), " |")
    add("| ", paste(rep("---", ncol(tbl)), collapse = " | "), " |")
    for (i in seq_len(nrow(tbl))) {
      add("| ", paste(tbl[i, ], collapse = " | "), " |")
    }
    add("")
  }

  add("### ROI-Level Post-Hoc Contrasts")
  add("")
  if (nrow(posthoc_df) > 0) {
    sig_posthoc <- posthoc_df %>% filter(significant == TRUE)
    if (nrow(sig_posthoc) > 0) {
      add("Significant ROI-level group differences (Holm-corrected q < 0.05):")
      add("")
      for (dv_name in unique(sig_posthoc$dv)) {
        dv_sig <- sig_posthoc %>% filter(dv == dv_name)
        add("#### ", dv_name)
        add("")
        add("| ROI | Estimate | SE | t | q | Hedges' g |")
        add("| --- | --- | --- | --- | --- | --- |")
        for (i in seq_len(nrow(dv_sig))) {
          row <- dv_sig[i, ]
          add(sprintf("| %s | %.4f | %.4f | %.2f | %.4f | %.2f |",
                      row$roi, row$estimate, row$SE, row$t_ratio,
                      row$q_value, row$hedges_g))
        }
        add("")
      }
    } else {
      add("No individual ROIs reached significance after Holm correction.")
      add("")
    }
    add("**Total ROIs tested:** ", length(unique(posthoc_df$roi)))
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

  all_omnibus <- bind_rows(
    if (has_region) omnibus_region_df %>% mutate(level = "region") else data.frame(),
    if (nrow(omnibus_df) > 0) omnibus_df %>% mutate(level = "ROI") else data.frame()
  )

  if (nrow(all_omnibus) > 0) {
    for (i in seq_len(nrow(all_omnibus))) {
      row <- all_omnibus[i, ]
      findings <- character()
      if (isTRUE(row$group_significant))
        findings <- c(findings, sprintf("group main effect (F=%.2f, p=%.4f)",
                                         row$group_F, row$group_p))
      if (isTRUE(row$interaction_significant))
        findings <- c(findings, sprintf("group x %s interaction (F=%.2f, p=%.4f)",
                                         row$level, row$interaction_F, row$interaction_p))
      if (length(findings) > 0) {
        any_sig <- TRUE
        add(sprintf("- **%s** [%s, %s-level]: %s", row$dv, row$contrast, row$level,
                    paste(findings, collapse = "; ")))
      }
    }
  }

  if (!any_sig) {
    add("- No significant group effects found for aperiodic exponent or offset.")
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


# ============================================================
# Main execution
# ============================================================

dvs <- c("exponent", "offset")

all_omnibus <- list()
all_posthoc <- list()
all_omnibus_region <- list()
all_posthoc_region <- list()

for (dv_name in dvs) {
  message("\n=== DV: ", dv_name, " ===")

  # ROI-level omnibus
  message("Running ROI-level omnibus LMM (group * roi)...")
  omnibus <- run_omnibus_lmm_aperiodic(ap_df, config$contrasts, dv_name)
  all_omnibus[[dv_name]] <- omnibus

  if (nrow(omnibus) > 0) {
    for (i in seq_len(nrow(omnibus))) {
      row <- omnibus[i, ]
      grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
      int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
      message(sprintf("  %s | %s: group F=%.2f p=%.4f%s | interaction F=%.2f p=%.4f%s",
                      row$contrast, row$dv,
                      row$group_F, row$group_p, grp_sig,
                      row$interaction_F, row$interaction_p, int_sig))
    }
  }

  # ROI-level post-hoc
  message("Running ROI-level post-hoc emmeans...")
  posthoc <- run_posthoc_emmeans_aperiodic(ap_df, config$contrasts, dv_name, omnibus)
  all_posthoc[[dv_name]] <- posthoc

  if (nrow(posthoc) > 0) {
    sig_count <- sum(posthoc$significant, na.rm = TRUE)
    message("  ", nrow(posthoc), " ROI contrasts, ", sig_count, " significant")
  } else {
    message("  No post-hoc tests (no significant omnibus effects)")
  }

  # Region-level (if roi_categories defined)
  if (length(config$roi_categories) > 0) {
    message("Running region-level omnibus LMM (group * region)...")
    omnibus_reg <- run_omnibus_lmm_region_aperiodic(ap_df, config$contrasts,
                                                     config$roi_categories, dv_name)
    all_omnibus_region[[dv_name]] <- omnibus_reg

    if (nrow(omnibus_reg) > 0) {
      for (i in seq_len(nrow(omnibus_reg))) {
        row <- omnibus_reg[i, ]
        grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
        int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
        message(sprintf("  %s | %s: group F=%.2f p=%.4f%s | interaction F=%.2f p=%.4f%s",
                        row$contrast, row$dv,
                        row$group_F, row$group_p, grp_sig,
                        row$interaction_F, row$interaction_p, int_sig))
      }
    }

    message("Running region-level post-hoc emmeans...")
    posthoc_reg <- run_posthoc_emmeans_region_aperiodic(
      ap_df, config$contrasts, config$roi_categories, omnibus_reg, dv_name)
    all_posthoc_region[[dv_name]] <- posthoc_reg

    if (nrow(posthoc_reg) > 0) {
      sig_count <- sum(posthoc_reg$significant, na.rm = TRUE)
      message("  ", nrow(posthoc_reg), " region contrasts, ", sig_count, " significant")
    } else {
      message("  No region post-hoc tests (no significant omnibus effects)")
    }
  }
}

# Combine results across DVs
omnibus_df <- bind_rows(all_omnibus)
posthoc_df <- bind_rows(all_posthoc)
omnibus_region_df <- bind_rows(all_omnibus_region)
posthoc_region_df <- bind_rows(all_posthoc_region)

# --- Export tables ---
message("\nExporting tables...")
if (nrow(omnibus_df) > 0) {
  write_csv(omnibus_df, file.path(tbl_dir, "aperiodic_omnibus.csv"))
  message("  Saved: tables/aperiodic_omnibus.csv")
}
if (nrow(posthoc_df) > 0) {
  write_csv(posthoc_df, file.path(tbl_dir, "aperiodic_posthoc_roi.csv"))
  message("  Saved: tables/aperiodic_posthoc_roi.csv")
}
if (nrow(omnibus_region_df) > 0) {
  write_csv(omnibus_region_df, file.path(tbl_dir, "aperiodic_omnibus_region.csv"))
  message("  Saved: tables/aperiodic_omnibus_region.csv")
}
if (nrow(posthoc_region_df) > 0) {
  write_csv(posthoc_region_df, file.path(tbl_dir, "aperiodic_posthoc_region.csv"))
  message("  Saved: tables/aperiodic_posthoc_region.csv")
}

# --- Figures ---
message("\nGenerating figures...")

# Boxplots for exponent and offset
for (dv_name in dvs) {
  plot_aperiodic_boxplot(ap_df, dv_name, group_colors, group_labels,
                          group_order, fig_dir)
}

# Region bar chart
if (length(config$roi_categories) > 0) {
  plot_aperiodic_by_region(ap_df, config$roi_categories, group_colors,
                            group_labels, group_order, fig_dir)
}

# Forest plots (ROI-level)
if (nrow(posthoc_df) > 0) {
  plot_aperiodic_roi_forest(posthoc_df, fig_dir)
}

# --- Summary report ---
message("\nWriting summary...")

n_subjects <- ap_df %>%
  dplyr::distinct(subject, group) %>%
  dplyr::count(group) %>%
  { setNames(.$n, .$group) }

sfreq <- if (!is.null(config$sfreq)) config$sfreq else 500
fitting_method <- if (nrow(ap_df) > 0) ap_df$method[1] else "unknown"

write_aperiodic_summary(omnibus_df, posthoc_df, config, n_subjects, sfreq,
                         fig_dir, file.path(output_dir, "ANALYSIS_SUMMARY.md"),
                         fitting_method,
                         omnibus_region_df = omnibus_region_df,
                         posthoc_region_df = posthoc_region_df)

message("\nDone. Output: ", output_dir)
