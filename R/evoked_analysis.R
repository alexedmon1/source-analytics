#!/usr/bin/env Rscript
# evoked_analysis.R — R entry point for evoked response statistical analysis
#
# Called by Python: Rscript R/evoked_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads evoked_measures.csv (scalar ITC/ERSP/STP values per subject/ROI/measure),
# runs LMM + emmeans for each measure, generates figures, writes summary.

library(argparse)
library(yaml)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggsignif)

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
source(file.path(script_dir, "report.R"))

# --- Argument parsing ---
parser <- ArgumentParser(description = "Evoked response statistical analysis (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing evoked_measures.csv")
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
args <- parser$parse_args()

data_dir <- args$data_dir
config_path <- args$config
output_dir <- args$output_dir
figures_only <- args$figures_only
no_figures <- args$no_figures

if (no_figures) {
  ggsave <- function(...) invisible(NULL)
}

fig_dir <- if (!is.null(args$fig_dir)) args$fig_dir else file.path(output_dir, "figures")
tbl_dir <- if (!is.null(args$tbl_dir)) args$tbl_dir else file.path(output_dir, "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

# --- Load data ---
message("Loading data...")
measures_df <- read_csv(file.path(data_dir, "evoked_measures.csv"), show_col_types = FALSE)
message("  evoked_measures.csv: ", nrow(measures_df), " rows")

# Exclude Corpus Callosum ROIs
cc_rois <- c("Corpus_Callosum_Genu_L", "Corpus_Callosum_Genu_R",
             "Corpus_Callosum_Body_L", "Corpus_Callosum_Body_R",
             "Corpus_Callosum_Splenium_L", "Corpus_Callosum_Splenium_R")
n_before <- nrow(measures_df)
measures_df <- measures_df %>% filter(!roi %in% cc_rois)
message("  Excluded CC ROIs: ", n_before, " -> ", nrow(measures_df), " rows")

# --- Load config ---
config <- read_yaml(config_path)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))

# Get unique measure names
measure_names <- unique(measures_df$measure_name)
message("Measures: ", paste(measure_names, collapse = ", "))

if (!figures_only) {
# --- Run LMM for each contrast x measure ---
all_omnibus <- list()
all_posthoc <- list()

for (contrast in config$contrasts) {
  cname <- contrast$name
  ga <- contrast$group_a
  gb <- contrast$group_b

  for (mname in measure_names) {
    mdata <- measures_df %>% filter(measure_name == mname, group %in% c(ga, gb))
    if (nrow(mdata) == 0) next

    mtype <- mdata$measure_type[1]
    n_a <- length(unique(mdata$subject[mdata$group == ga]))
    n_b <- length(unique(mdata$subject[mdata$group == gb]))
    n_rois <- length(unique(mdata$roi))

    mdata$group <- factor(mdata$group, levels = c(ga, gb))
    mdata$roi <- factor(mdata$roi)

    group_F <- NA; group_p <- NA
    roi_F <- NA; roi_p <- NA
    interaction_F <- NA; interaction_p <- NA
    converged <- TRUE; singular <- FALSE

    tryCatch({
      fit <- lmer(value ~ group * roi + (1 | subject), data = mdata)
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
      message("  LMM failed for ", cname, "/", mname, ": ", conditionMessage(e))
    })

    all_omnibus[[length(all_omnibus) + 1]] <- data.frame(
      contrast = cname,
      measure = mname,
      measure_type = mtype,
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
}

omnibus_df <- bind_rows(all_omnibus)

# FDR correction across measures within each contrast
if (nrow(omnibus_df) > 0) {
  omnibus_df <- omnibus_df %>%
    group_by(contrast) %>%
    mutate(
      group_q = p.adjust(group_p, method = "BH"),
      group_significant = group_q < 0.05,
      interaction_q = p.adjust(interaction_p, method = "BH"),
      interaction_significant = interaction_q < 0.05
    ) %>%
    ungroup()

  message("\n=== Omnibus Results ===")
  for (i in seq_len(nrow(omnibus_df))) {
    row <- omnibus_df[i, ]
    grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
    int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
    message(sprintf("  %s | %s: group F=%.2f q=%.4f%s | interaction F=%.2f q=%.4f%s",
                    row$contrast, row$measure,
                    row$group_F, row$group_q, grp_sig,
                    row$interaction_F, row$interaction_q, int_sig))
  }
}

# --- Post-hoc emmeans (gated on significant omnibus) ---
message("\nRunning post-hoc emmeans...")

for (contrast in config$contrasts) {
  cname <- contrast$name
  ga <- contrast$group_a
  gb <- contrast$group_b

  for (mname in measure_names) {
    # Gate on omnibus significance
    if (nrow(omnibus_df) > 0) {
      omni_row <- omnibus_df %>%
        filter(contrast == cname, measure == mname)
      if (nrow(omni_row) == 0) next
      if (!isTRUE(omni_row$group_significant[1]) &&
          !isTRUE(omni_row$interaction_significant[1])) next
    }

    mdata <- measures_df %>% filter(measure_name == mname, group %in% c(ga, gb))
    if (nrow(mdata) == 0) next

    mtype <- mdata$measure_type[1]
    mdata$group <- factor(mdata$group, levels = c(ga, gb))
    mdata$roi <- factor(mdata$roi)

    tryCatch({
      fit <- lmer(value ~ group * roi + (1 | subject), data = mdata)

      emm <- emmeans(fit, pairwise ~ group | roi)
      con_df <- as.data.frame(emm$contrasts)
      emm_df <- as.data.frame(emm$emmeans)

      resid_sd <- sigma(fit)
      con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

      for (i in seq_len(nrow(con_df))) {
        roi_name <- as.character(con_df$roi[i])

        emm_a <- emm_df %>%
          filter(roi == roi_name, group == ga) %>%
          pull(emmean)
        emm_b <- emm_df %>%
          filter(roi == roi_name, group == gb) %>%
          pull(emmean)

        # Explicitly compute estimate as ga - gb to guarantee correct direction
        est <- if (length(emm_a) > 0 && length(emm_b) > 0) emm_a[1] - emm_b[1] else con_df$estimate[i]
        t_val <- est / con_df$SE[i]
        hg <- est / resid_sd

        all_posthoc[[length(all_posthoc) + 1]] <- data.frame(
          contrast = cname,
          measure = mname,
          measure_type = mtype,
          roi = roi_name,
          estimate = est,
          SE = con_df$SE[i],
          df = con_df$df[i],
          t_ratio = t_val,
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
      # Continue on singular fit warnings
    }, error = function(e) {
      message("  Post-hoc failed for ", cname, "/", mname, ": ", conditionMessage(e))
    })
  }
}

posthoc_df <- bind_rows(all_posthoc)

if (nrow(posthoc_df) > 0) {
  sig_count <- sum(posthoc_df$significant, na.rm = TRUE)
  message("  ", nrow(posthoc_df), " ROI contrasts, ", sig_count, " significant")
}

# --- Region-level analysis (if roi_categories defined) ---
omnibus_region_df <- data.frame()
posthoc_region_df <- data.frame()

if (length(config$roi_categories) > 0) {
  message("\nRunning region-level analysis...")

  # Map ROIs to regions and average
  roi_to_region <- data.frame(
    roi = unlist(config$roi_categories),
    region = rep(names(config$roi_categories), lengths(config$roi_categories)),
    stringsAsFactors = FALSE
  )

  region_df <- measures_df %>%
    inner_join(roi_to_region, by = "roi") %>%
    group_by(subject, group, region, measure_name, measure_type) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop")

  all_omnibus_reg <- list()
  all_posthoc_reg <- list()

  for (contrast in config$contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (mname in measure_names) {
      rdata <- region_df %>% filter(measure_name == mname, group %in% c(ga, gb))
      if (nrow(rdata) == 0) next

      mtype <- rdata$measure_type[1]
      n_a <- length(unique(rdata$subject[rdata$group == ga]))
      n_b <- length(unique(rdata$subject[rdata$group == gb]))
      n_regions <- length(unique(rdata$region))

      rdata$group <- factor(rdata$group, levels = c(ga, gb))
      rdata$region <- factor(rdata$region)

      group_F <- NA; group_p <- NA
      region_F <- NA; region_p <- NA
      interaction_F <- NA; interaction_p <- NA
      converged <- TRUE; singular <- FALSE

      tryCatch({
        fit <- lmer(value ~ group * region + (1 | subject), data = rdata)
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

      all_omnibus_reg[[length(all_omnibus_reg) + 1]] <- data.frame(
        contrast = cname,
        measure = mname,
        measure_type = mtype,
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
  }

  omnibus_region_df <- bind_rows(all_omnibus_reg)
  if (nrow(omnibus_region_df) > 0) {
    omnibus_region_df <- omnibus_region_df %>%
      group_by(contrast) %>%
      mutate(
        group_q = p.adjust(group_p, method = "BH"),
        group_significant = group_q < 0.05,
        interaction_q = p.adjust(interaction_p, method = "BH"),
        interaction_significant = interaction_q < 0.05
      ) %>%
      ungroup()
  }

  # Region post-hoc
  for (contrast in config$contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (mname in measure_names) {
      if (nrow(omnibus_region_df) > 0) {
        omni_row <- omnibus_region_df %>%
          filter(contrast == cname, measure == mname)
        if (nrow(omni_row) == 0) next
        if (!isTRUE(omni_row$group_significant[1]) &&
            !isTRUE(omni_row$interaction_significant[1])) next
      }

      rdata <- region_df %>% filter(measure_name == mname, group %in% c(ga, gb))
      if (nrow(rdata) == 0) next

      mtype <- rdata$measure_type[1]
      rdata$group <- factor(rdata$group, levels = c(ga, gb))
      rdata$region <- factor(rdata$region)

      tryCatch({
        fit <- lmer(value ~ group * region + (1 | subject), data = rdata)

        emm <- emmeans(fit, pairwise ~ group | region)
        con_df <- as.data.frame(emm$contrasts)
        emm_df <- as.data.frame(emm$emmeans)

        resid_sd <- sigma(fit)
        con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

        for (i in seq_len(nrow(con_df))) {
          region_name <- as.character(con_df$region[i])

          emm_a <- emm_df %>%
            filter(region == region_name, group == ga) %>%
            pull(emmean)
          emm_b <- emm_df %>%
            filter(region == region_name, group == gb) %>%
            pull(emmean)

          # Explicitly compute estimate as ga - gb to guarantee correct direction
          est <- if (length(emm_a) > 0 && length(emm_b) > 0) emm_a[1] - emm_b[1] else con_df$estimate[i]
          t_val <- est / con_df$SE[i]
          hg <- est / resid_sd

          all_posthoc_reg[[length(all_posthoc_reg) + 1]] <- data.frame(
            contrast = cname,
            measure = mname,
            measure_type = mtype,
            region = region_name,
            estimate = est,
            SE = con_df$SE[i],
            df = con_df$df[i],
            t_ratio = t_val,
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
        # Continue on singular fit warnings
      }, error = function(e) {
        message("  Region post-hoc failed for ", cname, "/", mname, ": ", conditionMessage(e))
      })
    }
  }

  posthoc_region_df <- bind_rows(all_posthoc_reg)
}

# --- Export tables (clean up stale files first) ---
message("\nExporting tables...")
for (f in c("evoked_omnibus.csv", "evoked_posthoc_roi.csv",
            "evoked_omnibus_region.csv", "evoked_posthoc_region.csv")) {
  fp <- file.path(tbl_dir, f)
  if (file.exists(fp)) file.remove(fp)
}

if (nrow(omnibus_df) > 0) {
  write_csv(omnibus_df, file.path(tbl_dir, "evoked_omnibus.csv"))
  message("  Saved: tables/evoked_omnibus.csv")
}
if (nrow(posthoc_df) > 0) {
  write_csv(posthoc_df, file.path(tbl_dir, "evoked_posthoc_roi.csv"))
  message("  Saved: tables/evoked_posthoc_roi.csv")
}
if (nrow(omnibus_region_df) > 0) {
  write_csv(omnibus_region_df, file.path(tbl_dir, "evoked_omnibus_region.csv"))
  message("  Saved: tables/evoked_omnibus_region.csv")
}
if (nrow(posthoc_region_df) > 0) {
  write_csv(posthoc_region_df, file.path(tbl_dir, "evoked_posthoc_region.csv"))
  message("  Saved: tables/evoked_posthoc_region.csv")
}

# --- Global posthoc (marginal group comparisons averaged over ROIs) ---
message("\nComputing global posthoc (marginal group comparisons)...")
global_posthoc_list <- list()
for (mname in measure_names) {
  gp_data <- measures_df %>%
    filter(measure_name == mname,
           group %in% unlist(lapply(config$contrasts, function(c) c(c$group_a, c$group_b))))

  if (nrow(gp_data) == 0) next

  gp <- run_posthoc_global(gp_data, config$contrasts, spatial_col = "roi",
                            dv_col = "value", dv_label = mname)
  if (nrow(gp) > 0) global_posthoc_list[[mname]] <- gp
}
global_posthoc_df <- bind_rows(global_posthoc_list)

if (nrow(global_posthoc_df) > 0) {
  # Re-apply FDR across all measure x contrast tests
  global_posthoc_df <- global_posthoc_df %>%
    mutate(
      q_value = p.adjust(p_value, method = "BH"),
      significant = q_value < 0.05,
      sig_label = sig_stars(q_value)
    )

  write_csv(global_posthoc_df, file.path(tbl_dir, "evoked_posthoc_global.csv"))
  message("  Saved: tables/evoked_posthoc_global.csv")
  sig_global <- global_posthoc_df %>% filter(significant == TRUE)
  message("  ", nrow(global_posthoc_df), " global contrasts, ", nrow(sig_global), " significant")
}

} else {
  message("Figures-only mode: loading existing tables...")
  omnibus_df <- tryCatch(read_csv(file.path(tbl_dir, "evoked_omnibus.csv"), show_col_types = FALSE), error = function(e) data.frame())
  posthoc_df <- tryCatch(read_csv(file.path(tbl_dir, "evoked_posthoc_roi.csv"), show_col_types = FALSE), error = function(e) data.frame())
  omnibus_region_df <- tryCatch(read_csv(file.path(tbl_dir, "evoked_omnibus_region.csv"), show_col_types = FALSE), error = function(e) data.frame())
  posthoc_region_df <- tryCatch(read_csv(file.path(tbl_dir, "evoked_posthoc_region.csv"), show_col_types = FALSE), error = function(e) data.frame())
  global_posthoc_df <- tryCatch(read_csv(file.path(tbl_dir, "evoked_posthoc_global.csv"), show_col_types = FALSE), error = function(e) data.frame())
}

# --- Figures ---
message("\nGenerating figures...")

# Group comparison violin plots per measure
for (mname in measure_names) {
  mdata <- measures_df %>% filter(measure_name == mname)
  if (nrow(mdata) == 0) next

  mtype <- mdata$measure_type[1]
  y_label <- switch(mtype,
    "itc" = "ITC",
    "ersp" = "ERSP (dB)",
    "stp" = "Power",
    "Value"
  )

  # Aggregate to region means if roi_categories present
  if (length(config$roi_categories) > 0) {
    roi_to_region <- data.frame(
      roi = unlist(config$roi_categories),
      region = rep(names(config$roi_categories), lengths(config$roi_categories)),
      stringsAsFactors = FALSE
    )

    plot_data <- mdata %>%
      inner_join(roi_to_region, by = "roi") %>%
      group_by(subject, group, region) %>%
      summarise(value = mean(value, na.rm = TRUE), .groups = "drop")

    plot_data$group <- factor(plot_data$group, levels = group_order)
    plot_data$region <- factor(plot_data$region, levels = names(config$roi_categories))

    p <- ggplot(plot_data, aes(x = region, y = value, fill = group)) +
      geom_boxplot(width = 0.6, alpha = 0.7, position = position_dodge(0.8),
                   outlier.shape = NA) +
      geom_jitter(aes(color = group), size = 1.5, alpha = 0.5,
                  position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.15),
                  show.legend = FALSE) +
      scale_fill_manual(values = group_colors, labels = group_labels) +
      scale_color_manual(values = group_colors, labels = group_labels) +
      labs(title = paste0(mname, " by Region"),
           x = "Region", y = y_label, fill = "Group") +
      theme_minimal(base_size = 14) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))

    fname <- file.path(fig_dir, paste0("evoked_", mname, "_by_region.png"))
    ggsave(fname, p, width = 14, height = 6, dpi = 150)
    message("  Saved: ", basename(fname))
  }

  # Mean across all ROIs per subject — simple group comparison
  subj_means <- mdata %>%
    group_by(subject, group) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop")

  subj_means$group <- factor(subj_means$group, levels = group_order)

  # Use group labels for x-axis
  subj_means$group_label <- group_labels[as.character(subj_means$group)]
  label_order <- group_labels[group_order]
  subj_means$group_label <- factor(subj_means$group_label, levels = label_order)

  color_vals <- group_colors[group_order]
  names(color_vals) <- label_order

  p2 <- ggplot(subj_means, aes(x = group_label, y = value, fill = group_label)) +
    geom_boxplot(width = 0.5, alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.1, size = 1.5, alpha = 0.5) +
    scale_fill_manual(values = color_vals, name = NULL) +
    labs(title = paste0(mname, " — Group Comparison (mean across ROIs)"),
         x = NULL, y = y_label) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none")

  # Add significance brackets
  if (nrow(global_posthoc_df) > 0) {
    m_sig <- global_posthoc_df %>% filter(dv == mname, significant == TRUE)

    if (nrow(m_sig) > 0) {
      y_max <- max(subj_means$value, na.rm = TRUE)
      y_range <- diff(range(subj_means$value, na.rm = TRUE))
      y_step <- y_range * 0.08

      for (j in seq_len(nrow(m_sig))) {
        row <- m_sig[j, ]
        label_a <- group_labels[row$group_a]
        label_b <- group_labels[row$group_b]
        y_pos <- y_max + y_step * j

        p2 <- p2 + geom_signif(
          comparisons = list(c(label_a, label_b)),
          annotations = row$sig_label,
          y_position = y_pos,
          tip_length = 0.02,
          textsize = 5,
          color = "black"
        )
      }
    }
  }

  fname2 <- file.path(fig_dir, paste0("evoked_", mname, "_group.png"))
  ggsave(fname2, p2, width = 6, height = 5, dpi = 150)
  message("  Saved: ", basename(fname2))
}

# TF heatmap if evoked_tfr.csv exists
tfr_file <- file.path(data_dir, "evoked_tfr.csv")
if (file.exists(tfr_file)) {
  message("  Generating TF heatmaps...")
  tfr_df <- read_csv(tfr_file, show_col_types = FALSE)

  for (mtype in c("itc", "ersp", "stp")) {
    for (grp in group_order) {
      gdata <- tfr_df %>% filter(group == grp)
      if (nrow(gdata) == 0) next

      # Average across subjects
      avg_data <- gdata %>%
        group_by(freq, time) %>%
        summarise(val = mean(.data[[mtype]], na.rm = TRUE), .groups = "drop")

      fill_label <- switch(mtype,
        "itc" = "ITC",
        "ersp" = "ERSP (dB)",
        "stp" = "Power"
      )

      p <- ggplot(avg_data, aes(x = time, y = freq, fill = val)) +
        geom_tile() +
        scale_fill_viridis_c(option = "inferno") +
        labs(title = paste0(toupper(mtype), " — ", grp),
             x = "Time (s)", y = "Frequency (Hz)", fill = fill_label) +
        theme_minimal(base_size = 14)

      fname <- file.path(fig_dir, paste0("tfr_", mtype, "_", grp, ".png"))
      ggsave(fname, p, width = 10, height = 6, dpi = 150)
      message("    Saved: ", basename(fname))
    }
  }
}

# Post-hoc forest plots
if (nrow(posthoc_df) > 0) {
  sig_ph <- posthoc_df %>% filter(significant == TRUE)
  if (nrow(sig_ph) > 0) {
    sig_ph$label <- paste0(sig_ph$contrast, " | ", sig_ph$roi)
    sig_ph <- sig_ph %>% arrange(desc(abs(hedges_g)))

    if (nrow(sig_ph) > 30) sig_ph <- sig_ph[1:30, ]

    p <- ggplot(sig_ph, aes(x = hedges_g, y = reorder(label, hedges_g))) +
      geom_point(aes(color = measure), size = 2) +
      geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5) +
      labs(title = "Significant Post-Hoc Effects (Top 30)",
           x = "Hedges' g", y = "", color = "Measure") +
      theme_minimal(base_size = 10)

    ggsave(file.path(fig_dir, "evoked_posthoc_forest.png"), p,
           width = 10, height = max(4, nrow(sig_ph) * 0.25), dpi = 150)
    message("  Saved: evoked_posthoc_forest.png")
  }
}

if (!figures_only) {
# --- Summary report ---
message("\nWriting summary...")

lines <- character()
add <- function(...) lines <<- c(lines, paste0(...))

# Header
add("# Evoked Response Analysis \u2014 ", config$name)
add("")
add("**Generated:** ", format(Sys.time(), "%Y-%m-%d %H:%M"))
add("")

# Methods
add("## Methods")
add("")

n_subjects <- measures_df %>%
  distinct(subject, group) %>%
  count(group) %>%
  { setNames(.$n, .$group) }

group_str <- paste(
  sapply(names(n_subjects), function(g) paste0(config$groups[[g]], " (n=", n_subjects[g], ")")),
  collapse = ", "
)

add("**Analysis:** Evoked Response (ITC, ERSP, STP)")
add("")
add("**Groups:** ", group_str)
add("")
add("**Measures:** ", paste(measure_names, collapse = ", "))
add("")

if (!is.null(config$evoked)) {
  add("**Epoch samples:** ", config$evoked$epoch_samples)
  add("")
  add("**Sampling rate:** ", config$evoked$sfreq, " Hz")
  add("")
  add("**Baseline:** ", paste(config$evoked$baseline, collapse = " to "), " s")
  add("")
  if (!is.null(config$evoked$tf_params)) {
    add("**TF params:** freq_range = ",
        paste(config$evoked$tf_params$freq_range, collapse = "-"), " Hz, ",
        "n_cycles = ", config$evoked$tf_params$n_cycles)
    add("")
  }
}

add("**Statistics:** LMM (value ~ group * roi + (1|subject)), Type III ANOVA. ",
    "FDR (BH) correction across measures within each contrast. ",
    "Post-hoc: emmeans pairwise group contrasts, gated on significant omnibus. ",
    "Holm correction within each measure.")
add("")

# Omnibus results
if (nrow(omnibus_df) > 0) {
  add("## ROI-Level Omnibus Results")
  add("")
  add("| Contrast | Measure | Type | n_a | n_b | group_F | group_q | Sig? | interact_F | interact_q | Sig? |")
  add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
  for (i in seq_len(nrow(omnibus_df))) {
    row <- omnibus_df[i, ]
    g_sig <- if (isTRUE(row$group_significant)) "**Yes**" else "No"
    i_sig <- if (isTRUE(row$interaction_significant)) "**Yes**" else "No"
    add(sprintf("| %s | %s | %s | %d | %d | %.2f | %.4f | %s | %.2f | %.4f | %s |",
                row$contrast, row$measure, row$measure_type,
                row$n_a, row$n_b,
                row$group_F, row$group_q, g_sig,
                row$interaction_F, row$interaction_q, i_sig))
  }
  add("")
}

# Region omnibus
if (nrow(omnibus_region_df) > 0) {
  add("## Region-Level Omnibus Results")
  add("")
  add("| Contrast | Measure | Type | n_a | n_b | group_F | group_q | Sig? | interact_F | interact_q | Sig? |")
  add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
  for (i in seq_len(nrow(omnibus_region_df))) {
    row <- omnibus_region_df[i, ]
    g_sig <- if (isTRUE(row$group_significant)) "**Yes**" else "No"
    i_sig <- if (isTRUE(row$interaction_significant)) "**Yes**" else "No"
    add(sprintf("| %s | %s | %s | %d | %d | %.2f | %.4f | %s | %.2f | %.4f | %s |",
                row$contrast, row$measure, row$measure_type,
                row$n_a, row$n_b,
                row$group_F, row$group_q, g_sig,
                row$interaction_F, row$interaction_q, i_sig))
  }
  add("")
}

# Post-hoc results
if (nrow(posthoc_df) > 0) {
  add("## Post-Hoc Contrasts (ROI-Level)")
  add("")
  sig_ph <- posthoc_df %>% filter(significant == TRUE)
  if (nrow(sig_ph) > 0) {
    add("Significant ROI-level group differences (Holm-corrected q < 0.05):")
    add("")
    for (mname in unique(sig_ph$measure)) {
      msig <- sig_ph %>% filter(measure == mname)
      add("### ", mname, " (", msig$measure_type[1], ")")
      add("")
      add("| Contrast | ROI | Estimate | SE | t | q | Hedges' g |")
      add("| --- | --- | --- | --- | --- | --- | --- |")
      for (i in seq_len(nrow(msig))) {
        row <- msig[i, ]
        add(sprintf("| %s | %s | %.4f | %.4f | %.2f | %.4f | %.2f |",
                    row$contrast, row$roi, row$estimate, row$SE,
                    row$t_ratio, row$q_value, row$hedges_g))
      }
      add("")
    }
  } else {
    add("No individual ROIs reached significance after Holm correction.")
    add("")
  }
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
      findings <- c(findings, sprintf("group x ROI interaction (F=%.2f, q=%.4f)", row$interaction_F, row$interaction_q))
    if (length(findings) > 0) {
      any_sig <- TRUE
      add(sprintf("- **%s** [%s]: %s", row$measure, row$contrast,
                  paste(findings, collapse = "; ")))
    }
  }
}

if (!any_sig) {
  add("- No measures reached significance after FDR correction.")
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
} # end if (!figures_only)

message("\nDone. Output: ", output_dir)
