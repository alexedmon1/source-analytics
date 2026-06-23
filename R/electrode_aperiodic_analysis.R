#!/usr/bin/env Rscript
# electrode_aperiodic_analysis.R — Electrode-level aperiodic (1/f) analysis
#
# Reads electrode_aperiodic_params.csv, runs:
#   1. Channel-level omnibus: dv ~ group * channel + (1|subject)
#   2. Region-nested omnibus: dv ~ group * region + (1|subject/channel)
#   3. Post-hoc emmeans for significant effects

library(argparse)
library(yaml)
library(readr)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)
library(effectsize)

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

tryCatch(source(file.path(script_dir, "stats_utils.R")), error = function(e) NULL)
tryCatch(source(file.path(script_dir, "hypothesis.R")), error = function(e) NULL)

# --- Argument parsing ---
parser <- ArgumentParser(description = "Electrode aperiodic analysis (R)")
parser$add_argument("--data-dir", required = TRUE)
parser$add_argument("--config", required = TRUE)
parser$add_argument("--output-dir", required = TRUE)
parser$add_argument("--fig-dir", default = NULL)
parser$add_argument("--tbl-dir", default = NULL)
parser$add_argument("--no-figures", action = "store_true", default = FALSE)
parser$add_argument("--hypothesis", default = NULL,
                    help = "Run only the named hypothesis(es) (comma-separated) from the design spec; default = all")
args <- parser$parse_args()

data_dir <- args$data_dir
output_dir <- args$output_dir
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
ap_df <- read_csv(file.path(data_dir, "electrode_aperiodic_params.csv"), show_col_types = FALSE)
message("  electrode_aperiodic_params.csv: ", nrow(ap_df), " rows")

config <- read_yaml(args$config)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order
electrode_categories <- config$electrode_categories

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))
if (length(electrode_categories) > 0) {
  message("Electrode regions: ", length(electrode_categories),
          " (", paste(names(electrode_categories), collapse = ", "), ")")
}


# ============================================================
# Channel-level omnibus LMM: dv ~ group * channel + (1|subject)
# ============================================================
run_omnibus_channel <- function(ap_df, contrasts, dv_name) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    cdata <- ap_df %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    n_a <- length(unique(cdata$subject[cdata$group == ga]))
    n_b <- length(unique(cdata$subject[cdata$group == gb]))
    n_channels <- length(unique(cdata$channel))

    cdata$group <- factor(cdata$group, levels = c(ga, gb))
    cdata$channel <- factor(cdata$channel)
    cdata$dv <- cdata[[dv_name]]

    group_F <- NA; group_p <- NA
    interaction_F <- NA; interaction_p <- NA
    converged <- TRUE; singular <- FALSE

    tryCatch({
      fit <- lmer(dv ~ group * channel + (1 | subject), data = cdata)
      singular <- isSingular(fit)
      aov <- anova(fit, type = 3)

      if ("group" %in% rownames(aov)) {
        group_F <- aov["group", "F value"]
        group_p <- aov["group", "Pr(>F)"]
      }
      if ("group:channel" %in% rownames(aov)) {
        interaction_F <- aov["group:channel", "F value"]
        interaction_p <- aov["group:channel", "Pr(>F)"]
      }
    }, warning = function(w) {
      if (grepl("singular|converge", conditionMessage(w), ignore.case = TRUE)) {
        singular <<- TRUE
      }
    }, error = function(e) {
      converged <<- FALSE
      message("  LMM failed for ", cname, "/", dv_name, ": ", conditionMessage(e))
    })

    # Global Hedges' g
    subj_means <- cdata %>% group_by(subject, group) %>%
      summarise(mean_val = mean(dv, na.rm = TRUE), .groups = "drop")
    ga_vals <- subj_means$mean_val[subj_means$group == ga]
    gb_vals <- subj_means$mean_val[subj_means$group == gb]
    g_val <- tryCatch({
      as.numeric(hedges_g(ga_vals, gb_vals)$Hedges_g)
    }, error = function(e) NA)

    results[[length(results) + 1]] <- data.frame(
      contrast = cname, dv = dv_name,
      group_a = ga, group_b = gb,
      n_a = n_a, n_b = n_b, n_channels = n_channels,
      group_F = as.numeric(group_F), group_p = as.numeric(group_p),
      interaction_F = as.numeric(interaction_F), interaction_p = as.numeric(interaction_p),
      hedges_g = g_val,
      converged = converged, singular = singular,
      stringsAsFactors = FALSE
    )
  }

  df <- bind_rows(results)
  if (nrow(df) > 0) {
    df$group_significant <- df$group_p < 0.05
    df$interaction_significant <- df$interaction_p < 0.05
  }
  df
}


# ============================================================
# Channel-level post-hoc: emmeans per channel
# ============================================================
run_posthoc_channel <- function(ap_df, contrasts, dv_name, omnibus_df, gate = TRUE) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    if (gate && nrow(omnibus_df) > 0) {
      omni <- omnibus_df %>% filter(contrast == cname, dv == dv_name)
      if (nrow(omni) == 0) next
      if (!isTRUE(omni$group_significant[1]) && !isTRUE(omni$interaction_significant[1])) next
    }

    cdata <- ap_df %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    cdata$group <- factor(cdata$group, levels = c(ga, gb))
    cdata$channel <- factor(cdata$channel)
    cdata$dv <- cdata[[dv_name]]

    tryCatch({
      fit <- lmer(dv ~ group * channel + (1 | subject), data = cdata)
      emm <- emmeans(fit, pairwise ~ group | channel, lmer.df = "satterthwaite")
      contr <- as.data.frame(summary(emm$contrasts))

      for (i in seq_len(nrow(contr))) {
        ch <- contr$channel[i]
        ch_data <- cdata %>% filter(channel == ch)
        ga_vals <- ch_data$dv[ch_data$group == ga]
        gb_vals <- ch_data$dv[ch_data$group == gb]
        g_val <- tryCatch(as.numeric(hedges_g(ga_vals, gb_vals)$Hedges_g), error = function(e) NA)

        results[[length(results) + 1]] <- data.frame(
          contrast = cname, dv = dv_name, channel = as.character(ch),
          estimate = contr$estimate[i], SE = contr$SE[i],
          t_ratio = contr$t.ratio[i], p_value = contr$p.value[i],
          hedges_g = g_val,
          stringsAsFactors = FALSE
        )
      }
    }, error = function(e) {
      message("  Post-hoc failed for ", cname, "/", dv_name, ": ", conditionMessage(e))
    })
  }

  df <- bind_rows(results)
  if (nrow(df) > 0) {
    df$q_fdr <- p.adjust(df$p_value, method = "BH")
    df$significant <- df$q_fdr < 0.05
  }
  df
}


# ============================================================
# Region-nested omnibus: dv ~ group * region + (1|subject/channel)
# ============================================================
run_omnibus_region_nested <- function(ap_df, contrasts, electrode_categories, dv_name) {
  # Map channels to regions
  ch_to_region <- data.frame(
    channel = unlist(electrode_categories),
    region = rep(names(electrode_categories), lengths(electrode_categories)),
    stringsAsFactors = FALSE
  )
  region_df <- ap_df %>% inner_join(ch_to_region, by = "channel")

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

    cdata$group <- factor(cdata$group, levels = c(ga, gb))
    cdata$region <- factor(cdata$region)
    cdata$channel <- factor(cdata$channel)
    cdata$dv <- cdata[[dv_name]]

    group_F <- NA; group_p <- NA
    interaction_F <- NA; interaction_p <- NA
    converged <- TRUE; singular <- FALSE

    tryCatch({
      fit <- lmer(dv ~ group * region + (1 | subject / channel), data = cdata)
      singular <- isSingular(fit)
      aov <- anova(fit, type = 3)

      if ("group" %in% rownames(aov)) {
        group_F <- aov["group", "F value"]
        group_p <- aov["group", "Pr(>F)"]
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
      message("  Region-nested LMM failed for ", cname, "/", dv_name, ": ", conditionMessage(e))
    })

    # Global Hedges' g
    subj_means <- cdata %>% group_by(subject, group) %>%
      summarise(mean_val = mean(dv, na.rm = TRUE), .groups = "drop")
    ga_vals <- subj_means$mean_val[subj_means$group == ga]
    gb_vals <- subj_means$mean_val[subj_means$group == gb]
    g_val <- tryCatch(as.numeric(hedges_g(ga_vals, gb_vals)$Hedges_g), error = function(e) NA)

    results[[length(results) + 1]] <- data.frame(
      contrast = cname, dv = dv_name,
      group_a = ga, group_b = gb,
      n_a = n_a, n_b = n_b, n_regions = n_regions,
      group_F = as.numeric(group_F), group_p = as.numeric(group_p),
      interaction_F = as.numeric(interaction_F), interaction_p = as.numeric(interaction_p),
      hedges_g = g_val,
      converged = converged, singular = singular,
      stringsAsFactors = FALSE
    )
  }

  df <- bind_rows(results)
  if (nrow(df) > 0) {
    df$group_significant <- df$group_p < 0.05
    df$interaction_significant <- df$interaction_p < 0.05
  }
  df
}


# ============================================================
# Region-nested post-hoc: emmeans per region
# ============================================================
run_posthoc_region_nested <- function(ap_df, contrasts, electrode_categories, omnibus_df, dv_name, gate = TRUE) {
  ch_to_region <- data.frame(
    channel = unlist(electrode_categories),
    region = rep(names(electrode_categories), lengths(electrode_categories)),
    stringsAsFactors = FALSE
  )
  region_df <- ap_df %>% inner_join(ch_to_region, by = "channel")

  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    if (gate && nrow(omnibus_df) > 0) {
      omni <- omnibus_df %>% filter(contrast == cname, dv == dv_name)
      if (nrow(omni) == 0) next
      if (!isTRUE(omni$group_significant[1]) && !isTRUE(omni$interaction_significant[1])) next
    }

    cdata <- region_df %>% filter(group %in% c(ga, gb))
    if (nrow(cdata) == 0) next

    cdata$group <- factor(cdata$group, levels = c(ga, gb))
    cdata$region <- factor(cdata$region)
    cdata$channel <- factor(cdata$channel)
    cdata$dv <- cdata[[dv_name]]

    tryCatch({
      fit <- lmer(dv ~ group * region + (1 | subject / channel), data = cdata)
      emm <- emmeans(fit, pairwise ~ group | region, lmer.df = "satterthwaite")
      contr <- as.data.frame(summary(emm$contrasts))

      for (i in seq_len(nrow(contr))) {
        reg <- contr$region[i]
        reg_data <- cdata %>% filter(region == reg) %>%
          group_by(subject, group) %>%
          summarise(mean_val = mean(dv, na.rm = TRUE), .groups = "drop")
        ga_vals <- reg_data$mean_val[reg_data$group == ga]
        gb_vals <- reg_data$mean_val[reg_data$group == gb]
        g_val <- tryCatch(as.numeric(hedges_g(ga_vals, gb_vals)$Hedges_g), error = function(e) NA)

        results[[length(results) + 1]] <- data.frame(
          contrast = cname, dv = dv_name, region = as.character(reg),
          estimate = contr$estimate[i], SE = contr$SE[i],
          t_ratio = contr$t.ratio[i], p_value = contr$p.value[i],
          hedges_g = g_val,
          stringsAsFactors = FALSE
        )
      }
    }, error = function(e) {
      message("  Region posthoc failed for ", cname, "/", dv_name, ": ", conditionMessage(e))
    })
  }

  df <- bind_rows(results)
  if (nrow(df) > 0) {
    df$q_fdr <- p.adjust(df$p_value, method = "BH")
    df$significant <- df$q_fdr < 0.05
  }
  df
}


# ============================================================
# Main execution
# ============================================================
dvs <- c("exponent", "offset")

all_omnibus <- list()
all_omnibus_region <- list()
all_posthoc_region <- list()

for (dv_name in dvs) {
  message("\n=== DV: ", dv_name, " ===")

  # Channel-level omnibus [DIAGNOSTIC, not a hypothesis]
  message("Running channel-level omnibus LMM (group * channel) [diagnostic]...")
  omnibus <- run_omnibus_channel(ap_df, config$contrasts, dv_name)
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

  # Channel-level per-contrast post-hoc now comes from the hypothesis layer
  # (run_posthoc_channel retired). See the write_module_hypotheses call below.

  # Region-nested (if electrode_categories defined). KEPT on the legacy nested
  # model dv ~ group*region + (1|subject/channel); the hypothesis layer fits
  # (1|subject) only and cannot express this nesting.
  if (length(electrode_categories) > 0) {
    message("Running region-nested omnibus LMM (group * region, channels nested)...")
    omnibus_reg <- run_omnibus_region_nested(ap_df, config$contrasts, electrode_categories, dv_name)
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

    message("Running region-nested post-hoc emmeans...")
    posthoc_reg <- run_posthoc_region_nested(ap_df, config$contrasts, electrode_categories,
                                              omnibus_reg, dv_name)
    all_posthoc_region[[dv_name]] <- posthoc_reg

    if (nrow(posthoc_reg) > 0) {
      sig_count <- sum(posthoc_reg$significant, na.rm = TRUE)
      message("  ", nrow(posthoc_reg), " region contrasts, ", sig_count, " significant")
    } else {
      message("  No region post-hoc tests")
    }
  }
}

# --- Declarative hypotheses (hypothesis layer) — channel-level per-contrast ---
# Sole per-contrast engine for the channel level. electrode_aperiodic_posthoc_
# channel.csv is rebuilt from the contrast-kind rows (legacy schema via aliases;
# channel restored from spatial) so figures/report consume it unchanged.
message("\nRunning declarative hypotheses (hypothesis layer) — channel level...")
hyp_chan <- write_module_hypotheses(ap_df, config, tbl_dir, prefix = "electrode_aperiodic",
                                    dv_cols = dvs, spatial_col = "channel",
                                    band_col = NULL, hypothesis = args$hypothesis)

# Combine and export
omnibus_df <- bind_rows(all_omnibus)
posthoc_df <- if (!is.null(hyp_chan) && nrow(hyp_chan) > 0)
  hyp_chan[hyp_chan$kind == "contrast", , drop = FALSE] else data.frame()
if (nrow(posthoc_df) > 0) posthoc_df$channel <- posthoc_df$spatial
omnibus_region_df <- bind_rows(all_omnibus_region)
posthoc_region_df <- bind_rows(all_posthoc_region)

message("\nExporting tables...")
if (nrow(omnibus_df) > 0) {
  write_csv(omnibus_df, file.path(tbl_dir, "electrode_aperiodic_omnibus.csv"))
  message("  Saved: tables/electrode_aperiodic_omnibus.csv (diagnostic)")
}
if (nrow(posthoc_df) > 0) {
  write_csv(posthoc_df, file.path(tbl_dir, "electrode_aperiodic_posthoc_channel.csv"))
  message("  Saved: tables/electrode_aperiodic_posthoc_channel.csv (", nrow(posthoc_df),
          " rows, hypothesis-derived)")
}
if (nrow(omnibus_region_df) > 0) {
  write_csv(omnibus_region_df, file.path(tbl_dir, "electrode_aperiodic_omnibus_region_nested.csv"))
  message("  Saved: tables/electrode_aperiodic_omnibus_region_nested.csv")
}
if (nrow(posthoc_region_df) > 0) {
  write_csv(posthoc_region_df, file.path(tbl_dir, "electrode_aperiodic_posthoc_region_nested.csv"))
  message("  Saved: tables/electrode_aperiodic_posthoc_region_nested.csv")
}

# --- Summary ---
message("\nWriting summary...")
lines <- c(
  "# Electrode Aperiodic Analysis Summary\n",
  paste0("**Study:** ", config$name),
  paste0("**Groups:** ", paste(group_order, collapse = ", ")),
  paste0("**Channels:** ", length(unique(ap_df$channel))),
  ""
)

if (nrow(omnibus_df) > 0) {
  lines <- c(lines, "## Channel-Level Omnibus\n")
  for (i in seq_len(nrow(omnibus_df))) {
    row <- omnibus_df[i, ]
    lines <- c(lines, sprintf("- **%s** | %s: group F=%.2f, p=%.4f | interaction F=%.2f, p=%.4f | g=%.2f",
                               row$contrast, row$dv, row$group_F, row$group_p,
                               row$interaction_F, row$interaction_p, row$hedges_g))
  }
  lines <- c(lines, "")
}

if (nrow(omnibus_region_df) > 0) {
  lines <- c(lines, "## Region-Nested Omnibus\n")
  for (i in seq_len(nrow(omnibus_region_df))) {
    row <- omnibus_region_df[i, ]
    lines <- c(lines, sprintf("- **%s** | %s: group F=%.2f, p=%.4f | interaction F=%.2f, p=%.4f | g=%.2f",
                               row$contrast, row$dv, row$group_F, row$group_p,
                               row$interaction_F, row$interaction_p, row$hedges_g))
  }
}

writeLines(lines, file.path(output_dir, "ANALYSIS_SUMMARY.md"))
message("  Report written: ", file.path(output_dir, "ANALYSIS_SUMMARY.md"))

message("\nDone. Output: ", output_dir)
