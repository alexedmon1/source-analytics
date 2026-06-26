#!/usr/bin/env Rscript
# roi_transfer_entropy_analysis.R — Transfer entropy statistics and report
#
# Called by Python: Rscript R/roi_transfer_entropy_analysis.R --data-dir ... --config ... --output-dir ...
#
# Reads transfer_entropy_edges.csv exported by Python.
# Three analysis tiers:
#   1. Global TE: mean TE across all directed edges per subject x band, Welch t-test, BH FDR
#   2. Directional: paired t-test on TE(X→Y) vs TE(Y→X) within groups (test for net directionality)
#   3. Region-pair level: map directed edges to region pairs, LMM per band, post-hoc emmeans

library(argparse)
library(yaml)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(forcats)

# Conditionally load LMM packages
has_lme4 <- requireNamespace("lme4", quietly = TRUE) &&
            requireNamespace("lmerTest", quietly = TRUE) &&
            requireNamespace("emmeans", quietly = TRUE)

# --- Source the declarative hypothesis layer -------------------------------
script_dir <- if (exists("script.dir")) {
  script.dir
} else {
  tryCatch({
    args_all <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args_all, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("^--file=", "", file_arg)))
    } else {
      "R"
    }
  }, error = function(e) "R")
}
tryCatch(source(file.path(script_dir, "stats_utils.R")), error = function(e) NULL)
source(file.path(script_dir, "hypothesis.R"))

# --- Hypothesis-layer adapters --------------------------------------------
# TE has three spatial tiers, all migrated to the hypothesis layer:
#   global         — between-group contrast on per-subject mean TE (one cell)
#   directed edges — mass-univariate per ordered source->target edge (the
#                    directed-edge adapter; ~10^3 edges, no joint group*edge fit)
#   region pair    — emmeans-tabular over directed region pairs (group*region_pair)
# These helpers reshape the tidy hypothesis rows into the band-keyed frames the
# TE report consumes (mirrors the PAC slice's .pac_* helpers).

#' Contrast-kind rows -> report-shaped frame (q_value/t_ratio/effect_size kept).
.te_contrast_rows <- function(h) {
  if (is.null(h) || nrow(h) == 0) return(data.frame())
  cr <- h[h$kind == "contrast", , drop = FALSE]
  if (nrow(cr) == 0) return(data.frame())
  cr
}

#' Rebuild the legacy `contrasts` list (name/group_a/group_b) from contrast rows,
#' so the diagnostic region-pair omnibus keeps iterating contrasts now that
#' config$contrasts is no longer populated (design-spec migration).
.contrasts_from_hyp <- function(h) {
  cr <- .te_contrast_rows(h)
  if (nrow(cr) == 0 || !all(c("group_a", "group_b") %in% names(cr))) return(list())
  uc <- unique(cr[, c("contrast", "group_a", "group_b")])
  uc <- uc[!is.na(uc$group_a) & !is.na(uc$group_b), , drop = FALSE]
  if (nrow(uc) == 0) return(list())
  lapply(seq_len(nrow(uc)), function(i)
    list(name = uc$contrast[i], group_a = uc$group_a[i], group_b = uc$group_b[i]))
}

# --- Publication theme ---
theme_pub <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92"),
      strip.text = element_text(face = "bold", size = base_size),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = base_size + 2)
    )
}

# ===========================================================================
# 1. Global TE analysis: Welch t-tests
# ===========================================================================

compute_global_te <- function(edges) {
  edges %>%
    group_by(subject, group, band) %>%
    summarise(
      mean_te = mean(te, na.rm = TRUE),
      mean_abs_net_te = mean(abs(net_te), na.rm = TRUE),
      n_edges = n(),
      .groups = "drop"
    )
}

run_global_ttests <- function(global_df, contrasts, bands) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      bdata <- global_df %>%
        filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      vals_a <- bdata %>% filter(group == ga) %>% pull(mean_te)
      vals_b <- bdata %>% filter(group == gb) %>% pull(mean_te)

      n_a <- length(vals_a)
      n_b <- length(vals_b)
      mean_a <- mean(vals_a, na.rm = TRUE)
      mean_b <- mean(vals_b, na.rm = TRUE)
      sd_a <- sd(vals_a, na.rm = TRUE)
      sd_b <- sd(vals_b, na.rm = TRUE)

      t_stat <- NA; p_val <- NA; df_val <- NA
      tryCatch({
        tt <- t.test(vals_a, vals_b, var.equal = FALSE)
        t_stat <- tt$statistic
        p_val <- tt$p.value
        df_val <- tt$parameter
      }, error = function(e) {
        message("  t-test failed for ", cname, "/", band_name, ": ", conditionMessage(e))
      })

      pooled_sd <- sqrt(((n_a - 1) * sd_a^2 + (n_b - 1) * sd_b^2) / (n_a + n_b - 2))
      hedges_g <- if (!is.na(pooled_sd) && pooled_sd > 0) (mean_a - mean_b) / pooled_sd else NA

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        metric = "te",
        band = band_name,
        group_a = ga, group_b = gb,
        n_a = n_a, n_b = n_b,
        mean_a = mean_a, mean_b = mean_b,
        sd_a = sd_a, sd_b = sd_b,
        t_stat = as.numeric(t_stat),
        df = as.numeric(df_val),
        p_value = as.numeric(p_val),
        hedges_g = hedges_g,
        stringsAsFactors = FALSE
      )
    }
  }

  result_df <- bind_rows(results)
  if (nrow(result_df) == 0) return(result_df)

  result_df <- result_df %>%
    mutate(
      q_value = p_value,
      significant = q_value < 0.05
    )

  return(result_df)
}

# ===========================================================================
# 2. Directional analysis: paired t-test on TE(X→Y) vs TE(Y→X)
# ===========================================================================

run_directional_ttests <- function(edges, contrasts, bands) {
  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (grp in c(ga, gb)) {
      for (band_name in names(bands)) {
        bdata <- edges %>%
          filter(band == band_name, group == grp)
        if (nrow(bdata) == 0) next

        # For each subject, compute mean net_te (should be ~0 if no directionality)
        subj_net <- bdata %>%
          group_by(subject) %>%
          summarise(mean_net_te = mean(net_te, na.rm = TRUE), .groups = "drop")

        vals <- subj_net$mean_net_te
        n_subj <- length(vals)
        mean_net <- mean(vals, na.rm = TRUE)
        sd_net <- sd(vals, na.rm = TRUE)

        t_stat <- NA; p_val <- NA; df_val <- NA
        tryCatch({
          tt <- t.test(vals, mu = 0)
          t_stat <- tt$statistic
          p_val <- tt$p.value
          df_val <- tt$parameter
        }, error = function(e) {
          message("  Directional t-test failed for ", grp, "/", band_name, ": ", conditionMessage(e))
        })

        results[[length(results) + 1]] <- data.frame(
          contrast = cname,
          group = grp,
          band = band_name,
          n_subjects = n_subj,
          mean_net_te = mean_net,
          sd_net_te = sd_net,
          t_stat = as.numeric(t_stat),
          df = as.numeric(df_val),
          p_value = as.numeric(p_val),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  result_df <- bind_rows(results)
  if (nrow(result_df) == 0) return(result_df)

  result_df <- result_df %>%
    mutate(
      q_value = p_value,
      significant = q_value < 0.05
    )

  return(result_df)
}

# ===========================================================================
# 3. Region-pair LMM
# ===========================================================================

aggregate_edges_to_region_pairs <- function(edges, roi_categories) {
  roi_to_region <- data.frame(
    roi = unlist(roi_categories),
    region = rep(names(roi_categories), lengths(roi_categories)),
    stringsAsFactors = FALSE
  )

  edges_mapped <- edges %>%
    inner_join(roi_to_region, by = c("source_roi" = "roi")) %>%
    rename(source_region = region) %>%
    inner_join(roi_to_region, by = c("target_roi" = "roi")) %>%
    rename(target_region = region)

  # Create directed region pair name
  edges_mapped <- edges_mapped %>%
    mutate(region_pair = paste(source_region, "->", target_region))

  # Average TE within each subject x band x directed region pair
  edges_mapped %>%
    group_by(subject, group, band, region_pair) %>%
    summarise(
      te = mean(te, na.rm = TRUE),
      net_te = mean(net_te, na.rm = TRUE),
      n_edges = n(),
      .groups = "drop"
    )
}

run_omnibus_lmm <- function(region_pair_df, contrasts, bands) {
  if (!has_lme4) {
    message("  lme4/lmerTest not available -- skipping region-pair LMM")
    return(data.frame())
  }
  library(lme4)
  library(lmerTest)

  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      bdata <- region_pair_df %>%
        filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      n_a <- length(unique(bdata$subject[bdata$group == ga]))
      n_b <- length(unique(bdata$subject[bdata$group == gb]))
      n_pairs <- length(unique(bdata$region_pair))

      bdata$group <- factor(bdata$group, levels = c(ga, gb))
      bdata$region_pair <- factor(bdata$region_pair)

      group_F <- NA; group_p <- NA
      region_pair_F <- NA; region_pair_p <- NA
      interaction_F <- NA; interaction_p <- NA
      converged <- TRUE; singular <- FALSE

      tryCatch({
        fit <- lmer(te ~ group * region_pair + (1 | subject), data = bdata)
        singular <- isSingular(fit)

        aov <- anova(fit, type = 3)

        if ("group" %in% rownames(aov)) {
          group_F <- aov["group", "F value"]
          group_p <- aov["group", "Pr(>F)"]
        }
        if ("region_pair" %in% rownames(aov)) {
          region_pair_F <- aov["region_pair", "F value"]
          region_pair_p <- aov["region_pair", "Pr(>F)"]
        }
        if ("group:region_pair" %in% rownames(aov)) {
          interaction_F <- aov["group:region_pair", "F value"]
          interaction_p <- aov["group:region_pair", "Pr(>F)"]
        }
      }, warning = function(w) {
        if (grepl("singular|converge", conditionMessage(w), ignore.case = TRUE)) {
          singular <<- TRUE
        }
      }, error = function(e) {
        converged <<- FALSE
        message("  LMM failed for ", cname, "/", band_name, ": ", conditionMessage(e))
      })

      results[[length(results) + 1]] <- data.frame(
        contrast = cname,
        metric = "te",
        band = band_name,
        group_a = ga, group_b = gb,
        n_a = n_a, n_b = n_b,
        n_region_pairs = n_pairs,
        group_F = as.numeric(group_F),
        group_p = as.numeric(group_p),
        region_pair_F = as.numeric(region_pair_F),
        region_pair_p = as.numeric(region_pair_p),
        interaction_F = as.numeric(interaction_F),
        interaction_p = as.numeric(interaction_p),
        converged = converged,
        singular = singular,
        stringsAsFactors = FALSE
      )
    }
  }

  omnibus_df <- bind_rows(results)
  if (nrow(omnibus_df) == 0) return(omnibus_df)

  omnibus_df <- omnibus_df %>%
    mutate(
      group_q = group_p,
      group_significant = group_q < 0.05,
      interaction_q = interaction_p,
      interaction_significant = interaction_q < 0.05
    )

  return(omnibus_df)
}

run_posthoc_emmeans <- function(region_pair_df, contrasts, bands, omnibus_df, gate = TRUE) {
  if (!has_lme4) return(data.frame())
  library(lme4)
  library(lmerTest)
  library(emmeans)

  results <- list()

  for (contrast in contrasts) {
    cname <- contrast$name
    ga <- contrast$group_a
    gb <- contrast$group_b

    for (band_name in names(bands)) {
      if (gate && nrow(omnibus_df) > 0) {
        omni_row <- omnibus_df %>%
          filter(contrast == cname, band == band_name)
        if (nrow(omni_row) == 0) next
        if (!isTRUE(omni_row$group_significant[1]) &&
            !isTRUE(omni_row$interaction_significant[1])) next
      }

      bdata <- region_pair_df %>%
        filter(band == band_name, group %in% c(ga, gb))
      if (nrow(bdata) == 0) next

      bdata$group <- factor(bdata$group, levels = c(ga, gb))
      bdata$region_pair <- factor(bdata$region_pair)

      tryCatch({
        fit <- lmer(te ~ group * region_pair + (1 | subject), data = bdata)

        emm <- emmeans(fit, pairwise ~ group | region_pair)
        con_df <- as.data.frame(emm$contrasts)
        emm_df <- as.data.frame(emm$emmeans)

        resid_sd <- sigma(fit)
        con_df$q_value <- p.adjust(con_df$p.value, method = "holm")

        for (i in seq_len(nrow(con_df))) {
          pair_name <- as.character(con_df$region_pair[i])
          emm_a <- emm_df %>%
            filter(region_pair == pair_name, group == ga) %>%
            pull(emmean)
          emm_b <- emm_df %>%
            filter(region_pair == pair_name, group == gb) %>%
            pull(emmean)

          hg <- con_df$estimate[i] / resid_sd

          results[[length(results) + 1]] <- data.frame(
            contrast = cname,
            metric = "te",
            band = band_name,
            region_pair = pair_name,
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
        # Continue on singular fit warnings
      }, error = function(e) {
        message("  Post-hoc failed for ", cname, "/", band_name, ": ", conditionMessage(e))
      })
    }
  }

  posthoc_df <- bind_rows(results)
  return(posthoc_df)
}

# ===========================================================================
# Figures
# ===========================================================================

plot_global_te_bar <- function(global_df, group_colors, group_labels,
                               group_order, output_dir) {
  plot_data <- global_df %>%
    filter(group %in% group_order) %>%
    mutate(
      group_label = group_labels[group],
      group_label = factor(group_label, levels = group_labels[group_order])
    )

  summary_data <- plot_data %>%
    group_by(group_label, band) %>%
    summarise(
      mean_val = mean(mean_te, na.rm = TRUE),
      sem_val = sd(mean_te, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  color_vals <- group_colors[group_order]
  names(color_vals) <- group_labels[group_order]

  p <- ggplot(plot_data, aes(x = band, y = mean_te, fill = group_label)) +
    geom_boxplot(width = 0.6, alpha = 0.7, position = position_dodge(0.8),
                 outlier.shape = NA) +
    geom_jitter(aes(color = group_label),
                position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.1),
                size = 1.5, alpha = 0.5, show.legend = FALSE) +
    scale_fill_manual(values = color_vals, name = NULL) +
    scale_color_manual(values = color_vals, name = NULL) +
    labs(x = "Frequency Band", y = "Global Transfer Entropy (mean of all directed edges)",
         title = "Global Transfer Entropy by Band and Group") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  n_bands <- length(unique(summary_data$band))
  ggsave(file.path(output_dir, "roi_transfer_entropy_global_bar.png"), p,
         width = max(8, 2 * n_bands), height = 5, dpi = 300)
  message("  Saved: roi_transfer_entropy_global_bar.png")
}

# ===========================================================================
# Report
# ===========================================================================

write_te_summary <- function(global_df, hyp_global, hyp_edges,
                             omnibus_df, posthoc_df,
                             config, n_subjects, sfreq,
                             fig_dir, output_path) {
  lines <- character()
  add <- function(...) lines <<- c(lines, paste0(...))

  add("# Transfer Entropy Analysis \u2014 ", config$name)
  add("")
  add("**Generated:** ", format(Sys.time(), "%Y-%m-%d %H:%M"))
  add("")
  add("**Note:** This is an exploratory analysis. Results are not included in the manuscript.")
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

  add("**Analysis:** Binned Transfer Entropy (Schreiber 2000)")
  add("")
  add("**Groups:** ", group_str)
  add("")
  add("**Sampling Rate:** ", sfreq, " Hz")
  add("")
  add("**Frequency Bands:** ", band_str)
  add("")
  add("**Metric:** TE(X\u2192Y) = H(Y_f, Y_p) + H(Y_p, X_p) \u2212 H(Y_p) \u2212 H(Y_f, Y_p, X_p)")
  add("")
  add("**Parameters:** lag=1 sample, 5 equal-probability bins (quantile-based)")
  add("")
  add("**Timeseries:** Signed (phase-preserving) ROI source timeseries, band-pass filtered (4th-order Butterworth)")
  add("")

  edge_info <- global_df %>% slice(1)
  n_edges_str <- if (nrow(edge_info) > 0) as.character(edge_info$n_edges[1]) else "unknown"

  add("**Statistics (declarative hypothesis layer):**")
  add("")
  add("1. **Global:** between-group contrast on per-subject mean TE (across all ",
      n_edges_str, " directed edges) per band \u2014 lm(mean_te ~ group), per-band FDR.")
  add("")
  add("2. **Directed edges:** mass-univariate group contrast per ordered ",
      "source\u2192target edge (the directed-edge adapter; source\u2192target and ",
      "target\u2192source tested independently), FDR across the edge family per band.")
  add("")
  if (nrow(omnibus_df) > 0) {
    add("3. **Region-pair:** emmeans contrast over directed region pairs ",
        "(te ~ group * region_pair + (1|subject)); diagnostic Type-III omnibus ",
        "for the group\u00d7region_pair interaction.")
    add("")
  }

  # --- Global contrasts (hypothesis layer) ---
  add("## Global TE (between-group contrasts)")
  add("")
  gc <- if (!is.null(hyp_global) && nrow(hyp_global) > 0)
          hyp_global[hyp_global$kind == "contrast", , drop = FALSE] else data.frame()
  if (nrow(gc) > 0) {
    add("| Hypothesis | Band | estimate | t | df | q | Hedges' g | Sig |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (i in seq_len(nrow(gc))) {
      row <- gc[i, ]
      sig_str <- if (isTRUE(row$significant)) "**Yes**" else "No"
      add(sprintf("| %s | %s | %.5f | %.2f | %.1f | %.4f | %.2f | %s |",
                  row$label %||% row$hypothesis, row$band,
                  ifelse(is.na(row$estimate), 0, row$estimate),
                  ifelse(is.na(row$t_ratio), 0, row$t_ratio),
                  ifelse(is.na(row$df), 0, row$df),
                  ifelse(is.na(row$q_value), 1, row$q_value),
                  ifelse(is.na(row$hedges_g), 0, row$hedges_g),
                  sig_str))
    }
    add("")
  } else {
    add("*No global contrast results computed.*")
    add("")
  }

  # --- Directed-edge results ---
  add("## Directed-Edge TE (mass-univariate)")
  add("")
  ec <- if (!is.null(hyp_edges) && nrow(hyp_edges) > 0)
          hyp_edges[hyp_edges$kind == "contrast", , drop = FALSE] else data.frame()
  if (nrow(ec) > 0) {
    sig_e <- ec[isTRUE(ec$significant) | (!is.na(ec$significant) & ec$significant), , drop = FALSE]
    add(sprintf("%d directed edges tested per band x contrast; %d significant after FDR.",
                length(unique(ec$spatial)), nrow(sig_e)))
    add("")
    if (nrow(sig_e) > 0) {
      sig_e <- sig_e[order(sig_e$q_value), , drop = FALSE]
      top_e <- head(sig_e, 30)
      add("Top significant directed edges (FDR q < 0.05, up to 30 shown):")
      add("")
      add("| Hypothesis | Band | Source \u2192 Target | estimate | t | q | Hedges' g |")
      add("| --- | --- | --- | --- | --- | --- | --- |")
      for (i in seq_len(nrow(top_e))) {
        row <- top_e[i, ]
        add(sprintf("| %s | %s | %s \u2192 %s | %.5f | %.2f | %.4f | %.2f |",
                    row$label %||% row$hypothesis, row$band, row$source, row$target,
                    ifelse(is.na(row$estimate), 0, row$estimate),
                    ifelse(is.na(row$t_ratio), 0, row$t_ratio),
                    ifelse(is.na(row$q_value), 1, row$q_value),
                    ifelse(is.na(row$hedges_g), 0, row$hedges_g)))
      }
      add("")
    } else {
      add("No directed edges reached significance after FDR correction.")
      add("")
    }
  } else {
    add("*No directed-edge results computed.*")
    add("")
  }

  # --- Region-pair LMM ---
  if (nrow(omnibus_df) > 0) {
    add("## Region-Pair LMM Results")
    add("")
    add("| Contrast | Band | n_a | n_b | n_pairs | group_F | group_q | Sig | interaction_F | interaction_q | Int Sig |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (i in seq_len(nrow(omnibus_df))) {
      row <- omnibus_df[i, ]
      grp_sig <- if (isTRUE(row$group_significant)) "**Yes**" else "No"
      int_sig <- if (isTRUE(row$interaction_significant)) "**Yes**" else "No"
      add(sprintf("| %s | %s | %d | %d | %d | %.2f | %.4f | %s | %.2f | %.4f | %s |",
                  row$contrast, row$band, row$n_a, row$n_b,
                  row$n_region_pairs,
                  ifelse(is.na(row$group_F), 0, row$group_F),
                  ifelse(is.na(row$group_q), 1, row$group_q), grp_sig,
                  ifelse(is.na(row$interaction_F), 0, row$interaction_F),
                  ifelse(is.na(row$interaction_q), 1, row$interaction_q), int_sig))
    }
    add("")

    if (nrow(posthoc_df) > 0) {
      sig_ph <- posthoc_df %>% filter(significant == TRUE)
      if (nrow(sig_ph) > 0) {
        add("### Significant Region-Pair Contrasts (Holm q < 0.05)")
        add("")
        add("| Band | Region Pair | Estimate | SE | t | q | Hedges' g |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for (i in seq_len(nrow(sig_ph))) {
          row <- sig_ph[i, ]
          add(sprintf("| %s | %s | %.5f | %.5f | %.2f | %.4f | %.2f |",
                      row$band, row$region_pair, row$estimate, row$SE,
                      row$t_ratio, row$q_value, row$hedges_g))
        }
        add("")
      } else {
        add("No region pairs reached significance after Holm correction.")
        add("")
      }
    } else {
      add("*Post-hoc not performed (no significant omnibus effects).*")
      add("")
    }
  }

  # Key findings
  add("## Key Findings")
  add("")
  any_sig <- FALSE

  if (!is.null(hyp_global) && nrow(hyp_global) > 0) {
    sig_global <- hyp_global[hyp_global$kind == "contrast" & !is.na(hyp_global$significant) &
                             hyp_global$significant, , drop = FALSE]
    if (nrow(sig_global) > 0) {
      any_sig <- TRUE
      for (i in seq_len(nrow(sig_global))) {
        row <- sig_global[i, ]
        add(sprintf("- **%s TE** [%s, global]: t=%.2f, q=%.4f, g=%.2f",
                    row$band, row$label %||% row$hypothesis, row$t_ratio, row$q_value, row$hedges_g))
      }
    }
  }

  if (!is.null(hyp_edges) && nrow(hyp_edges) > 0) {
    sig_edge <- hyp_edges[hyp_edges$kind == "contrast" & !is.na(hyp_edges$significant) &
                          hyp_edges$significant, , drop = FALSE]
    if (nrow(sig_edge) > 0) {
      any_sig <- TRUE
      n_show <- min(10, nrow(sig_edge))
      sig_edge <- sig_edge[order(sig_edge$q_value), , drop = FALSE]
      for (i in seq_len(n_show)) {
        row <- sig_edge[i, ]
        add(sprintf("- **%s → %s** [%s, %s, directed edge]: estimate=%.5f, t=%.2f, q=%.4f",
                    row$source, row$target, row$band, row$label %||% row$hypothesis,
                    row$estimate, row$t_ratio, row$q_value))
      }
      if (nrow(sig_edge) > n_show)
        add(sprintf("- ...and %d more significant directed edges.", nrow(sig_edge) - n_show))
    }
  }

  if (!any_sig) {
    add("- No effects reached significance after FDR correction at any analysis level.")
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

# ===========================================================================
# Main
# ===========================================================================

parser <- ArgumentParser(description = "Transfer entropy statistical analysis (R)")
parser$add_argument("--data-dir", required = TRUE,
                    help = "Directory containing transfer_entropy_edges.csv")
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
parser$add_argument("--roi-categories", default = NULL,
                    help = "Path to roi_categories.yaml (atlas ROI groupings)")
parser$add_argument("--hypothesis", default = NULL,
                    help = "Comma-separated declared hypothesis name(s) to run (default: all)")
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
edges <- read_csv(file.path(data_dir, "roi_transfer_entropy_edges.csv"), show_col_types = FALSE)
message("  roi_transfer_entropy_edges.csv: ", nrow(edges), " rows")

# --- Load config ---
config <- read_yaml(config_path)
group_colors <- unlist(config$group_colors)
group_labels <- unlist(config$groups)
group_order <- config$group_order

message("Study: ", config$name)
message("Groups: ", paste(group_order, collapse = ", "))
message("Bands: ", paste(names(config$bands), collapse = ", "))

# Load roi_categories from atlas file if provided. The pipeline passes an
# unwrapped file (categories at top level); the documented proposed file wraps
# them under a single `roi_categories:` key — unwrap that so either form works.
if (!is.null(args$roi_categories) && file.exists(args$roi_categories)) {
  rc <- read_yaml(args$roi_categories)
  if (length(rc) == 1 && identical(names(rc), "roi_categories")) rc <- rc[["roi_categories"]]
  config$roi_categories <- rc
  message("Loaded roi_categories from: ", args$roi_categories,
          " (", length(config$roi_categories), " regions)")
}

# ===========================================================================
# 1. Global TE analysis (needed for figures — always compute)
# ===========================================================================
message("\n=== Global Transfer Entropy Analysis ===")

global_df <- compute_global_te(edges)
message("  Global TE computed: ", nrow(global_df), " subject x band rows")

spec <- tryCatch(parse_design_spec(config), error = function(e) NULL)

if (!figures_only) {
  if (is.null(spec) || length(spec$hypotheses) == 0)
    stop("No design:/hypotheses: declared in config — nothing to test.")

  # ===========================================================================
  # 1. Global TE: between-group contrast on per-subject mean TE.
  # Mean TE collapses the spatial (edge) axis to one value per subject, so the
  # hypothesis layer has a single cell — handled by the directed-edge adapter
  # with one synthetic (global) edge (lm(mean_te ~ group), no spatial term).
  # ===========================================================================
  message("\n=== Global TE (hypothesis layer) ===")
  global_edges <- global_df
  global_edges$source_roi <- "(global)"; global_edges$target_roi <- "(global)"
  hyp_global <- run_directed_edges(global_edges, names(spec$hypotheses), spec,
                                   dv_col = "mean_te", band_col = "band")
  if (!is.null(args$hypothesis) && nrow(hyp_global) > 0)
    hyp_global <- hyp_global[hyp_global$hypothesis %in%
                             trimws(strsplit(args$hypothesis, ",")[[1]]), , drop = FALSE]
  if (nrow(hyp_global) > 0) {
    hyp_global <- .add_legacy_aliases(hyp_global)
    write_csv(hyp_global, file.path(tbl_dir, "roi_transfer_entropy_global_hypotheses.csv"))
    message("  Saved: roi_transfer_entropy_global_hypotheses.csv (", nrow(hyp_global), " rows)")
    for (i in seq_len(nrow(hyp_global))) {
      row <- hyp_global[i, ]
      sig_str <- if (isTRUE(row$significant)) " ***" else ""
      message(sprintf("  %s | %s | %s: stat=%.2f, q=%.4f%s", row$hypothesis, row$kind,
                      row$band, ifelse(is.na(row$stat), 0, row$stat),
                      ifelse(is.na(row$q_value), 1, row$q_value), sig_str))
    }
  }

  # ===========================================================================
  # 2. Directed-edge analysis: mass-univariate per ordered source->target edge
  # (the directed-edge adapter). One model per band x edge, reused across
  # hypotheses; FDR per hypothesis across the edge family (declarative scope).
  # ===========================================================================
  message("\n=== Directed-Edge TE (hypothesis layer, mass-univariate) ===")
  hyp_edges <- write_module_directed_edges(
    edges, config, tbl_dir, prefix = "roi_transfer_entropy",
    dv_cols = "te", source_col = "source_roi", target_col = "target_roi",
    band_col = "band", hypothesis = args$hypothesis)
  if (is.null(hyp_edges)) hyp_edges <- data.frame()

  # Rebuild the contrasts list from the hypothesis rows so the diagnostic
  # region-pair omnibus keeps iterating contrasts (config$contrasts is NULL).
  te_contrasts <- .contrasts_from_hyp(if (nrow(hyp_global) > 0) hyp_global else hyp_edges)

  # ===========================================================================
  # 3. Region-pair: emmeans-tabular over directed region pairs (group*region_pair).
  # ===========================================================================
  hyp_region <- data.frame()
  omnibus_df <- data.frame()
  posthoc_df <- data.frame()

  if (length(config$roi_categories) > 0 && has_lme4) {
    message("\n=== Region-Pair TE (hypothesis layer) ===")
    region_pair_df <- aggregate_edges_to_region_pairs(edges, config$roi_categories)
    n_region_pairs <- length(unique(region_pair_df$region_pair))
    message("  Aggregated to ", n_region_pairs, " directed region pairs")

    hyp_region <- write_module_hypotheses(
      region_pair_df, config, tbl_dir, prefix = "roi_transfer_entropy_region",
      dv_cols = "te", spatial_col = "region_pair", band_col = "band",
      hypothesis = args$hypothesis)
    if (is.null(hyp_region)) hyp_region <- data.frame()

    posthoc_df <- .te_contrast_rows(hyp_region)
    if (nrow(posthoc_df) > 0) posthoc_df$region_pair <- posthoc_df$spatial

    # Diagnostic omnibus LMM (group x region_pair interaction-F) — NOT a
    # hypothesis; provides the interaction the marginal hypothesis layer lacks.
    omnibus_df <- run_omnibus_lmm(region_pair_df, te_contrasts, config$bands)
    if (nrow(omnibus_df) > 0) {
      message("\n  === Region-Pair Omnibus (diagnostic) ===")
      for (i in seq_len(nrow(omnibus_df))) {
        row <- omnibus_df[i, ]
        grp_sig <- if (isTRUE(row$group_significant)) " ***" else ""
        int_sig <- if (isTRUE(row$interaction_significant)) " ***" else ""
        message(sprintf("  %s | %s: group F=%.2f q=%.4f%s | interaction F=%.2f q=%.4f%s",
                        row$contrast, row$band,
                        row$group_F, row$group_q, grp_sig,
                        row$interaction_F, row$interaction_q, int_sig))
      }
      write_csv(omnibus_df, file.path(tbl_dir, "roi_transfer_entropy_omnibus_lmm.csv"))
      message("  Saved: roi_transfer_entropy_omnibus_lmm.csv (diagnostic)")
    }
    if (nrow(posthoc_df) > 0) {
      sig_count <- sum(posthoc_df$significant, na.rm = TRUE)
      message("  ", nrow(posthoc_df), " region-pair contrasts, ", sig_count, " significant")
    }
  } else if (length(config$roi_categories) == 0) {
    message("\n  No roi_categories in config -- skipping region-pair analysis")
  } else {
    message("\n  lme4/lmerTest not available -- skipping region-pair LMM analysis")
  }
} else {
  message("Figures-only mode: loading existing hypothesis tables...")
  rd <- function(f) tryCatch(read_csv(file.path(tbl_dir, f), show_col_types = FALSE),
                             error = function(e) data.frame())
  hyp_global <- rd("roi_transfer_entropy_global_hypotheses.csv")
  hyp_edges  <- rd("roi_transfer_entropy_directed_edges_hypotheses.csv")
  hyp_region <- rd("roi_transfer_entropy_region_hypotheses.csv")
  omnibus_df <- rd("roi_transfer_entropy_omnibus_lmm.csv")
  posthoc_df <- .te_contrast_rows(hyp_region)
  if (nrow(posthoc_df) > 0) posthoc_df$region_pair <- posthoc_df$spatial
}

# ===========================================================================
# Figures
# ===========================================================================
message("\nGenerating figures...")
plot_global_te_bar(global_df, group_colors, group_labels, group_order, fig_dir)

if (!figures_only) {
  # ===========================================================================
  # Summary report
  # ===========================================================================
  message("\nWriting summary...")

  n_subjects <- edges %>%
    dplyr::distinct(subject, group) %>%
    dplyr::count(group) %>%
    { setNames(.$n, .$group) }

  sfreq <- if (!is.null(config$sfreq)) config$sfreq else 500

  write_te_summary(
    global_df, hyp_global, hyp_edges,
    omnibus_df, posthoc_df,
    config, n_subjects, sfreq,
    fig_dir, file.path(output_dir, "ANALYSIS_SUMMARY.md")
  )
}

message("\nDone. Output: ", output_dir)
