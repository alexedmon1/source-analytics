#!/usr/bin/env Rscript
# Known-scenario test for the hypothesis-testing engine in R/stats_utils.R.
# Run: Rscript tests/test_gating.R   (exits non-zero on failure)

suppressMessages(source("R/stats_utils.R"))

mkrow <- function(contrast, band, roi, est, se, q, hg = est / 0.5) data.frame(
  contrast = contrast, band = band, power_type = "relative", roi = roi,
  estimate = est, SE = se, df = 30, q_value = q, hedges_g = hg,
  significant = q < 0.05, stringsAsFactors = FALSE)

df <- do.call(rbind, list(
  # phenotype significant only at Low Gamma (both ROIs); estimate = deficit
  mkrow("disease_effect", "Delta", "A", 0.1, 0.2, 0.40),
  mkrow("disease_effect", "Delta", "B", 0.1, 0.2, 0.50),
  mkrow("disease_effect", "Low Gamma", "A", 1.0, 0.2, 0.001),
  mkrow("disease_effect", "Low Gamma", "B", 1.2, 0.2, 0.002),
  # rescue significant only at Low Gamma A
  mkrow("hd_icv_rescue", "Delta", "A", 0.05, 0.2, 0.6),
  mkrow("hd_icv_rescue", "Delta", "B", 0.05, 0.2, 0.7),
  mkrow("hd_icv_rescue", "Low Gamma", "A", 0.9, 0.2, 0.004),
  mkrow("hd_icv_rescue", "Low Gamma", "B", 0.2, 0.2, 0.30),
  # normalization (equivalence): near-zero (equivalent) at Low Gamma A only
  mkrow("hd_icv_normalization", "Low Gamma", "A", 0.05, 0.05, 0.9),
  mkrow("hd_icv_normalization", "Low Gamma", "B", 0.8, 0.2, 0.10)
))

contrasts <- list(
  list(name = "disease_effect", group_a = "KO_VEH", group_b = "WT_VEH",
       role = "phenotype", test = "difference"),
  list(name = "hd_icv_rescue", group_a = "KO_HD_ICV", group_b = "KO_VEH",
       role = "rescue", test = "difference", gate_on = "disease_effect"),
  list(name = "hd_icv_normalization", group_a = "KO_HD_ICV", group_b = "WT_VEH",
       role = "normalization", test = "equivalence",
       gate_on = list("disease_effect", "hd_icv_rescue"))
)
hyp <- list(gate_alpha = 0.05,
            default_equivalence_margin = list(mode = "gap_fraction", value = 0.25))

g <- apply_hypothesis_gating(df, contrasts, hyp,
                             cell_cols = c("band", "power_type", "roi"))
cellget <- function(d, cn, b, r, col) d[d$contrast == cn & d$band == b & d$roi == r, col]

# Rescue gating: only inside the phenotype (Low Gamma) mask.
stopifnot(cellget(g, "hd_icv_rescue", "Delta", "A", "gated_in") == FALSE)
stopifnot(cellget(g, "hd_icv_rescue", "Low Gamma", "A", "gated_in") == TRUE)
stopifnot(cellget(g, "hd_icv_rescue", "Low Gamma", "B", "gated_in") == TRUE)

# Normalization: gated only where BOTH disease and rescue are significant.
stopifnot(cellget(g, "hd_icv_normalization", "Low Gamma", "A", "gated_in") == TRUE)
stopifnot(cellget(g, "hd_icv_normalization", "Low Gamma", "B", "gated_in") == FALSE)
# gap_fraction margin = 0.25 * |deficit| ; TOST equivalent at the normalized cell.
stopifnot(abs(cellget(g, "hd_icv_normalization", "Low Gamma", "A", "margin_used") - 0.25) < 1e-9)
stopifnot(cellget(g, "hd_icv_normalization", "Low Gamma", "A", "equivalent") == TRUE)

# Rescue verdicts.
v <- build_rescue_verdicts(g, contrasts, cell_cols = c("band", "power_type", "roi"))
vget <- function(b, r) v[v$band == b & v$roi == r, "verdict"]
stopifnot(vget("Delta", "A") == "not_in_phenotype")
stopifnot(vget("Low Gamma", "A") == "rescued_normalized")
stopifnot(vget("Low Gamma", "B") == "not_rescued")

cat("test_gating.R: all assertions passed\n")
