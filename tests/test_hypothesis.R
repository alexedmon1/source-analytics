#!/usr/bin/env Rscript
# Verification test for R/hypothesis.R (the emmeans adapter of the hypothesis layer).
# Asserts the new runner reproduces legacy stats_utils.R numbers and that the new
# kinds (omnibus / regression / equivalence) behave. Exits non-zero on failure.
# Run: Rscript tests/test_hypothesis.R

suppressMessages({
  source("R/stats_utils.R")   # legacy run_omnibus_lmm/run_posthoc_emmeans + kept primitives
  source("R/hypothesis.R")    # module under test
})

ok <- function(cond, msg) {
  if (!isTRUE(cond)) { cat("FAIL:", msg, "\n"); quit(status = 1) }
  cat("ok  :", msg, "\n")
}
near <- function(a, b, tol = 1e-6) isTRUE(abs(a - b) < tol)

set.seed(42)

# ---- Synthetic data: 4 groups x 3 ROIs x 1 band, random-intercept subjects ----
groups <- c("WT_VEH", "KO_VEH", "KO_HD_ICV", "KO_HD_IV")
geff   <- c(WT_VEH = 0.0, KO_VEH = 1.0, KO_HD_ICV = 0.4, KO_HD_IV = 0.6)  # true group means
rois   <- c("A", "B", "C"); reff <- c(A = 0.0, B = 0.3, C = -0.2)
rows <- list(); sid <- 0
for (g in groups) for (s in 1:12) {
  sid <- sid + 1; subj <- sprintf("s%03d", sid); ri <- rnorm(1, 0, 0.4)
  for (r in rois) {
    val <- 2 + geff[[g]] + reff[[r]] + ri + rnorm(1, 0, 0.5)
    rows[[length(rows) + 1]] <- data.frame(
      subject = subj, group = g, roi = r, band = "B1",
      relative = val, dv = val, dose = geff[[g]], stringsAsFactors = FALSE)
  }
}
df <- do.call(rbind, rows)
bands <- list(B1 = c(1, 4))

# ---- 1. parse_hypothesis: legacy group_a/group_b sugar -> contrast kind ----
h_legacy <- parse_hypothesis(list(name = "ko_vs_wt", group_a = "KO_VEH", group_b = "WT_VEH"))
ok(h_legacy$kind == "contrast", "legacy group_a/group_b parses to kind=contrast")
ok(near(h_legacy$weights[["KO_VEH"]], 1) && near(h_legacy$weights[["WT_VEH"]], -1),
   "legacy sugar -> weights {KO_VEH:+1, WT_VEH:-1}")

# ---- 2. weight_vector alignment (0-fill, level order) ----
wv <- weight_vector(h_legacy, c("WT_VEH", "KO_VEH", "KO_HD_ICV"))
ok(identical(unname(wv), c(-1, 1, 0)), "weight_vector aligns + zero-fills in level order")

spec <- list(factor = "group", reference = NULL, levels = groups,
             covariates = character(0),
             hypotheses = list(
               ko_vs_wt = parse_hypothesis(list(name = "ko_vs_wt", kind = "contrast",
                                                weights = list(KO_VEH = 1, WT_VEH = -1))),
               grp_omni = parse_hypothesis(list(name = "grp_omni", kind = "omnibus")),
               dose_resp = parse_hypothesis(list(name = "dose_resp", kind = "regression",
                                                 predictor = "dose")),
               norm = parse_hypothesis(list(name = "norm", kind = "equivalence",
                                            weights = list(KO_HD_ICV = 1, WT_VEH = -1),
                                            margin = list(mode = "sd", value = 2.0)))))

# ---- 3. contrast estimate (per_contrast) == legacy posthoc emmeans ----
legacy_con <- list(list(name = "ko_vs_wt", group_a = "KO_VEH", group_b = "WT_VEH"))
omni_legacy <- run_omnibus_lmm(df, legacy_con, bands, power_type = "relative")
ph_legacy <- run_posthoc_emmeans(df, legacy_con, bands, omni_legacy,
                                 power_type = "relative", gate = FALSE)
new_con <- run_hypothesis(df, "ko_vs_wt", spec, dv_col = "dv", spatial_col = "roi",
                          band_col = "band", fit_scope = "per_contrast")
for (r in rois) {
  e_old <- ph_legacy$estimate[ph_legacy$roi == r]
  e_new <- new_con$estimate[new_con$spatial == r]
  ok(near(e_old, e_new, 1e-4), sprintf("contrast estimate matches legacy @ roi %s (%.4f)", r, e_new))
}
ok(all(new_con$effect_size_type == "hedges_g"), "contrast reports hedges_g")

# ---- 4. omnibus F (2-group, per_contrast) == legacy group_F ----
new_omni2 <- run_hypothesis(df, parse_hypothesis(list(name = "o2", kind = "omnibus",
                              groups = list("KO_VEH", "WT_VEH"))), spec,
                            dv_col = "dv", spatial_col = "roi", band_col = "band")
ok(near(new_omni2$stat, omni_legacy$group_F, 1e-3),
   sprintf("omnibus F matches legacy group_F (%.3f)", new_omni2$stat))

# ---- 5. 4-group omnibus runs; partial omega^2 in [0,1) ----
new_omni4 <- run_hypothesis(df, "grp_omni", spec, dv_col = "dv", spatial_col = "roi",
                            band_col = "band")
ok(nrow(new_omni4) == 1 && is.finite(new_omni4$stat), "4-group omnibus yields a finite F")
ok(new_omni4$effect_size >= 0 && new_omni4$effect_size < 1,
   sprintf("omnibus partial omega2 in [0,1): %.3f", new_omni4$effect_size))
ok(new_omni4$df_num == 3 && new_omni2$df_num == 1,
   "omnibus numerator df reflects group count (4-group=3 df vs 2-group=1 df)")

# ---- 6. regression recovers the synthetic slope (dose -> dv, true slope 1) ----
new_reg <- run_hypothesis(df, "dose_resp", spec, dv_col = "dv", spatial_col = "roi",
                          band_col = "band")
ok(new_reg$estimate > 0.5 && new_reg$p_value < 0.05,
   sprintf("regression slope positive & significant (b=%.3f, p=%.4g)", new_reg$estimate, new_reg$p_value))
ok(new_reg$effect_size_type == "std_beta", "regression reports std_beta")

# ---- 7. equivalence (sd margin) yields a logical verdict + margin ----
new_eq <- run_hypothesis(df, "norm", spec, dv_col = "dv", spatial_col = "roi", band_col = "band")
ok(all(c("margin_used", "equivalent") %in% names(new_eq)), "equivalence emits margin_used + equivalent")
ok(all(new_eq$margin_used > 0), "equivalence margin (sd mode) resolved > 0")
ok(is.logical(new_eq$equivalent), "equivalence verdict is logical")

# ---- 8. within-run FDR present ----
ok(all(c("q_value", "significant", "fdr_family") %in% names(new_con)), "within-run FDR columns present")

cat("\nAll hypothesis-layer checks passed.\n")
