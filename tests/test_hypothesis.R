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

# ---- 9. FDR family scope: aggressiveness is driven by family SIZE ----
# Band b1 carries a real signal (p=0.008) among weak cells; band b2 is all-weak.
# The signal survives a per-band family (n=5) but is diluted to non-significance
# when pooled into the hypothesis-wide family (n=10) — the toggle recovers power
# without changing the test (PAC's pre-specified-freq_pair argument, in miniature).
fdf <- data.frame(
  band    = rep(c("b1", "b2"), each = 5),
  spatial = rep(c("r1", "r2", "r3", "r4", "r5"), 2),
  p_value = c(0.008, 0.04, 0.2, 0.4, 0.6,
              0.50,  0.60, 0.7, 0.8, 0.9),
  stringsAsFactors = FALSE)

q_hyp  <- .apply_fdr(fdf, "BH", "hypothesis")
q_band <- .apply_fdr(fdf, "BH", "band")
q_none <- .apply_fdr(fdf, "BH", "none")
sig_cell <- fdf$band == "b1" & fdf$spatial == "r1"   # the p=0.008 signal
ok(q_band$significant[sig_cell] && !q_hyp$significant[sig_cell],
   sprintf("signal cell significant per-band (q=%.3f) but not hypothesis-wide (q=%.3f)",
           q_band$q_value[sig_cell], q_hyp$q_value[sig_cell]))
ok(sum(q_band$significant) > sum(q_hyp$significant),
   sprintf("per-band scope recovers significance (band=%d sig vs hypothesis=%d sig)",
           sum(q_band$significant), sum(q_hyp$significant)))
ok(all(abs(q_none$q_value - fdf$p_value) < 1e-9), "scope=none leaves q == p (no correction)")
ok(grepl("scope=band", q_band$fdr_family[1]) && grepl("scope=hypothesis", q_hyp$fdr_family[1]),
   "fdr_family label records the scope + method used")

# ---- 9b. W1: fully-qualified fdr_family label — byte-parity with Python ----
# The two builders duplicate BH/`p.adjust`, so their labels must match to the
# byte; these exact strings are also asserted in tests/test_tabular.py.
lab_band <- .fdr_family_label("BH", "band", "Alpha", "disease_effect", "relative",
                              c("Motor_L", "Auditory_L", "Auditory_R"),
                              c("Motor_L", "Auditory_L", "Auditory_R"), "roi")
ok(identical(lab_band,
             "scope=band method=BH key=Alpha|disease_effect|relative members=roi[3] hash=7463bb9d"),
   "W1 band-scope label byte-matches the Python builder (incl. member hash)")
lab_hyp <- .fdr_family_label("BH", "hypothesis", NA, "hd_icv_rescue", "offset",
                             c(NA, NA), c("Auditory_L", "Auditory_R"), "roi")
ok(identical(lab_hyp,
             "scope=hypothesis method=BH key=all|hd_icv_rescue|offset members=cell[2] hash=11f8cfba"),
   "W1 band-less hypothesis-scope label byte-matches Python (NA band token)")
# §10b property: 20-ROI vs 32-ROI family → different hash despite same band/hyp/dv
r20 <- sprintf("R%02d", 0:19); r32 <- sprintf("R%02d", 0:31)
h20 <- .fdr_family_label("BH", "band", "Alpha", "h", "rel", r20, r20, "roi")
h32 <- .fdr_family_label("BH", "band", "Alpha", "h", "rel", r32, r32, "roi")
ok(grepl("members=roi\\[20\\]", h20) && grepl("members=roi\\[32\\]", h32) &&
     sub(".*hash=", "", h20) != sub(".*hash=", "", h32),
   "W1 member-set hash distinguishes a 20-ROI family from a 32-ROI one (§10b)")

# ---- 10. .resolve_fdr precedence: per-hyp > design > built-in default ----
sp_fdr <- list(fdr = list(method = "BH", scope = "hypothesis"))
r_over <- .resolve_fdr(list(fdr = list(scope = "band")), sp_fdr)
ok(r_over$scope == "band" && r_over$method == "BH",
   "per-hyp scope overrides design default; method inherited field-by-field")
r_def <- .resolve_fdr(list(), sp_fdr)
ok(r_def$scope == "hypothesis", "design default applies when the hypothesis is silent")
r_builtin <- .resolve_fdr(list(), list())
ok(r_builtin$scope == "hypothesis" && r_builtin$method == "BH",
   "built-in default is {scope=hypothesis, method=BH} (pre-toggle behaviour)")
ok(inherits(try(.resolve_fdr(list(fdr = list(scope = "bogus")), list()), silent = TRUE),
            "try-error"), "invalid scope is rejected")

# ---- 11. contrasts_from_spec: legacy pairwise list from the design spec ----
# Fixture has 4 hyps: ko_vs_wt (contrast, pairwise), grp_omni (omnibus -> skip),
# dose_resp (regression -> skip), norm (equivalence, pairwise). Pairwise = 2.
cl <- contrasts_from_spec(spec)
nm <- vapply(cl, function(c) c$name, character(1))
ok(length(cl) == 2 && all(c("ko_vs_wt", "norm") %in% nm),
   "contrasts_from_spec keeps pairwise contrast+equivalence, drops omnibus/regression")
ko <- cl[[which(nm == "ko_vs_wt")]]
ok(ko$group_a == "KO_VEH" && ko$group_b == "WT_VEH",
   "contrasts_from_spec maps +weight->group_a, -weight->group_b")

# ---- 12. directed-edge adapter (asymmetric, mass-univariate) ----
# Synthetic directed edges among 3 ROIs (6 ordered pairs) x 2 groups x 1 band.
# Edge A->B carries a real KO>WT signal; its reverse B->A is null — the adapter
# must treat them as distinct edges (directed). One obs per subject per edge.
set.seed(7)
de_groups <- c("WT_VEH", "KO_VEH", "KO_HD_ICV", "KO_HD_IV")
de_eff <- c(WT_VEH = 0.0, KO_VEH = 0.0, KO_HD_ICV = 0.0, KO_HD_IV = 0.0)
edge_rows <- list(); sid <- 0
ord_pairs <- list(c("A","B"), c("B","A"), c("A","C"), c("C","A"), c("B","C"), c("C","B"))
for (g in de_groups) for (s in 1:12) {
  sid <- sid + 1; subj <- sprintf("e%03d", sid)
  for (pr in ord_pairs) {
    src <- pr[1]; tgt <- pr[2]
    # Signal lives ONLY on A->B for KO_VEH (mean shift +1.0); everything else null.
    sig <- if (src == "A" && tgt == "B" && g == "KO_VEH") 1.0 else 0.0
    edge_rows[[length(edge_rows) + 1]] <- data.frame(
      subject = subj, group = g, source_roi = src, target_roi = tgt,
      band = "B1", te = 0.5 + sig + rnorm(1, 0, 0.4), stringsAsFactors = FALSE)
  }
}
edf <- do.call(rbind, edge_rows)
de_spec <- list(factor = "group", reference = "WT_VEH", levels = de_groups,
                covariates = character(0),
                hypotheses = list(
                  ko_vs_wt = parse_hypothesis(list(name = "ko_vs_wt", kind = "contrast",
                                                   weights = list(KO_VEH = 1, WT_VEH = -1))),
                  grp_omni = parse_hypothesis(list(name = "grp_omni", kind = "omnibus"))))

de <- run_directed_edges(edf, c("ko_vs_wt", "grp_omni"), de_spec,
                         dv_col = "te", band_col = "band")
ok(all(c("spatial", "source", "target") %in% names(de)),
   "directed-edge result carries spatial/source/target")
ok(nrow(de[de$kind == "contrast", ]) == length(ord_pairs),
   sprintf("one contrast row per ordered edge (%d edges)", length(ord_pairs)))

# Gold check: contrast estimate on A->B == a hand-fit lm(te~group) emmeans contrast
ab <- edf[edf$source_roi == "A" & edf$target_roi == "B", ]
ab$grp <- factor(ab$group, levels = de_groups)
gold_fit <- lm(te ~ grp, data = ab)
gold_emm <- emmeans::emmeans(gold_fit, ~ grp)
gold_w <- weight_vector(de_spec$hypotheses$ko_vs_wt, de_groups)
gold_con <- as.data.frame(summary(emmeans::contrast(gold_emm,
              method = setNames(list(gold_w), "ko_vs_wt"))))
de_ab <- de[de$kind == "contrast" & de$source == "A" & de$target == "B", ]
ok(near(de_ab$estimate, gold_con$estimate, 1e-6),
   sprintf("A->B contrast estimate matches hand-fit lm emmeans (%.4f)", de_ab$estimate))
ok(near(de_ab$stat, gold_con$t.ratio, 1e-6), "A->B contrast t-ratio matches hand-fit")

# Directed asymmetry: signal edge A->B significant, reverse B->A not
de_ba <- de[de$kind == "contrast" & de$source == "B" & de$target == "A", ]
ok(de_ab$significant && !de_ba$significant,
   sprintf("directed asymmetry: A->B sig (q=%.4f) but reverse B->A not (q=%.4f)",
           de_ab$q_value, de_ba$q_value))
ok(de_ab$effect_size_type == "hedges_g" && de_ab$estimate > 0.5,
   "signal edge reports hedges_g with a positive KO>WT effect")

# Omnibus rows: finite F, partial omega^2 in [0,1), 3 numerator df (4 groups)
de_om <- de[de$kind == "omnibus", ]
ok(nrow(de_om) == length(ord_pairs) && all(is.finite(de_om$stat)),
   "omnibus yields a finite F per edge")
ok(all(de_om$effect_size >= 0 & de_om$effect_size < 1) && all(de_om$df_num == 3),
   "omnibus partial omega2 in [0,1) with 3 numerator df (4 groups)")

# FDR per hypothesis across the edge family (declarative scope)
ok(all(c("q_value", "significant", "fdr_family") %in% names(de)),
   "directed-edge FDR columns present")

# Equal-mean edge (no group effect) should not be flagged by ko_vs_wt
de_ac <- de[de$kind == "contrast" & de$source == "A" & de$target == "C", ]
ok(!de_ac$significant, "null edge A->C not flagged significant")

# ---- 13. minimum detectable effect (DUET #151): power 0.80, alpha 0.05 ----
# Reference values shared with the Python helper (tests/test_power.py) and with
# neurofaune / neurovrai: one contract, one set of numbers.
mde_ref <- list(c(1.0, 10, 3.107197), c(1.0, 26, 2.911195), c(2.09, 24, 6.104376),
                c(1.0, Inf, 2.801585), c(0.5, 4, 1.858705))
for (r in mde_ref)
  ok(near(mde_from_se(r[1], r[2]), r[3], 1e-6),
     sprintf("mde_from_se(se=%g, df=%g) = %.6f", r[1], r[2], r[3]))
ok(near(mde_from_se(1.0, NA), 2.801585, 1e-6), "mde: missing df falls back to normal quantiles")
ok(is.na(mde_from_se(NA, 10)) && is.na(mde_from_se(0, 10)) && is.na(mde_from_se(1, 0)),
   "mde: NA for missing/zero SE or non-positive df")
ok(identical(.mde_reference_weights(c(A = -1, B = 1), "B"), c(A = 0, B = 1)),
   "reference weights: the design reference when the contrast weights it")
ok(isTRUE(all.equal(.mde_reference_weights(c(A = 1, B = -0.5, C = -0.5), NULL),
                    c(A = 0, B = 0.5, C = 0.5))),
   "reference weights: otherwise the |w|-weighted negative side")
ok(is.null(.mde_reference_weights(c(A = 1, B = 1), NULL)), "reference weights: none without a negative side")

# mde_percent boundary: blank when |reference mean| < residual SD, computed at or above
bnd <- .add_mde(data.frame(SE = rep(1, 6), df = rep(10, 6)),
                ref_mean = c(2 * (1 - 1e-9), 2, 2 * (1 + 1e-9), -2 * (1 - 1e-9), -2, 0),
                resid_sd = 2)
ok(is.na(bnd$mde_percent[1]) && is.na(bnd$mde_percent[4]) && is.na(bnd$mde_percent[6]),
   "mde_percent blank when |reference mean| is just below one residual SD")
ok(near(bnd$mde_percent[2], 100 * 3.107197 / 2, 1e-4) &&
     near(bnd$mde_percent[3], 100 * 3.107197 / 2, 1e-4) &&
     near(bnd$mde_percent[5], 100 * 3.107197 / 2, 1e-4),
   "mde_percent computed at and just above one residual SD")
ok(all(abs(bnd$mde - 3.107197) < 1e-6) && all(bnd$mde_note == ""),
   "the residual-SD rule blanks only the percent, never the mde")
ok(is.na(.add_mde(data.frame(SE = 1, df = 10), ref_mean = 5)$mde_percent),
   "mde_percent blank without a residual SD to judge the reference by")

mde_cols <- c("mde", "mde_percent", "mde_power", "mde_alpha", "mde_note")
# contrast rows: mde from the row's own SE/df; percent of the WT_VEH emmean in the same ROI
ok(all(mde_cols %in% names(new_con)), "contrast rows carry the mde columns")
ok(isTRUE(all.equal(new_con$mde, mde_from_se(new_con$SE, new_con$df))),
   "contrast mde = (qt(.975,df) + qt(.80,df)) * SE per row")
ok(all(new_con$mde_power == 0.8 & new_con$mde_alpha == 0.05 & new_con$mde_note == ""),
   "contrast rows: mde_power 0.8, mde_alpha 0.05, empty note")
pc <- df[df$group %in% c("KO_VEH", "WT_VEH"), ]
pc$grp <- factor(pc$group, levels = c("KO_VEH", "WT_VEH"))
gold_ed <- as.data.frame(emmeans::emmeans(suppressMessages(
  lmer(dv ~ grp * roi + (1 | subject), data = pc)), ~ grp | roi))
for (r in rois) {
  wt <- gold_ed$emmean[gold_ed$grp == "WT_VEH" & gold_ed$roi == r]
  row <- new_con[new_con$spatial == r, ]
  ok(near(row$mde_percent, 100 * row$mde / abs(wt), 1e-6),
     sprintf("mde_percent @ roi %s = 100*mde/|WT_VEH emmean| (%.2f%%)", r, row$mde_percent))
}
con_marg <- run_hypothesis(df, "ko_vs_wt", spec, dv_col = "dv", spatial_col = "roi",
                           band_col = "band", marginal = TRUE)
marg_wt <- as.data.frame(emmeans::emmeans(suppressMessages(
  lmer(dv ~ grp * roi + (1 | subject),
       data = transform(df, grp = relevel(factor(group, levels = groups), "WT_VEH")))),
  ~ grp))
ok(nrow(con_marg) == 1 &&
     near(con_marg$mde_percent,
          100 * con_marg$mde / abs(marg_wt$emmean[marg_wt$grp == "WT_VEH"]), 1e-6),
   "marginal contrast: mde_percent against the WT_VEH emmean marginalized over ROI")
ok(all(is.na(new_omni4$mde)) && all(new_omni4$mde_note == "not computed: omnibus F test"),
   "omnibus rows: mde NA with a 'not computed' note")
ok(near(new_reg$mde, mde_from_se(new_reg$SE, new_reg$df)) && is.na(new_reg$mde_percent) &&
     new_reg$mde_note == "", "regression: slope mde computed, no percent")
ok(isTRUE(all.equal(new_eq$mde, mde_from_se(new_eq$SE, new_eq$df))) && all(new_eq$mde_percent > 0),
   "equivalence (TOST) rows carry mde and mde_percent")
ok(all(!is.na(new_con$mde) | grepl("^not computed: ", new_con$mde_note)) &&
     all(!is.na(new_omni4$mde) | grepl("^not computed: ", new_omni4$mde_note)),
   "mde is never NA without a note")

# legacy descriptive post-hoc tables (stats_utils.R): group_b's emmean is the reference
ok(isTRUE(all.equal(ph_legacy$mde, mde_from_se(ph_legacy$SE, ph_legacy$df))) &&
     isTRUE(all.equal(ph_legacy$mde_percent, 100 * ph_legacy$mde / abs(ph_legacy$emmean_b))) &&
     all(ph_legacy$mde_note == ""),
   "legacy run_posthoc_emmeans rows carry mde, percent of emmean_b")
ph_glob <- run_posthoc_global(df[df$group %in% c("KO_VEH", "WT_VEH"), ], legacy_con,
                              spatial_col = "roi", dv_col = "dv", dv_label = "dv")
ok(nrow(ph_glob) == 1 && near(ph_glob$mde, mde_from_se(ph_glob$SE, ph_glob$df)) &&
     near(ph_glob$mde_percent, 100 * ph_glob$mde / abs(ph_glob$emmean_b)),
   sprintf("legacy run_posthoc_global row carries mde (%.3f, %.1f%% of emmean_b)",
           ph_glob$mde, ph_glob$mde_percent))

# directed edges (analytic contrast): reference WT_VEH cell mean; omnibus note
de_ab_wt <- mean(ab$te[ab$group == "WT_VEH"])
ok(near(de_ab$mde, mde_from_se(de_ab$SE, de_ab$df)) &&
     near(de_ab$mde_percent, 100 * de_ab$mde / abs(de_ab_wt)),
   sprintf("directed edge A->B: mde %.3f, %.1f%% of the WT_VEH mean", de_ab$mde, de_ab$mde_percent))
ok(all(de_om$mde_note == "not computed: omnibus F test") && all(is.na(de_om$mde)),
   "directed-edge omnibus rows: note, no mde")

cat("\nAll hypothesis-layer checks passed.\n")
