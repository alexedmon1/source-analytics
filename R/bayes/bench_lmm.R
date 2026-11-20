#!/usr/bin/env Rscript
# R/bayes/bench_lmm.R — the benchmark's frequentist arms with the engine's structure (D10 plan step 1, M1; port of
# bayes-sl R/f1_fit.R and R/f1c_fit.R).
#
#   z ~ 0 + a + b + (0 + x | roi) + (1 | roi) + (1 | subject), lme4 REML, one band
#
# The engine's z (bayes_prepare) and its partial pooling of the group effect across ROIs, one residual variance; no
# priors, and one band alone, so (1 | roi:band)'s variance is not shared with another band. Two ways to get the
# uncertainty (--method):
#   resample     arm F1: mice resampled within group (--b resamples), each resample refitted from the fit's theta;
#                percentile intervals. Draws: the resamples.
#   conditional  arm F1c: the joint covariance of (beta-hat, u-hat - u) from the mixed-model equations with the
#                variances held at their estimates (Henderson's prediction error variance), sigma^2 C^-1 in lme4's
#                spherical coordinates mapped back through Lambda. Allows for the fixed effects' uncertainty but not
#                the variances', so expected to be too narrow, most at small N. Draws: --draws from the joint normal.
# A variance estimated at 0 (a singular fit) pools fully: reported per fit (singular), not avoided.
#
#   Rscript R/bayes/bench_lmm.R --cohorts <set> --out <dir> --method resample --todo <todo.csv> --part 1/4
#
# Per fit: <out>/fits/<cell>/rep<NNN>/N<NN>_<tag>.csv (tag F1 or F1c; columns band, spatial, estimate, lo, hi,
# post_sd, unit_db, sd_slope, singular, [boot_singular], cell, rep, n_per_group, fit_seconds; SD units) and
# N<NN>_<tag>_draws.csv.gz (columns <band>|<spatial>). Seeds: 1000 * rep + N, as bayes-sl's.
suppressPackageStartupMessages({ library(optparse); library(lme4) })
here <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else file.path("R", "bayes")
})
opt <- parse_args(OptionParser(option_list = list(
  make_option("--cohorts"), make_option("--out"), make_option("--todo", default = NULL),
  make_option("--method", default = "resample"), make_option("--band", default = "Low Gamma"),
  make_option("--part", default = "1/1"), make_option("--b", type = "integer", default = 200),
  make_option("--draws", type = "integer", default = 1000),
  make_option("--sa", default = normalizePath(file.path(here, ".."))))))
if (is.null(opt$cohorts) || is.null(opt$out) || is.null(opt$todo)) stop("bench_lmm: --cohorts, --out, --todo required")
if (!opt$method %in% c("resample", "conditional")) stop("bench_lmm: --method resample or conditional")
tag <- if (opt$method == "resample") "F1" else "F1c"
suppressPackageStartupMessages(source(file.path(opt$sa, "bayes", "engine.R")))
spec <- parse_design_spec(list(
  design = list(factor = "group", reference = "A", levels = list("A", "B"), covariates = list()),
  hypotheses = list(effect = list(name = "effect", kind = "contrast", label = "Plant: B vs A", role = "phenotype",
                                  weights = list(B = 1.0, A = -1.0)))))
ctl <- lmerControl(calc.derivs = FALSE, check.conv.singular = "ignore", check.nobs.vs.nlev = "ignore",
                   check.nobs.vs.nRE = "ignore", check.conv.grad = "ignore", check.conv.hess = "ignore")
form <- z ~ 0 + a + b + (0 + x | roi) + (1 | roi) + (1 | subject)

point <- function(m) {
  x <- ranef(m, drop = TRUE)$roi
  if (is.data.frame(x)) x <- setNames(x[["x"]], rownames(x))
  d <- unname(fixef(m)["b"] - fixef(m)["a"])
  c(`ROI-average` = d, d + x)
}

slope_sd <- function(m) {
  v <- as.data.frame(VarCorr(m))
  v$sdcor[v$var1 == "x" & !is.na(v$var1)]
}

# L: contrasts x (fixed, random) coefficients. Row 1 the ROI average (b - a), then b - a + slope_r per ROI.
pev <- function(m) {
  X <- getME(m, "X"); Z <- t(as.matrix(getME(m, "Zt"))); Lam <- t(as.matrix(getME(m, "Lambdat")))
  ZL <- Z %*% Lam; p <- ncol(X); q <- ncol(Z)
  C <- rbind(cbind(crossprod(X), crossprod(X, ZL)), cbind(crossprod(ZL, X), crossprod(ZL) + diag(q)))
  Tm <- as.matrix(Matrix::bdiag(diag(p), Lam))
  V <- sigma(m)^2 * Tm %*% solve(C) %*% t(Tm)
  # which random coefficients are the ROI slopes: lme4's term order, from the per-term Zt blocks
  zl <- getME(m, "Ztlist"); off <- cumsum(c(0, vapply(zl, nrow, 1L)))
  k <- which(names(zl) == "roi.x")
  stopifnot(length(k) == 1)
  roi <- rownames(zl[[k]]); idx <- p + off[k] + seq_along(roi)
  L <- matrix(0, length(roi) + 1, p + q, dimnames = list(c("ROI-average", roi), NULL))
  L[, match("b", colnames(X))] <- 1; L[, match("a", colnames(X))] <- -1
  L[cbind(seq_along(roi) + 1, idx)] <- 1
  coef <- c(fixef(m), as.vector(Lam %*% getME(m, "u")))
  list(est = drop(L %*% coef), V = L %*% V %*% t(L))
}

load_cohort <- function(cid, n) {
  d <- read.csv(file.path(opt$cohorts, cid, "band_power.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  d <- d[d$band == opt$band & d$order <= n, ]
  if (length(unique(d$subject)) != 2 * n) stop(sprintf("%s: %d subjects at N = %d", cid, length(unique(d$subject)), n))
  prep <- bayes_prepare(d, spec, "absolute", "roi", "band", "W", NULL)
  D <- prep$data
  D <- data.frame(z = D$z, roi = D$roi, subject = as.character(D$subject), group = as.character(D$.grp))
  D$a <- as.numeric(D$group == "A"); D$b <- as.numeric(D$group == "B"); D$x <- D$b
  list(D = D, unit = prep$scale$scale[1])
}

fit_resample <- function(D, n, seed) {
  m <- lmer(form, D, control = ctl)
  est <- point(m)
  rows <- split(seq_len(nrow(D)), D$subject)
  grp <- vapply(rows, function(i) D$group[i[1]], "")
  set.seed(seed)
  th <- getME(m, "theta")
  sing <- logical(opt$b)
  B <- t(sapply(seq_len(opt$b), function(i) {
    ids <- unlist(lapply(c("A", "B"), function(g) sample(names(grp)[grp == g], n, replace = TRUE)))
    R <- D[unlist(rows[ids], use.names = FALSE), ]
    R$subject <- rep(as.character(seq_along(ids)), lengths(rows[ids]))
    b <- lmer(form, R, control = ctl, start = th)
    sing[i] <<- isSingular(b)
    point(b)[names(est)]
  }))
  list(r = data.frame(band = opt$band, spatial = names(est), estimate = est,
                      lo = apply(B, 2, quantile, .025), hi = apply(B, 2, quantile, .975), post_sd = apply(B, 2, sd),
                      sd_slope = slope_sd(m), singular = isSingular(m), boot_singular = mean(sing), row.names = NULL),
       B = B)
}

fit_conditional <- function(D, n, seed) {
  m <- lmer(form, D, control = ctl)
  est0 <- point(m)
  P <- pev(m)
  # checks: the same point estimates as F1, and the ROI average's variance equal to lme4's own vcov
  stopifnot(max(abs(P$est[names(est0)] - est0)) < 1e-8)
  cv <- as.matrix(vcov(m)); w <- c(a = -1, b = 1)[colnames(cv)]
  stopifnot(abs(P$V[1, 1] - drop(t(w) %*% cv %*% w)) < 1e-8 * max(1, P$V[1, 1]))
  est <- P$est; se <- sqrt(pmax(diag(P$V), 0))
  # SVD, not eigen(): LAPACK's dsyevr fails (error code 3) on some of these matrices (bayes-sl, Hippocampus_Ant_R_1.0
  # rep 6 N 24)
  e <- svd((P$V + t(P$V)) / 2)
  set.seed(seed)
  B <- sweep(matrix(rnorm(opt$draws * length(est)), opt$draws) %*% t(e$u %*% diag(sqrt(e$d))), 2, est, "+")
  colnames(B) <- names(est)
  list(r = data.frame(band = opt$band, spatial = names(est), estimate = est, lo = est - qnorm(.975) * se,
                      hi = est + qnorm(.975) * se, post_sd = se, sd_slope = slope_sd(m), singular = isSingular(m),
                      row.names = NULL),
       B = B)
}

todo <- read.csv(opt$todo, stringsAsFactors = FALSE)
pp <- as.integer(strsplit(opt$part, "/")[[1]])
todo <- todo[seq_len(nrow(todo)) %% pp[2] == pp[1] %% pp[2], , drop = FALSE]
for (k in seq_len(nrow(todo))) {
  cid <- todo$cohort[k]; n <- todo$n_per_group[k]
  cell <- dirname(cid); rep <- as.integer(sub("^rep", "", basename(cid)))
  of <- file.path(opt$out, "fits", cell, sprintf("rep%03d", rep), sprintf("N%02d_%s.csv", n, tag))
  if (file.exists(of)) next
  t0 <- Sys.time()
  co <- load_cohort(cid, n)
  x <- if (opt$method == "resample") fit_resample(co$D, n, 1000 * rep + n) else fit_conditional(co$D, n, 1000 * rep + n)
  x$r$unit_db <- co$unit
  x$r$cell <- cell; x$r$rep <- rep; x$r$n_per_group <- n
  x$r$fit_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  dir.create(dirname(of), recursive = TRUE, showWarnings = FALSE)
  colnames(x$B) <- paste(opt$band, colnames(x$B), sep = "|")
  gz <- gzfile(sub("\\.csv$", "_draws.csv.gz", of), "w")
  write.csv(signif(x$B, 5), gz, row.names = FALSE); close(gz)
  write.csv(x$r, paste0(of, ".part"), row.names = FALSE); file.rename(paste0(of, ".part"), of)
  cat(sprintf("bench %s %s N %d: %.0f s, slope SD %.3f, singular %s %s\n", tag, cid, n, x$r$fit_seconds[1],
              x$r$sd_slope[1], x$r$singular[1], format(Sys.time(), "%H:%M")))
}
cat("bench", tag, "part", opt$part, "finished\n")
