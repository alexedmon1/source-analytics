#!/usr/bin/env Rscript
# R/bayes/bench_fit.R — the benchmark's engine arm (D10 plan step 1, M1; port of bayes-sl R/ee3_fit.R).
#
# The engine on a frozen cohort set's simulated studies: two groups (A reference, B), one contrast B - A, the bands
# given, nested N (each group's first N mice by `order`). Run by source_analytics.bayes.arms, which compiles the
# template first (--template) and then runs the parts in parallel, each in its own TMPDIR, retrying a part whose
# cmdstan CSVs vanish.
#
#   Rscript R/bayes/bench_fit.R --cohorts <set> --out <dir> --set W --template --template-cohort noise/rep001 --template-n 8
#   Rscript R/bayes/bench_fit.R --cohorts <set> --out <dir> --set W --todo <todo.csv> --part 1/4
#
# --sa: the R directory whose engine is sourced (default: this script's own); point it at a frozen copy to repeat old
# fits digit for digit. Per fit: <out>/fits/<cell>/rep<NNN>/N<NN>_<set>.csv, the engine's readout rows plus post_sd
# (the estimand's posterior SD, SD units), unit_db (the engine's scale at this N, per band), fit_seconds, cell, rep,
# n_per_group; and N<NN>_<set>_draws.csv.gz, every 4th joint draw of the estimands (SD units, 5 significant digits).
# A fit already written is skipped. Seeds: 1000 * rep + N, as bayes-sl's.

suppressMessages({ library(optparse); library(jsonlite); library(brms); library(posterior) })
here <- local({
  a <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else file.path("R", "bayes")
})
opt <- parse_args(OptionParser(option_list = list(
  make_option("--cohorts"), make_option("--out"), make_option("--todo", default = NULL),
  make_option("--set", default = "W"), make_option("--bands", default = "Low Gamma,Theta"),
  make_option("--part", default = "1/1"), make_option("--sa", default = normalizePath(file.path(here, ".."))),
  make_option("--chains", type = "integer", default = 4), make_option("--iter", type = "integer", default = 2000),
  make_option("--warmup", type = "integer", default = 1000), make_option("--rope", type = "double", default = 0.1),
  make_option("--informative", default = NULL, help = "JSON file: x_<band>_<group> -> [mean, sd], for --set I"),
  make_option("--template", action = "store_true", default = FALSE),
  make_option("--template-cohort", dest = "template_cohort", default = NULL),
  make_option("--template-n", dest = "template_n", type = "integer", default = 8))))
if (is.null(opt$cohorts) || is.null(opt$out)) stop("bench_fit: --cohorts and --out are required")
source(file.path(opt$sa, "hypothesis.R"))
if (file.exists(file.path(opt$sa, "bayes", "engine.R"))) source(file.path(opt$sa, "bayes", "engine.R")) else
  source(file.path(opt$sa, "bayes_hypothesis.R"))                     # an engine copy from before E1
if (opt$set == "I" && is.null(opt$informative)) stop("--set I needs --informative: without entries it fits W's priors")
informative <- if (is.null(opt$informative)) NULL else lapply(fromJSON(opt$informative), as.numeric)
bands <- trimws(strsplit(opt$bands, ",")[[1]])
spec <- parse_design_spec(list(
  design = list(factor = "group", reference = "A", levels = list("A", "B"), covariates = list()),
  hypotheses = list(effect = list(name = "effect", kind = "contrast", label = "Plant: B vs A", role = "phenotype",
                                  weights = list(B = 1.0, A = -1.0)))))
dir.create(file.path(opt$out, "stan"), recursive = TRUE, showWarnings = FALSE)
tpl_f <- file.path(opt$out, sprintf("template_%s.rds", opt$set))

cohort <- function(cohort_id, n) {
  d <- read.csv(file.path(opt$cohorts, cohort_id, "band_power.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  d <- d[d$band %in% bands & d$order <= n, ]
  if (length(unique(d$subject)) != 2 * n) stop(sprintf("%s: %d subjects at N = %d", cohort_id,
                                                       length(unique(d$subject)), n))
  bayes_prepare(d, spec, "absolute", "roi", "band", opt$set, informative)
}

if (opt$template) {
  if (file.exists(tpl_f)) { cat("template exists:", tpl_f, "\n"); quit(save = "no") }
  if (is.null(opt$template_cohort)) stop("--template needs --template-cohort")
  prep <- cohort(opt$template_cohort, opt$template_n)
  saveRDS(bayes_fit(prep, chains = 2, iter = 400, warmup = 200, seed = 1, model_dir = file.path(opt$out, "stan")),
          tpl_f)
  cat("wrote", tpl_f, "\n"); quit(save = "no")
}
if (is.null(opt$todo)) stop("bench_fit: --todo (cohort, n_per_group) is required")
tpl <- readRDS(tpl_f)

# the estimands' draws (one column per readout row), from the same draws and contrast weights as bayes_readout
contrast_draws <- function(fit, prep, r) {
  dr <- as.data.frame(as_draws_df(fit))
  B <- as.matrix(dr[, paste0("b_", prep$fixed$col)]); colnames(B) <- prep$fixed$col
  w <- weight_vector(spec$hypotheses[[1]], prep$levels)
  D <- sapply(seq_len(nrow(r)), function(k) {
    b <- .safe(r$band[k])
    R <- if (r$spatial[k] == "ROI-average") NULL else {
      m <- as.matrix(dr[, sprintf("r_roi[%s,%s]", r$spatial[k], prep$terms$col)]); colnames(m) <- prep$terms$col; m
    }
    .contrast_draws(B, R, prep, b, w)
  })
  colnames(D) <- paste(r$band, r$spatial, sep = "|")
  D
}

todo <- read.csv(opt$todo, stringsAsFactors = FALSE)
pp <- as.integer(strsplit(opt$part, "/")[[1]])
todo <- todo[seq_len(nrow(todo)) %% pp[2] == pp[1] %% pp[2], , drop = FALSE]
for (k in seq_len(nrow(todo))) {
  cid <- todo$cohort[k]; n <- todo$n_per_group[k]
  cell <- dirname(cid); rep <- as.integer(sub("^rep", "", basename(cid)))
  of <- file.path(opt$out, "fits", cell, sprintf("rep%03d", rep), sprintf("N%02d_%s.csv", n, opt$set))
  if (file.exists(of)) next
  prep <- cohort(cid, n)
  t0 <- Sys.time()
  fit <- bayes_fit(prep, chains = opt$chains, iter = opt$iter, warmup = opt$warmup, seed = 1000 * rep + n,
                   template = tpl, cores = opt$chains)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  r <- bayes_readout(fit, prep, spec, rope_sd = opt$rope)
  D <- contrast_draws(fit, prep, r)
  r$post_sd <- apply(D, 2, sd)
  r$unit_db <- prep$scale$scale[match(.safe(r$band), prep$scale$band)]
  r$cell <- cell; r$rep <- rep; r$n_per_group <- n; r$fit_seconds <- secs
  dir.create(dirname(of), recursive = TRUE, showWarnings = FALSE)
  # the joint draws (every 4th, SD units): what location statements across ROIs need; the .csv marks the fit as done
  gz <- gzfile(sub("\\.csv$", "_draws.csv.gz", of), "w")
  write.csv(signif(D[seq(1, nrow(D), by = 4), , drop = FALSE], 5), gz, row.names = FALSE); close(gz)
  write.csv(r, paste0(of, ".part"), row.names = FALSE); file.rename(paste0(of, ".part"), of)
  cat(sprintf("bench engine %s N %d done %s: %.0f s, estimand R-hat max %.3f, scales %.3f, %d divergent\n", cid, n,
              format(Sys.time(), "%H:%M"), secs, max(r$rhat), r$max_rhat_scale[1], r$n_divergent[1]))
  rm(fit); invisible(gc(verbose = FALSE))
}
cat("bench engine part", opt$part, "finished\n")
