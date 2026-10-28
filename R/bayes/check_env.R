#!/usr/bin/env Rscript
# R/bayes/check_env.R — `source-analytics bayes --check-env`: what the engine needs, and what is installed (decision B5:
# record and check). Prints one JSON object; exit status 1 if anything required is missing.
#   Rscript R/bayes/check_env.R [--against <manifest.json>]   # also compare with the versions a run recorded
`%||%` <- function(a, b) if (is.null(a)) b else a
args <- commandArgs(trailingOnly = TRUE)
against <- if (length(args) >= 2 && args[1] == "--against") args[2] else NULL
REQUIRED <- c("brms", "cmdstanr", "posterior", "digest", "optparse", "jsonlite", "dplyr")
pkgs <- vapply(REQUIRED, function(p) if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else
  NA_character_, character(1))
cs <- if (!is.na(pkgs[["cmdstanr"]])) tryCatch(list(version = as.character(cmdstanr::cmdstan_version()),
                                                    path = cmdstanr::cmdstan_path()),
                                               error = function(e) list(version = NA, path = NA)) else
  list(version = NA, path = NA)
out <- list(R = paste(R.version$major, R.version$minor, sep = "."), packages = as.list(pkgs), cmdstan = cs,
            missing = c(names(pkgs)[is.na(pkgs)], if (is.na(cs$version)) "cmdstan"))
if (!is.null(against)) {
  rec <- jsonlite::fromJSON(against, simplifyVector = FALSE)
  rec <- if (!is.null(rec$versions)) rec$versions else rec[[1]]$versions        # a manifest, or one merged by dv
  now <- list(R = out$R, brms = pkgs[["brms"]], cmdstanr = pkgs[["cmdstanr"]], posterior = pkgs[["posterior"]],
              digest = pkgs[["digest"]], cmdstan = cs$version)
  out$differs_from_record <- Filter(Negate(is.null), lapply(setNames(names(now), names(now)), function(k)
    if (!identical(as.character(rec[[k]] %||% NA), as.character(now[[k]]))) list(recorded = rec[[k]], now = now[[k]])))
}
cat(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null"), "\n")
quit(status = if (length(out$missing)) 1L else 0L)
