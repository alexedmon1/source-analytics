# R/bayes/fit.R — fitting: one compiled template per prior set, guarded (part of R/bayes/engine.R)

#' A template's priors are in its compiled code: brms writes prior values into the Stan program, and
#' update(..., recompile = FALSE) runs them whatever `prior =` says. Stops unless `template` was compiled with
#' exactly the prior table `pr` (and this prep's formula); returns the template.
bayes_check_template <- function(template, prep, pr) {
  want <- stancode(bayes_formula(prep), data = prep$data, family = gaussian(), prior = pr, threads = template$threads)
  if (!identical(as.character(stancode(template)), as.character(want)))
    stop("bayes: the template's compiled code differs from these priors and formula (a template from another prior ",
         "set or an older prior?); update() would keep the template's priors. Refit the template.")
  template
}

#' model_dir: where cmdstanr keeps the compiled executable, so a saved fit can serve as an update() template in a
#' later process (brms' default is a session temp dir, gone when the process ends).
bayes_fit <- function(prep, chains = 4, iter = 2000, warmup = 1000, seed = 1,
                      backend = "cmdstanr", threads = NULL, template = NULL, model_dir = NULL, ...) {
  pr <- bayes_priors(prep)
  if (!is.null(template))
    return(update(bayes_check_template(template, prep, pr), newdata = prep$data, recompile = FALSE, chains = chains,
                  iter = iter, warmup = warmup, seed = seed, refresh = 0, ...))
  brm(bayes_formula(prep), data = prep$data, family = gaussian(), prior = pr, chains = chains, iter = iter,
      warmup = warmup, cores = chains, seed = seed, backend = backend, refresh = 0,
      threads = if (is.null(threads)) NULL else threading(threads),
      stan_model_args = if (is.null(model_dir)) list() else { dir.create(model_dir, FALSE, TRUE); list(dir = model_dir) },
      ...)
}
