# Bayesian hypothesis adapter (`feat/bayes-group`): design, G1

**Status:** design agreed 2026-10-14 (section 8); no code yet. Plan and decisions:
`/mnt/e/research/EEG/bayes-sl/PLAN.md`, Phase G. Method precedent: bayes-sl Track A (`R/model.R`, `R/03_recovery_sbc.R`)
and the tectum-pair model (`R/tp_model.R`, `R/tp_gate.R`).

## 1. What it is

A fourth adapter for the hypothesis layer (`HYPOTHESIS.md`). It sits beside emmeans, permutation and edge/NBS. It
takes the same `design:` / `hypotheses:` blocks and the same per-subject long tables. It answers each declared
hypothesis with a posterior instead of a p-value.

It does not replace the emmeans adapter, and it changes no existing output. It runs only when asked for.

**G1 scope** (the Conductor, 2026-10-14):
- FORGE rest, `roi_based` arm;
- two outcome families: absolute band power (dB, `roi_psd/data/band_power.csv`) and aperiodic exponent
  (`roi_aperiodic/data/aperiodic_params.csv`);
- bands pooled in one model;
- omnibus deferred.

Sessions and within-subject factors are G2.

## 2. Entry point

A separate CLI step, not a flag on the existing modules:

```bash
source-analytics bayes --study study.yaml --paradigm resting --analysis roi_psd --dv absolute [--hypothesis NAME]
source-analytics bayes --study study.yaml --paradigm resting --analysis roi_aperiodic --dv exponent
source-analytics bayes ... --gate            # prior predictive, SBC, null, equivalence, recovery at this design; no real fit
source-analytics bayes --study study.yaml --paradigm resting   # every outcome in bayes.outcomes
```

**Settings (E2, 2026-10-28):** everything that changes a number is in the study YAML's `bayes:` block, and the CLI
says only what to run and where (`--analysis`, `--dv`, `--hypothesis`, `--prior-sets` as a subset, `--gate`,
`--workers`, `--out`). A bad block stops before R starts. Schema: `src/source_analytics/bayes/config.py`.

```yaml
bayes:
  rope: {mode: sd, value: 0.1}
  prior_sets: [W, I]                 # default: [W, I] when priors.I is set, else [W]
  priors: {I: priors/informative.yaml}   # relative to the study YAML
  fit: {chains: 4, iter: 2000, warmup: 1000, seed: 1}
  gate: {reps: {sbc: 100, null: 100, recovery: 10, equivalence: 100}, reduced_bands: [Theta, Low Gamma]}
  outcomes: {roi_psd: [absolute], roi_aperiodic: [exponent]}
```

The set-I file is written in hypothesis x band terms, each entry with its source (`bayes/priors.py`):

```yaml
version: 1
entries:
  - {dv: absolute, hypothesis: disease_effect, band: Low Gamma, prior: {mean: 0.5, sd: 0.8},
     source: priors/lit/2026-10-07_fmr1_resting_state.md, derivation: "pooled rule ..."}
```

An entry's hypothesis must be one group minus the reference; the engine maps it to its internal term. The pre-E2
JSON (`{x_<band>_<group>: [mean, sd]}`) is still read, unchecked.

- **Inputs:** the step reads the module's existing `data/*.csv` (the analytics working tree). It needs no subject
  processing, so the frequentist pipeline is untouched.
- **Python:** a thin wrapper, the subpackage `source_analytics.bayes` (`tables.py`: the outcome tables; `job.py`:
  paths, design spec to JSON; `run.py`: `Rscript`).
- **R:** `R/bayes/`. `engine.R` loads the engine: `model.R` (coding, formula, priors), `fit.R` (templates and the
  guard), `readout.R` (readout and the decision rules). The entry scripts are `analysis.R` (fits) and `gate.R` (the
  gate). Python calls `Rscript`, as the emmeans modules do.
- **Regression:** `tests/bayes/test_e0_regression.py` replays the E0 fixtures (FORGE's fits and gate verdicts, frozen
  2026-10-28 outside the repository; `SA_BAYES_E0`) and requires them digit for digit.
- **Tests (E5):** `tests/bayes/`. Fast tests (no sampling) run by default: the job, the `bayes:` block and prior file,
  coding and priors, gate ids, the template guard on an empty fit, the gate's boundary draw. Tests marked `slow`
  compile and sample (minutes; `SA_BAYES_SLOW=1`, scratch under `SA_BAYES_SCRATCH`): a tiny portable fit with a
  single-band and a multi-band outcome (same seed gives the same draws, fresh or through a template; the guard on a
  compiled template; the HDI covers the simulated effect), a gate smoke run with an equivalence hypothesis and both
  prior sets through the CLI, and the E3 end-to-end run.

## 3. Model

The outcome is standardised before fitting, per band: z = (y - band mean) / s_b. Here s_b is the within-group
between-subject SD within ROI x band, pooled over ROIs and groups. This is the unit of a per-cell Hedges g. The total SD over all rows is not used,
because ROI differences in absolute dB would inflate it. Effects, priors and the ROPE are in these units. A `mode: sd`
equivalence margin is not: it is margin x the model's residual SD (sigma, per band and draw; section 4), as in the
emmeans adapter (`.equivalence_margin()`: margin x the residual SD of its subject-intercept model). In FORGE sigma is
0.55-0.88 of s_b, so a 0.25 margin is 0.14-0.22 in these units.

**Band power, bands pooled:**

```
z ~ 0 + band + band:grp                                   group effect per band (reference level = design.reference)
    + (0 + band:grp || roi)                               ROI deviations of each band's group effects (partial pooling)
    + (1 | roi:band)                                      ROI level per band
    + (0 + band || subject)                               subject level per band
    , sigma ~ 0 + band                                    residual SD per band
```

**Aperiodic exponent:** the same model without `band`.

**Covariates:** `design.covariates` enter mean-centred, as in `.hyp_fit()`.

**Choices to confirm:**
- **Family:** Gaussian on dB and on the exponent; Student-t only if the posterior predictive check asks for it
  (section 8).
- **ROI deviations:** uncorrelated across group x band (`||`). A full correlation matrix has 28 x 28 terms.
- **Partial pooling across ROIs.** This is what the per-cell LMM lacks. A cell's contrast is
  `band:grp + r_roi[band:grp]`, shrunk toward the band's ROI-average.

## 4. Hypothesis kinds

| kind | readout | per |
|---|---|---|
| contrast | posterior of the weighted contrast Σ w·μ: median, 95% HDI, P(direction), P(ROPE), Savage-Dickey BF10; decision rules as Track A (`decide.R`) | band (ROI-average) and band x ROI cell |
| equivalence | ROPE = margin. `mode: sd` → margin × the posterior residual SD (per draw); decision: HDI inside ±margin, and P(inside) | as contrast |
| regression | slope posterior (+`by`: per-group slopes); same readouts | band (and cell) |
| omnibus | **deferred**: one row, `status = not_implemented_bayes` | n/a |

- **ROPE for plain contrasts:** study-level `bayes: {rope: {mode: sd, value: 0.1}}`, default 0.1 SD (section 8). An
  equivalence hypothesis uses its own margin.
- **Multiplicity:** no FDR. Partial pooling across ROIs is the adjustment. The output states this per row
  (`fdr_family = none (partial pooling)`).

**Output:** `tables/.../<analysis>_hypotheses_bayes.csv`, written beside `<analysis>_hypotheses.csv` (untouched):

```
hypothesis, kind, role, band, spatial, dv, prior_set, estimate (posterior median), hdi_lo, hdi_hi, p_direction,
p_rope, rope_lo, rope_hi, log10_bf10, decision, rhat, ess_bulk, max_rhat_scale, max_rhat, n_divergent,
gate_passed, label
```

`rhat` and `ess_bulk` are the row's estimand's own (its draws split by chain); `max_rhat_scale` is the max over every
SD and log sigma; `max_rhat` is the max over every parameter, reported only (section 6, convergence).

Fits, the prior table used and the gate summary go to `data/bayes/` (provenance).

## 5. Priors: generated from the spec, two sets

**Coding: group levels (the Conductor, 2026-10-19).** The fixed part has one level per band x group, the reference
included (`lv_<band>_<group>`). It used to be a reference level plus effects against it, with N(0, 1) on each effect.
Jointly, those priors pooled the reference level toward the other groups' levels: in FORGE's exponent every contrast
against WT_VEH came out 0.10-0.18 SD smaller than emmeans (bayes-sl `reports/G1_forge_exponent.md`; checked against
REML and an N(0, 10) refit). Fits and gate verdicts carry the coding (`BAYES_CODING`); a verdict attaches only to fits
of the same coding.

**W (weakly informative, SD units):**
- group levels N(0, 1/sqrt(2)) per band, so every pairwise contrast is a priori N(0, 1). N(0, 1) on contrasts is
  Track A's weak default; the first draft had 0.5, which was tighter than the library's informative entries (SD 0.8).
  The Conductor set it, 2026-10-14. Levels are pooled toward the band's grand mean (z is centred per band), each by its
  own standard error, so small arms shrink more; no group is privileged;
- ROI-deviation SDs half-N(0, 0.25);
- ROI level per band SD half-N(0, 3) (`BAYES_SD_ROI_BAND`). Raised from 0.5 (the Conductor, 2026-10-15): FORGE's
  source power ROI means spread 0.9-3.0 between-subject SD per band (largest in gamma and epsilon), and the real spread
  fell at the 97th-100th prior predictive percentile. Group effects were not involved (group-mean SD 0.35-0.40);
- subject SDs half-N(0, 0.5);
- log sigma N(log 0.5, 0.5).

These are proposals. Each must pass the prior predictive check at the study design. The tectum model needed three
rounds.

**I (informative):** only from a logged literature search. An entry `x_<band>_<group>` is a prior on group -
reference. That group's level is written as reference level + `dx_<band>_<group>` with `dx` ~ the entry, so the
contrast's prior is exactly the entry; groups without an entry keep W levels and are not tied to the reference. **For FORGE (the Conductor, 2026-10-14): pooled.**
- **The gap:** library entries are regional (AC, FC, S1, deep), and no source measured the ROI-average.
- **The rule:** a band-level entry is the median of that band's regional entries when they agree in sign, with the
  largest regional scale.
- **Derivation:** by `bayes-sl/scripts/derive_forge_I.py`. Files: `bayes-sl/priors/library/2026-10-14_forge_I_*`. For FORGE the source is
`bayes-sl/priors/lit/2026-10-07_fmr1_resting_state.md`, the Fmr1 resting-state log, which predates FORGE. The
entries are derived by script, as Track A's set I was. Bands and contrasts without logged evidence stay at W.

**Both sets are always fitted and reported side by side** (bayes-sl `CLAUDE.md`).

## 6. Gate, generated from the spec (`--gate`)

At the study's actual design (subjects per group, ROIs, bands), before any real fit:
1. **Prior predictive:** each band's spread (SD, MAD) against the real spread. The 5-95% band is the pass rule.
2. **SBC:** parameters drawn in R from the stated priors, data simulated at the design, the template refit, ranks of
   the truth among 100 thinned draws. Covers every hypothesis estimand, per band and per cell. Uniformity by
   chi-square, and coverage of the 95% HDI.
3. **Null:** every group effect 0. Rates of each decision rule claiming an effect, per hypothesis.
4. **Equivalence (E5, 2026-10-28; only with equivalence hypotheses):** the null draw, then each band's group levels
   moved (the least shift) so that every equivalence contrast lies exactly at +-margin (x the drawn sigma), signs drawn
   per hypothesis; hypotheses whose weights are linearly dependent are placed one per replication, in turn. At the
   margin "equivalent" is the wrong call. Rule: its rate per hypothesis x band (ROI-average) at most 0.05
   (`equiv_max_false`). Reported: the cells' rate, and from null replications how often equivalence is shown when the
   groups are equal (`shown_when_equal_*`: the design's power to show equivalence). Equivalence hypotheses are scored by
   their own rule (`.decide_equivalence`); before E5 the gate scored them only by the contrast rule (still read by the
   null rule), so a false "equivalent" was never measured.
5. **Recovery:** at the full design and the job's real-fit chains/iter/warmup. HDI coverage of the ROI-average
   estimands at least 0.85, and convergence (below).

**Convergence: R-hat on what is decided (the Conductor, 2026-10-21).** The rule is R-hat < 1.05 (`BAYES_RHAT_MAX`)
for every estimand (each hypothesis x band, ROI-average and cell, computed on its draws split by chain) and for every
scale parameter (SDs, log sigma). The max over every parameter is reported as `max_rhat_raw` and not gated.
- **Why:** the model has a direction the data do not identify. A per-band constant trades between the group levels
  and the `(1 | roi:band)` offsets, held only by the prior, and no contrast sees it.
- **Case:** FORGE power recovery W, rep 9 (bayes-sl `R/g1_power_rhat_diag.R`, `R/g1_power_rhat_directions.R`). All 26
  Low Gamma offsets and 4 levels were at 1.07-1.076, moving together with bulk ESS about 50. Every pairwise contrast
  was at or below 1.016 (min bulk ESS 295); the band's grand level was at or below 1.01; scales were at or below 1.021.
- **Consequence:** absolute group levels mix slowly. A report that shows levels rather than contrasts has to check
  their own R-hat.
- Recovery reps written before the rule carry no estimand R-hat; `summarize` refuses them.

**One template per prior set.** brms writes prior values into the compiled Stan code, and `update(..., recompile =
FALSE)` keeps the template's priors whatever `prior =` says. The gate compiles one template per set and design
(`template_<bands><_SET>.rds`), and `bayes_check_template()` stops a fit whose template's code differs from the
requested priors (a template from the other set, or from before a prior change). The tectum model in bayes-sl hit
this on 2026-10-15: its set I fits ran W priors.

**Workers are separate R processes** (a PSOCK cluster), not forks. With cmdstanr fits in forked `mclapply` workers,
the gate's R process was SIGKILLed with its whole process group as it exited (3 of 4 trials in a reduced test). That
took the calling queue down at the end of a stage three times (2026-10-14 to 10-16), each time with every replication
already saved.

A gate result is cached per (spec hash, design hash, prior hash). The real-fit table carries `gate_passed`.

**Cost.** This is the open risk:
- FORGE power is 60 subjects x 26 ROIs x 7 bands = 10,920 rows, with about 900 ROI-deviation terms.
- The fit time is not measured yet. A first template fit gives it.
- If one fit takes more than about 10 min, 100-200 SBC replications on rstan (no cmdstanr installed) is a day or
  more.
- Options: install cmdstanr (within-chain threading, faster); run SBC per band subset; or a reduced SBC (100) with
  the null at 50.

### 6.1 Gate ids, manifests and the environment (E3, 2026-10-28)

**Gate id.** A gate belongs to one design and model. `gate.json` carries `gate_id`, a hash of: the engine version
(`BAYES_ENGINE_VERSION`, bumped when the model, priors, readout or gate change what they compute), the coding, the prior
set and its prior table, the design rows, the outcome values (the prior predictive reads them), the hypotheses, the
ROPE (the null rule's claims use it), the gate settings (replications, reduced bands, recovery's sampler settings) and
the thresholds (`BAYES_GATE_RULES`, locked defaults). Each component's hash is kept as `gate_inputs`. A real fit
computes the id of its own design and records `gate_passed` (and `gate_id`) only for a gate with that id. Otherwise
`gate_passed` is `NA`, with a warning naming what differs ("the gate was made for different prior table, ROPE"). Not in
the id: the fit's seed, the package's git commit (a docs commit must not invalidate gates). Gates written before E3 have
no id and count as `NA` until their `summarize` is rerun.

**Completeness.** `summarize` reads only the configured design's replication directories and fails the gate unless
every configured replication is present (`complete` in `gate.json`): a resumable gate summarized half way does not pass.

**Manifest.** Each run writes `manifest_<dv>.json` in the work dir and merges it by dv into
`<analysis>_hypotheses_bayes_manifest.json` beside the table: engine version and coding; R, brms, cmdstanr, posterior,
digest and CmdStan versions; the package version and git describe; the study YAML, set-I file, job file and input
table with sha256; rows, subjects and groups used; the settings (ROPE, sampler, gate, thresholds); per prior set the
gate check (id wanted, id found, verdict, reason), the prior table and fit with sha256, and convergence; the table's
sha256.

**Environment.** `source-analytics bayes --check-env` reports R, the engine's R packages and CmdStan and exits 1 if one
is missing; `--check-env <manifest>` also lists versions that differ from the ones a run recorded (decision B5: record
and check; `renv` only if two machines disagree).

## 7. FORGE test (the G1 acceptance test)

1. **Inputs:** the G0 sha256 hashes match (`bayes-sl/reports/G0_forge_inputs.md`), or work from a copy.
2. **Gate passes** for power and exponent under W, and for power under I.
3. **Real fits, both sets.** The test compares against `analyses-v2`'s frozen emmeans tables:
   - direction agreement for each contrast x band (and cell);
   - posterior median against the emmeans estimate in SD units;
   - where they differ, the interval and the pooling are checked before any explanation (bayes-sl `CLAUDE.md`).

   Agreement is not required. Disagreement has to be explained.
4. **Unit tests** (pytest, no Stan): spec → model terms; weights → contrast vector; prior table from the spec; and
   the output schema. A tiny-design brms fit runs as a slow, marked test.

## 8. Decisions (the Conductor, 2026-10-14)

1. **ROPE for plain contrasts: 0.1 SD** (`bayes: {rope: {mode: sd, value: 0.1}}` default). An equivalence hypothesis
   uses its own margin.
2. **cmdstanr is installed** for the engine. rstan stays available; Track A and the tectum model keep rstan.
3. **Family: Gaussian first.** Student-t only if the posterior predictive check asks for it.
4. **Entry point: a separate `source-analytics bayes` step,** not an engine flag on the modules.
