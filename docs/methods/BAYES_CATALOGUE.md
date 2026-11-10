# Bayesian engine catalogue

Every model, prior and decision rule of the Bayesian group engine, each with a stable **id** and a **version**. Engine version described: `2026.11.10.1`. Units: between-subject SD of the outcome (the unit of a per-cell Hedges g) unless stated.

*Generated from `source_analytics/bayes/catalogue.py` (`python -m source_analytics.bayes.catalogue > docs/methods/BAYES_CATALOGUE.md`); do not edit by hand. `tests/bayes/test_catalogue.py` checks it against the R code and checks that it is current.*

## Models

| id | version | what it is |
|---|---|---|
| `model.group_levels` | 1 | hierarchical group model over bands and ROIs |
| `model.group_levels_noroi` | 1 | the same model with no ROI: one value per subject, e.g. along a located field |
| `model.group_levels_uncentred` | 1 | the same model on an uncentred outcome: group levels against 0 |

### `model.group_levels` (version 1)

Hierarchical group model over bands and ROIs.

- **Definition:** The outcome is standardised per band by the pooled within-group between-subject SD within ROI x band (so effects are in the unit of a per-cell Hedges g): z = (y - band mean) / that SD. Then z ~ Normal(mu, sigma_band) with mu = one level per band x group + each band's group effect deviating by ROI (partial pooling) + a level per ROI x band + a subject level per band; log sigma per band. A set-I entry re-expresses that group's level as reference level + dx.
- **Rests on:** `prior.w.group_level`, `prior.w.covariate`, `prior.w.roi_deviation_sd`, `prior.w.roi_band_sd`, `prior.w.subject_sd`, `prior.w.log_sigma`, `prior.i.contrast_entry`
- **Validity:** Gaussian outcomes per band. Each study design must pass the gate (gate.rules) before a real fit.
- **Implementation:** `R/bayes/model.R:bayes_formula`
- **Tests:** `tests/bayes/test_bayes_job.py, tests/bayes/test_e5_slow.py`
- **Reference:** docs/methods/BAYES_ENGINE.md section 3

### `model.group_levels_noroi` (version 1)

The same model with no ROI: one value per subject, e.g. along a located field.

- **Definition:** model.group_levels with spatial_col = NULL: the two ROI terms are dropped; the subject term is kept only when some subject has more than one row in a band (with one row it would be the residual under another name). The readout has one spatial cell, "all".
- **Rests on:** `prior.w.group_level`, `prior.w.covariate`, `prior.w.log_sigma_noroi`, `prior.i.contrast_entry`
- **Validity:** As model.group_levels. Reachable from R only until the config path exists (field plan, phase F6).
- **Implementation:** `R/bayes/model.R:bayes_prepare`
- **Tests:** `tests/bayes/test_no_roi.py`
- **Reference:** docs/methods/BAYES_ENGINE.md section 3

### `model.group_levels_uncentred` (version 1)

The same model on an uncentred outcome: group levels against 0.

- **Definition:** model.group_levels (or its no-ROI form) with center = FALSE (a declared table's center: false): z = y / the same SD, not centred, so 0 keeps its meaning (a difference score such as structured - random ITC). A hypothesis whose weights do not sum to 0 (e.g. {KO: 1}) then reads a group's own level against 0, decided by rule.contrast_decision; in a ROI cell it includes that cell's ROI x band level. The ROI x band levels are the ROI deviations of the level shared by all groups. With every group at 0 the gate's null puts every level at 0.
- **Rests on:** `prior.w.group_level`, `prior.w.covariate`, `prior.w.roi_deviation_sd`, `prior.w.roi_band_sd_uncentred`, `prior.w.subject_sd`, `prior.w.log_sigma`, `prior.w.log_sigma_noroi`, `prior.i.contrast_entry`, `prior.i.level_entry`
- **Validity:** As model.group_levels. Only for outcomes whose 0 means "no effect"; a centred outcome refuses level hypotheses.
- **Implementation:** `R/bayes/model.R:bayes_prepare`
- **Tests:** `tests/bayes/test_uncentred.py`
- **Reference:** engine 2026.11.10.1 (bayes-sl, 2026-11-10: SL learning effect per genotype)

## Priors (set W, weakly informative; set I, informative)

| id | version | what it is |
|---|---|---|
| `prior.w.group_level` | 1 | set W: each group level per band |
| `prior.w.covariate` | 1 | set W: covariate slopes |
| `prior.w.roi_deviation_sd` | 1 | set W: how much a group effect varies by ROI |
| `prior.w.roi_band_sd` | 1 | set W: how much ROI means vary within a band |
| `prior.w.roi_band_sd_uncentred` | 1 | set W: how much the shared level varies by ROI, uncentred outcome |
| `prior.w.subject_sd` | 1 | set W: subject levels |
| `prior.w.log_sigma` | 1 | set W: residual SD per band, with ROI or subject terms |
| `prior.w.log_sigma_noroi` | 1 | set W: residual SD per band, no ROI and no subject term |
| `prior.i.contrast_entry` | 1 | set I: an informative prior on one contrast against the reference |
| `prior.i.level_entry` | 1 | set I: an informative prior on one group's own level (uncentred outcome) |

### `prior.w.group_level` (version 1)

Set W: each group level per band.

- **Definition:** Each band x group level ~ Normal(0, 1/sqrt(2)), independently, so every pairwise contrast is a priori Normal(0, 1) and no group is privileged.
- **Values:** mean 0; SD 1/sqrt(2) = 0.707 between-subject SD (contrast SD 1)
- **Implementation:** `R/bayes/model.R:bayes_priors`
- **Reference:** docs/methods/BAYES_ENGINE.md section 5; group-level coding the Conductor, 2026-10-19

### `prior.w.covariate` (version 1)

Set W: covariate slopes.

- **Definition:** Each centred covariate's slope ~ Normal(0, 1) in SD units of the outcome per unit of the covariate.
- **Values:** mean 0; SD 1
- **Implementation:** `R/bayes/model.R:bayes_priors`

### `prior.w.roi_deviation_sd` (version 1)

Set W: how much a group effect varies by ROI.

- **Definition:** SD of each band's group-effect deviations across ROIs ~ half-Normal(0, 0.25).
- **Values:** half-Normal scale 0.25 between-subject SD
- **Implementation:** `R/bayes/model.R:bayes_priors`

### `prior.w.roi_band_sd` (version 1)

Set W: how much ROI means vary within a band.

- **Definition:** SD of the ROI x band levels ~ half-Normal(0, 3).
- **Values:** half-Normal scale 3 between-subject SD
- **Implementation:** `R/bayes/model.R:bayes_priors`
- **Reference:** raised from 0.5 (the Conductor, 2026-10-15): FORGE source-power ROI means spread 0.9-3.0 SD

### `prior.w.roi_band_sd_uncentred` (version 1)

Set W: how much the shared level varies by ROI, uncentred outcome.

- **Definition:** SD of the ROI x band levels ~ half-Normal(0, 0.5) when the outcome is uncentred: a difference score has no anatomical baseline for its ROI means to spread around, so the scale is the subject term's, not 3.
- **Values:** half-Normal scale 0.5 between-subject SD
- **Implementation:** `R/bayes/model.R:bayes_priors`
- **Reference:** engine 2026.11.10.1 (2026-11-10); checked at each design by the gate's prior predictive

### `prior.w.subject_sd` (version 1)

Set W: subject levels.

- **Definition:** SD of the subject level per band ~ half-Normal(0, 0.5).
- **Values:** half-Normal scale 0.5 between-subject SD
- **Implementation:** `R/bayes/model.R:bayes_priors`

### `prior.w.log_sigma` (version 1)

Set W: residual SD per band, with ROI or subject terms.

- **Definition:** log sigma per band ~ Normal(log 0.5, 0.5): the residual is the part of the standardised spread that the ROI and subject terms do not take.
- **Values:** median sigma 0.5 between-subject SD; SD 0.5 on the log scale
- **Implementation:** `R/bayes/model.R:bayes_log_sigma_mean`

### `prior.w.log_sigma_noroi` (version 1)

Set W: residual SD per band, no ROI and no subject term.

- **Definition:** log sigma per band ~ Normal(log 1, 0.5). With one row per subject and band, the standardisation makes the within-group SD 1, so the residual is all of it.
- **Values:** median sigma 1 between-subject SD; SD 0.5 on the log scale
- **Implementation:** `R/bayes/model.R:bayes_log_sigma_mean`
- **Reference:** engine 2026.11.09.1 (the Conductor, 2026-11-09): log 0.5 put 1 at the prior's 92nd percentile

### `prior.i.contrast_entry` (version 1)

Set I: an informative prior on one contrast against the reference.

- **Definition:** An entry x_<band>_<group> = Normal(mean, sd) on (group - reference) in between-subject SD. The group's level becomes reference level + dx with dx ~ the entry, so that contrast's prior is exactly the entry and no other group is tied to the reference. Terms without an entry stay as set W.
- **Values:** mean and SD per entry, from the study's informative-prior file
- **Validity:** Only from a logged literature search: each entry carries its source (priors.py requires it). The file is identified in the run manifest by its path and sha256.
- **Implementation:** `R/bayes/model.R:bayes_prepare`
- **Tests:** `tests/bayes/test_e2_config.py`
- **Reference:** docs/methods/BAYES_ENGINE.md section 5

### `prior.i.level_entry` (version 1)

Set I: an informative prior on one group's own level (uncentred outcome).

- **Definition:** An entry lv_<band>_<group> = Normal(mean, sd) on that group's level in between-subject SD replaces its set-W prior. Only with an uncentred outcome; a group cannot carry both a level and a contrast entry in one band.
- **Values:** mean and SD per entry, from the study's informative-prior file
- **Validity:** As prior.i.contrast_entry: only from a logged literature search, each entry with its source.
- **Implementation:** `R/bayes/model.R:bayes_prepare`
- **Tests:** `tests/bayes/test_uncentred.py, tests/bayes/test_declared_tables.py`
- **Reference:** engine 2026.11.10.1 (2026-11-10)

## Decision rules and the gate

| id | version | what it is |
|---|---|---|
| `rule.contrast_decision` | 1 | effect, null or undecided, from the HDI against the ROPE |
| `rule.equivalence_decision` | 1 | equivalent or not shown, from the HDI against a margin |
| `rule.convergence` | 1 | what must converge before a result is read |
| `gate.rules` | 1 | the simulation checks a design must pass before a real fit |

### `rule.contrast_decision` (version 1)

Effect, null or undecided, from the HDI against the ROPE.

- **Definition:** 95% highest-density interval (HDI) of the contrast. "effect" when it lies wholly outside the ROPE, "null" when wholly inside, else "undecided". Reported beside it: median, P(direction), P(inside ROPE), and BF10 by Savage-Dickey (prior density at 0 over posterior density at 0, from draws).
- **Values:** HDI 95%; ROPE +-0.1 between-subject SD by default (a study may set its own)
- **Caveat:** BF10 from draws is unreliable far in the tails; read very large values only as decisive.
- **Implementation:** `R/bayes/readout.R:.decide`
- **Tests:** `tests/bayes/test_e5_slow.py`
- **Reference:** docs/methods/BAYES_ENGINE.md section 4

### `rule.equivalence_decision` (version 1)

Equivalent or not shown, from the HDI against a margin.

- **Definition:** "equivalent" when the 95% HDI lies inside +-margin, the margin per draw being margin x the posterior residual SD (mode: sd); else "not_shown".
- **Values:** HDI 95%; margin set by the hypothesis
- **Implementation:** `R/bayes/readout.R:.decide_equivalence`
- **Tests:** `tests/bayes/test_e5_gate.py, tests/bayes/test_e5_slow.py`
- **Reference:** docs/methods/BAYES_ENGINE.md section 4

### `rule.convergence` (version 1)

What must converge before a result is read.

- **Definition:** R-hat (split by chain) of every decided estimand and of every scale parameter (SDs, log sigma) below 1.05. The maximum over every parameter is reported, not gated.
- **Values:** R-hat < 1.05
- **Implementation:** `R/bayes/readout.R:.estimand_conv`
- **Reference:** the Conductor, 2026-10-21

### `gate.rules` (version 1)

The simulation checks a design must pass before a real fit.

- **Definition:** At the study's own design and prior set: the real outcome's spread (SD and MAD per band) within the 5th-95th percentile of the prior predictive; simulation-based calibration rank-uniformity chi-square p above 0.01 / number of estimands and 95% HDI coverage at least 0.85; with every group effect 0, false "effect" claims at most 5% and direction claims at most 10% per estimand; recovery coverage at least 0.85 at the real fit's settings; false "equivalent" at most 5% at the margin; every configured replication present.
- **Values:** prior predictive 5-95th percentile; SBC alpha 0.01, Bonferroni over estimands; coverage >= 0.85; null effect claims <= 0.05; null direction claims <= 0.10; recovery coverage >= 0.85; R-hat < 1.05; false equivalent <= 0.05; default replications: SBC 100, null 100, recovery 10, equivalence 100
- **Rests on:** `rule.contrast_decision`, `rule.equivalence_decision`, `rule.convergence`
- **Implementation:** `R/bayes/gate.R:run_reps`
- **Tests:** `tests/bayes/test_e3_provenance.py, tests/bayes/test_e5_gate.py`
- **Reference:** docs/methods/BAYES_ENGINE.md section 6
