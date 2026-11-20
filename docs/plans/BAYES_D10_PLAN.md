# Bayesian coverage for every outcome: the D10 plan (20 Nov 2026)

**Status:** step 1 (M1) built and its check passed on branch `d10-m1-arms` (20 Nov), waiting for your yes to merge; steps 2-17 not started. Written from the coverage table ([BAYES_COVERAGE.md](BAYES_COVERAGE.md)). It is
step 3 of bayes-sl's close-out plan section 9 (`/mnt/e/research/EEG/bayes-sl/reports/2026-11-18_closeout_migration_plan.md`).

**What was decided (the user):**
- **D10 (19 Nov):** nothing is deferred. Every outcome source-analytics handles gets a Bayesian path. That includes
  the per-region gate rule for single-group claims, the equivalence margin's unit, sessions and within-subject factors,
  the other outcome types, omnibus tests and the convergence work.
- **D11 (19 Nov):** not tied to source-localization 1.0.0. Only reading its new file format waits for that release.
- **D12 (19 Nov):** main stays fully working throughout.
- **20 Nov (answer to "should signatures stay frequentist-only?"):** no. The classifier signature modules
  (`roi_signature`, `electrode_signature`) and the source-vs-sensor comparison (`electrode_comparison`) get
  Bayesian paths too. The coverage table's section 4, item 4 is withdrawn.

## 1. How the work is run

- One branch per step, off main. A step merges into main only when its check passes, the full test suite passes and
  the user says so (D12). No step changes an existing frequentist output. The Bayesian results are written beside
  the frequentist ones, as now (`*_hypotheses_bayes.csv`).
- **Every step ends with a check.** Either it reproduces a recorded number, or the Bayesian and frequentist answers
  are compared on real data the way the FORGE G1 test did: the same direction for each contrast, and the posterior
  median inside the frequentist 95% CI. A step that adds a model also runs the gate (prior predictive check,
  simulation-based calibration, the no-difference null and recovery) at a real study's design.
- **Frequentist gaps go in the same step** as the outcome they belong to (D12). They are listed with that step.
- **Version numbers.** Each merged step is a minor release (0.11.0, 0.12.0, ...). 1.0.0 comes once every step is in
  and source-localization 1.0.0's file format is read.
- bayes-sl supplies reference numbers only (decision D6). The work happens in source-analytics.

## 2. Order

Steps 1-3 come first because they close bayes-sl. Their checks reproduce bayes-sl's numbers from its stored fits,
which may be deleted once M1 and M4 pass (decision D5). They also need the engine exactly as it is, so the engine's
recoding (step 4) comes after them. From step 5 on, the order is what the studies need:
- **FORGE's treatment study** (KO vs WT plus rescue, five groups) is the most active study. Its MS2 rerun of
  9-10 Nov is not yet frozen. It runs every region-level module: `roi_psd`, `roi_aperiodic`, `roi_connectivity`,
  `roi_cross_freq`, `roi_directed`, `roi_graph`, `roi_nbs` and `roi_signature`. It also has an exploratory group
  omnibus and electrode spectra.
- **The evoked studies** (surface-evoked's Sentinel and Autifony sets, and SL_Batterink) use `roi_evoked` and
  `electrode_evoked` on source and field runs. SL's learning effect compares two conditions within each mouse, and its
  October retest is a second session. Both need within-subject factors.
- No study has a due date. The manuscripts built on this work are paused (the user, 10 Nov), so "first" means
  "used by the most active analyses", not "due soonest".

## 3. Steps

Sizes are my estimates, not measured.

| # | Step | Serves | Work | Check | Size |
|---|---|---|---|---|---|
| 1 | **M1: running and scoring benchmark arms** | bayes-sl close-out | Arm runner (the engine with both prior sets, the frequentist `roi_psd` step, and frequentist models built like the engine), with the retry for temporary CSVs that vanish; "which region differs most" probabilities from posterior draws; the paired gain over the no-difference twin in `benchmark location` | The two-source benchmark's scorecard and the final location scorecard rows reproduced from bayes-sl's stored fits (to 1e-9 where deterministic) | 1 day |
| 2 | **M2: design analysis command** | bayes-sl close-out; SL's confirmatory design | Mice per group needed for a target rate of decisions, over the engine and the gate | The 11 Nov source-level design table (`reports/2026-11-11_source_design.md`) reproduced at the same seeds | 0.5 day |
| 3 | **M3: coarse contrasts** (cortex vs deep, per compartment) | bayes-sl close-out; surface-evoked | Declared hypotheses if they can express them; otherwise a port of `coarse_fit.R` | The 26 Oct compartment numbers | 0.5 day |
| 4 | **Engine recoding and the convergence rule** (needs your yes, section 4) | every later step | The model rewrite from the convergence report, section 6: offsets split into a mean plus deviations that sum to zero, with the region offsets sampled centred and Stan's `sum_to_zero_vector`. Gate rule: bulk ESS (effective sample size) of at least 400 on every estimand and scale parameter, beside R-hat < 1.05. The dense mass matrix becomes an option for small models, not the default. New engine version | Exactness test: the log density equals brms's own model's up to the Jacobian, at 20 points, as a test in the engine; the E0 replay of the FORGE G1 fits within Monte Carlo error; step 2's table within Monte Carlo error; bayes-sl's five convergence fits meet the rule | 1 day |
| 5 | **FORGE region outcomes the engine can already read** | FORGE treatment | Built-in tables (`BAYES_TABLES`) for `roi_psd` relative power and delta-referenced power, the `roi_aperiodic` offset, phase-amplitude coupling (`roi_cross_freq` PAC) and `roi_graph`'s nodal and global metrics. Posterior predictive check for each; if the Gaussian model fails it (relative power is a fraction, degree and betweenness are counts), that outcome goes to step 14 | Each compared with its frequentist result on FORGE's treatment study, as in G1 | 1.5 days |
| 6 | **Decisions D10 left open** | FORGE treatment, SL | (a) The equivalence margin's unit: the residual SD (as now, and as the frequentist adapter does) or the between-mouse SD. (b) The gate rule for one group's level against 0, per region. Both need a proposal and your answer, section 4 | Gate results for both at FORGE's design | 0.5 day |
| 7 | **Omnibus** | FORGE treatment (exploratory `group_omnibus`) | Bayesian counterpart of the F test: a Bayes factor for "every group equal" in each band, by Savage-Dickey on the vector of group differences (the prior is Gaussian, so its density at 0 is exact; the posterior density at 0 comes from a multivariate normal fit to the draws). Also reported: the posterior probability that every difference is inside the ROPE. The engine's `not_implemented_bayes` row goes | Gate: the null rate at FORGE's design; on real data, agreement in direction with the F test's verdict | 1 day |
| 8 | **Electrode outcomes** | FORGE electrode spectra; evoked studies | Built-in tables with a channel spatial column for `electrode_psd`, `electrode_aperiodic`, `electrode_evoked` and `electrode_connectivity`'s FCD. **Frequentist gaps:** a test that runs `electrode_psd`'s R script, and one that runs `electrode_connectivity` as a module | FORGE electrode KO vs WT (bayes-sl 10 Nov report) for spectra; a Sentinel set for evoked | 1 day |
| 9 | **Field runs and located-field region values** | surface-evoked | A test that runs `roi_evoked` on a field run, both frequentist and Bayesian. **Frequentist gap:** a frequentist path for located-field region values, using the same model as the tabular evoked path | The a97d field job (exact, already recorded) for the Bayesian side; a97d's study-script frequentist numbers for the new frequentist path | 0.5 day |
| 10 | **Sessions and within-subject factors (G2)** | SL learning (structured vs random within each mouse); SL October retest (a session term); FORGE's sessions, now coded as groups (e.g. `G2_D1pre`) | Engine terms for a within-subject factor (its effect per group, a per-mouse deviation from it) and for sessions. Step 4's rewrite assumes each mouse sits in one cell, so "the mouse's own mean" has to be defined per cell first. **Frequentist gaps:** check whether a mouse with sessions gets one random intercept in the linear mixed models, and model sessions explicitly in the frequentist path (they are coded as groups now) | The SL learning per parcel result (bayes-sl 10 Nov, fitted on per-mouse differences) reproduced from the two conditions; the gate at the SL retest design | 2 days |
| 11 | **Regression in the tabular path** (frequentist gap) | FORGE `roi_graph`; evoked | Covariate regression in `hypothesis/tabular.py`, as the R layer has it (the engine already does it) | Against the R layer on the same table | 0.5 day |
| 12 | **Per-pair outcomes** (connectivity, AAC/PPC, transfer entropy, electrode pairs) | FORGE treatment (wPLI region heatmaps, directed) | A model with two spatial columns. Each pair's effect = the two regions' node effects + the pair's own deviation, partially pooled. Directed pairs keep the order (A to B differs from B to A) | Gate at FORGE's 26 regions (325 pairs); on real data, compared with the frequentist per-edge and region-pair results | 2-3 days |
| 13 | **Network-Based Statistic counterpart** | FORGE treatment (`roi_nbs`) | From step 12's draws: in each draw, the pairs whose effect is outside the ROPE and the connected sets they form. Reported: each pair's probability of belonging to such a set, and the posterior of the largest set's size | Gate null at FORGE's design (how often a set appears with no difference); against NBS on FORGE | 1 day |
| 14 | **Non-Gaussian outcomes, if step 5 or 12 asks for them** | as found | Families chosen by what failed: logit or beta for fractions and 0-1 coupling values, negative binomial for counts | The posterior predictive check that failed now passes; the gate at the design | 1-2 days, only if needed |
| 15 | **Signature modules** (`roi_signature`, `electrode_signature`) | FORGE treatment | Needs a design first (section 4). Proposed: a Bayesian logistic regression of group on the signature's features, with a shrinkage prior. Readouts: the Bayes factor against a model with no features; the cross-validated accuracy from PSIS-LOO beside the frequentist LOOCV accuracy; each feature's posterior weight | Gate null (labels shuffled): how often evidence for a signature appears with none; on FORGE, against the permutation test | 1.5 days |
| 16 | **Source-vs-sensor comparison** (`electrode_comparison`) | FORGE | Per band, a Bayesian correlation between electrode and source power, with the bands partially pooled (Fisher z). The level comparison is source vs electrode as a within-mouse factor, so it uses step 10 | Against Pearson r and Hedges g on FORGE | 1 day |
| 17 | **Docs** | everyone | Each step updates the catalogue and `BAYES_ENGINE.md`. At the end, the coverage table is redone with every "possible" or "missing" mark changed or explained (with bayes-sl's M5) | read-through | 0.5 day |

Total: about 17-19 days of work, plus fitting time for the gates (FORGE-sized gates have taken hours each).

## 4. Decisions this plan needs from you

| Where | Question | Proposed |
|---|---|---|
| Step 4 | Adopt the recoding (rewrite, centred region offsets, `sum_to_zero_vector`), and the rule "bulk ESS at least 400 on every estimand and scale parameter"? | Yes. In bayes-sl's tests it fixes the slow mixing in all five fits, FORGE included, without changing the posterior (convergence report section 6) |
| Step 6a | Equivalence margin's unit: residual SD or between-mouse SD? | To be written up in step 6, with the gate numbers for both |
| Step 6b | Gate rule for one group's level against 0 in a region | To be written up in step 6 |
| Step 15 | The signature model: Bayesian logistic regression with a shrinkage prior (proposed), or a model of each mouse's held-out score? | To be decided before step 15 starts |
| Order | Steps 1-3 before the recoding, so that bayes-sl's stored fits can be retired sooner | Yes |

## 5. What is not in this plan

- M4 (the simulated-study generator) and M6 (closing bayes-sl). They are in source-localization and bayes-sl
  (close-out plan section 4).
- Rerunning the benchmarks on the fixed head model, and moving the real-data source results to their studies
  (bayes-sl items 441d, Stage 2-3). They come after M4 and are not conditions for closing bayes-sl.
