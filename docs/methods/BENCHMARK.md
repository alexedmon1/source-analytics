# Benchmark: one scoring contract for every method (Stage 2)

Plan and decisions: bayes-sl `reports/2026-10-28_stage2_benchmark_plan.md` (BD1-BD7, the Conductor 2026-10-28).
Code: `source_analytics.bayes.benchmark`. Tests: `tests/bayes/test_benchmark.py`.

Every method ("arm") is scored by the same code against the same frozen simulated cohorts with known truth: the
frequentist LMM + emmeans step, the Bayesian engine (sets W and I), the field-model locator, and later methods as
they arrive. The scorer reads files only: claims tables written by the arms, and the cohort set.

```bash
source-analytics benchmark score --cohorts <set>/v1 --claims <dir or csv> ... --out <dir> [--compare engine/W:lmm]
```

## Running the arms (D10 step 1, M1)

Ported from bayes-sl's scripts (`R/ee3_fit.R`, `R/f1_fit.R`, `R/f1c_fit.R`, `scripts/bm_b1.py`, `bm_f0.py`,
`bm_l4.py`, `bm_l4_contrasts.py`, the retry in `r2_arms.sh`). Code: `source_analytics.bayes.arms`,
`source_analytics.bayes.benchmark_probs`, `R/bayes/bench_fit.R`, `R/bayes/bench_lmm.R`.
Tests: `tests/bayes/test_benchmark_arms.py`.

```bash
source-analytics benchmark run --arm engine --cohorts <set> --out runs/engine [--prior-set I --informative I.json]
source-analytics benchmark run --arm F0  --cohorts <set> --out runs/f0       # roi_psd's own R step, unchanged
source-analytics benchmark run --arm F1  --cohorts <set> --out runs/f1       # engine structure, lme4, mice resampled
source-analytics benchmark run --arm F1c --cohorts <set> --out runs/f1c      # same fit, conditional covariance
#   all: [--cells a,b] [--reps 1:7] [--n 8,30] [--parts 4] [--r-dir <frozen R copy>]; resumes
source-analytics benchmark claims --arm engine --runs runs/engine --out claims/engine.csv
source-analytics benchmark claims --arm F0 --runs runs/f0 --cohorts <set> --out claims/f0.csv
source-analytics benchmark probs --cohorts <set> --fits engine/W=runs/engine F1=runs/f1 F1c=runs/f1c \
    --f0-claims claims/f0.csv --hierarchy regions.csv --out probs/
source-analytics benchmark location --cohorts <set> --hierarchy regions.csv --probs probs/probs.csv \
    --question difference --twins [--group-by spacing_mm pair] [--reps 1:7] --out scores/
```

| Arm | What it is | Uncertainty |
|---|---|---|
| `engine` (W, I) | the Bayesian engine, two groups, contrast B - A, the planted and control bands | posterior; every 4th joint draw kept |
| `F0` | source-analytics' `roi_psd` step: per band `dv ~ group * roi + (1 \| subject)`, emmeans per ROI, BH across ROIs within band | SE; no joint draws |
| `F1` | the engine's model without priors, one band: `z ~ 0 + a + b + (0 + x \| roi) + (1 \| roi) + (1 \| subject)`, lme4 REML | mice resampled within group, 200 refits |
| `F1c` | F1's fit | the conditional (Henderson) covariance of fixed and random effects, variances held fixed: too narrow at small N |

Every arm cuts each cohort to the first N mice of each group for each N in its `n_nested`, seeds each fit
`1000 * rep + N`, and skips fits already written. R parts run in parallel, each with its own `TMPDIR`, and a part
that fails is retried up to 3 times: cmdstan's output CSVs on `/mnt/e` have vanished right after sampling (bayes-sl,
4, 13 and 14 Nov; cause unknown), and a retry redoes only the fit that was lost. `--r-dir` sources the engine (or
`roi_psd`) from another R directory, so a frozen copy repeats old fits.

**Which region differs most** (`benchmark probs`, `probs.csv`, for `benchmark location --question difference`): from
an arm's joint draws of B - A per ROI (SD units), P(r) is the share of draws in which |contrast| is largest at r, and
P(none) the share in which every |contrast| is below the ROPE (0.1 SD). Tied draws are shared (a singular lme4 fit
gives every ROI the same contrast). F0 has no joint draws: 1000 independent normal draws from its estimates and SEs,
seeded by cohort and N, put in SD units with the engine's unit for the same cohort and N; this ignores correlation
between ROIs. `contrasts.csv` (with `--hierarchy`): P(cortex > deep) and P(right > left) from the draws' means over
each set of ROIs (midline left out, a tie counts half), with the same reduction of the path truth.

**Gain over the no-difference twin** (`benchmark location --twins`, `gain.csv`): a cell with a difference is paired
with its twin, the same source(s) in both groups with no difference and the same seeds (the manifest's `twin`, else
`<condition>_nodiff`). Gain = the study's hit - the twin's `top_at_source`, per method x N x level: the hits the
difference caused, not the ones the source would have won anyway. Its 95% interval resamples reps within each cell.
Twins in another cohort set: `--twin-cohorts <set> --twin-probs <csv>`.

**Check (M1).** With `SA_BAYES_BENCH=<bayes-sl>`: the probabilities from bayes-sl's stored draws and F0 claims equal
its `location_probs/{v1,v1_l5,r2_v1}/L4/{probs,contrasts}.csv`; scored here, they give its stored L4 scorer tables;
the gain table gives its R2 two-source scorecard (`location_scores/R2/accuracy.csv`, test reps) and its final location
scorecard (`L6_final/accuracy.csv` gain rows and `pairs.csv`), bootstrap intervals included, all to 1e-9. The
field-model arms' stored tables are read beside the readout arms' there, because the bootstrap's generator runs
through every arm in order; porting those arms is M4 (source-localization). Rerun through `benchmark run` (20 Nov
2026), a v1 study (Thalamus_1.0 rep 1, N 8 and 30) repeats bayes-sl's stored fits exactly, draws included: F0 with
today's `roi_psd` step, F1 and F1c, and the engine's sets W and I with the frozen copy (`--r-dir`, a450d5a). The
current engine gives the same W fits too (zero difference at both N): nothing changed since a450d5a touches this
model. The rerun is not a test (the fits take minutes); the commands are above.

## Inputs

**Cohort set** (written by bayes-sl `scripts/bm_cohort.py`): `MANIFEST.json` (cells with their planted ROIs, the
ROI and band lists, the per-band real-data units, every file's sha256) and per cohort `path_truth.csv` and
`plant.csv`.

**Claims table:** one row per arm x cohort x N x band x unit (`ROI-average`, an ROI, or `located`), with the
estimate of group B - group A and its 95% interval in dB, the arm's own decision (`effect_pos`, `effect_neg`,
`null`, `undecided`) and the matched decision (`excludes_0`, `includes_0`). Optional: `se_db`, `p_direction`,
`log10_bf10`, `rope_db`, positions (`x_mm`, `y_mm`, `z_mm`), `runtime_s`, `converged`. The full list is in the
module docstring. A table that breaks the contract stops with a message (unknown cohort, unit or decision; duplicated
rows; lo > hi; located rows without positions).

`claims_from_engine()` turns an engine readout (SD units) into claims rows, using the fit's unit per band.

## What it scores

| Table | Question | Against |
|---|---|---|
| `noise` | False claims at the ROI average and in any ROI, own and matched rule; pass: claim_effect <= 0.05, claim_dir <= 0.10 | noise-only cohorts |
| `calibration` | Bias, RMSE, coverage, interval width, SD of standardised errors, variance inflation; ROI-average and per ROI; pass: ROI-average coverage >= 0.85, per ROI median >= 0.85, min >= 0.75 | path truth |
| `attribution` | Where per-ROI claims in the planted band fall: planted, leaked but real (path truth beyond 0.1 x the band's real-data unit), elsewhere, wrong sign; usable as a location if planted >= 0.80 | source truth (planted ROIs) |
| `location` | For located arms: distance to the nearest planted source (median, 90th percentile, share within 1 mm) | plant positions |
| `design` | Decision rate by N, own and matched rule; N for 80% from a logistic fit in log N, never beyond 2x the largest N | planted cells |
| `compare_A_vs_B` | B - A paired by cohort: width, absolute error, coverage, claim rates; bootstrap 95% interval over cohorts | same cohorts |

**Two decision rules.** Each arm's own rule (engine: HDI outside the ROPE; frequentist: p after FDR) and a matched
rule, "the interval excludes 0", the same for every arm, so a difference between arms is not just a different
threshold.

## Checks

- Synthetic cohorts and claims with known answers: an honest arm is calibrated and its claims are attributed as
  planted; biased and over-confident arms fail calibration; wrong-place and wrong-sign claims are counted as such;
  noise claims fail the noise rule; located distances; paired differences; N80 recovers a known logistic curve;
  every contract error.
- Parity (`SA_BAYES_BENCH=<bayes-sl>`): the engine's EE3 pilot fits (175) give bayes-sl `scripts/ee3_score.py`'s
  tables (noise, calibration, attribution, design rates) to 1e-9, with EE3's leak rule (`--leak-rule row_rope`).

**Changed from EE3:** a claim against the sign of a negligible truth counts by where it is, not as "wrong sign"
(wrong sign needs a real or planted difference); the leak threshold is arm-independent by default (EE3 used each
fit's ROPE in its cohort's unit). Neither changes the pilot's numbers.
