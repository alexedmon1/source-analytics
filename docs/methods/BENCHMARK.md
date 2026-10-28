# Benchmark: one scoring contract for every method (Stage 2, `feat/bayes-group`)

Plan and decisions: bayes-sl `reports/2026-10-28_stage2_benchmark_plan.md` (BD1-BD7, the Conductor 2026-10-28).
Code: `source_analytics.bayes.benchmark`. Tests: `tests/bayes/test_benchmark.py`.

Every method ("arm") is scored by the same code against the same frozen simulated cohorts with known truth: the
frequentist LMM + emmeans step, the Bayesian engine (sets W and I), the field-model locator, and later methods as
they arrive. The scorer reads files only: claims tables written by the arms, and the cohort set.

```bash
source-analytics benchmark score --cohorts <set>/v1 --claims <dir or csv> ... --out <dir> [--compare engine/W:lmm]
```

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
