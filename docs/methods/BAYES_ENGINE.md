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
source-analytics bayes ... --gate            # prior predictive + recovery/SBC + null at this design, no real fit
```

- **Inputs:** the step reads the module's existing `data/*.csv` (the analytics working tree). It needs no subject
  processing, so the frequentist pipeline is untouched.
- **Python:** a thin wrapper (`hypothesis/bayes.py`: paths, design spec to JSON, `Rscript`).
- **R:** the model, priors, gate and readout in `R/bayes_hypothesis.R` and `R/bayes_gate.R`. Python calls `Rscript`,
  as the emmeans modules do.

## 3. Model

The outcome is standardised before fitting, per band: z = (y - band mean) / s_b. Here s_b is the within-group
between-subject SD within ROI x band, pooled over ROIs and groups. This is the unit of a per-cell Hedges g. The total SD over all rows is not used,
because ROI differences in absolute dB would inflate it. Effects, priors, the ROPE and `mode: sd` margins are all in
these units.

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
p_rope, rope_lo, rope_hi, log10_bf10, decision, max_rhat, n_divergent, min_ess_bulk, gate_passed, label
```

Fits, the prior table used and the gate summary go to `data/bayes/` (provenance).

## 5. Priors: generated from the spec, two sets

**W (weakly informative, SD units):**
- band-specific group effects N(0, 1). This is Track A's weak default; the first draft had 0.5, which was
  tighter than the library's informative entries (SD 0.8). The Conductor set it, 2026-10-14;
- band intercepts N(0, 1);
- ROI-deviation SDs half-N(0, 0.25);
- ROI-level and subject SDs half-N(0, 0.5);
- log sigma N(log 0.5, 0.5).

These are proposals. Each must pass the prior predictive check at the study design. The tectum model needed three
rounds.

**I (informative):** only from a logged literature search. **For FORGE (the Conductor, 2026-10-14): pooled.**
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

A gate result is cached per (spec hash, design hash, prior hash). The real-fit table carries `gate_passed`.

**Cost.** This is the open risk:
- FORGE power is 60 subjects x 26 ROIs x 7 bands = 10,920 rows, with about 900 ROI-deviation terms.
- The fit time is not measured yet. A first template fit gives it.
- If one fit takes more than about 10 min, 100-200 SBC replications on rstan (no cmdstanr installed) is a day or
  more.
- Options: install cmdstanr (within-chain threading, faster); run SBC per band subset; or a reduced SBC (100) with
  the null at 50.

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
