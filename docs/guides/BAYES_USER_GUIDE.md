# User guide: the Bayesian group engine

From installation to reading a result. The engine answers each hypothesis your study declares with a posterior
instead of a p-value: an estimate with its 95% interval, the probability of its sign, the probability that it is too
small to matter, and a decision (effect, null or undecided). Before it fits your data it checks itself by simulation
at your exact design (the *gate*). The method, the model and every prior are in
[BAYES_ENGINE.md](../methods/BAYES_ENGINE.md) and the [engine catalogue](../methods/BAYES_CATALOGUE.md).

## 1. Install

```bash
uv sync                                                        # the Python package
Rscript -e 'install.packages(c("brms", "posterior", "digest", "optparse", "jsonlite", "dplyr"))'
Rscript -e 'install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))'
Rscript -e 'cmdstanr::install_cmdstan()'
source-analytics bayes --check-env                             # lists what is installed; exit 1 if anything is missing
```

If your R packages live in a personal library, set `R_LIBS` (for example `export R_LIBS=$HOME/R`) in every shell that
runs the engine; background services do not read your shell profile (section 6).

## 2. Declare what you want to know

The engine reads the same `design:` and `hypotheses:` blocks as the frequentist analyses
([DESIGN_SPEC.md](../methods/DESIGN_SPEC.md)):

```yaml
design:
  factor: group
  reference: WT_VEH
  levels: [WT_VEH, KO_VEH]
hypotheses:
  - {name: disease_effect, kind: contrast, weights: {KO_VEH: 1, WT_VEH: -1}}
```

Contrasts, equivalence (with a margin) and regression are supported; omnibus is not.

**Is one group's effect different from 0?** When the outcome is itself an effect (a difference such as structured -
random ITC), a declared table with `center: false` keeps its 0, and a contrast with one group, `weights: {KO: 1}`,
tests that group's level against 0 ([BAYES_ENGINE.md, section 4](../methods/BAYES_ENGINE.md)). Without
`center: false` the engine refuses it: the outcome is centred, so a group's level only means something relative to
the others.

## 3. The `bayes:` block

Everything that changes a number is in the study YAML; command-line flags only say what to run and where.

```yaml
bayes:
  rope: {mode: sd, value: 0.1}          # differences within +-0.1 between-subject SD count as too small to matter
  prior_sets: [W, I]                    # W: weakly informative (engine defaults); I: informative, from a file
  priors: {I: priors/informative.yaml}
  fit: {chains: 4, iter: 2000, warmup: 1000, seed: 1}
  gate:
    reps: {sbc: 100, null: 100, recovery: 10}
  outcomes:
    roi_aperiodic: [exponent]           # what `source-analytics bayes` runs by default
```

- **Units.** Each outcome is standardised by its pooled within-group between-subject SD, so estimates, priors and the
  ROPE are in SD units (a per-cell Hedges g). The report also gives each estimate in the outcome's own unit.
- **Set W** uses the engine's default priors, listed in the [engine catalogue](../methods/BAYES_CATALOGUE.md).
- **Set I** needs a file of entries, each with its source; informative priors come from a logged literature search,
  never from the data being analysed:

  ```yaml
  version: 1
  entries:
    - {dv: exponent, hypothesis: disease_effect, prior: {mean: 0.5, sd: 0.8},
       source: "lit/2026-10-07_fmr1_resting_state.md, table 3"}
  ```

  `band:` names the band for banded outcomes. Both sets are reported side by side.

**Other tables** (anything per subject: evoked measures, field runs, a CSV from another tool) are declared under
`bayes.tables` ([BAYES_ENGINE.md, section 2.1](../methods/BAYES_ENGINE.md)): which rows make each outcome, whether
the model has ROIs, and how the table's groups map (or pool) onto the design's levels.

## 4. Run: gate, then fit

The analysis that writes the table must have run (`source-analytics run --analysis roi_aperiodic ...`), or the
declared CSV must exist.

```bash
# 1. the gate: prior predictive, simulation-based calibration, null, recovery, at your design (hours)
source-analytics bayes --study study.yaml --paradigm resting --analysis roi_aperiodic --dv exponent --gate

# 2. the fit (minutes): the table, its manifest and the report
source-analytics bayes --study study.yaml --paradigm resting --analysis roi_aperiodic --dv exponent
```

A declared table needs no `--paradigm`. `--out DIR` writes the table and fits elsewhere (for example when the study's
results are frozen). A fit records whether a gate *with its own id* passed (`gate_passed`): the id covers the design
rows, the outcome values, the hypotheses, the priors, the ROPE, the gate settings and the engine version, so a gate
run for other data or settings does not count. Gate replications already written are skipped on a re-run.

**If the gate fails,** do not loosen a threshold. Look at which rule and which outcome failed: a prior predictive
failure says a prior does not fit the scale of the data (a structural fix, recorded as a new engine version); a single
calibration or null failure in one outcome among identical ones can be chance, and the honest next step is more
replications with the thresholds unchanged (the first ones are kept). A fit outside a passed gate is reported as such
in every table row and in the report.

## 5. Read the result

Each fit writes, beside the table:

- `<analysis>_hypotheses_bayes.csv`: one row per hypothesis × band × spatial cell × prior set (schema 1, columns
  below);
- `<analysis>_hypotheses_bayes_manifest.json`: what made it (engine version, inputs and their sha256, settings, gate
  ids, convergence, package versions);
- `<analysis>_hypotheses_bayes_<dv>_report.md`: the readable report (also `source-analytics bayes-report --table ...`).

| column | meaning |
|---|---|
| `hypothesis`, `kind`, `label`, `role` | the declared hypothesis |
| `dv`, `band`, `spatial` | outcome, band (or declared outcome label), spatial cell (`all` = across ROIs) |
| `prior_set` | W or I |
| `estimate`, `hdi_lo`, `hdi_hi` | posterior median and 95% highest-density interval, between-subject SD |
| `p_direction`, `p_rope` | probability of the sign; probability inside the ROPE |
| `rope_lo`, `rope_hi` | the ROPE (or the equivalence margin) |
| `log10_bf10` | Savage-Dickey Bayes factor for a non-zero effect; read very large values only as decisive |
| `decision` | `effect` (HDI outside the ROPE), `null` (inside), `undecided`; for equivalence `equivalent` / `not_shown` |
| `rhat`, `ess_bulk`, `max_rhat_scale`, `max_rhat`, `n_divergent` | convergence (R-hat < 1.05 on estimands and scales) |
| `status`, `fdr_family` | `ok` or why not; the family (partial pooling is the adjustment, no FDR) |
| `gate_passed`, `gate_id` | whether this fit's own gate passed (TRUE, FALSE, or empty when none ran) |

The report shows the run (study, inputs, model and priors by catalogue id), the results with both prior sets side by
side and the gate's false-claim rate beside each claim, the gate per set, convergence, and the prior and posterior
predictive checks (observed spread and mean against data replicated from the model).

## 6. Long runs

A gate takes hours. Run it so that it survives the terminal closing:

```bash
P=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/c' | paste -sd:)     # on WSL: a Linux-only PATH, quoted
mkdir -p $PWD/scratch/rtmp
systemd-run --user --unit=my-gate --collect "--setenv=PATH=$P" --setenv=HOME=$HOME --setenv=R_LIBS=$HOME/R \
  --setenv=TMPDIR=$PWD/scratch/rtmp -p WorkingDirectory=$PWD \
  nohup /bin/bash -c "exec source-analytics bayes --study study.yaml --analysis my_table --gate >> gate.log 2>&1"
systemctl --user show -p MainPID --value my-gate          # then: cat /proc/<pid>/cgroup -> .../my-gate.service
```

- **`R_LIBS` and `PATH`:** a user service does not read your profile; without them R cannot find its packages.
- **`TMPDIR`:** cmdstanr writes its data files to R's temporary directory; point it at the project.
- **No forked workers:** the gate runs its parallel fits in separate R processes (PSOCK). Do not wrap cmdstanr fits in
  forked workers (`mclapply`): their process group is killed when the parent exits.
- A run started from an interactive session's process group can be killed when that session ends; the unit above
  is outside it.

## 7. Worked example: FORGE's aperiodic exponent

FORGE (rest, `roi_based` arm) was the engine's acceptance test (G1). With the FORGE study YAML and its `roi_aperiodic`
output in place:

```bash
source-analytics bayes --check-env
source-analytics bayes --study forge.yaml --paradigm resting --analysis roi_aperiodic --dv exponent --gate
source-analytics bayes --study forge.yaml --paradigm resting --analysis roi_aperiodic --dv exponent
```

Then read `roi_aperiodic_hypotheses_bayes_exponent_report.md`. The frozen reference fits and gate verdicts of this
example are replayed by `tests/bayes/test_e0_regression.py` (with `SA_BAYES_E0` pointing at them), which requires every
readout row, gate verdict and simulator draw to reproduce; engine 2026.11.09.1 passes it (2026-11-09).

## 8. Worked example: a field readout

A study that projected its recordings onto field templates (source-localization's field mode) scores them with
`roi_evoked` like parcels; source-analytics refuses to pool field runs with source runs. To size a deficit along the
fields, declare the measures as a table:

```yaml
bayes:
  tables:
    deficit_along_field:
      source: roi_evoked
      paradigms: [assr_40hz, chirp]
      value: value
      spatial: none
      outcomes:
        40 Hz ASSR ITC:  {paradigm: assr_40hz, roi: assr40_D, measure_name: itc_40hz}
        chirp 40 Hz ITC: {paradigm: chirp, roi: chirp_D, measure_name: itc_40hz_sweep}
      groups: {G2_D1pre: KO, G3_D1pre: KO, G4_D1pre: KO, G5_D1pre: KO, G6_D1pre: WT}
  outcomes:
    deficit_along_field: [value]
```

and run `source-analytics bayes --study study.yaml --analysis deficit_along_field --gate`, then without `--gate`.
Region values from source-localization's `field region-values` are a `csv:` table with `outcomes` per region.
