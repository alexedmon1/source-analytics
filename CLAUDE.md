# CLAUDE.md — source-analytics

Group-level statistics for source-localized EEG. **Python** orchestrates, loads
reconstructions, does signal processing, and runs the permutation/cluster stats
for vertex and sensor maps. **R** (lme4/lmerTest/emmeans, ggplot2) does the LMM
statistics and figures for the ROI/electrode modules. Python calls `Rscript`.

`README.md` is the user manual and is kept in sync with the code. For methods,
`docs/methods/` is authoritative. When the two disagree with the code, fix the doc.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[all]"        # mne / scikit-learn / networkx / nibabel extras
Rscript -e 'install.packages(c("ggplot2","dplyr","tidyr","readr","forcats","lme4","lmerTest","effectsize","emmeans","yaml","argparse","optparse","patchwork","scales","ggsignif"))'
```

Tests: `.venv/bin/python -m pytest -q`. R syntax check:
`for f in R/*.R; do Rscript -e "invisible(parse(file='$f'))"; done`.

## Usage

```bash
source-analytics init /path/to/rest_roi --name study        # writes rest_roi/analysis/study.yaml
source-analytics validate --study study.yaml
source-analytics list --study study.yaml
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd
source-analytics run ... --steps statistics,figures,summary  # figures are OFF by default
scripts/run_study.sh study.yaml                             # whole study, dependency order
```

## Layout

```
src/source_analytics/
  cli.py            run / validate / list / figure / init
  core.py           ANALYSIS_REGISTRY (+ deprecated aliases), ANALYSIS_METADATA, StudyAnalyzer
  config.py         StudyConfig, DesignSpec (design:/hypotheses:), profiles, paradigm scoping
  analyses/         one module per analysis, all subclass analyses/base.BaseAnalysis
  hypothesis/       Python adapters (tabular / permutation / edge) for declared hypotheses
  spectral/, stats/, io/, viz/, atlas/
R/                  <module>_analysis.R scripts, hypothesis.R (emmeans adapter), stats_utils.R
scripts/run_study.sh
docs/methods/       APERIODIC_FIT_WINDOW, CONNECTIVITY_METHODS, HYPOTHESIS, DESIGN_SPEC
```

Lifecycle per module: `setup → process(_subject) → aggregate → statistics →
figures → summary`. `DEFAULT_RUN_STEPS` excludes `figures`.

## Output contract

- Working tree: `paths.analytics/[<profile>/]<paradigm>/<analysis>/` holds
  `data/` (per-subject CSVs + `study_config.yaml` snapshot) and `ANALYSIS_SUMMARY.md`.
- Published: `paths.results/[<profile>/]{tables,figures}/<paradigm>/<analysis>/`
  (`BaseAnalysis.tbl_dir` / `fig_dir`). source-lightbox reads these.
- Every inferential module writes `<analysis>_hypotheses.csv` (one row per
  band × spatial cell, or per cluster for map modules) into `tables/`.
- Every tabular row carries the minimum detectable effect, because DUET (OPEN.md
  #151) reads each finding's size against it and only the fitted model has the SE
  and df: `mde` = `(qt(1-α/2, df) + qt(power, df)) * SE` (normal quantiles when
  df is NA/Inf), `mde_percent` = `100 * mde / |reference mean|` (NA for slopes,
  omnibus rows, or a reference mean within one residual SD of zero), `mde_power`
  0.80, `mde_alpha` 0.05 **nominal (uncorrected), by the user's decision**, and
  `mde_note` (empty, or `not computed: <reason>` -- mde is never NA without one).
  `SE` is the contract's `se`; do not rename it or add a lowercase twin. Omnibus,
  cluster-permutation and NBS rows get the note, not a number. Helpers:
  `mde_from_se` / `.add_mde` in `R/hypothesis.R`, `hypothesis/power.py` in
  Python. The column names are shared with neurofaune and neurovrai.
- Band power CSVs carry `absolute` (mean power density, dB/Hz) and `relative`.
  There is no `dB` column.

## Conventions

- Known defects and open follow-ups are in `docs/KNOWN_ISSUES.md`; check it before
  touching effect sizes, the MDE columns or the lock file.

- Deprecated analysis names (`psd`, `pac`, `vertex_mvpa`, …) are in
  `core._DEPRECATED_NAMES`; output always goes under the canonical name.
- R scripts derive contrasts with `contrasts_from_spec(parse_design_spec(config))`
  from `R/hypothesis.R`; never read `config$contrasts` directly.
- Optional deps are lazy: mne (evoked/TFR), scikit-learn (signature), networkx
  (graph). The package must import without them.
- Adding an analysis: subclass `BaseAnalysis`, register in `ANALYSIS_REGISTRY`,
  add an `ANALYSIS_METADATA` entry (`category`, `level`, `domain`, and
  `supplements`/`requires` for secondaries), declare `SELECTABLE`, add the R
  script if it uses emmeans, then add it to the README catalog and
  `scripts/run_study.sh`. Keep the timeout value and its log message in step.
