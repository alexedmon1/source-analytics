# source-analytics

Group-level statistical analysis toolkit for **source-localized EEG**. It is the
middle stage of a three-package pipeline:

```
source-localization  ──►  source-analytics  ──►  source-lightbox
(reconstruct sources)     (stats + figures)       (render the gallery)
```

[`source-localization`](../source-localization) turns raw EEG into per-subject
source reconstructions (ROI timeseries, whole-brain source estimates).

> **📖 Methods documentation:** [`docs/`](docs/index.md) — the decisions behind
> the numbers, with primary-literature citations. Start with the
> [aperiodic fit window](docs/methods/APERIODIC_FIT_WINDOW.md) and
> [connectivity metrics](docs/methods/CONNECTIVITY_METHODS.md).
> Build the site locally with
> `uv run --no-project --with "mkdocs-material>=9.5,<10" mkdocs serve`.
**source-analytics** reads those reconstructions, runs group-level analyses
(spectral, connectivity, cross-frequency, directed, …), and writes
publication-quality statistics tables and figures. [`source-lightbox`](../source-lightbox)
then reads the *same study config* and the stat tables to build a browsable gallery.

**Python** handles orchestration, signal processing, and I/O. **R** handles the
linear-mixed-model statistics (lme4/lmerTest/emmeans) and ggplot2 figures. Vertex-
and sensor-map modules do their statistics in Python (cluster permutation) and use
R only for the markdown report. Python calls `Rscript` automatically — there is no
manual R step.

---

## Contents

- [Quickstart (after source-localization)](#quickstart-after-source-localization)
- [Installation](#installation)
- [Core concepts](#core-concepts) — the mental model
- [Input: the source-localization handoff](#input-the-source-localization-handoff)
- [Study configuration](#study-configuration)
- [The CLI](#the-cli)
- [Analysis catalog — what exists](#analysis-catalog--what-exists)
- [Retired: the vertex level](#retired-the-vertex-level)
- [Hypothesis testing](#hypothesis-testing)
- [Selecting metrics, bands & hypotheses](#selecting-metrics-bands--hypotheses)
- [Output structure](#output-structure)
- [Running a full study, in order](#running-a-full-study-in-order)
- [Architecture](#architecture)
- [Extending: add an analysis](#extending-add-an-analysis)
- [Reference documents](#reference-documents)

---

## Quickstart (after source-localization)

You have run `source-localization` and have a derivatives tree of per-subject
reconstructions. Three commands take you from there to results:

```bash
# 1. Scaffold a study config from the reconstruction directory. It is WRITTEN to
#    <dir>/analysis/<name>.yaml (status goes to stderr); pass `--output -` to
#    print the YAML to stdout instead. --groups-from reuses the subject→group
#    mapping from the source-localization config.
source-analytics init /path/to/localization/rest_roi \
    --name study \
    --groups-from /path/to/localization/study_config.yaml
#   -> /path/to/localization/rest_roi/analysis/study.yaml

# 2. Edit that file — it already has groups, `design:`/`hypotheses:` (an omnibus
#    plus every pairwise contrast), the canonical bands, and one `paradigms:`
#    block (`resting`, with roi_psd / roi_aperiodic / roi_connectivity). Add
#    paradigms and analyses as needed (see "Study configuration").

# 3. Sanity-check config + subject discovery before any long run.
source-analytics validate --study rest_roi/analysis/study.yaml

# 4. Run an analysis. --paradigm picks the block under `paradigms:` in the config.
source-analytics run --study rest_roi/analysis/study.yaml --paradigm resting --analysis roi_psd
```

`init` discovers subjects in either layout the toolkit reads: BIDS-style
`derivatives/sub-*/` (groups come from `--groups-from`, otherwise `UNKNOWN` until
you edit them) or `derivatives/<Group>/<Subject>/` (the folder name is the group).

Each run writes per-subject data + `ANALYSIS_SUMMARY.md` under `paths.analytics`
and the published `tables/` + `figures/` under `paths.results` (see
[Output structure](#output-structure)). **Figures are not produced by a default
run** — add `--steps …,figures` or use `source-analytics figure`.
`source-analytics list` shows every analysis you can run; `--atlases`, `--plugins` and `--all` widen it to what else this install offers.

---

## Installation

### Python (3.10+)

```bash
pip install -e ".[all]"   # or:  uv pip install -e ".[all]"
```

Core dependencies (always installed): numpy, scipy, pandas, pyyaml, specparam,
joblib, matplotlib. The rest are **extras** — the package imports and the CLI
works without them, and a module that needs one fails with an ImportError naming
the extra to install:

| Extra | Pulls in | Needed by |
|---|---|---|
| `mne` | mne | `roi_evoked`, `electrode_evoked` (Morlet TFR) |
| `mvpa` | scikit-learn | `electrode_signature`, `roi_signature` |
| `network` | networkx | `roi_graph`, `roi_nbs`, `roi_network` |
| `atlas` | nibabel | atlas readers |
| `all` | all of the above + dev tools | a full study |

Wheels/sdists ship the R scripts under `<prefix>/share/source-analytics/R`; an
editable checkout uses `R/` beside `src/`. `SOURCE_ANALYTICS_R_DIR` overrides
the lookup.

> **uv users:** run the CLI with `uv run --no-sync source-analytics …`. Plain
> `uv run` can trip on the lockfile; `--no-sync` avoids the re-resolve.

### Reproducing a published analysis

`specparam` has no final 2.0 release — the index carries `2.0.0rc7` at the
latest, and PEP 440 sorts every release candidate *below* `2.0`. A declaration
of `specparam>=2.0` therefore resolves to nothing at all, which is what v0.4.0
shipped with. The floor is now `>=2.0.0rc6`.

That makes the package installable, but it does not make a result reproducible:
the aperiodic numbers depend on the exact specparam build, and `mne` is
unpinned. Install against a lockfile that pins both, then add the package
without letting it re-resolve:

```bash
uv venv .venv
uv pip install --python .venv -r <lockfile>
uv pip install --python .venv --no-deps "source-analytics @ git+https://github.com/alexedmon1/source-analytics.git@<tag>"
```

⚠ **`v0.4.0` is a scientific pin, not merely an old version.** It hardcodes
`freq_range=(2, 50)` for aperiodic fitting; later releases resolve the window
dynamically and default to **12–45 Hz** (`spectral.aperiodic.DEFAULT_FREQ_RANGE`;
see [`docs/methods/APERIODIC_FIT_WINDOW.md`](docs/methods/APERIODIC_FIT_WINDOW.md)),
which changes every aperiodic number. Do not "upgrade" it to reproduce work that
cites it.

### R

Statistics and most figures are R. Install once:

```r
install.packages(c(
  "ggplot2", "dplyr", "tidyr", "readr", "forcats", "ggsignif",
  "lme4", "lmerTest", "effectsize", "emmeans",
  "yaml", "argparse", "optparse", "patchwork", "scales"
))
```

(`ggsignif` draws the significance brackets in the PSD/aperiodic/evoked figures;
`argparse` is used by the ROI/electrode report scripts.)

---

## Core concepts

Five ideas explain the whole toolkit.

**1. Levels × Domains.** Every analysis sits at one **level** (the data it reads)
and in one **domain** (what it measures).

- **Level** — **ROI** (atlas parcels: 32 for Allen32, 26 for Allen26, 46 for the
  legacy Antwerp atlas) or **electrode** (raw scalp channels, for validation /
  source-vs-sensor comparison). There is no vertex level: the vertex analyses
  were retired in v0.8.0 (see [Retired: the vertex level](#retired-the-vertex-level)).
- **Domain** — **Spectral**, **Connectivity**, **Cross-frequency**, **Directed**,
  **Sensor-level**, or **Evoked**.

`ANALYSIS_METADATA` in `core.py` is the single source of truth for this map;
`source-lightbox` reads it to group the gallery.

**2. Paradigms.** A study config groups analyses under `paradigms:` keys (e.g.
`resting`, `evoked`). A paradigm names *where the reconstruction data lives* and
*which analyses run on it* — it is **not** the same as level. Two paradigms can
read two different reconstructions of the same recordings (e.g. a shell and a
surface localization), which is how one study compares them. `--paradigm` selects
the block; the analysis must also be listed in that block's `analyses:`.

**3. Primary vs supplementary.** Most analyses are **primary** — they read
reconstructions directly. A few are **supplementary**: they consume another
analysis's output and must run *after* it. The graph-theory modules are the main
case — `roi_graph`/`roi_nbs` need `roi_connectivity`; `electrode_comparison`
needs `electrode_psd`. The
toolkit does **not** auto-run dependencies — run the primary first or you get a
"missing edges CSV" error. The dependency is recorded as `supplements` in
`ANALYSIS_METADATA`.

**4. Two statistics adapters, one declaration.** Inference runs through the shared
**hypothesis layer**. You declare hypotheses once (in `design:`/`hypotheses:`) and
each module tests them with whichever adapter matches how it computes its statistic:

- **emmeans** (R LMM modules — `roi_psd`, `roi_aperiodic`, `roi_cross_freq`,
  `roi_directed`, `electrode_psd`, `electrode_aperiodic`): a **tabular** result —
  per-cell estimate / CI / p, effect size, declarative-scope FDR.
- **permutation** (sensor map modules — `electrode_connectivity`): a **map +
  clusters** result — per-unit statistic map with cluster extent/mass and
  cluster-p (max-stat or TFCE).

The declaration is shared across adapters, so a tabular and a map module can test
the *same* hypothesis. (Until v0.8.0 that was how `vertex_connectivity` and
`electrode_connectivity` gave a source-vs-sensor comparison; the vertex side has
since been retired.)

**5. Declare once, run by name.** Nothing auto-fires. You declare the hypotheses,
then run them one (or a few) at a time with `--hypothesis NAME`. There is no gating
and no "run-everything-and-adjudicate"; the scientific judgment (which post-hoc
follows an omnibus, whether a band/region matters) stays with you. See
[Hypothesis testing](#hypothesis-testing).

---

## Input: the source-localization handoff

source-analytics reads the per-subject files written by `source-localization`.
What it looks for depends on the level:

**ROI-level** (`roi_psd`, `roi_aperiodic`, `roi_connectivity`, `roi_cross_freq`,
`roi_directed`, `roi_evoked`):

| File | Format | Contents |
|------|--------|----------|
| `step6_roi_timeseries_magnitude.pkl` | pickle | `Dict[str, ndarray]` — ROI timeseries, **unsigned** (PSD/aperiodic) |
| `step6_roi_timeseries_signed.pkl` | pickle | `Dict[str, ndarray]` — ROI timeseries, **signed** (connectivity/PAC/directed need phase) |
| `roi_timeseries_magnitude.set` | EEGLAB | same data + metadata (sfreq) |

A run with no `step6_*` files falls back to extracting the parcel series on the
fly from `step5_stc_*.pkl` + `step3_source_coords_mm.npy` using the atlas.

**Run provenance** (every level):

| File | Format | Contents |
|------|--------|----------|
| `config_resolved.yaml` | YAML | The fully resolved source-localization config, written from 0.4.2 on: atlas, head model, source space, inverse method, orientation, sampling mode. Read via `io.run_manifest.read_run_manifest`; absent for older runs, which read as *unknown*, never as fixed |
| `monte_carlo_report.json` | JSON | Monte Carlo runs only. Per parcel: SNR `gain` over a single draw, `coverage`, and `collinear_with` |

**An analysis refuses a cohort that was not localized the same way.** Pooling a
fixed-grid subject with a Monte Carlo one, or two atlases, is a group statistic
over two different measurements, and is otherwise silent — the arrays have the
same shape and the parcel names line up. Subjects with no manifest are skipped
rather than guessed at.

### Monte Carlo runs

source-localization 0.5.0 added `source_space.source_sampling: monte_carlo`,
which averages the ROI operator over many sparse source draws instead of solving
one arbitrary grid. Its parcel series need nothing special here — same
`step6_roi_timeseries_signed.pkl`, same epoch-major layout, and every analysis
asks for `signed=True`. Two things do follow from it:

- **It is ROI-only by construction.** No single grid is solved, so there is no
  `step5_stc_*.pkl` and no source coordinates. Asking for them raises
  `MonteCarloRunError` naming the method, rather than a `FileNotFoundError`
  suggesting a re-run that would produce the same absence.
- **Two per-parcel flags must be read before the tables are.** A parcel
  near-collinear with another has no meaningful *individual* value — the inverse
  splits their shared signal arbitrarily, and the split moves with the draw. A
  parcel with `coverage < 0.5` has its amplitude scaled down, so a low value
  there means "rarely sampled", not "quiet source". Both are logged before the
  analysis produces anything; `SubjectLoader.parcel_caveats()` returns them.


**Electrode-level** (`electrode_psd`, `electrode_aperiodic`,
`electrode_connectivity`, `electrode_comparison`, `electrode_evoked`):

| File | Format | Contents |
|------|--------|----------|
| `*.set` / `*.fdt` | EEGLAB | raw scalp EEG `(channels × timepoints)` |

These need a `subject_roster.csv` (`subject_id, group, eeg_filename, eeg_dir`) set
via `electrode.subject_roster` in the config.

Two discovery layouts are supported. **Grouped** (the folder name is the group):

```
data_dir/
  Group_A/Subject_001/<data_subdir>/…
  Group_A/Subject_002/<data_subdir>/…
  Group_B/Subject_003/<data_subdir>/…
```

**Flat** (BIDS-style `sub-*`, what source-localization writes; groups come from a
`subjects:` map on the paradigm — `init` fills it from `--groups-from`):

```
data_dir/
  sub-001/<data_subdir>/…
  sub-002/<data_subdir>/…
```

`data_subdir` defaults to `pipeline/data`. Only subjects whose group is
referenced by a declared hypothesis/contrast are analysed
(`StudyConfig.referenced_groups()`); a group nobody tests is silently skipped.

---

## Study configuration

One YAML drives the whole study — and the **same file** is read by
`source-lightbox`. Study-design keys (groups, `design:`/`hypotheses:`, bands) are
global; the **per-paradigm `analyses:` block** gives each analysis its data
location and parameters. Minimal shape:

```yaml
name: "My Study"

# ── Study design (global) ──────────────────────────────────────────
groups:                              # raw group id → display label
  WT_VEH: "WT Vehicle"
  KO_VEH: "KO Vehicle"
group_order:  [WT_VEH, KO_VEH]       # plot / x-axis order
group_colors: {WT_VEH: "#3498DB", KO_VEH: "#E74C3C"}

# Declarative hypotheses — the `hypothesis` layer. Tested one at a time by name
# (--hypothesis NAME); nothing auto-fires. See "Hypothesis testing".
design:
  factor: group                      # the categorical factor tests are taken over
  reference: WT_VEH                  # reference level (effect orientation)
  levels: [WT_VEH, KO_VEH]           # explicit level order (optional)
  fdr: { scope: band, method: BH }   # FDR family scope (optional; default scope=hypothesis)
hypotheses:
  - name: group_omnibus              # "do any groups differ?"  (ANOVA / permutation-F)
    kind: omnibus
    role: phenotype
  - name: disease_effect             # name is used in table/file names
    kind: contrast                   # a linear comparison of group means
    label: "Disease effect (KO vs WT)"
    weights: { KO_VEH: 1, WT_VEH: -1 }   # KO − WT (sign from the weights)
    role: phenotype                  # display/grouping tag only — no gating

bands:                               # name → [fmin, fmax] Hz
  Delta: [1, 4]
  Theta: [4, 10]
  Alpha: [10, 13]
  Beta: [13, 30]
  Low Gamma: [30, 55]
  High Gamma: [65, 80]

circos_metrics: [imag_coherence, dwpli, pli, aec, coherence]   # gallery circos chords
                                     # (read by source-lightbox only; source-analytics
                                     #  passes it through untouched)

jobs: -1                             # default worker count for --jobs (-1/0 = all but one core)

# ── Atlas (optional) ───────────────────────────────────────────────
# The parcellation the ROI data were extracted with. Resolved BY NAME to that
# atlas's own files through source-localization's registry.yaml, so atlases that
# share a directory (allen32 / allen26 / allen64 all live in allen/) never borrow
# each other's labels or categories. An unknown name is an error, not a guess.
pipeline:
  atlas: allen26
# An atlas that is not in the registry names its files instead (relative paths
# are taken from atlas_dir, else from the source-localization atlas data):
# atlas_files:
#   brain_labels:   /path/to/labels.nii.gz
#   roi_mapping:    /path/to/roi_mapping.json
#   roi_categories: /path/to/roi_categories.yaml   # optional; the study's own
#                                                  # roi_categories always win

# ── Random epoch sampling (global default; per-analysis override below) ──
# Code defaults when the block is absent: enabled: false, n_bootstrap: 1.
epoch_sampling:
  enabled: true
  epoch_duration_sec: 2.0
  n_epochs: 80
  n_bootstrap: 500                   # 0 = use the full timeseries, no sampling

# ── Output locations (shared with source-lightbox) ─────────────────
paths:
  analytics: ./analytics             # working dir: ANALYSIS_SUMMARY.md + data/
  results:   ./results               # published tables/ + figures/  (gallery reads this)

# ── Paradigms: where the data is + which analyses to run ───────────
paradigms:
  resting:
    data_dir:    ./localization/rest_roi/derivatives   # reconstruction output root
    data_subdir: pipeline/data
    analyses:
      roi_psd: {}
      roi_aperiodic: {}
      roi_connectivity:
        metrics: [imag_coherence, dwpli, pli, aec, coherence]   # subset of the ROI metric set
        epoch_sampling: {n_bootstrap: 0}               # per-analysis override
      roi_graph:        {connectivity_metrics: [imag_coherence, dwpli, pli, aec, coherence]}
      roi_nbs:          {nbs_threshold: 2.5, nbs_permutations: 5000}
      roi_cross_freq: {}
      roi_directed: {}
      electrode_psd: {}
      electrode_comparison: {}                         # needs electrode_psd AND roi_psd
      electrode_connectivity: {}
      roi_signature: {}
```

### What each key feeds

| Key | Consumed by | Purpose |
|---|---|---|
| `groups`, `group_order`, `group_colors` | all analyses | group identity, plot order/colour |
| `design` `{factor, reference, levels, covariates, fdr}` | hypothesis layer | the factor + design tests are taken over; FDR family scope |
| `hypotheses[]` `{name, kind, weights/groups/predictor}` | hypothesis layer | the declarative tests, run by name via `--hypothesis` |
| `hypotheses[]` `{label, role}` | figures, gallery | readable labels + grouping tag (no gating) |
| `bands` | all spectral/connectivity | frequency bands analysed |
| `pipeline.atlas`, `atlas_files`, `atlas_dir` | atlas I/O, R region tier, mosaics | which parcellation the ROI data use: resolved by name through source-localization's `registry.yaml` to that atlas's own labels / mapping / categories / anatomy; `atlas_files` names the files of an unregistered atlas. The 10× voxel convention is read from each NIfTI header, never inferred from its filename |
| `roi_categories` | region tier (Python + R), mosaics | category → ROI map. The study's map (or a profile's narrowing) always wins over the atlas default, on the Python and R sides alike |
| `<analysis>.r_timeout_sec` | R-backed analyses | wall-clock limit for that module's R statistics step; unset = no limit. A step that fails or times out fails the run (exit 1); a paradigm-wide run finishes its other modules first |
| `epoch_sampling` | spectral/connectivity, all levels | random-epoch resampling (`n_bootstrap: 0` = full timeseries). Precedence: global → `<paradigm>.epoch_sampling` → per-analysis block |
| `jobs` | `run --jobs` default | worker count when `--jobs` is not given |
| `<profile>.{include_analyses, include_hypotheses, bands, rois}` | `run --profile` | a narrowed study written to its own tree (see below) |
| `paths.{analytics, results}` | I/O + gallery | working vs published output trees |
| `paradigms.<p>.data_dir` / `data_subdir` | discovery | where subject reconstructions live |
| `paradigms.<p>.analyses.<a>` | that analysis | enables it + sets its parameters |

The per-analysis block is merged into `config.raw[<analysis>]` by
`config.for_paradigm_analysis()`, so any analysis-specific key (`connectivity_metrics`,
`nbs_permutations`, …) **must live under
`paradigms.<paradigm>.analyses.<analysis>`**, not at the top level.

> **Connectivity metrics.** Graph/NBS supplements run on every metric in their
> `connectivity_metrics`. Set `roi_connectivity.metrics` to the same list so the
> primary precomputes all of them in one shared-STFT pass; `roi_graph`/`roi_nbs`
> then load them per metric instead of recomputing. `aec` is computed outside the
> shared STFT and is the slow one — drop it if runtime matters more than
> completeness.

---

## The CLI

Everything runs through one entry point with five subcommands.

| Subcommand | Purpose |
|---|---|
| `run` | run an analysis (the workhorse) |
| `validate` | check config + subject discovery without running |
| `list` | what this install can run. Analyses by default (+ selectable dims; paradigm-aware with `--study`); `--atlases` the registered parcellations with parcel counts and coverage; `--plugins` installed plugins and the analyses that left core; `--all` everything |
| `figure` | regenerate on-demand summary figures from existing tables |
| `init` | scaffold a study config from a reconstruction directory |

### `run`

```bash
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd [options]
```

| Flag | Meaning |
|---|---|
| `--study PATH` | study YAML (required) |
| `--paradigm NAME` | paradigm block under `paradigms:`. Omit it (and `--analysis`) on a multi-paradigm config to run **every** listed analysis of every paradigm; `--analysis` without `--paradigm` is an error |
| `--analysis NAME` | analysis to run (see [catalog](#analysis-catalog--what-exists)). Omit it with `--paradigm` to run everything listed for that paradigm |
| `--steps a,b,…` | lifecycle steps to run. Valid: `setup, process, aggregate, statistics, figures, summary` |
| `--jobs N`, `-j N` | worker processes for the per-subject `process` step. `0`/`-1` = all but one core. Explicit `N` wins over the YAML `jobs:`; omitted = YAML value, else serial. Used by `roi_connectivity`, `electrode_connectivity`; results are identical to serial |
| `--profile NAME` | run under the top-level `<NAME>:` profile block (narrowed bands / ROIs / hypotheses / analyses) and write to a separate tree, `analytics/<NAME>/…` + `results/<NAME>/…`. Narrowing ROIs changes the FDR family, so profile q-values are not comparable to the default run's |
| `--metric m,…` | restrict a module's metrics (shorthand for `--select metric=…`) |
| `--band b,…` | restrict bands, case/format-insensitive (shorthand for `--select band=…`) |
| `--hypothesis n,…` | test only these declared hypotheses (shorthand for `--select hypothesis=…`) |
| `--select DIM=v,…` | generic sub-output selection, repeatable (see `list` for a module's dims) |
| `--force` | remove the analysis's previous output first: its published `tables/` + `figures/` always, and its working dir (`data/` + summary) when the `process` step runs. Also overrides `--strict-output` |
| `--strict-output` | error if the working dir already holds output (unless `--force`) |

**Lifecycle steps.** The full lifecycle is `setup → process → aggregate →
statistics → figures → summary`. **A default `run` executes everything except
`figures`** (`DEFAULT_RUN_STEPS` in `analyses/base.py`), so tables and the summary
appear but no images do. Render figures with an explicit step list, or with
`source-analytics figure` for the on-demand summary figures:

```bash
# full run including figures
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd \
    --steps setup,process,aggregate,statistics,figures,summary
# recompute only statistics + figures + report from persisted data/ (no reprocessing)
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd \
    --steps statistics,figures,summary
```

`--steps` re-runs a subset against the on-disk `data/` of an earlier run; the
`figures` step clears the module's figure dir before regenerating so stale images
never linger. Deprecated analysis names (`psd`, `pac`, …) still
resolve, and their output always lands under the **canonical** name (`roi_psd/`,
`roi_cross_freq/`, `roi_signature/`).

### `validate`, `list`, `figure`, `init`

```bash
source-analytics validate --study study.yaml [--paradigm resting]
source-analytics list [--study study.yaml]          # paradigm-aware when --study given
source-analytics list --atlases                     # parcellations, from source-localization's registry
source-analytics list --plugins                     # plugins, and what left core in v0.8.0
source-analytics list --all                         # analyses + atlases + plugins
source-analytics figure --study study.yaml --paradigm resting --analysis roi_psd --list
source-analytics figure --study study.yaml --paradigm resting --analysis roi_psd \
    --type effect_heatmap [--contrast disease_effect --band low_gamma]
source-analytics init /path/to/reconstruction_dir --name study --groups-from sl_config.yaml \
    [--paradigm resting] [--analyses roi_psd,roi_aperiodic] [--output PATH | -]
```

`init` writes `<reconstruction_dir>/analysis/<name>.yaml` (or stdout with
`--output -`) and parses it back to prove the scaffold loads. `list` groups the
catalog by paradigm category and level, tagging each module's `--select` dims.

---

## Analysis catalog — what exists

Grouped by **domain** (what they measure). Levels: ROI / electrode (elec).
*Supplementary* analyses are indented under their primary and
must run after it. Method provenance for the connectivity / cross-frequency /
directed families is tracked, equation-checked, in
[`docs/methods/CONNECTIVITY_METHODS.md`](docs/methods/CONNECTIVITY_METHODS.md).

### Spectral

| Analysis | Level | Computes | Reference |
|---|---|---|---|
| `roi_psd`, `electrode_psd` | ROI, elec | band power (Welch PSD). CSV columns: `absolute` = mean power density in dB/Hz, `relative` = fraction of total; optional `delta_ref` under a profile. There is no separate `dB` column | Welch 1967 |
| `roi_aperiodic`, `electrode_aperiodic` | ROI, elec | 1/f aperiodic (offset, exponent) + oscillatory peaks; default fit window **12–45 Hz** | Donoghue 2020 (specparam) |
| `roi_signature` | ROI | per-parcel neural signature: multi-model decoding on band power, permutation p | — |

### Connectivity (same-frequency functional connectivity)

| Analysis | Level | Computes | Reference |
|---|---|---|---|
| `roi_connectivity` | ROI | FC-six + more: coherence, imaginary coherence, PLI, wPLI, dwPLI, dPLI, AEC, partial correlation | Nolte 2004; Stam 2007; Vinck 2011; Stam & van Straaten 2012; Hipp 2012; Marrelec 2006 |
| ↳ `roi_graph` *(suppl.)* | ROI | graph-theoretic nodal metrics (degree/clustering/betweenness) | Rubinov & Sporns 2010 |
| ↳ `roi_nbs` *(suppl.)* | ROI | Network-Based Statistic (sub-network test) | Zalesky 2010 |
| `electrode_connectivity` | elec | FC-six all-pairs + per-channel FCD — the **source-vs-sensor comparator** | as above |

> `roi_network` is a **combined alias** that runs graph + NBS together and writes
> a Python summary (there is no R report for it); the split modules (`roi_graph`,
> `roi_nbs`) are preferred for the gallery. `dpli` is directed and is
> auto-excluded from the undirected graph/NBS layer. `roi_connectivity` takes an
> optional `metrics:` list in its config block; the R report adapts to whichever
> metric columns the edge CSV carries, so `--metric aec` alone is fine.

### Cross-frequency

| Analysis | Level | Computes | Reference |
|---|---|---|---|
| `roi_cross_freq` | ROI | PAC (Modulation Index, surrogate-z); cross-frequency AAC; n:m PPC. ROI: PAC hypotheses via `roi_pac_analysis.R`; AAC/PPC via `roi_cross_freq_edges_analysis.R`, three tiers each: `roi_cross_freq_{aac,ppc}_{global,directed_edges,region}_hypotheses.csv` (PPC has DVs `ppc` + `ppc_z`) | Tort 2010; Bruns 2000 / Masimore 2004; Tass 1998 / Palva 2005 |

### Directed

| Analysis | Level | Computes | Reference |
|---|---|---|---|
| `roi_directed` | ROI | transfer entropy (`te`, `net_te`); DTF (`dtf`, ridge-MVAR). `--metric te,dtf`; hypothesis tables `roi_directed_{global,directed_edges,region}_hypotheses.csv` carry a `dv` column covering every exported DV (`te`, `net_te`, `dtf`) | Schreiber 2000; Kamiński & Blinowska 1991 |

> Source ROIs/vertices are strongly collinear (mean inter-node |corr| ≈ 0.64), so
> DTF uses a **ridge-regularized** MVAR — plain LS-MVAR is non-stationary; the
> module warns if a fit is unstable.

### Source vs sensor (validation)

| Analysis | Level | Reads | Computes |
|---|---|---|---|
| `electrode_comparison` *(suppl.)* | elec | `electrode_psd` **and** `roi_psd` (same paradigm) | source-vs-electrode band-power concordance + effect-size validation |
| `electrode_signature` *(suppl. of `electrode_psd`)* | elec | `electrode_psd` | sensor-level neural signature (decoding on electrode band power) — the sensor counterpart of `roi_signature`, compared when one ran in the same paradigm |
| `roi_signature` | roi | — | ROI-level neural signature (decoding on per-parcel band power) — the source-side counterpart of `electrode_signature`; runs on any ROI output, including Monte Carlo operators. Set `sensor_paradigm:` to compare against an `electrode_signature` run in another paradigm |

`ANALYSIS_METADATA` records these as `supplements` (the primary the gallery nests
them under) plus `requires` (every upstream module, for run ordering).

### Evoked (trial-based paradigms only)

| Analysis | Level | Computes |
|---|---|---|
| `roi_evoked`, `electrode_evoked` | ROI, elec | ITC (raw + debiased), ERSP, single-trial power, induced power, ERP amplitude/latency. Descriptive `group × unit` LMM in R plus declared hypotheses (`_evoked_hypotheses.py`, measure as facet) |

### Retired: the vertex level

The 12 vertex analyses — `vertex_cluster`, `vertex_connectivity`,
`vertex_cross_freq`, `vertex_directed`, `vertex_evoked`, `vertex_graph`,
`vertex_nbs`, `vertex_network`, `vertex_signature`, `vertex_spatial`,
`vertex_specparam`, and `fcd_comparison` (which read `vertex_connectivity`'s
output) — **left this package in v0.8.0**, along with `spectral.vertex`,
`spectral.vertex_aperiodic`, the five `R/vertex_*.R` scripts and the glass-brain
summary figures.

A vertex map describes one arbitrary placement of the source grid. The ROI
analyses over Monte Carlo source operators integrate over placement instead, and
match what a 30-channel dorsal array can actually resolve, so they replaced them.

- A config or `--analysis` naming one now fails with a message naming the plugin.
- `source-analytics list --plugins` prints them, and what provides them.
- They live in the unmaintained `source-analytics-vertex` plugin. Installing it
  registers them again through the `source_analytics.plugins` entry point.
- **v0.7.1 reproduces published vertex results** and stays available.

**Renames (2026-06).** `roi_pac` → `roi_cross_freq` (now also AAC + PPC);
`roi_transfer_entropy` → `roi_directed`. Old names still work as deprecated aliases
(`psd`/`aperiodic`/`pac`/`roi_pac`/`mvpa`/`vertex_mvpa`/`wholebrain`/`spatial_lmm`/
`specparam_vertex`/`transfer_entropy`/`roi_transfer_entropy`/`evoked`/`electrode`
map to the canonical names; output always lands under the canonical directory).
Two R scripts keep their legacy filenames on purpose: `roi_pac_analysis.R` (for
`roi_cross_freq`) and `roi_transfer_entropy_analysis.R` (for `roi_directed`).

---

## Hypothesis testing

The `hypothesis` layer is a shared inference engine (peer to `R/stats_utils.R` and
`src/source_analytics/stats/`, **not** a registry module) that turns the declarative
`design:`/`hypotheses:` blocks into tests. Full reference:
[`docs/methods/HYPOTHESIS.md`](docs/methods/HYPOTHESIS.md); design rationale: [`docs/methods/DESIGN_SPEC.md`](docs/methods/DESIGN_SPEC.md).

**Four kinds.** A hypothesis carries a `kind` and the payload it needs:

| kind | question | payload | effect size |
|---|---|---|---|
| `omnibus` | do these groups differ at all? | `groups` (default: all levels) | partial ω² |
| `contrast` | a specific linear comparison (post-hoc) | `weights: {level: w}` | Hedges g |
| `regression` | slope of a continuous predictor | `predictor` (+ optional `by`) | standardized β |
| `equivalence` | is a contrast within a margin? (TOST) | `weights` + `margin` | — |

The legacy `group_a`/`group_b` form is still accepted as sugar for a pairwise
`weights` map.

**FDR family scope (declarative).** Multiple-comparison correction happens *within a
single hypothesis run's cells* (band × spatial). The family is declarative via an
`fdr:` block — study-level under `design:` and/or per-hypothesis:

```yaml
design:
  fdr: { scope: hypothesis, method: BH }   # default = the whole band×spatial grid
hypotheses:
  - name: disease_effect
    fdr: { scope: band }                   # per-band/freq-pair family (override)
```

`scope` is `hypothesis` (one family over the whole grid — most conservative,
default), `band` (a family per band/freq-pair — the principled choice when bands
are pre-specified independent hypotheses, e.g. PAC), `spatial`, or `none`. `method`
is `BH` (default), `BY`, `holm`, `bonferroni`, or `none`. The permutation/map
adapter uses cluster-extent correction, so `fdr:` is a no-op there. Aggressiveness
is driven by family **size**, not just the method — declaring the family in the
spec keeps it pre-registered.

**Running.** `--hypothesis NAME[,NAME]` runs one (or a few) by name; with no flag a
module runs all declared hypotheses. It composes with `--metric` / `--band` /
`--select`:

```bash
source-analytics run --study study.yaml --paradigm resting \
    --analysis roi_psd --hypothesis disease_effect
```

Output is **additive**: a `<module>_hypotheses.csv` written alongside the module's
other tables, with one tidy row per band × spatial cell (estimate, SE, CI, stat, p,
q, effect size, `fdr_family`). Modules with multiple spatial tiers emit one table
per tier — `roi_directed` writes `roi_directed_global_hypotheses.csv`,
`roi_directed_directed_edges_hypotheses.csv`, and `roi_directed_region_hypotheses.csv`.

Two limits worth knowing: `kind: regression` is accepted by the config and the
emmeans (R) adapter, but the Python permutation (map) and edge (NBS) adapters
return **no rows** for it yet — a continuous predictor is not wired into those
paths. And the R evoked scripts and every emmeans module derive their contrast
list from `design:`/`hypotheses:`; a legacy `contrasts:` block is lifted into the
same spec, so both config styles work.

---

## Selecting metrics, bands & hypotheses

Multi-output analyses honour a sub-output filter, so you compute exactly what you
want **without losing the shared STFT/Hilbert compute pass**:

```bash
# two connectivity metrics only
source-analytics run … --analysis roi_connectivity --metric dwpli,wpli
# one band, one cross-frequency measure
source-analytics run … --analysis roi_cross_freq --metric ppc --band low_gamma
# one declared hypothesis
source-analytics run … --analysis roi_psd --hypothesis disease_effect
# generic form (repeatable)
source-analytics run … --select metric=pli --select band=beta,low_gamma
```

Selectable dimensions vary by module (`metric` / `band` / `hypothesis` /
`measure`). `source-analytics list` tags each analysis with its dimensions, e.g.
`roi_psd [--select: band, hypothesis]` — that listing is the source of truth.

---

## Output structure

Two trees, both segmented by paradigm (and by profile when `--profile` is used).
The **working tree** under `paths.analytics` holds the per-subject data and the
narrative; the **published tree** under `paths.results` holds the tables and
figures that `source-lightbox` reads. `tables/` and `figures/` do **not** live
inside the analysis working directory.

```
<analytics>/[<profile>/]<paradigm>/<analysis>/
  ANALYSIS_SUMMARY.md          # methods + results narrative (markdown)
  data/
    <analysis>_*.csv           # the computed per-subject measures (the inputs to stats)
    study_config.yaml          # the resolved config snapshot used for this run

<results>/[<profile>/]tables/<paradigm>/<analysis>/
    <analysis>_hypotheses.csv  # the hypothesis-layer result (one row per band×cell)
    <analysis>_subnetwork_edges.csv  # roi_nbs only: per-edge membership of each NBS component
    provenance.json            # what produced these tables (see below)
    …                          # any module-specific diagnostic tables
<results>/[<profile>/]figures/<paradigm>/<analysis>/
    *.png                      # ggplot2 / glass-brain / matplotlib figures (figures step only)
```

`config.for_paradigm_analysis()` sets the working dir to `analytics/<paradigm>`;
`BaseAnalysis.tbl_dir` / `fig_dir` resolve the published dirs. A legacy
single-paradigm config (no `paradigms:`) omits the paradigm segment.

The `<analysis>_hypotheses.csv` is the canonical statistical contract across all
emmeans/permutation modules; figure and gallery consumers read it (plus legacy
column aliases during the migration).

> Figures are **render-on-demand** — they are not auto-regenerated when data
> changes. Re-run the `figures` step (or `source-analytics figure …`) before
> rebuilding a manuscript/gallery from updated tables.

### `provenance.json` — what produced these tables

A stats CSV cannot say what made it, and the numbers depend on more than the
config: on the source-analytics version that computed them, on the plugin that
provided the analysis if it was not built in, and above all on **how the
recordings were localized**. A different atlas, inverse method or source-sampling
mode is a different measurement, not a different view of the same one.

Every run writes one next to its tables, carrying forward what source-localization
recorded in each subject's `config_resolved.yaml`:

```json
{
  "schema": 1,
  "written": "2026-09-18T13:56:19-04:00",
  "analysis": "roi_psd",
  "paradigm": "resting",
  "source_analytics": {"version": "v0.8.2", "git_describe": "v0.8.2"},
  "steps": ["aggregate", "process", "setup", "statistics", "summary"],
  "subjects": {"n": 3, "groups": {"KO": 1, "WT": 2}, "ids": ["sub-901", "…"]},
  "localization": {
    "n_with_manifest": 3, "n_unrecorded": 0,
    "description": "ellipsoid_surface_anatomical, allen26, ellipsoid/surface/anatomical, sLORETA (fixed), Monte Carlo (K=100, 160 sources/draw)",
    "version": "0.5.1", "atlas": "allen26", "source_sampling": "monte_carlo",
    "inverse_method": "sLORETA", "orientation": "fixed",
    "monte_carlo": {"n_draws": 100, "n_sources": 160, "seed": 20260821}
  },
  "parcel_caveats": {
    "Thalamus": "sampled in 31% of draws; amplitude is scaled down accordingly; …"
  }
}
```

Three things worth noting:

- **`parcel_caveats` outlives the log.** A Monte Carlo run flags parcels it
  rarely sampled and parcels it cannot separate from a neighbour. Those warnings
  are printed when the analysis runs, but a log is not what someone reading the
  table six months later has.
- **`n_unrecorded` counts subjects localized before source-localization 0.4.2**,
  which left no manifest. They are counted, never guessed at — a record that
  quietly claimed settings for them would be worse than one that says how many
  are unaccounted for.
- **Writing it cannot fail a run.** It is a record *about* a run, written last,
  and both assembling and writing it swallow errors. Nothing downstream requires
  the file to exist; `provenance.read_provenance(tbl_dir)` returns `None` when it
  is absent or corrupt.

---

## Running a full study, in order

A study run is just the analyses invoked in dependency order (primaries before
their supplements). [`scripts/run_study.sh study.yaml [run flags]`](scripts/run_study.sh)
is the canonical recipe; the essential order:

```bash
SA="source-analytics run --study study.yaml --paradigm"

# Resting paradigm — ROI + electrode
$SA resting --analysis roi_psd
$SA resting --analysis roi_aperiodic
$SA resting --analysis roi_connectivity        # PRIMARY
$SA resting --analysis roi_graph               # ↳ after roi_connectivity
$SA resting --analysis roi_nbs                 # ↳ after roi_connectivity
$SA resting --analysis roi_cross_freq          # PAC + AAC + PPC  (--metric to pick one)
$SA resting --analysis roi_directed            # transfer entropy + DTF  (--metric te|dtf)
$SA resting --analysis electrode_psd           # PRIMARY
$SA resting --analysis electrode_aperiodic
$SA resting --analysis electrode_comparison    # ↳ after electrode_psd AND roi_psd
$SA resting --analysis electrode_connectivity  # sensor FC comparator
$SA resting --analysis electrode_signature     # ↳ after electrode_psd
$SA resting --analysis roi_signature           # source side of the decoding comparison

# Evoked paradigm (trial-based data only)
$SA evoked  --analysis roi_evoked
$SA evoked  --analysis electrode_evoked
```

---

## Architecture

```
Python                                         R
──────────────────────────────────────        ──────────────────────────────
1. Load YAML config, discover subjects
2. Load reconstructions (pickle/.set/.npy)
3. Signal processing (scipy/mne/sklearn)
4. Export per-subject CSVs ───────────────►    5. Read CSVs + config
   (sensor maps: also do cluster-perm          6. LMMs (lme4/lmerTest), emmeans
    stats in Python)                           7. Hypothesis layer: effect sizes, FDR
                                               8. ggplot2 figures
                                               9. Markdown ANALYSIS_SUMMARY.md
```

Python calls `Rscript` automatically. ROI/electrode LMM modules delegate stats +
figures to R; sensor map modules do statistics in Python and use R only for the
report.

---

## Extending: add an analysis

1. Create `src/source_analytics/analyses/my_analysis.py` subclassing `BaseAnalysis`.
2. Implement the lifecycle hooks: `setup → process_subject → aggregate →
   statistics → figures → summary` (any subset; the base no-ops the rest).
3. If it does LMM stats, add `R/my_analysis.R`, `source()` `R/hypothesis.R`, and
   emit `<module>_hypotheses.csv` via `write_module_hypotheses()` (or
   `write_module_directed_edges()` for asymmetric directed edges). If it does map
   stats, use the Python permutation adapter (`write_module_hypotheses_perm`).
4. Register the class in `ANALYSIS_REGISTRY` and add an `ANALYSIS_METADATA` entry
   (`level`, `domain`, optional `supplements`) in `core.py`.
5. Declare a `SELECTABLE` dict on the class for any `--metric`/`--band`/`hypothesis`
   sub-output dimensions.
6. Add it to the catalog above.

---

## Reference documents

| Doc | What it covers |
|---|---|
| [`docs/`](docs/index.md) | **methods documentation site** (MkDocs) — start here |
| [`docs/methods/APERIODIC_FIT_WINDOW.md`](docs/methods/APERIODIC_FIT_WINDOW.md) | why the 1/f fit range is 12–45 Hz; overriding it; `offset_centered` |
| [`docs/methods/HYPOTHESIS.md`](docs/methods/HYPOTHESIS.md) | the hypothesis layer — kinds, adapters, usage |
| [`docs/methods/DESIGN_SPEC.md`](docs/methods/DESIGN_SPEC.md) | design rationale for `design:`/`hypotheses:` + FDR scope |
| [`docs/methods/CONNECTIVITY_METHODS.md`](docs/methods/CONNECTIVITY_METHODS.md) | equation-checked provenance for every connectivity / coupling / directed metric |
| [`CHANGELOG.md`](CHANGELOG.md) | version history |
| [`docs/archive/`](docs/archive/README.md) | completed/superseded planning docs — **not current** |

## License

MIT
