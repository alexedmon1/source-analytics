# source-analytics

Group-level statistical analysis toolkit for **source-localized EEG**. It is the
middle stage of a three-package pipeline:

```
source-localization  ──►  source-analytics  ──►  source-lightbox
(reconstruct sources)     (stats + figures)       (render the gallery)
```

[`source-localization`](../source-localization) turns raw EEG into per-subject
source reconstructions (ROI timeseries, whole-brain source estimates).
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
# 1. Scaffold a study config from the reconstruction directory.
#    (--groups-from reuses the group mapping from the source-localization config.)
source-analytics init /path/to/localization/rest_roi \
    --name "My Study" \
    --groups-from /path/to/localization/study_config.yaml  > study.yaml

# 2. Edit study.yaml — declare groups, bands, the design/hypotheses, and which
#    analyses to run under each paradigm (see "Study configuration").

# 3. Sanity-check config + subject discovery before any long run.
source-analytics validate --study study.yaml

# 4. Run an analysis. --paradigm picks the block under `paradigms:` in the config.
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd
```

Each analysis writes a self-contained output directory with a `data/`, `tables/`,
`figures/`, and an `ANALYSIS_SUMMARY.md` (see [Output structure](#output-structure)).
`source-analytics list` shows every analysis you can run.

---

## Installation

### Python (3.10+)

```bash
pip install -e .          # or:  uv pip install -e .
```

Core dependencies: numpy, scipy, pandas, pyyaml, scikit-learn, networkx,
matplotlib, mne.

> **uv users:** run the CLI with `uv run --no-sync source-analytics …`. Plain
> `uv run` can trip on the lockfile; `--no-sync` avoids the re-resolve.

### R

Statistics and most figures are R. Install once:

```r
install.packages(c(
  "ggplot2", "dplyr", "tidyr", "readr", "stringr", "forcats",
  "lme4", "lmerTest", "effectsize", "emmeans",
  "yaml", "argparse", "patchwork", "scales"
))
```

---

## Core concepts

Five ideas explain the whole toolkit.

**1. Levels × Domains.** Every analysis sits at one **level** (the data it reads)
and in one **domain** (what it measures).

- **Level** — **ROI** (atlas ROIs: 32 for Allen32, 46 for the legacy Antwerp atlas),
  **vertex** (whole-brain source vertices), or **electrode** (raw scalp channels,
  for validation / source-vs-sensor comparison).
- **Domain** — **Spectral**, **Connectivity**, **Cross-frequency**, **Directed**,
  **Sensor-level**, or **Evoked**.

`ANALYSIS_METADATA` in `core.py` is the single source of truth for this map;
`source-lightbox` reads it to group the gallery.

**2. Paradigms.** A study config groups analyses under `paradigms:` keys (e.g.
`resting`, `vertex`, `evoked`). A paradigm names *where the reconstruction data
lives* and *which analyses run on it* — it is **not** the same as level. In the
canonical FORGE setup the `resting` paradigm holds the ROI + electrode analyses
(they read `rest_roi/derivatives`) and the `vertex` paradigm holds the vertex
analyses (they read `rest_shell/derivatives`, a different reconstruction).
`--paradigm` selects the block; the analysis must also be listed in that block's
`analyses:`.

**3. Primary vs supplementary.** Most analyses are **primary** — they read
reconstructions directly. A few are **supplementary**: they consume another
analysis's output and must run *after* it. The graph-theory modules are the main
case — `roi_graph`/`roi_nbs` need `roi_connectivity`; `vertex_graph`/`vertex_nbs`
need `vertex_connectivity`; `electrode_comparison` needs `electrode_psd`. The
toolkit does **not** auto-run dependencies — run the primary first or you get a
"missing edges CSV" error. The dependency is recorded as `supplements` in
`ANALYSIS_METADATA`.

**4. Two statistics adapters, one declaration.** Inference runs through the shared
**hypothesis layer**. You declare hypotheses once (in `design:`/`hypotheses:`) and
each module tests them with whichever adapter matches how it computes its statistic:

- **emmeans** (R LMM modules — `roi_psd`, `roi_aperiodic`, `roi_cross_freq`,
  `roi_directed`, `electrode_psd`, `electrode_aperiodic`): a **tabular** result —
  per-cell estimate / CI / p, effect size, declarative-scope FDR.
- **permutation** (vertex & sensor map modules — `vertex_cluster`,
  `vertex_connectivity`, `vertex_directed`, `vertex_specparam`,
  `electrode_connectivity`): a **map + clusters** result — per-unit statistic map
  with cluster extent/mass and cluster-p (max-stat or TFCE).

Because the *declaration* is shared, `vertex_connectivity` (source FCD) and
`electrode_connectivity` (sensor FCD) test the **same** hypothesis with the **same**
cluster adapter — a clean source-vs-sensor comparison.

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

**Vertex-level** (`vertex_cluster`, `vertex_connectivity`, `vertex_cross_freq`,
`vertex_directed`, `vertex_specparam`, `vertex_mvpa`, `vertex_evoked`):

| File | Format | Contents |
|------|--------|----------|
| `step5_stc_signed.pkl` | pickle | MNE `SourceEstimate` `(n_vertices, n_times)`, signed (falls back to `step5_stc_magnitude.pkl`, legacy `step5_stc.pkl`) |
| `step3_source_coords_mm.npy` | NumPy | source coordinates `(n_vertices, 3)` in mm |

**Electrode-level** (`electrode_psd`, `electrode_aperiodic`,
`electrode_connectivity`, `electrode_comparison`, `electrode_evoked`):

| File | Format | Contents |
|------|--------|----------|
| `*.set` / `*.fdt` | EEGLAB | raw scalp EEG `(channels × timepoints)` |

These need a `subject_roster.csv` (`subject_id, group, eeg_filename, eeg_dir`) set
via `electrode.subject_roster` in the config.

Expected discovery layout (group folders → subject folders → `data/`):

```
data_dir/
  Group_A/Subject_001/data/…
  Group_A/Subject_002/data/…
  Group_B/Subject_003/data/…
```

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

# ── Random epoch sampling (global default; per-analysis override below) ──
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
        epoch_sampling: {n_bootstrap: 0}               # per-analysis override
      roi_graph:        {connectivity_metrics: [imag_coherence, dwpli, pli, aec, coherence]}
      roi_nbs:          {nbs_threshold: 2.5, nbs_permutations: 5000}
      roi_cross_freq: {}
      roi_directed: {}
      electrode_psd: {}
      electrode_comparison: {}

  vertex:
    data_dir:    ./localization/rest_shell/derivatives
    data_subdir: pipeline/data
    analyses:
      vertex_connectivity:                             # PRIMARY for the vertex graph theory
        vertex_filter: {z_min: 0.0}
        metrics: [imag_coherence, dwpli, pli, aec, coherence]   # all share one STFT pass
      vertex_graph:     {connectivity_metrics: [imag_coherence, dwpli, pli, aec, coherence]}
      vertex_nbs:       {nbs_threshold: 3.0, nbs_permutations: 5000}
      vertex_cluster: {}
      vertex_specparam: {}
```

### What each key feeds

| Key | Consumed by | Purpose |
|---|---|---|
| `groups`, `group_order`, `group_colors` | all analyses | group identity, plot order/colour |
| `design` `{factor, reference, levels, covariates, fdr}` | hypothesis layer | the factor + design tests are taken over; FDR family scope |
| `hypotheses[]` `{name, kind, weights/groups/predictor}` | hypothesis layer | the declarative tests, run by name via `--hypothesis` |
| `hypotheses[]` `{label, role}` | figures, gallery | readable labels + grouping tag (no gating) |
| `bands` | all spectral/connectivity | frequency bands analysed |
| `epoch_sampling` | spectral/connectivity | random-epoch resampling (`n_bootstrap: 0` = full timeseries) |
| `paths.{analytics, results}` | I/O + gallery | working vs published output trees |
| `paradigms.<p>.data_dir` / `data_subdir` | discovery | where subject reconstructions live |
| `paradigms.<p>.analyses.<a>` | that analysis | enables it + sets its parameters |

The per-analysis block is merged into `config.raw[<analysis>]` by
`config.for_paradigm_analysis()`, so any analysis-specific key (`connectivity_metrics`,
`nbs_permutations`, `vertex_filter`, …) **must live under
`paradigms.<paradigm>.analyses.<analysis>`**, not at the top level.

> **Connectivity metrics.** Graph/NBS supplements run on every metric in their
> `connectivity_metrics`. At the vertex level, set `vertex_connectivity.metrics` to
> the same list so the primary precomputes all of them in one shared-STFT pass;
> `vertex_graph`/`vertex_nbs` then load them per metric instead of recomputing.
> `aec` is computed outside the shared STFT and is the slow one — drop it if
> runtime matters more than completeness.

---

## The CLI

Everything runs through one entry point with five subcommands.

| Subcommand | Purpose |
|---|---|
| `run` | run an analysis (the workhorse) |
| `validate` | check config + subject discovery without running |
| `list` | list available analyses (+ selectable dims; paradigm-aware with `--study`) |
| `figure` | regenerate on-demand summary figures from existing tables |
| `init` | scaffold a study config from a reconstruction directory |

### `run`

```bash
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd [options]
```

| Flag | Meaning |
|---|---|
| `--study PATH` | study YAML (required) |
| `--paradigm NAME` | paradigm block under `paradigms:` (required for multi-paradigm configs) |
| `--analysis NAME` | analysis to run (see [catalog](#analysis-catalog--what-exists)) |
| `--steps a,b,…` | lifecycle steps to run. Valid: `setup, process, aggregate, statistics, figures, summary` |
| `--metric m,…` | restrict a module's metrics (shorthand for `--select metric=…`) |
| `--band b,…` | restrict bands, case/format-insensitive (shorthand for `--select band=…`) |
| `--hypothesis n,…` | test only these declared hypotheses (shorthand for `--select hypothesis=…`) |
| `--select DIM=v,…` | generic sub-output selection, repeatable (see `list` for a module's dims) |
| `--force` | overwrite the output directory if it exists |
| `--strict-output` | error if the output directory exists (unless `--force`) |

**Lifecycle steps.** A run is `setup → process → aggregate → statistics → figures →
summary`. `--steps` re-runs a subset against existing on-disk data — e.g. recompute
only the statistics and report after a config change, without reprocessing subjects:

```bash
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd \
    --steps statistics,summary,figures
```

### `validate`, `list`, `figure`, `init`

```bash
source-analytics validate --study study.yaml [--paradigm resting]
source-analytics list [--study study.yaml]          # paradigm-aware when --study given
source-analytics figure --study study.yaml --paradigm resting --analysis roi_psd --list
source-analytics figure --study study.yaml --paradigm resting --analysis roi_psd \
    --type effect_heatmap [--contrast disease_effect --band low_gamma]
source-analytics init /path/to/reconstruction_dir --name "Study" --groups-from sl_config.yaml
```

---

## Analysis catalog — what exists

Grouped by **domain** (what they measure). Levels: ROI / vertex (vtx) /
electrode (elec). *Supplementary* analyses are indented under their primary and
must run after it. Method provenance for the connectivity / cross-frequency /
directed families is tracked, equation-checked, in
[`CONNECTIVITY_METHODS.md`](CONNECTIVITY_METHODS.md).

### Spectral

| Analysis | Level | Computes | Reference |
|---|---|---|---|
| `roi_psd`, `electrode_psd` | ROI, elec | band power (Welch PSD: absolute/relative/dB) | Welch 1967 |
| `roi_aperiodic`, `electrode_aperiodic`, `vertex_specparam` | ROI, elec, vtx | 1/f aperiodic (offset, exponent) + oscillatory peaks | Donoghue 2020 (specparam) |
| `vertex_cluster` | vtx | per-vertex band power / fALFF / slope / peak-α, cluster-corrected maps | Maris & Oostenveld 2007 |
| `vertex_mvpa` | vtx | whole-brain pattern decoding (linear SVM, LOOCV, permutation) | — (linear SVM) |
| `vertex_spatial` *(RETIRED)* | vtx | was: spatial-covariance GLS robustness check | — |

> `vertex_spatial` is **retired** (it produced a spatial-covariance robustness
> table, never a manuscript result, and did not survive the design-spec migration).
> Spatially-resolved vertex inference is `vertex_cluster` (glass-brain clusters) +
> `vertex_nbs` (network-based statistic). The module exits cleanly with empty
> frames + a note.

### Connectivity (same-frequency functional connectivity)

| Analysis | Level | Computes | Reference |
|---|---|---|---|
| `roi_connectivity`, `vertex_connectivity` | ROI, vtx | FC-six + more: coherence, imaginary coherence, PLI, wPLI, dwPLI, dPLI, AEC, partial correlation | Nolte 2004; Stam 2007; Vinck 2011; Stam & van Straaten 2012; Hipp 2012; Marrelec 2006 |
| ↳ `roi_graph`, `vertex_graph` *(suppl.)* | ROI, vtx | graph-theoretic nodal metrics (degree/clustering/betweenness; vtx: multi-density AUC) | Rubinov & Sporns 2010 |
| ↳ `roi_nbs`, `vertex_nbs` *(suppl.)* | ROI, vtx | Network-Based Statistic (sub-network test) | Zalesky 2010 |
| `electrode_connectivity` | elec | FC-six all-pairs + per-channel FCD — the **source-vs-sensor comparator** | as above |

> `roi_network` / `vertex_network` are **combined aliases** that run graph + NBS
> together; the split modules (`*_graph`, `*_nbs`) are preferred for the gallery.
> `dpli` is directed and is auto-excluded from the undirected graph/NBS layer.

### Cross-frequency

| Analysis | Level | Computes | Reference |
|---|---|---|---|
| `roi_cross_freq`, `vertex_cross_freq` | ROI, vtx | PAC (Modulation Index, surrogate-z); cross-frequency AAC; n:m PPC | Tort 2010; Bruns 2000 / Masimore 2004; Tass 1998 / Palva 2005 |

### Directed

| Analysis | Level | Computes | Reference |
|---|---|---|---|
| `roi_directed` | ROI | transfer entropy (`te`, `net_te`); DTF (`dtf`, ridge-MVAR) | Schreiber 2000; Kamiński & Blinowska 1991 |
| `vertex_directed` | vtx | DTF outflow / inflow / netflow (ridge-MVAR), cluster-corrected | Kamiński & Blinowska 1991 |

> Source ROIs/vertices are strongly collinear (mean inter-node |corr| ≈ 0.64), so
> DTF uses a **ridge-regularized** MVAR — plain LS-MVAR is non-stationary; the
> module warns if a fit is unstable.

### Sensor-level (validation)

| Analysis | Level | Computes |
|---|---|---|
| `electrode_comparison` *(suppl. of `electrode_psd`)* | elec | source-vs-electrode effect-size validation |

### Evoked (trial-based paradigms only)

| Analysis | Level | Computes |
|---|---|---|
| `roi_evoked`, `vertex_evoked`, `electrode_evoked` | ROI, vtx, elec | ITC, ERSP, single-trial power |

**Renames (2026-06).** `roi_pac` → `roi_cross_freq` (now also AAC + PPC);
`roi_transfer_entropy` → `roi_directed`. Old names still work as deprecated aliases
(`psd`/`aperiodic`/`pac`/`mvpa`/`wholebrain`/… also map to the canonical names).

---

## Hypothesis testing

The `hypothesis` layer is a shared inference engine (peer to `R/stats_utils.R` and
`src/source_analytics/stats/`, **not** a registry module) that turns the declarative
`design:`/`hypotheses:` blocks into tests. Full reference:
[`HYPOTHESIS.md`](HYPOTHESIS.md); design rationale: [`DESIGN_SPEC.md`](DESIGN_SPEC.md).

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
q, effect size, `fdr_family`, plus legacy-named aliases for existing figure
consumers). Modules with multiple spatial tiers emit one table per tier — e.g.
`roi_directed` writes `…_global_hypotheses.csv`, `…_directed_edges_hypotheses.csv`,
and `…_region_hypotheses.csv`.

---

## Selecting metrics, bands & hypotheses

Multi-output analyses honour a sub-output filter, so you compute exactly what you
want **without losing the shared STFT/Hilbert compute pass**:

```bash
# two connectivity metrics only
source-analytics run … --analysis vertex_connectivity --metric dwpli,wpli
# one band, one cross-frequency measure
source-analytics run … --analysis vertex_cross_freq --metric ppc --band low_gamma
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

Each analysis writes a self-contained directory under `paths.analytics`
(working tree). The published `tables/` + `figures/` are mirrored to
`paths.results`, which `source-lightbox` reads.

```
<analytics>/<analysis>/
  ANALYSIS_SUMMARY.md          # methods + results narrative (markdown)
  data/
    <analysis>_*.csv           # the computed per-subject measures (the inputs to stats)
    study_config.yaml          # the resolved config snapshot used for this run
  tables/
    <analysis>_hypotheses.csv  # the hypothesis-layer result (one row per band×cell)
    …                          # any module-specific diagnostic tables
  figures/
    *.png                      # ggplot2 / glass-brain / matplotlib figures
```

The `<analysis>_hypotheses.csv` is the canonical statistical contract across all
emmeans/permutation modules; figure and gallery consumers read it (plus legacy
column aliases during the migration).

> Figures are **render-on-demand** — they are not auto-regenerated when data
> changes. Re-run the `figures` step (or `source-analytics figure …`) before
> rebuilding a manuscript/gallery from updated tables.

---

## Running a full study, in order

A study run is just the analyses invoked in dependency order (primaries before
their supplements). The canonical recipes live in `scripts/`; the essential order:

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
$SA resting --analysis electrode_comparison    # ↳ after electrode_psd
$SA resting --analysis electrode_connectivity  # sensor FC comparator

# Vertex paradigm — whole-brain
$SA vertex  --analysis vertex_connectivity     # PRIMARY (slow; computes matrices)
$SA vertex  --analysis vertex_graph            # ↳ after vertex_connectivity
$SA vertex  --analysis vertex_nbs              # ↳ after vertex_connectivity
$SA vertex  --analysis vertex_cluster
$SA vertex  --analysis vertex_specparam
$SA vertex  --analysis vertex_mvpa
$SA vertex  --analysis vertex_cross_freq        # local PAC + AAC + PPC
$SA vertex  --analysis vertex_directed         # vertex DTF (outflow/inflow/netflow)

# Evoked paradigm (trial-based data only)
$SA evoked  --analysis roi_evoked
$SA evoked  --analysis vertex_evoked
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
   (vertex/sensor: also do cluster-perm        6. LMMs (lme4/lmerTest), emmeans
    stats + glass-brain figures in Python)     7. Hypothesis layer: effect sizes, FDR
                                               8. ggplot2 figures
                                               9. Markdown ANALYSIS_SUMMARY.md
```

Python calls `Rscript` automatically. ROI/electrode LMM modules delegate stats +
figures to R; vertex/sensor map modules do statistics and glass-brain figures in
Python and use R only for the report.

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
| [`HYPOTHESIS.md`](HYPOTHESIS.md) | the hypothesis layer — kinds, adapters, usage |
| [`DESIGN_SPEC.md`](DESIGN_SPEC.md) | design rationale for `design:`/`hypotheses:` + FDR scope |
| [`CONNECTIVITY_METHODS.md`](CONNECTIVITY_METHODS.md) | equation-checked provenance for every connectivity / coupling / directed metric |
| [`CHANGELOG.md`](CHANGELOG.md) | version history |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | current work / migration status |

## License

MIT
