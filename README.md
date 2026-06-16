# source-analytics

Statistical analysis toolkit for source-localized EEG data. Reads pipeline output from the [source-localization](../source-localization) package and runs group-level analyses with publication-quality statistics and figures. Its stat tables feed directly into [source-lightbox](../source-lightbox), which renders the gallery.

Analyses are organized along **two orthogonal axes** — see
[Analyses by category](#analyses-by-category-with-implementation-references) for the full map:

- **Level** — the data an analysis reads: **ROI** (32–46 atlas ROIs), **vertex**
  (whole-brain source vertices), or **electrode** (scalp, for validation).
- **Domain** — what it measures: **Spectral**, **Connectivity**,
  **Cross-frequency**, **Directed**, **Sensor-level**, **Evoked**.

A few analyses are **supplementary** (secondary): they consume another analysis's
output and therefore must run *after* it. The two graph-theory modules are the
main case — `roi_network` needs `roi_connectivity`, `vertex_network` needs
`vertex_connectivity`. The toolkit records this in `ANALYSIS_METADATA` (`core.py`)
so both the run order and the gallery's grouping follow from it.

**Python** handles orchestration, signal processing, and data I/O. **R** handles statistics (linear mixed models via lme4) and visualization (ggplot2). Vertex-level modules use Python for statistics (cluster permutation) and visualization (glass brain plots), with R for report generation.

## AI-Assisted Workflow (IRL)

This repo ships an [IRL](https://github.com/drpedapati/irl-template) plan template for running source-analytics inside an Idempotent Research Loop: [`irl-template.md`](irl-template.md) (author `analysis.yaml` → run ROI/vertex/electrode modules → copy summaries to `$RESULTS`). Initialize a study with `irl init -t source-analytics "<project-name>"` after dropping the template into `~/research/_templates/`.

## Installation

### Python

```bash
pip install -e .
# or with uv
uv pip install -e .
```

Requires Python 3.10+. Dependencies: numpy, scipy, pandas, pyyaml.

### R

```r
install.packages(c(
  "ggplot2", "dplyr", "tidyr", "readr", "stringr", "forcats",
  "lme4", "lmerTest", "effectsize", "emmeans",
  "yaml", "argparse", "patchwork", "scales"
))
```

## Usage

Everything runs through one CLI. Subcommands: `run`, `validate`, `list`,
`figure`, `init`.

```bash
# One analysis. --paradigm selects the block under paradigms: in the config.
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd

# Re-run an analysis whose output dir already exists
source-analytics run --study study.yaml --paradigm resting --analysis roi_network --force

# Re-run only some lifecycle steps (setup, process, aggregate, statistics, figures, summary)
source-analytics run --study study.yaml --paradigm resting --analysis roi_psd --steps statistics,summary

# Validate the config + data discovery before a long run
source-analytics validate --study study.yaml

# List available analyses
source-analytics list
```

`--paradigm` is required for multi-paradigm configs and names a key under
`paradigms:` (e.g. `resting`, `vertex`). It is **not** the same as *level*: the
`resting` paradigm holds the ROI and electrode analyses; the `vertex` paradigm
holds the vertex analyses (they read a different reconstruction). The analysis
must also be listed under that paradigm's `analyses:` block.

> **Supplementary analyses must run after their primary** — see the run order in
> [Analyses by category](#analyses-by-category-with-implementation-references). The toolkit does
> not auto-run dependencies; if you run `roi_network` before `roi_connectivity`
> it errors that the edges CSV is missing.

### Running everything, in order

A study run is just the analyses invoked in dependency order. The canonical
FORGE recipe lives at `scripts/run_treatment_analyses.sh` (study) /
`scripts/run_ms1_analytics_parallel.sh`; the essential order is:

```bash
SA="source-analytics run --study study.yaml --paradigm"

# Spectral + connectivity primaries (any order)
$SA resting --analysis roi_psd
$SA resting --analysis roi_aperiodic
$SA resting --analysis roi_connectivity        # PRIMARY
$SA resting --analysis roi_cross_freq          # PAC + AAC + PPC (--metric to pick one)
$SA resting --analysis roi_directed            # transfer entropy + DTF (--metric te|dtf)
$SA resting --analysis roi_graph               # supplements roi_connectivity → run after it
$SA resting --analysis roi_nbs                 # supplements roi_connectivity → run after it

$SA resting --analysis electrode_psd           # PRIMARY
$SA resting --analysis electrode_comparison    # supplements electrode_psd → run after it

$SA vertex  --analysis vertex_cluster
$SA vertex  --analysis vertex_mvpa
$SA vertex  --analysis vertex_specparam
$SA vertex  --analysis vertex_spatial
$SA vertex  --analysis vertex_connectivity     # PRIMARY (slow; computes connectivity matrices)
$SA vertex  --analysis electrode_connectivity  # sensor FC-six comparator (source-vs-sensor)
$SA vertex  --analysis vertex_cross_freq        # local PAC + AAC + PPC (full vertex resolution)
$SA vertex  --analysis vertex_directed         # vertex DTF (ridge-MVAR; outflow/inflow/netflow)
$SA vertex  --analysis vertex_graph            # supplements vertex_connectivity → run after it
$SA vertex  --analysis vertex_nbs              # supplements vertex_connectivity → run after it

# Evoked paradigm (trial-based data only)
$SA evoked  --analysis roi_evoked              # ITC / ERSP / STP
$SA evoked  --analysis vertex_evoked           # per-vertex ITC / ERSP / STP (cluster-corrected)
$SA evoked  --analysis electrode_evoked
```

## Study Configuration

One YAML file drives the whole study — and the **same file** is read by
`source-lightbox` to build the gallery. The study-design keys (groups, contrasts,
bands) are global; the **per-paradigm `analyses:` block** is where each analysis
gets its data location and parameters. The minimal shape:

```yaml
name: "My Study"

# ── Study design (global) ──────────────────────────────────────────
groups:                              # raw group id → display label
  WT_VEH: "WT Vehicle"
  KO_VEH: "KO Vehicle"
group_order: [WT_VEH, KO_VEH]        # plot/x-axis order
group_colors: {WT_VEH: "#3498DB", KO_VEH: "#E74C3C"}

contrasts:                           # each = one group-vs-group test
  - name: disease_effect             # used in table/file names
    label: "Disease effect (KO vs WT)"   # human label (axes, digests)
    group: "Disease effect"          # tier heading (groups contrasts in the gallery)
    group_a: KO_VEH                  # "A − B": positive effect = higher in A
    group_b: WT_VEH

bands:                               # name → [fmin, fmax] Hz
  Delta: [1, 4]
  Theta: [4, 8]
  Alpha: [8, 13]
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
  analytics: ./analytics             # working dir: ANALYSIS_SUMMARY.md, data/
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
      roi_network:                                     # supplements roi_connectivity
        threshold_method: proportional
        threshold_value: 0.15
        nbs_threshold: 2.5
        nbs_permutations: 5000
        connectivity_metrics: [imag_coherence, dwpli, pli, aec, coherence]
      electrode_psd: {}
      electrode_comparison: {}                         # supplements electrode_psd

  vertex:
    data_dir:    ./localization/rest_shell/derivatives
    data_subdir: pipeline/data
    analyses:
      vertex_connectivity:                             # PRIMARY for the vertex graph theory
        vertex_filter: {z_min: 0.0}
        metrics: [imag_coherence, dwpli, pli, aec, coherence]   # all 5 share one STFT pass
      vertex_network:                                  # supplements vertex_connectivity
        nbs_threshold: 3.0
        nbs_permutations: 5000
        connectivity_metrics: [imag_coherence, dwpli, pli, aec, coherence]
```

### What each key feeds

| Key | Consumed by | Purpose |
|---|---|---|
| `groups`, `group_order`, `group_colors` | all analyses | group identity, plot order/colour |
| `contrasts[]` `{name, group_a, group_b}` | statistics | the group-vs-group tests (A − B) |
| `contrasts[]` `{label, group}` | figures, gallery | readable labels + tier grouping |
| `bands` | all spectral/connectivity | frequency bands analysed |
| `epoch_sampling` | spectral/connectivity | random-epoch resampling (`n_bootstrap: 0` = full timeseries) |
| `paths.{analytics, results}` | I/O + gallery | working vs published output trees |
| `paradigms.<p>.data_dir` / `data_subdir` | discovery | where subject reconstructions live |
| `paradigms.<p>.analyses.<a>` | that analysis | enables it + sets its parameters |

The per-analysis block is merged into `config.raw[<analysis>]` by
`config.for_paradigm_analysis()`, so any analysis-specific key (e.g.
`connectivity_metrics`, `nbs_permutations`, `vertex_filter`) **must live under
`paradigms.<paradigm>.analyses.<analysis>`**, not at the top level.

> **Connectivity metrics.** The network/graph analyses run on every metric in
> `connectivity_metrics`. At the vertex level, set `vertex_connectivity.metrics`
> to the same list so the primary precomputes all of them in one shared-STFT pass
> (`compute_..._epochs_multi`); `vertex_network` then loads them per metric
> instead of recomputing. (Note: `aec` is computed outside the shared STFT and is
> the slow one — drop it from the list if runtime matters more than completeness.)

## Input Data

source-analytics reads output files produced by the source_localization pipeline. Each subject directory contains:

**ROI-level analyses** (roi_psd, roi_aperiodic, roi_connectivity, roi_directed, roi_cross_freq, roi_evoked) -- default discovery:

| File | Format | Contents |
|------|--------|----------|
| `step6_roi_timeseries_magnitude.pkl` | Python pickle | Dict[str, ndarray] -- ROI timeseries (unsigned, for PSD) |
| `step6_roi_timeseries_signed.pkl` | Python pickle | Dict[str, ndarray] -- ROI timeseries (signed, for connectivity) |
| `roi_timeseries_magnitude.set` | EEGLAB .set | Same data + metadata (sfreq) |

**Vertex-level analyses** (vertex_cluster, vertex_connectivity, vertex_cross_freq, vertex_directed, vertex_specparam, vertex_mvpa, network, vertex_spatial, vertex_evoked) -- uses `discovery.required_files` in config:

| File | Format | Contents |
|------|--------|----------|
| `step5_stc.pkl` | Python pickle | MNE SourceEstimate (n_vertices, n_times) |
| `step3_source_coords_mm.npy` | NumPy array | Source coordinates (n_vertices, 3) in mm |

**Electrode-level analyses** (electrode_psd, electrode_aperiodic, electrode_connectivity, electrode_comparison, electrode_evoked) -- uses `electrode.subject_roster` in config:

| File | Format | Contents |
|------|--------|----------|
| `*.set / *.fdt` | EEGLAB | Raw scalp EEG (channels x timepoints) |

Expected directory layout:

```
root_dir/
  Group_A/
    Subject_001/data/
    Subject_002/data/
  Group_B/
    Subject_003/data/
```

## Architecture

```
Python                                         R
──────────────────────────────────────         ──────────────────────────────
1. Load YAML config, discover subjects
2. Load ROI timeseries (pickle/.set)
3. Signal processing (scipy)
4. Export CSVs ───────────────────────────►   5. Read CSVs + config
                                              6. LMMs (lme4/lmerTest)
                                              7. Effect sizes, FDR correction
                                              8. ggplot2 figures
                                              9. Markdown summary
```

Python calls `Rscript` automatically -- no manual R interaction needed.

## Analyses by category, with implementation references

Analyses are grouped by **category (domain)** — what they measure. Each row lists
the analyses in that category, the level(s) they run at, the measures/metrics
they compute, and the **primary literature** the implementation follows.

> **Method provenance is tracked in [`CONNECTIVITY_METHODS.md`](CONNECTIVITY_METHODS.md)** —
> the source of truth for every connectivity / coupling metric: canonical
> reference, defining equation, our `file:function`, and any deviation, each
> verified against fetched primary sources. Citations below for the
> connectivity / cross-frequency / directed families are condensed from it.
> References for the spectral / graph families are the standard canonical sources
> (verify before manuscript submission — only the connectivity family has been
> formally equation-checked).

| Category | Analyses (level) | Measures | Implementation reference(s) |
|---|---|---|---|
| **Spectral** | `roi_psd`, `electrode_psd` (ROI/electrode); `vertex_cluster`, `vertex_spatial` (vertex) | band power (Welch PSD), spatial GLS, cluster-corrected vertex maps | Welch 1967; cluster permutation Maris & Oostenveld 2007 |
| | `roi_aperiodic`, `electrode_aperiodic`, `vertex_specparam` | 1/f aperiodic (offset, exponent) + oscillatory peaks | Donoghue et al. 2020, *Nat Neurosci* (specparam) |
| | `vertex_mvpa` | multivariate pattern decoding (linear SVM) | standard MVPA (linear SVM, permutation-tested) |
| **Connectivity** (same-frequency FC) | `roi_connectivity`, `vertex_connectivity` | coherence; imaginary coherence; PLI; wPLI; dwPLI; dPLI; AEC; partial correlation | Nolte 2004 (imcoh); Stam 2007 (PLI); Vinck 2011 (wPLI/dwPLI); Stam & van Straaten 2012 (dPLI); Hipp 2012 (AEC); Marrelec 2006 (partial corr) — see `CONNECTIVITY_METHODS.md` |
| | `roi_graph`/`roi_nbs`, `vertex_graph`/`vertex_nbs` *(supplements `*_connectivity`)* | graph-theoretic metrics; Network-Based Statistic | Rubinov & Sporns 2010 (graph); Zalesky et al. 2010 (NBS) |
| **Cross-frequency** | `roi_cross_freq`, `vertex_cross_freq` | PAC (Modulation Index); cross-frequency AAC; n:m PPC | Tort et al. 2010 (PAC MI); Bruns 2000 / Masimore 2004 (AAC); Tass 1998 / Palva 2005 (PPC) — see `CONNECTIVITY_METHODS.md` |
| **Directed** | `roi_directed` (ROI); `vertex_directed` (vertex) | transfer entropy (`te`, `net_te`); DTF (`dtf`; vertex: outflow/inflow/netflow) via ridge-MVAR | Schreiber 2000 (transfer entropy); Kaminski & Blinowska 1991 (DTF) |
| **Connectivity** (sensor) | `electrode_connectivity` | FC-six (coherence/imcoh/PLI/wPLI/dwPLI/dPLI) + per-channel FCD — the source-vs-sensor comparator | as Connectivity row — see `CONNECTIVITY_METHODS.md` |
| **Sensor-level** | `electrode_comparison` *(supplements `electrode_psd`)* | source-vs-electrode validation comparison | — (internal comparison) |
| **Evoked** | `roi_evoked`, `vertex_evoked`, `electrode_evoked` | ITC, ERSP, single-trial power (trial paradigms) | standard time-frequency (Hilbert/wavelet ITC/ERSP) |

Each analysis is `--metric`-selectable where it computes multiple measures (e.g.
`--metric wpli`, `--metric pac`); see [Selecting metrics & bands](#selecting-metrics--bands).

**Renamed (2026-06):** `roi_pac` → `roi_cross_freq` (now also AAC + PPC);
`roi_transfer_entropy` → `roi_directed`. Old names still work as deprecated
aliases.

This grouping is generated from `ANALYSIS_METADATA` in `core.py` (`domain` +
`supplements`), the single source of truth; `analysis_meta()` exposes it, and
`source-lightbox` reads it to group the gallery by domain and nest each
supplementary analysis under its primary. **Domain** decides where an analysis is
*listed*; **`supplements`** is a real *dependency* — that's what sets the run
order in [Running everything, in order](#running-everything-in-order).

### Selecting metrics & bands

Multi-output analyses honour a sub-output filter so you compute exactly what you
want, not the whole group — without losing the shared STFT/Hilbert compute pass:

```bash
# just two connectivity metrics
source-analytics run --study study.yaml --paradigm vertex --analysis vertex_connectivity --metric dwpli,wpli
# one band; one cross-frequency measure
source-analytics run --study study.yaml --paradigm vertex --analysis vertex_cross_freq --metric ppc --band low_gamma
# generic form
source-analytics run ... --select metric=pli --select band=beta,low_gamma
```

`source-analytics list` tags each analysis with its selectable dimensions
(`[--select: metric, band]`).

## Analysis Modules

### ROI-Level Analyses

These analyses operate on 46 source-localized ROI timeseries (`step6_roi_timeseries_*.pkl`). They use the standard `analysis.yaml` config pointing to the `roi_based_ellipsoid/` pipeline output.

#### ROI PSD (Power Spectral Density)

Computes power spectral density via Welch's method and extracts band power across ROIs.

**Python side:**
- Welch PSD (2s Hann windows, 50% overlap) via `scipy.signal.welch`
- Band power extraction (absolute, relative, dB) via trapezoidal integration
- Exports `band_power.csv` and `psd_curves.csv`

**R side:**
- Omnibus LMM: `relative ~ group * roi + (1|subject)` (lme4/lmerTest)
- Type III ANOVA with Satterthwaite degrees of freedom
- Post-hoc: emmeans pairwise group contrasts per ROI (gated on significant omnibus)
- FDR (Benjamini-Hochberg) correction across bands; Holm correction across ROIs
- Hedges' g effect sizes (emmean difference / residual SD)
- PSD curve plots, band power boxplots, regional heatmaps, ROI forest plots, significance heatmaps (ggplot2)
- Markdown summary with methods, omnibus table, post-hoc results, and key findings

**Output:**

```
output_dir/roi_psd/
  ANALYSIS_SUMMARY.md
  data/
    band_power.csv
    psd_curves.csv
    study_config.yaml
  tables/
    psd_omnibus.csv            # Omnibus LMM results (group x ROI interaction)
    psd_posthoc_roi.csv        # emmeans post-hoc contrasts per ROI
  figures/
    psd_by_region.png
    band_power_relative.png
    band_power_absolute.png
    band_power_dB.png
    heatmap_relative_*.png
    roi_forest_plot_*.png      # Group contrast per ROI (dot-and-whisker)
    roi_significance_heatmap_*.png  # ROI x band heatmap (Hedges' g)
```

#### ROI Aperiodic (1/f Spectral Decomposition)

Decomposes PSD into periodic and aperiodic (1/f) components using specparam (FOOOF) with linear regression fallback.

**Python side:**
- Aperiodic fitting via specparam or linreg on log-log PSD
- Extracts exponent (spectral slope) and offset per ROI
- Exports `aperiodic.csv`

**R side:**
- Omnibus LMM: `exponent ~ group * roi + (1|subject)` (and same for offset)
- Region-level aggregation and LMM if roi_categories defined
- Post-hoc emmeans, Hedges' g, Holm correction
- Boxplots, regional summaries, forest plots
- Markdown summary

**Output:**

```
output_dir/roi_aperiodic/
  ANALYSIS_SUMMARY.md
  data/
    aperiodic.csv
    study_config.yaml
  tables/
    aperiodic_omnibus.csv
    aperiodic_posthoc_roi.csv
    aperiodic_omnibus_region.csv
    aperiodic_posthoc_region.csv
  figures/
    aperiodic_boxplot_*.png
    aperiodic_by_region_*.png
    aperiodic_roi_forest_*.png
```

#### ROI Connectivity (Functional Connectivity)

ROI-to-ROI functional connectivity using **signed** (phase-preserving) source timeseries. Computes six complementary connectivity metrics for all 1,035 unique ROI pairs (46 ROIs):

| Metric | Description | Volume conduction resistant |
|--------|-------------|:--:|
| **Coherence** | Magnitude-squared coherence (Welch CSD) | No |
| **Imaginary Coherence** | Im(Cxy)/√(Sxx·Syy) -- zero-lag immune (Nolte 2004) | Yes |
| **PLI** | Phase Lag Index \|⟨sign(Im(Pxy))⟩\| (Stam 2007) | Yes |
| **wPLI** | Weighted PLI \|E{Im}\|/E{\|Im\|} (Vinck 2011) | Yes |
| **dwPLI** | Debiased weighted PLI² (Vinck 2011) | Yes |
| **dPLI** | Directed PLI ⟨H(Im)⟩ -- **asymmetric**, i>0.5 leads j (Stam & van Straaten 2012) | Yes |
| **AEC** | Orthogonalized amplitude envelope correlation, imag-projection + log-power (Hipp 2012) | Yes |
| **Partial Correlation** | Conditional independence via precision matrix, LW shrinkage (Marrelec 2006) | Yes |

Full equations, our implementation, and provenance: [`CONNECTIVITY_METHODS.md`](CONNECTIVITY_METHODS.md).
`dpli` is directed and is auto-excluded from the undirected graph/NBS layer.

**Python side:**
- Cross-spectral density via `scipy.signal.csd` (Welch, 2s Hann, 50% overlap)
- All 6 metrics computed simultaneously per subject
- Exports `roi_connectivity_edges.csv` (subject x edge x band x metric)

**R side:**
- **Global analysis:** Mean connectivity across all edges per subject x band; Welch t-test per band x metric, BH FDR across bands
- **Region-pair analysis:** Edges mapped to region pairs via roi_categories, averaged within; LMM `dv ~ group * region_pair + (1|subject)`, post-hoc emmeans per region pair, Holm correction
- Connectivity matrix heatmaps, global bar charts, region-pair forest plots
- Markdown summary

**Output:**

```
output_dir/roi_connectivity/
  ANALYSIS_SUMMARY.md
  data/
    roi_connectivity_edges.csv      # subject x roi_pair x band (all 6 metrics)
    study_config.yaml
  tables/
    roi_connectivity_global.csv     # global t-tests per band x metric
    roi_connectivity_omnibus_region_pair.csv   # LMM results (if roi_categories)
    roi_connectivity_posthoc_region_pair.csv   # post-hoc per region pair (if significant)
  figures/
    roi_connectivity_matrix_coherence_*.png
    roi_connectivity_matrix_imag_coherence_*.png
    roi_connectivity_global_bar.png
    roi_connectivity_region_pair_forest_*.png
```

#### ROI Directed (Transfer Entropy + DTF)

Directed connectivity between all ROI pairs. `--metric` selects the measure:
**`te`** (transfer entropy, model-free/pairwise) and/or **`dtf`** (directed
transfer function, multivariate via ridge-MVAR). Uses **signed** (phase-preserving)
ROI timeseries.

**Transfer entropy (`te`):**
- Bandpass filtering per frequency band, quantile-based discretization (5 bins)
- TE(X→Y) = H(Y_future, Y_past) + H(Y_past, X_past) − H(Y_past) − H(Y_future, Y_past, X_past)
- All n×(n-1) directed pairs; net TE: TE(X→Y) − TE(Y→X) for directional asymmetry
- R side does the group/directional/region-pair stats (below)

**DTF (`dtf`):**
- One ridge-regularized MVAR fit over all ROIs (order 8, ridge 0.05 by default;
  config `roi_directed.mvar_order` / `mvar_ridge`), then DTF read out across bands
- Ridge is required because source ROIs are strongly collinear (mean inter-node
  |corr| ≈ 0.64) — plain LS-MVAR is non-stationary; the module warns if the fit
  is unstable. `dtf[i,j]` = directed influence source i → target j
- Exported as a `dtf` column in the directed edge CSV; group-level DTF stats are
  not yet wired (a DTF-only run skips the TE R script)

**R side:**
- **Global analysis:** Mean TE across all directed edges per subject × band; Welch t-test per band, BH FDR across bands
- **Directional analysis:** Paired t-test on TE(X→Y) vs TE(Y→X) within groups (test for net directionality)
- **Region-pair analysis:** Directed edges mapped to region pairs via roi_categories; LMM per band, post-hoc emmeans
- Markdown summary

**Output:**

```
output_dir/roi_directed/
  ANALYSIS_SUMMARY.md
  data/
    transfer_entropy_edges.csv    # subject x directed roi_pair x band (TE + net TE)
    study_config.yaml
  tables/
    transfer_entropy_global.csv   # global t-tests per band
    transfer_entropy_directional.csv  # paired t-tests on directionality
    transfer_entropy_omnibus_region_pair.csv   # LMM results (if roi_categories)
    transfer_entropy_posthoc_region_pair.csv   # post-hoc per region pair
  figures/
    transfer_entropy_global_bar.png
    transfer_entropy_region_pair_forest_*.png
```

#### ROI PAC (Phase-Amplitude Coupling)

Cross-frequency phase-amplitude coupling via the Modulation Index (Tort et al., 2010) with surrogate-based z-scoring. Uses **signed** (phase-preserving) ROI timeseries.

**Python side:**
- Bandpass filtering (Butterworth, zero-phase; auto-reduces order for narrow bands)
- Hilbert transform for instantaneous phase and amplitude envelope
- Phase binning (18 bins, 20° each), mean amplitude per bin
- MI = KL divergence from uniform / log(N)
- 200 surrogate MIs via circular time-shifts of amplitude envelope (min 1 sec shift)
- z-score = (observed MI - mean(surrogates)) / std(surrogates)
- Auto-generates valid frequency pairs from config bands (amplitude center >= 2.5× phase center)
- Exports `pac_values.csv` (subject x roi x freq_pair)

**R side:**
- **Global analysis:** Mean z-scored MI across all ROIs per subject x freq_pair; Welch t-test per freq_pair, BH FDR across pairs
- **Region-level analysis:** ROIs mapped to regions via roi_categories, averaged within; LMM `z_score ~ group * region + (1|subject)`, BH FDR across freq_pairs, post-hoc emmeans per region gated on significance, Holm correction
- Global bar chart, comodulogram heatmaps (per group + difference), region forest plots
- Markdown summary

**Output:**

```
output_dir/roi_cross_freq/
  ANALYSIS_SUMMARY.md
  data/
    pac_values.csv              # subject x roi x freq_pair (z-scored MI)
    study_config.yaml
  tables/
    pac_global.csv              # global t-tests per freq_pair
    pac_omnibus_region.csv      # region-level LMM omnibus (if roi_categories)
    pac_posthoc_region.csv      # post-hoc per region (if significant)
  figures/
    pac_global_bar.png
    pac_comodulogram_*.png
    pac_region_forest_*.png
```

### Vertex-Level Analyses

These analyses operate on 154-vertex source estimates from the `shell_ellipsoid/` pipeline. They **require a separate study config** with `discovery.required_files` pointing to `step5_stc.pkl` and `step3_source_coords_mm.npy`.

#### Vertex Cluster (Vertex-Level Spectral Analysis)

Vertex-level spectral analysis with cluster-based permutation testing (Maris & Oostenveld, 2007). All metrics derived from a single PSD computation per subject.

**Python side (signal processing + statistics + visualization):**
- PSD via `scipy.signal.welch` with axis=-1 broadcasting on (n_vertices, n_times) arrays
- Per-vertex metrics: relative/absolute band power, fALFF (high-gamma/total ratio), spectral slope (1/f exponent via log-log regression), peak alpha frequency
- Voxel-wise Welch's t-tests + Hedges' g per vertex
- Cluster-based permutation correction: spatial adjacency from source coordinates, BFS connected components, max cluster statistic null distribution
- Glass brain figures: 3-view (axial/coronal/sagittal) scatter, 6-panel band comparison (group means, difference, t-map, significant clusters, histogram), multi-band summary

**R side (report generation only):**
- Reads pre-computed CSVs
- Effect size summary table
- Formatted ANALYSIS_SUMMARY.md with methods, results tables, figure references

**Study config (`analysis_vertex_cluster.yaml`):**

```yaml
discovery:
  root_dir: "/path/to/source_localization/shell_ellipsoid"
  group_mapping:
    "KO ICV": KO_VEH
    "WT ICV": WT_VEH
  required_files:
    - "step5_stc.pkl"
    - "step3_source_coords_mm.npy"

vertex:
  correction_method: cluster  # "cluster" (default) or "tfce"
  cluster_threshold: 2.0      # only used when correction_method: cluster
  n_permutations: 1000
  adjacency_distance_mm: 5.0
  noise_exclude_hz: [55, 65]
  tfce:                        # only used when correction_method: tfce
    E: 0.5
    H: 2.0
    dh: 0.1
```

**Output:**

```
output_dir/vertex_cluster/
  ANALYSIS_SUMMARY.md
  data/
    vertex_cluster_values.csv       # subject x vertex x band (relative, absolute, dB)
    vertex_cluster_features.csv     # subject x vertex (fALFF, spectral slope, peak alpha)
    source_coords.csv           # vertex coordinates in mm
    vertex_cluster_results.pkl      # full results dict for reuse
    study_config.yaml
  tables/
    voxelwise_stats.csv         # per-vertex t, p, Hedges' g per contrast x metric
    cluster_results.csv         # cluster summaries with permutation-corrected p-values
    effect_size_summary.csv     # aggregated effect sizes (from R)
  figures/
    vertex_cluster_delta.png
    vertex_cluster_theta.png
    vertex_cluster_alpha.png
    vertex_cluster_beta.png
    vertex_cluster_low_gamma.png
    vertex_cluster_high_gamma.png
    vertex_cluster_falff.png
    vertex_cluster_spectral_slope.png
    vertex_cluster_peak_alpha.png
    vertex_cluster_summary.png
```

##### TFCE Correction Option

The vertex_cluster analysis supports TFCE (Smith & Nichols, 2009) as an alternative to cluster-based permutation testing. Set `correction_method: tfce` in the vertex config section. TFCE eliminates the arbitrary cluster-forming threshold by integrating cluster extent and height across all thresholds: `TFCE(v) = sum_h { e(h)^E * h^H * dh }`. When using TFCE, additional output includes `tfce_scores_*.png` glass brains and per-vertex corrected p-values in `voxelwise_stats.csv`.

#### Vertex Connectivity (Functional Connectivity Density)

All-to-all connectivity between dorsal source vertices for one or more of the
FC-six metrics, deriving Functional Connectivity Density (FCD) maps showing how
connected each vertex is to the rest of the brain.

**Python side:**
- One shared STFT pass computes the spectral metrics (coherence, imag_coherence,
  PLI, wPLI, dwPLI, dPLI); AEC is computed separately — all metrics in `metrics:`
- FCD: fraction of connections above threshold per vertex (directed-aware, so dPLI
  thresholds |dPLI − 0.5| — see `FCD_CENTER` in `spectral/vertex_connectivity.py`)
- Cluster-based permutation testing on FCD maps; glass-brain FCD visualizations
- Saves the full connectivity matrices (consumed by vertex_graph / vertex_nbs)

**Config:**
```yaml
vertex_connectivity:
  metrics: [imag_coherence, dwpli, wpli, pli, dpli, aec, coherence]
  fcd_threshold: 0.05
  n_permutations: 1000
```

**Output:**
```
output_dir/vertex_connectivity/
  ANALYSIS_SUMMARY.md
  data/vertex_fcd.csv, vertex_connectivity_matrices.pkl, source_coords.csv
  tables/vertex_connectivity_stats.csv
  figures/fcd_*.png
```

#### Vertex Directed (DTF)

Vertex-level directed connectivity via the directed transfer function. One
ridge-regularized MVAR fit per subject over the dorsal vertices (ridge mandatory:
mean inter-vertex |corr| ≈ 0.64 makes plain MVAR explosive), then DTF read out
across bands and reduced to three per-vertex maps — **outflow** (driver strength),
**inflow** (receiver strength), **netflow** (out − in). Cluster-permutation stats +
glass-brain maps, like vertex_connectivity. Warns if the MVAR is non-stationary.

```yaml
vertex_directed:
  mvar_order: 8
  mvar_ridge: 0.05
  n_permutations: 1000
```
Output: `data/vertex_directed.csv`, `vertex_dtf_matrices.pkl`, `source_coords.csv`;
`tables/vertex_directed_stats.csv`; `figures/dtf_*.png`.

#### Vertex Evoked (ITC / ERSP / STP)

Per-vertex inter-trial coherence, event-related spectral perturbation, and
single-trial power for trial-based (evoked) paradigms — the vertex companion to
`roi_evoked`. Morlet TFR per vertex, scalar measures extracted per band/time
window, group differences via spatial cluster permutation + glass-brain maps.
Requires an `evoked:` config section (epoch_samples, sfreq, baseline, tf_params,
measures). Output: `data/vertex_evoked_measures.csv`,
`tables/vertex_evoked_stats.csv`, `figures/evoked_*.png`.

#### Vertex Specparam (Vertex-Level Spectral Parameterization)

Determines whether gamma elevation is a true oscillatory peak vs. broadband shift by fitting aperiodic (1/f) models at each vertex using specparam (FOOOF) or linear regression fallback.

**Python side:**
- Per-vertex specparam fit: exponent, offset, R², peak detection
- Gamma peak presence detection per vertex
- Cluster-based permutation on exponent/offset maps
- Chi-squared tests on gamma peak presence rates
- Glass brain maps: aperiodic parameters, gamma peak prevalence

**R side:**
- Group summary of aperiodic parameters, method distribution
- ANALYSIS_SUMMARY.md

**Config:**
```yaml
vertex_specparam:
  freq_range: [1, 100]
  peak_width_limits: [1.0, 12.0]
  max_n_peaks: 6
```

**Output:**
```
output_dir/vertex_specparam/
  ANALYSIS_SUMMARY.md
  data/vertex_specparam.csv, source_coords.csv
  tables/vertex_specparam_stats.csv, gamma_peak_chi2.csv
  figures/specparam_*.png, gamma_peak_presence.png
```

#### Vertex MVPA (Multivariate Pattern Analysis)

Single omnibus test per band: can the whole-brain spatial pattern classify KO vs WT? Uses linear SVM + Leave-One-Out Cross-Validation with permutation testing.

**Python side:**
- Feature matrix: 154-vertex relative band power per subject
- Linear SVM with LOOCV
- Permutation test (shuffled group labels, 1000 permutations)
- Reports: accuracy, p-value, sensitivity, specificity, AUC, 95% CI
- Feature importance from SVM coefficients
- Figures: null distribution histograms, importance glass brains, confusion matrices

**R side:**
- Classification results table, significant bands
- ANALYSIS_SUMMARY.md

**Config:**
```yaml
vertex_mvpa:
  classifier: svm_linear
  cv_method: loocv
  n_permutations: 1000
```

**Output:**
```
output_dir/vertex_mvpa/
  ANALYSIS_SUMMARY.md
  data/vertex_mvpa_features.csv, source_coords.csv
  tables/vertex_mvpa_results.csv
  figures/vertex_mvpa_importance_*.png, vertex_mvpa_null_*.png, vertex_mvpa_confusion_*.png
```

#### Network (Graph-Theoretic Analysis + NBS)

Graph-theoretic metrics from thresholded connectivity matrices and Network-Based Statistic (Zalesky et al., 2010) for subnetwork identification.

**Python side:**
- Graph metrics via networkx: degree, clustering, betweenness, global efficiency, modularity, small-worldness
- Cluster-based permutation on nodal metrics
- NBS: edge-wise t-tests + connected component permutation testing
- Glass brain nodal metric visualizations

**R side:**
- Global metric group comparisons (t-tests)
- NBS subnetwork results
- ANALYSIS_SUMMARY.md

**Config:**
```yaml
network:
  threshold_method: proportional
  threshold_value: 0.1
  nbs_threshold: 3.0
  nbs_permutations: 5000
```

**Output:**
```
output_dir/network/
  ANALYSIS_SUMMARY.md
  data/network_nodal_metrics.csv, network_global_metrics.csv, source_coords.csv
  tables/network_stats.csv, nbs_results.csv
  figures/network_*.png
```

#### Vertex Spatial (Spatial Mixed Effects Models)

Single model per band accounting for spatial correlation, avoiding the multiple comparison problem entirely. Primary computation in R using `nlme::gls` with exponential spatial correlation.

**R side (primary computation):**
- `gls(relative ~ group, correlation = corExp(form = ~x+y+z | subject))`
- Spatial vs non-spatial model comparison via AIC/BIC
- Group effect coefficient, SE, t-value, p-value
- Estimated spatial range from correlation structure
- Variogram plots (empirical vs fitted)
- Fallback to GAM with `s(x,y,z, bs="tp")` if GLS fails

**Python side:**
- Data preparation: vertex power + coordinates CSVs
- Spatial residual glass brain maps (from R output)

**Config:**
```yaml
vertex_spatial:
  correlation_structure: exponential
  spatial_range_mm: 3.0
```

**Output:**
```
output_dir/vertex_spatial/
  ANALYSIS_SUMMARY.md
  data/vertex_spatial_data.csv, source_coords.csv
  tables/vertex_spatial_results.csv, spatial_residuals.csv
  figures/variogram_*.png, spatial_residuals_*.png
```

##### Cross-Cutting: Random Epoch Sampling

All vertex-level analyses support optional random epoch sampling. Instead of computing PSD/connectivity on full continuous recordings, randomly sample non-overlapping epochs of fixed duration. Enable in the vertex config section:

```yaml
vertex:
  epoch_sampling:
    enabled: true
    epoch_duration_sec: 2.0
    n_epochs: 80
    seed: 42
```

When enabled, PSD is computed per-epoch then averaged (more robust spectral estimate). Connectivity is computed per-epoch then averaged (standard approach in connectivity literature).

### Electrode-Level Analyses

These analyses operate on raw scalp EEG channels (pre-source-localization) for validation purposes. They require a `subject_roster.csv` mapping subjects to their raw `.set/.fdt` files.

#### Electrode (Electrode-Level PSD)

Mirrors the PSD analysis but operates on raw scalp EEG channels instead of source-localized ROI timeseries.

**Python side:**
- Per-channel Welch PSD and band power extraction (absolute, relative, dB)
- Exports `electrode_band_power.csv` and `electrode_psd_curves.csv`

**R side:**
- Omnibus LMM: `relative ~ group * channel + (1|subject)` (reuses stats_utils infrastructure)
- Post-hoc emmeans per channel, FDR/Holm correction
- PSD curve plots, band power boxplots, heatmaps
- Markdown summary

**Config:**
```yaml
electrode:
  subject_roster: /path/to/subject_roster.csv  # columns: subject_id, group, eeg_filename, eeg_dir
```

**Output:**

```
output_dir/electrode/
  ANALYSIS_SUMMARY.md
  data/
    electrode_band_power.csv      # subject x channel x band (absolute, relative, dB)
    electrode_psd_curves.csv      # subject x channel x freq_hz x psd
    study_config.yaml
  tables/
    electrode_omnibus.csv
    electrode_posthoc.csv
  figures/
    electrode_psd_by_channel.png
    electrode_band_power_*.png
```

#### Electrode Comparison (Electrode vs Source Validation)

Compares electrode-level and source-localized analysis results to validate that source localization provides spatial specificity beyond scalp-level recordings. Requires both `electrode` and `roi_psd` analyses to be run first.

**Python side:**
- Merges per-subject mean power from electrode and source analyses
- Pearson correlations between electrode and source power per band
- Hedges' g effect sizes at both levels (with 95% CIs)
- Regional source effect sizes vs global electrode baseline
- Publication-quality matplotlib figures: correlation scatters, effect size comparisons, regional forest plots, spatial advantage heatmaps, ROI-level forest plots

**R side:**
- Formatted comparison report with methods and interpretation
- ANALYSIS_SUMMARY.md

**Output:**

```
output_dir/electrode_comparison/
  ANALYSIS_SUMMARY.md
  data/
    comparison_data.csv           # subject x band (electrode + source power)
    regional_source_power.csv     # subject x band x region (if roi_categories)
    study_config.yaml
  tables/
    comparison_stats.csv          # correlations + effect sizes per band
    regional_effect_sizes.csv     # per-region source vs electrode Hedges' g
  figures/
    fig1_correlation_*.png        # electrode vs source scatter per band
    fig2_effect_sizes_*.png       # side-by-side Hedges' g comparison
    fig3_regional_forest_*.png    # regional source effects vs electrode reference
    fig4_spatial_advantage_*.png  # heatmap: |region g| - |electrode g|
    fig5_roi_forest_dB.png        # per-ROI disease effects (Low/High Gamma)
```

#### Electrode Connectivity (Source-vs-Sensor Comparator)

Sensor-level functional connectivity — the comparator the source-vs-sensor thesis
is tested against. Runs the FC-six metrics all-pairs on the raw electrode montage
using the **same** array kernel, metric list, and FCD threshold as
`vertex_connectivity`, so the head-to-head is apples-to-apples. Reads raw `.set`
via `electrode.subject_roster`. Per-channel FCD maps; group differences by
per-channel Welch t + BH-FDR across channels (cluster permutation is vertex-only —
the montage is too coarse). Emits the montage coordinates for topomap rendering.

```yaml
electrode_connectivity:
  epoch_sampling: { n_bootstrap: 0 }   # full timeseries, matches vertex_connectivity
  metrics: [imag_coherence, dwpli, wpli, pli, dpli, aec, coherence]
  fcd_threshold: 0.05
```
Output: `data/electrode_fcd.csv`, `electrode_connectivity_matrices.pkl`,
`electrode_layout.csv`; `tables/electrode_connectivity_stats.csv`.

### Atlas Integration

The `source_analytics.atlas` module maps vertex coordinates to anatomical ROI labels from the C57BL/6 MRI atlas. Used by analysis modules to annotate vertices with brain region names.

```python
from source_analytics.atlas import load_vertex_roi_labels, find_atlas_dir

atlas_dir = find_atlas_dir()  # auto-detects from source_localization package
labels = load_vertex_roi_labels(coords_mm, atlas_dir)
```

## Adding a New Analysis

1. Create `src/source_analytics/analyses/my_analysis.py` subclassing `BaseAnalysis`
2. Implement the lifecycle: `setup` -> `process_subject` -> `aggregate` -> `statistics` -> `figures` -> `summary`
3. Create `R/my_analysis.R` for statistics and visualization
4. Register in `core.py` `ANALYSIS_REGISTRY`
5. Update this README with the new module description

## License

MIT
