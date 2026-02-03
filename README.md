# source-analytics

Statistical analysis toolkit for source-localized EEG data. Reads pipeline output from the [source_localization](https://github.com/alexedmon1/AlexProjects) package and runs group-level analyses with publication-quality statistics and figures. Supports both ROI-level analyses (PSD, aperiodic, connectivity, PAC) and whole-brain vertex-level analysis with cluster-based permutation testing.

**Python** handles orchestration, signal processing, and data I/O. **R** handles statistics (linear mixed models via lme4) and visualization (ggplot2). The wholebrain module uses Python for statistics (cluster permutation) and visualization (glass brain plots), with R for report generation.

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

```bash
# Run PSD analysis
source-analytics run --study /path/to/analysis.yaml --analysis psd

# Validate study config and data discovery
source-analytics validate --study /path/to/analysis.yaml

# List available analyses
source-analytics list
```

## Study Configuration

Each study is configured with a YAML file (`analysis.yaml`) that lives alongside the study data. The config defines groups, contrasts, frequency bands, ROI categories, and data discovery settings.

```yaml
name: "My Study"
output_dir: "/path/to/output"

groups:
  GROUP_A: "Group A Label"
  GROUP_B: "Group B Label"

group_order: [GROUP_A, GROUP_B]
group_colors:
  GROUP_A: "#3498DB"
  GROUP_B: "#E74C3C"

contrasts:
  - name: main_effect
    group_a: GROUP_A
    group_b: GROUP_B

bands:
  Delta: [1, 4]
  Theta: [4, 8]
  Alpha: [8, 13]
  Beta: [13, 30]
  Low Gamma: [30, 55]
  High Gamma: [65, 100]

roi_categories:
  Frontal:
    - Cortex_Frontal_L
    - Cortex_Frontal_R

discovery:
  root_dir: "/path/to/source_localization/output"
  group_mapping:
    "Raw Group Name": GROUP_A
```

## Input Data

source-analytics reads output files produced by the source_localization pipeline. Each subject directory contains:

**ROI-level analyses** (psd, aperiodic, connectivity, pac) — default discovery:

| File | Format | Contents |
|------|--------|----------|
| `step6_roi_timeseries_magnitude.pkl` | Python pickle | Dict[str, ndarray] -- ROI timeseries (unsigned, for PSD) |
| `step6_roi_timeseries_signed.pkl` | Python pickle | Dict[str, ndarray] -- ROI timeseries (signed, for connectivity) |
| `roi_timeseries_magnitude.set` | EEGLAB .set | Same data + metadata (sfreq) |

**Wholebrain analysis** — uses `discovery.required_files` in config:

| File | Format | Contents |
|------|--------|----------|
| `step5_stc.pkl` | Python pickle | MNE SourceEstimate (n_vertices, n_times) |
| `step3_source_coords_mm.npy` | NumPy array | Source coordinates (n_vertices, 3) in mm |

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

## Analysis Modules

### PSD (Power Spectral Density) -- Implemented

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
output_dir/psd/
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

### Aperiodic (1/f Spectral Decomposition) -- Implemented

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
output_dir/aperiodic/
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

### Connectivity (Functional Connectivity) -- Implemented

ROI-to-ROI coherence and imaginary coherence using **signed** (phase-preserving) source timeseries.

**Python side:**
- Cross-spectral density via `scipy.signal.csd` (Welch, 2s Hann, 50% overlap)
- Magnitude-squared coherence and absolute imaginary coherence for all 1035 unique ROI pairs
- Exports `connectivity_edges.csv` (subject x edge x band)

**R side:**
- **Global analysis:** Mean connectivity across all edges per subject x band; Welch t-test per band, BH FDR across bands
- **Region-pair analysis:** Edges mapped to region pairs via roi_categories, averaged within; LMM `dv ~ group * region_pair + (1|subject)`, post-hoc emmeans per region pair, Holm correction
- Connectivity matrix heatmaps, global bar charts, region-pair forest plots
- Markdown summary

**Output:**

```
output_dir/connectivity/
  ANALYSIS_SUMMARY.md
  data/
    connectivity_edges.csv          # subject x roi_pair x band (full edge data)
    study_config.yaml
  tables/
    connectivity_global.csv         # global t-tests per band x metric
    connectivity_omnibus_region_pair.csv   # LMM results (if roi_categories)
    connectivity_posthoc_region_pair.csv   # post-hoc per region pair (if significant)
  figures/
    connectivity_matrix_coherence_*.png
    connectivity_matrix_imag_coherence_*.png
    connectivity_global_bar.png
    connectivity_region_pair_forest_*.png
```

### PAC (Phase-Amplitude Coupling) -- Implemented

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
output_dir/pac/
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

### Wholebrain (Vertex-Level Spectral Analysis) -- Implemented

Vertex-level spectral analysis on shell_ellipsoid source data (154 vertices) with cluster-based permutation testing (Maris & Oostenveld, 2007). All metrics derived from a single PSD computation per subject.

**Requires a separate study config** pointing to `shell_ellipsoid/` pipeline output (not the default `roi_based_ellipsoid/`). Discovery uses `required_files` to locate `step5_stc.pkl` and `step3_source_coords_mm.npy`.

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

**Study config (`analysis_wholebrain.yaml`):**

```yaml
discovery:
  root_dir: "/path/to/source_localization/shell_ellipsoid"
  group_mapping:
    "KO ICV": KO_VEH
    "WT ICV": WT_VEH
  required_files:
    - "step5_stc.pkl"
    - "step3_source_coords_mm.npy"

wholebrain:
  cluster_threshold: 2.0
  n_permutations: 1000
  adjacency_distance_mm: 5.0
  noise_exclude_hz: [55, 65]
```

**Output:**

```
output_dir/wholebrain/
  ANALYSIS_SUMMARY.md
  data/
    wholebrain_values.csv       # subject x vertex x band (relative, absolute, dB)
    wholebrain_features.csv     # subject x vertex (fALFF, spectral slope, peak alpha)
    source_coords.csv           # vertex coordinates in mm
    wholebrain_results.pkl      # full results dict for reuse
    study_config.yaml
  tables/
    voxelwise_stats.csv         # per-vertex t, p, Hedges' g per contrast x metric
    cluster_results.csv         # cluster summaries with permutation-corrected p-values
    effect_size_summary.csv     # aggregated effect sizes (from R)
  figures/
    wholebrain_delta.png
    wholebrain_theta.png
    wholebrain_alpha.png
    wholebrain_beta.png
    wholebrain_low_gamma.png
    wholebrain_high_gamma.png
    wholebrain_falff.png
    wholebrain_spectral_slope.png
    wholebrain_peak_alpha.png
    wholebrain_summary.png
```

## Adding a New Analysis

1. Create `src/source_analytics/analyses/my_analysis.py` subclassing `BaseAnalysis`
2. Implement the lifecycle: `setup` -> `process_subject` -> `aggregate` -> `statistics` -> `figures` -> `summary`
3. Create `R/my_analysis.R` for statistics and visualization
4. Register in `core.py` `ANALYSIS_REGISTRY`
5. Update this README with the new module description

## License

MIT
