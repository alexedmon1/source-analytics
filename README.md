# source-analytics

Statistical analysis toolkit for source-localized EEG data. Reads pipeline output from the [source_localization](https://github.com/alexedmon1/AlexProjects) package and runs group-level analyses with publication-quality statistics and figures. Supports both ROI-level analyses (PSD, aperiodic, roi_connectivity, PAC) and whole-brain vertex-level analysis with cluster-based permutation testing.

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

**ROI-level analyses** (psd, aperiodic, roi_connectivity, pac) — default discovery:

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

### ROI Connectivity (Functional Connectivity) -- Implemented

ROI-to-ROI coherence and imaginary coherence using **signed** (phase-preserving) source timeseries.

**Python side:**
- Cross-spectral density via `scipy.signal.csd` (Welch, 2s Hann, 50% overlap)
- Magnitude-squared coherence and absolute imaginary coherence for all 1035 unique ROI pairs
- Exports `roi_connectivity_edges.csv` (subject x edge x band)

**R side:**
- **Global analysis:** Mean connectivity across all edges per subject x band; Welch t-test per band, BH FDR across bands
- **Region-pair analysis:** Edges mapped to region pairs via roi_categories, averaged within; LMM `dv ~ group * region_pair + (1|subject)`, post-hoc emmeans per region pair, Holm correction
- Connectivity matrix heatmaps, global bar charts, region-pair forest plots
- Markdown summary

**Output:**

```
output_dir/roi_connectivity/
  ANALYSIS_SUMMARY.md
  data/
    roi_connectivity_edges.csv      # subject x roi_pair x band (full edge data)
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

#### TFCE Correction Option

The wholebrain analysis supports TFCE (Smith & Nichols, 2009) as an alternative to cluster-based permutation testing. Set `correction_method: tfce` in the wholebrain config section. TFCE eliminates the arbitrary cluster-forming threshold by integrating cluster extent and height across all thresholds: `TFCE(v) = sum_h { e(h)^E * h^H * dh }`. When using TFCE, additional output includes `tfce_scores_*.png` glass brains and per-vertex corrected p-values in `voxelwise_stats.csv`.

### Vertex Connectivity (Functional Connectivity Density) -- Implemented

All-to-all imaginary coherence between 154 vertices, deriving Functional Connectivity Density (FCD) maps showing how connected each vertex is to the rest of the brain.

**Python side:**
- CSD-based imaginary coherence for all 11,781 unique vertex pairs per band
- FCD: fraction of connections above threshold per vertex
- Cluster-based permutation testing on FCD maps
- Glass brain FCD visualizations

**R side:**
- FCD summary by group, cluster statistics
- ANALYSIS_SUMMARY.md

**Config:**
```yaml
vertex_connectivity:
  metric: imag_coherence
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

### Specparam Vertex (Vertex-Level Spectral Parameterization) -- Implemented

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
specparam_vertex:
  freq_range: [1, 100]
  peak_width_limits: [1.0, 12.0]
  max_n_peaks: 6
```

**Output:**
```
output_dir/specparam_vertex/
  ANALYSIS_SUMMARY.md
  data/specparam_vertex.csv, source_coords.csv
  tables/specparam_vertex_stats.csv, gamma_peak_chi2.csv
  figures/specparam_*.png, gamma_peak_presence.png
```

### MVPA (Multivariate Pattern Analysis) -- Implemented

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
mvpa:
  classifier: svm_linear
  cv_method: loocv
  n_permutations: 1000
```

**Output:**
```
output_dir/mvpa/
  ANALYSIS_SUMMARY.md
  data/mvpa_features.csv, source_coords.csv
  tables/mvpa_results.csv
  figures/mvpa_importance_*.png, mvpa_null_*.png, mvpa_confusion_*.png
```

### Network (Graph-Theoretic Analysis + NBS) -- Implemented

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

### Spatial LMM (Spatial Mixed Effects Models) -- Implemented

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
spatial_lmm:
  correlation_structure: exponential
  spatial_range_mm: 3.0
```

**Output:**
```
output_dir/spatial_lmm/
  ANALYSIS_SUMMARY.md
  data/spatial_lmm_data.csv, source_coords.csv
  tables/spatial_lmm_results.csv, spatial_residuals.csv
  figures/variogram_*.png, spatial_residuals_*.png
```

### Cross-Cutting: Random Epoch Sampling

All wholebrain-based analyses support optional random epoch sampling. Instead of computing PSD/connectivity on full continuous recordings, randomly sample non-overlapping epochs of fixed duration. Enable in the wholebrain config section:

```yaml
wholebrain:
  epoch_sampling:
    enabled: true
    epoch_duration_sec: 2.0
    n_epochs: 80
    seed: 42
```

When enabled, PSD is computed per-epoch then averaged (more robust spectral estimate). Connectivity is computed per-epoch then averaged (standard approach in connectivity literature).

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
