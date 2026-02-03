# Source Analytics — Project Status

**Last Updated:** 2026-02-03
**Version:** 0.2.0
**Status:** Production Ready (PSD Analysis Module)

---

## Current State

### Package Architecture: Python + R

| Layer | Language | Components | Status |
|-------|----------|------------|--------|
| **Orchestration** | Python | CLI, config, discovery, analysis lifecycle | Complete |
| **Signal Processing** | Python | Welch PSD (scipy), band power extraction | Complete |
| **Data I/O** | Python | Pickle/numpy/.set readers, subject discovery | Complete |
| **Statistics** | R | lme4/lmerTest LMMs, t-tests, Hedges' g, BH FDR | Complete |
| **Visualization** | R | ggplot2 figures, regional heatmaps | Complete |
| **Reporting** | R | Markdown summary with methods + results | Complete |

### Analysis Modules

| Module | Status | Description |
|--------|--------|-------------|
| **PSD** | Complete | Power spectral density, band power, group comparisons |
| Connectivity | Planned | ROI-to-ROI coherence, phase-lag index |
| Aperiodic | Planned | 1/f decomposition (specparam) |
| Cross-frequency coupling | Planned | Phase-amplitude coupling |
| Treatment effects | Planned | Multi-group comparisons with contrasts |

---

## Pipeline Data Consumed

Source-analytics reads output files from the `source_localization` pipeline (AlexProjects monorepo).

### Input Files (per subject)

| File | Format | Contents |
|------|--------|----------|
| `step6_roi_timeseries_magnitude.pkl` | Python pickle | Dict[str, ndarray] — 46 ROIs, shape (n_times,) each |
| `step6_roi_timeseries_signed.pkl` | Python pickle | Same with sign preservation (for connectivity) |
| `roi_timeseries_magnitude.set` | EEGLAB MAT | Same data + metadata (sfreq, channel names) |
| `step1_info.pkl` | MNE Info pickle | Electrode info, sfreq (requires MNE to read) |

### Data Specifications

| Parameter | Value | Source |
|-----------|-------|--------|
| ROI count | 46 | Antwerp mouse brain atlas (excluding Exterior) |
| Electrode channels | 32 | NeuroNexus array |
| Sampling rate | 500 Hz | Read from .set file header |
| Coordinate system | mm | MRI-space millimeters |
| ROI timecourse dtype | float32 | Continuous magnitude signal |

### Study Directory Layout

```
root_dir/
├── Group_A/
│   ├── Subject_001/
│   │   └── data/
│   │       ├── step6_roi_timeseries_magnitude.pkl
│   │       ├── roi_timeseries_magnitude.set
│   │       └── ...
│   └── Subject_002/data/...
└── Group_B/
    └── ...
```

---

## FORGE FXS Study Validation

**Dataset:** 61 subjects, 5 treatment groups
**Contrast tested:** KO Vehicle (n=14) vs WT Vehicle (n=14)

### PSD Analysis Results

| Band | t-stat | p-value | q-value (FDR) | Hedges' g | LMM p |
|------|--------|---------|---------------|-----------|-------|
| Delta | -0.51 | 0.617 | 0.934 | -0.19 | 0.617 |
| Theta | -0.22 | 0.829 | 0.934 | -0.08 | 0.829 |
| Alpha | 0.16 | 0.874 | 0.934 | 0.06 | 0.874 |
| Beta | -0.26 | 0.800 | 0.934 | -0.09 | 0.800 |
| Low Gamma | 0.60 | 0.552 | 0.934 | 0.22 | 0.552 |
| High Gamma | 0.08 | 0.934 | 0.934 | 0.03 | 0.934 |

No bands reached significance after FDR correction (relative power, subject-level means).

### Output Files Produced

```
output_dir/psd/
├── ANALYSIS_SUMMARY.md          # Methods, statistics table, key findings
├── data/
│   ├── band_power.csv           # 7,728 rows (28 subj × 46 ROI × 6 bands)
│   ├── psd_curves.csv           # 283,360 rows (per-subject PSDs)
│   └── study_config.yaml        # Config copy for R
├── tables/
│   └── psd_statistics.csv       # t-tests, LMMs, effect sizes, FDR q-values
└── figures/
    ├── psd_by_region.png        # PSD curves faceted by region (ggplot2)
    ├── band_power_relative.png  # Boxplots by band (ggplot2)
    ├── band_power_absolute.png
    ├── band_power_dB.png
    ├── heatmap_relative_WT_VEH.png
    └── heatmap_relative_KO_VEH.png
```

---

## Dependencies

### Python
- numpy, scipy, pandas, pyyaml
- Optional: mne (for reading step1_info.pkl directly)

### R (CRAN)
- ggplot2, dplyr, tidyr, readr, stringr
- lme4, lmerTest (linear mixed models)
- effectsize (Hedges' g)
- yaml, argparse, patchwork, scales

---

## Known Issues & Limitations

### Resolved
- ~~MNE dependency for sfreq~~ → Reads from .set file via scipy.io.loadmat
- ~~Duplicate subject IDs across groups~~ → Composite keys (group_subject)
- ~~ROI name mismatch in config~~ → Updated analysis.yaml with actual atlas names
- ~~lme4 Matrix ABI mismatch~~ → Reinstalled lme4 from source
- ~~Incorrect sfreq in R report~~ → Python exports sfreq in study_config.yaml

### Current Limitations
1. **Single analysis module** — Only PSD implemented; connectivity, aperiodic planned
2. **No parallel subject processing** — Subjects processed sequentially in Python
3. **R package loading noise** — lme4/lmerTest print attachment messages to stderr
4. **Band order in YAML** — R reads YAML alphabetically; band order in plots follows YAML key order

---

## Next Steps

### Immediate
1. Add treatment group contrasts to FORGE config (HD-ICV, HD-IV, LD-IV/ICV vs KO Vehicle)
2. Add absolute power analysis alongside relative
3. Consider ROI-level LMMs: `power ~ group + (1|subject/roi)`

### Short-term
4. Connectivity analysis module (coherence, PLI)
5. Aperiodic analysis module (specparam/FOOOF)
6. Parallel subject processing

### Long-term
7. Cross-frequency coupling module
8. Whole-brain voxel-wise analysis (shell source space)
9. R package formalization (DESCRIPTION, NAMESPACE, roxygen2)
