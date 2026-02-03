# CLAUDE.md — source-analytics

## What This Is

Statistical analysis toolkit for source-localized EEG data. **Python** handles signal processing (PSD, band power) and data I/O. **R** handles statistics (lme4/lmerTest LMMs, t-tests, effect sizes) and visualization (ggplot2).

## Setup

```bash
# Python
cd /home/edm9fd/sandbox/source-analytics
uv venv && source .venv/bin/activate
uv pip install -e .

# R packages (one-time)
Rscript -e 'install.packages(c("ggplot2","dplyr","tidyr","readr","stringr","lme4","lmerTest","effectsize","yaml","argparse","patchwork","scales"))'
```

## Usage

```bash
source-analytics validate --study configs/forge_fxs.yaml
source-analytics run --study configs/forge_fxs.yaml --analysis psd
source-analytics list
```

## Architecture

```
Python (orchestration + signal processing)     R (statistics + visualization)
─────────────────────────────────────────      ─────────────────────────────
1. Discovery (find subjects, load YAML)
2. Load ROI timeseries (pickle/.set)
3. Compute PSD (Welch's, scipy)
4. Extract band power (abs, rel, dB)
5. Export CSVs ──────────────────────────►  6. Read CSVs + YAML config
                                            7. LMMs (lme4/lmerTest)
                                            8. t-tests, Hedges' g, FDR
                                            9. ggplot2 figures
                                            10. Markdown summary
```

### Python modules (`src/source_analytics/`)
- `cli.py` — CLI entry point
- `core.py` — StudyAnalyzer orchestrator
- `config.py` — YAML study config loader
- `io/` — Data loading (pkl/npy/.set readers, subject discovery)
- `spectral/` — PSD and band power extraction
- `analyses/` — BaseAnalysis ABC + PSD module (exports CSV, calls Rscript)

### R scripts (`R/`)
- `psd_analysis.R` — Main entry point (called by Python)
- `stats_utils.R` — LMMs, t-tests, effect sizes, FDR correction
- `plot_psd.R` — ggplot2 figures (PSD curves, boxplots, heatmaps)
- `report.R` — Markdown summary writer

## CSV Interface (Python → R)

Exported to `{output_dir}/psd/data/`:
- `band_power.csv` — subject, group, roi, band, absolute, relative, dB
- `psd_curves.csv` — subject, group, roi, freq_hz, psd
- `study_config.yaml` — copy of study config for R

## R packages required

ggplot2, dplyr, tidyr, readr, stringr, lme4, lmerTest, effectsize, yaml, argparse, patchwork, scales

## Adding a New Analysis

1. Create `src/source_analytics/analyses/my_analysis.py` (Python: data extraction + CSV export)
2. Create `R/my_analysis.R` (R: statistics + visualization)
3. Register in `core.py` ANALYSIS_REGISTRY
