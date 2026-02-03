# CLAUDE.md — source-analytics

## What This Is

General-purpose statistical analysis toolkit for source-localized EEG data. Reads output files from the `source_localization` pipeline (pickle, numpy) and runs group-level analyses with publication-quality figures.

## Setup

```bash
cd /home/edm9fd/sandbox/source-analytics
uv venv && source .venv/bin/activate
uv pip install -e .
source-analytics --help
```

## Usage

```bash
# Validate study config
source-analytics validate --study configs/forge_fxs.yaml

# Run PSD analysis
source-analytics run --study configs/forge_fxs.yaml --analysis psd

# List available analyses
source-analytics list
```

## Architecture

```
src/source_analytics/
├── cli.py              # CLI entry point
├── core.py             # StudyAnalyzer orchestrator
├── config.py           # YAML study config loader
├── io/                 # Data loading (pkl/npy readers, subject discovery)
├── spectral/           # PSD and band power extraction
├── stats/              # t-tests, LMMs, effect sizes, FDR correction
├── viz/                # Publication figures and markdown reports
└── analyses/           # Analysis modules (BaseAnalysis ABC + implementations)
```

## Adding a New Analysis

1. Create `src/source_analytics/analyses/my_analysis.py`
2. Subclass `BaseAnalysis` and implement: `setup`, `process_subject`, `aggregate`, `statistics`, `figures`, `summary`
3. Register in `core.py` ANALYSIS_REGISTRY
4. Run: `source-analytics run --study configs/my_study.yaml --analysis my_analysis`

## Key Design Decisions

- **No dependency on source_localization** — only reads its output files
- **YAML-driven** — groups, contrasts, bands, ROI categories all in config
- **Subject-level effect sizes** — averages across ROIs per subject before Cohen's d (avoids pseudoreplication)
- **LMMs via statsmodels** — `power ~ group + (1|subject)`
- **FDR correction** — Benjamini-Hochberg across bands within each contrast

## Pipeline Output Files Consumed

From each subject's `data/` directory:
- `step6_roi_timeseries_magnitude.pkl` — Dict[str, ndarray(n_times,)]
- `step1_info.pkl` — MNE Info (for sfreq)

## Dependencies

numpy, scipy, pandas, matplotlib, seaborn, statsmodels, pyyaml
