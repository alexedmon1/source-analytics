---
author: Alex Edmondson
affiliation: CCHMC
email: alex.edmondson@cchmc.org
study: {{STUDY_NAME}}
project_name: {{PROJECT_NAME}}        # e.g. "group-psd", "gamma-connectivity"
project_slug: {{PROJECT_SLUG}}        # filesystem-safe: matches project_dir leaf
phase: analysis
project_dir: ~/research/{{STUDY_NAME}}/{{PROJECT_SLUG}}
study_dir: /mnt/arborea/{{STUDY_NAME}}
sl_deriv: /mnt/arborea/{{STUDY_NAME}}/derivatives/source_localization/{{SL_PRESET}}
pipeline_dir: /home/edm9fd/sandbox/source-analytics
pipeline_venv: /home/edm9fd/sandbox/source-analytics/.venv/bin/python
---

<!-- AI Instructions:
Analysis-phase IRL project that consumes source-localization output.
- Source localization is DONE — $SL_DERIV is populated and maintained by a source-localization project
- $PROJECT (this repo) is small, git-tracked; holds study configs (analysis.yaml) + result writeups
- $STUDY on /mnt/arborea holds big data, SHARED across projects
- This project's outputs go to $ANALYTICS_OUT (namespaced by project slug)
- Result summaries (markdown, small figures, small tables) live in $RESULTS inside $PROJECT
- source-analytics is invoked via the `source-analytics` CLI (installed in $PIPELINE/.venv) or `uv run source-analytics ...` from $PIPELINE
- Python calls Rscript automatically — no manual R interaction needed
- Long jobs (permutation NBS, cluster permutation on vertex analyses) MUST use nohup, logs to $STUDY/logs
- Before launching CPU-heavy jobs: `ps aux | grep -E 'source-analytics|Rscript'`
-->

# {{PROJECT_NAME}} ({{STUDY_NAME}}) — Source Analytics Plan

## 📁 Paths — Single source of truth

### `$PROJECT` — this IRL project (small, git-tracked on home drive)

- **`$PROJECT`** — `~/research/{{STUDY_NAME}}/{{PROJECT_SLUG}}` — this repo root
- **`$PLAN`** — `$PROJECT/plans` — main-plan.md, activity log, CSV log
- **`$CONFIGS`** — `$PROJECT/configs` — `analysis.yaml` and vertex-level configs (authoritative copies)
- **`$RESULTS`** — `$PROJECT/results` — markdown summaries, selected figures, small tables copied from outputs
- **`$SCRIPTS`** — `$PROJECT/scripts` — wrappers, helper scripts specific to this analysis

### `$STUDY` — study data (big, on arborea, shared across projects)

- **`$STUDY`** — `/mnt/arborea/{{STUDY_NAME}}`
- **`$SL_DERIV`** — `$STUDY/derivatives/source_localization/{{SL_PRESET}}` — upstream source-localization output (read-only from this project's POV)
- **`$ANALYTICS_OUT`** — `$STUDY/analytics/{{PROJECT_SLUG}}` — full analysis outputs, namespaced to THIS project
- **`$SHELL_DERIV`** — `$STUDY/derivatives/source_localization/shell_ellipsoid` — used by vertex-level analyses (if in scope)
- **`$ELECTRODE_ROSTER`** — `$STUDY/electrode/subject_roster.csv` — only if electrode-level analyses in scope
- **`$LOGS`** — `$STUDY/logs`

### Pipeline (read-only)

- **`$PIPELINE`** — `/home/edm9fd/sandbox/source-analytics` — source-analytics repo
- **`$SA`** — `source-analytics` CLI; invoke as `uv run source-analytics ...` from `$PIPELINE`

Rule: every section below refers to these by shorthand. Add new paths here before using them.

---

## 🔧 First Time Setup — Run once when starting this analysis

<!-- 👤 AUTHOR AREA: Fill in scope before first loop -->

### Research question
<!-- 2–4 sentences: what does this project investigate? What's the hypothesis? -->

### In-scope
- **Analysis levels:** <!-- ROI (46 ROIs) | vertex (154 vertices) | electrode (raw EEG) — pick subset -->
- **ROI modules:** <!-- roi_psd, roi_aperiodic, roi_connectivity, roi_transfer_entropy, roi_pac -->
- **Vertex modules:** <!-- vertex_cluster, vertex_connectivity, vertex_specparam, vertex_mvpa, network, vertex_spatial -->
- **Electrode modules:** <!-- electrode, electrode_comparison -->
- **Groups / contrasts:** <!-- e.g. KO vs WT -->
- **Frequency bands:** <!-- Delta / Theta / Alpha / Beta / Low Gamma / High Gamma -->

### Out-of-scope
<!-- What this project will NOT touch; prevents scope creep -->

### Verify upstream source-localization is complete
```bash
ls $SL_DERIV | wc -l                                           # subject count
ls $SL_DERIV/*/roi_timeseries/step6_roi_timeseries_magnitude.pkl | wc -l  # ROI ts present
# vertex-level analyses additionally need shell_ellipsoid outputs:
ls $SHELL_DERIV/*/pipeline/step5_stc.pkl 2>/dev/null | wc -l
ls $SHELL_DERIV/*/pipeline/step3_source_coords_mm.npy 2>/dev/null | wc -l
```
If missing, redirect to the source-localization project.

### Build study configs
```bash
mkdir -p $CONFIGS $ANALYTICS_OUT
# Primary ROI-level analysis.yaml:
#   output_dir: $ANALYTICS_OUT
#   groups / group_order / group_colors / contrasts
#   bands (Delta..High Gamma)
#   roi_categories (optional, but needed for region-pair analyses)
#   discovery.root_dir: $SL_DERIV
#   discovery.group_mapping: raw folder name -> group key
# Vertex-level (if in scope): analysis_vertex_cluster.yaml (or similar) with
#   discovery.root_dir: $SHELL_DERIV
#   discovery.required_files: [step5_stc.pkl, step3_source_coords_mm.npy]
#   vertex.correction_method (cluster | tfce), cluster_threshold, n_permutations
```

### Validate config + data discovery
```bash
cd $PIPELINE
uv run source-analytics validate --study $CONFIGS/analysis.yaml
uv run source-analytics list
```

### Common skill library
<!-- Uncomment to use -->
<!-- Install Scientific Writing: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/scientific-writing -->
<!-- Install PubMed Search: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/pubmed-database -->
<!-- Install PPTX Posters: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/pptx-posters -->

---

## ✅ Before Each Loop

- **Clean git tree** in `$PROJECT`: `git status`
- **Running-jobs check**: `ps aux | grep -E 'source-analytics|Rscript'`
- **Disk check**: `df -h /mnt/arborea`
- **Upstream freshness**: `ls -la $SL_DERIV/group/*.csv 2>/dev/null` — changes since last loop?
- **Pipeline version**: `cd $PIPELINE && git log -1` — record hash
- Re-running an analysis module must be **idempotent** (output dir is overwritten; no partial state)
- Only `## One-Time Instructions` is plan-editable without explicit permission

---

## 🔁 Instruction Loop — Define the work for each iteration

<!-- 👤 AUTHOR AREA: Edit each loop. -->

### Loop task (current)

- **Module:** <!-- roi_psd | roi_aperiodic | roi_connectivity | roi_transfer_entropy | roi_pac | vertex_cluster | vertex_connectivity | vertex_specparam | vertex_mvpa | network | vertex_spatial | electrode | electrode_comparison -->
- **Config file:** <!-- $CONFIGS/analysis.yaml or $CONFIGS/analysis_vertex_cluster.yaml -->
- **Contrast:** <!-- name from config -->
- **Bands:** <!-- subset or all -->
- **Output dir:** `$ANALYTICS_OUT/{module}`

### Command templates

**Run a single analysis module:**
```bash
cd $PIPELINE
nohup uv run source-analytics run \
    --study $CONFIGS/analysis.yaml \
    --analysis {{MODULE}} \
    > $LOGS/sa_{{MODULE}}_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Vertex-level module (separate config with `discovery.required_files`):**
```bash
cd $PIPELINE
nohup uv run source-analytics run \
    --study $CONFIGS/analysis_vertex_cluster.yaml \
    --analysis vertex_cluster \
    > $LOGS/sa_vertex_cluster_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Electrode-level module (requires `electrode.subject_roster` in config):**
```bash
cd $PIPELINE
uv run source-analytics run \
    --study $CONFIGS/analysis_electrode.yaml \
    --analysis electrode
```

**Electrode vs source comparison** (requires `electrode` and `roi_psd` already run):
```bash
cd $PIPELINE
uv run source-analytics run \
    --study $CONFIGS/analysis.yaml \
    --analysis electrode_comparison
```

**Validate/list modules (non-destructive):**
```bash
cd $PIPELINE
uv run source-analytics validate --study $CONFIGS/analysis.yaml
uv run source-analytics list
```

### One-Time Instructions — Tasks that should only execute once

<!-- 👤 AUTHOR AREA: Add tasks. Move to Completed once done. -->

- [ ] Verify upstream `$SL_DERIV` completeness (+ `$SHELL_DERIV` if vertex-level)
- [ ] Author `$CONFIGS/analysis.yaml` (groups, contrasts, bands, roi_categories, discovery)
- [ ] Author `$CONFIGS/analysis_vertex_*.yaml` if vertex-level in scope
- [ ] `source-analytics validate` on each config
- [ ] Draft preregistration / analysis plan summary in `$RESULTS/00_analysis_plan.md`
- [ ] First pilot run of primary module (e.g. `roi_psd`) on a sample subset

#### Completed (don't re-run)
<!-- Move checked items here with date -->

### Formatting Guidelines

- **Result summaries** → `$RESULTS/{module}.md` reading from `$ANALYTICS_OUT/{module}/ANALYSIS_SUMMARY.md`: design, N per group, exclusions, omnibus result, post-hoc highlights (q<0.05), figure refs
- **Figures** → copy key figures to `$RESULTS/figures/` (PNG/SVG); never inline binary blobs or symlink
- **Tables** → copy small CSVs to `$RESULTS/tables/`; render markdown summary tables in the `.md`
- **Large outputs** (cluster permutation pkl, full CSV matrices) stay in `$ANALYTICS_OUT`, not in `$PROJECT`
- **Paths in reports** — always shorthand from `## Paths` or absolute; never `../../`
- **Number formatting** — p-values 3 sig figs, Hedges' g 2 decimals, N as integer

---

## 📝 After Each Loop

- **Update activity log** (`$PLAN/main-plan-activity.md`, append 1–2 lines):
  - Module, config file, contrast, output path
  - Timestamp (UTC), `$PROJECT` hash, `$PIPELINE` hash
  - Any modifications to configs since last run

- **Update plan log** (`$PLAN/main-plan-log.csv`):
  `timestamp,module,config,contrast,n_subjects,output_dir,status,project_hash,pipeline_hash`

- **Commit `$PROJECT`** — plan edits, config files, new result writeups, selected figures/tables, scripts
  - Never commit anything from `$STUDY`; only `$PROJECT` is under this repo's git
  - Message format: `{module}: {one-line result}` (e.g. `roi_psd: KO>WT gamma in frontal + parietal`)

- **Feedback to AUTHOR**:
  1. What was done, omnibus + key post-hoc, next steps
  2. Idempotency or stale `## One-Time Instructions` issues
  3. Critical reasoning/stats concerns (interpretation, FDR choice, outliers)
  4. Pipeline quirks worth filing upstream in source-analytics

---

## 📚 Skill Library — Community skills (optional)
<!-- Uncomment to use -->
<!-- Install Scientific Writing -->
<!-- Install BioRx Search -->
<!-- Install Flowcharts -->

---

## 📌 Study-specific conventions

### Analysis module map (what reads what)

| Level | Modules | Input files |
|-------|---------|-------------|
| ROI | `roi_psd`, `roi_aperiodic`, `roi_connectivity`, `roi_transfer_entropy`, `roi_pac` | `step6_roi_timeseries_{magnitude,signed}.pkl` |
| Vertex | `vertex_cluster`, `vertex_connectivity`, `vertex_specparam`, `vertex_mvpa`, `network`, `vertex_spatial` | `step5_stc.pkl` + `step3_source_coords_mm.npy` |
| Electrode | `electrode`, `electrode_comparison` | Raw `.set/.fdt` via `subject_roster.csv` |

### Config versioning
- `$CONFIGS/*.yaml` are authoritative and committed in `$PROJECT`
- Every run snapshots the config into `$ANALYTICS_OUT/{module}/data/study_config.yaml` — cross-check against `$PROJECT` when reviewing results

### Output namespacing
All outputs that this project writes to `$STUDY` go under `{{PROJECT_SLUG}}`:
- `$STUDY/analytics/{{PROJECT_SLUG}}/{module}/...`

This prevents collisions with other analytics projects on the same study.

### Stats conventions
- Omnibus LMMs via lme4/lmerTest, Type III ANOVA with Satterthwaite DoF
- Post-hoc gated on significant omnibus; emmeans pairwise contrasts
- FDR (Benjamini-Hochberg) across bands; Holm correction across ROIs/region-pairs
- Hedges' g effect sizes from emmean difference / residual SD
- Vertex-level: cluster-based permutation (default) or TFCE (Smith & Nichols 2009) — choice recorded in config
