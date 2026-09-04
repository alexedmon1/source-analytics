# Changelog

## Unreleased — 2026-09 audit remediation

A repo audit (2026-09-04) compared the README/CLAUDE.md against the code and
found 24 defects plus a dozen false README claims. All verified and fixed here.

### Behaviour changes (read these before re-running a study)

- **Vertex `absolute` band power is now a density (dB/Hz)**, `10*log10(integral / bandwidth)`,
  matching the ROI/electrode definition. Previously `vertex_cluster` / `vertex_specparam`
  reported `10*log10(integral)`. Within-band group statistics are unaffected (a per-band
  constant shift); absolute values and their plots move by `10*log10(bandwidth)` per band.
- **Vertex modules now honour the top-level `epoch_sampling:` block and per-analysis
  overrides** (precedence: global → `vertex.epoch_sampling` → analysis block). Previously only
  the `vertex:` block was read. `vertex_cluster` now epoch-samples when the merged config
  enables it (it never did before). `n_bootstrap: 0` = full timeseries on the vertex sampler
  too (it used to fall through to a single random draw).
- **`--jobs`**: an explicit CLI value (including `1`) wins over the YAML `jobs:`; when omitted
  the YAML value is used. `--jobs 0` / `-1` now actually auto-parallelize (they were coerced
  to `1`). `roi_connectivity` and `electrode_connectivity` are parallel-capable too.
- **`--force` now removes previous output** (published `tables/` + `figures/` always; the
  working `data/` when the `process` step runs). It used to only bypass `--strict-output`.
- **`vertex_spatial` is retired in Python as well**: no subjects are loaded and R is not
  called; it writes empty result tables + a note. (R already did this after Python had
  processed every subject.)
- **`roi_connectivity` reads a `metrics:` list** from its config block (like
  `vertex_connectivity`). Unknown names error.
- **`roi_directed` hypothesis tables are renamed** to the canonical prefix
  (`roi_directed_{global,directed_edges,region}_hypotheses.csv`, `roi_directed_omnibus_lmm.csv`,
  `roi_directed_global_bar.png`) and cover DTF as well as TE; DTF-only runs no longer skip R.
- **`fcd_comparison` finds its two primaries across paradigm dirs** (they normally live under
  `resting` and `vertex`); `sensor_dir` / `source_dir` overrides are accepted.
- Deprecated analysis names print/check the **canonical** output directory.

### Fixed

- Evoked R scripts (`roi_evoked`, `electrode_evoked`) looped `config$contrasts`, which is NULL
  under `design:`/`hypotheses:`, so their LMM/post-hoc tables came out empty. They now derive
  contrasts from the design spec like `roi_psd_analysis.R`.
- `vertex_evoked` joins the hypothesis contract: `--hypothesis` is accepted and
  `vertex_evoked_hypotheses.csv` is written via the permutation adapter; `list` shows an
  "Evoked Response (Vertex Level)" heading.
- AAC / PPC from `roi_cross_freq` now get hypothesis statistics
  (`R/roi_cross_freq_edges_analysis.R`: `roi_cross_freq_{aac,ppc}_{global,directed_edges,region}_hypotheses.csv`);
  the old log claimed they ran "via the gating path".
- `R/roi_connectivity_analysis.R` no longer requires coherence columns; it adapts to the
  metric columns present (`--metric aec` alone works).
- `aggregate_to_regions()` keeps `delta_ref`, so region-level hypotheses work for that DV.
- The package imports without `mne` (lazy TFR import with a clear ImportError); `matplotlib`
  is a declared core dependency; extras are documented.
- R scripts are packaged into wheels/sdists (`share/source-analytics/R`); `find_r_script_dir()`
  checks that location and honours `SOURCE_ANALYTICS_R_DIR`.
- `init` writes a config that parses as-is: `design:`/`hypotheses:` (omnibus + pairwise
  contrasts), canonical bands, one `paradigms:` block, both discovery layouts; `--output -`
  streams YAML to stdout. It no longer emits the legacy `contrasts:` form.
- R-timeout log messages now report the real timeout (they said 600 s for 3600 s runs).
- `ANALYSIS_METADATA` `about` text: aperiodic default window is 12–45 Hz (not 2–50); PSD
  `absolute` is described as dB/Hz density; `electrode_comparison` / `fcd_comparison` carry
  `supplements` + `requires`.
- Helper-script `source()` calls in the R modules are hard failures again (were silently
  swallowed by `tryCatch`).
- Removed the orphaned `R/network_analysis.R` and the dead lookups for non-existent
  `roi_network_analysis.R` / `vertex_network_analysis.R`; removed the stale
  `run_connectivity_network.sh`; added `scripts/run_study.sh` (dependency-ordered recipe).

### Docs

- README synced to the code: `init` behaviour, figures off by default, real analytics/results
  trees (paradigm + profile segments), install extras, R package list (`ggsignif`, `optparse`;
  `stringr` dropped), `vertex_signature` naming, `fcd_comparison` / `electrode_signature` in
  the catalog, `electrode_comparison` needing `roi_psd`, `--jobs` / `--profile` /
  `--paradigm` semantics, no signed-STC fallback, no `dB` column, referenced-group subject
  filtering, `circos_metrics` passthrough, epoch-sampling defaults.
- CLAUDE.md rewritten (it described the original PSD-only package).

## v0.6.0 — 2026-08

- `fcd_comparison`: source-vs-sensor functional-connectivity-density comparison module.
- Evoked build-out: ERP amplitude/latency measures, induced power, debiased ITC, cycle ramps and
  tiled extraction, wired into all three evoked modules; declared hypotheses for `roi_evoked` /
  `electrode_evoked` with the measure as the FDR facet.
- `vertex_specparam`: two-fit peak detection, fit-window diagnostic, persisted `offset_centered`.
- Aperiodic default fit window 12–45 Hz (cited in `docs/methods/APERIODIC_FIT_WINDOW.md`);
  `vertex_specparam` no longer fits through the line-noise notch.
- `electrode_signature` module; `vertex_mvpa` renamed to `vertex_signature` (multi-model neural
  signature, true AUC, valid permutation p, balanced accuracy).
- `delta_ref` (delta-referenced power) DV for `roi_psd` / `electrode_psd` under a profile.
- `--profile` runs via `StudyConfig.for_profile()` writing to `analytics/<profile>/` and
  `results/<profile>/`; profile-narrowed hypotheses forwarded to R.
- `--jobs` parallel per-subject processing for the vertex modules, `roi_connectivity` and
  `electrode_connectivity`; precomputed-connectivity cache for `vertex_graph`.
- Fully-qualified `fdr_family` (member-set identity); canonical low→high band order everywhere.
- ROI PSD `absolute` switched to power density (dB/Hz) with a restricted relative-power range.
- Figures: module figure dir cleared before regeneration; figures regenerable from persisted
  data; effect-size mosaics; anatomical-coverage labels for significant clusters; NBS
  subnetwork figures; circos polish.
- NBS significant-edge mask is now filled (it was allocated and never written).
- Packaging: `specparam` floor made resolvable (`>=2.0.0rc6`); version single-sourced from git.

## v0.5.0 — 2026-06

- Declarative hypothesis layer: `design:` / `hypotheses:` (kinds `omnibus`, `contrast`,
  `regression`, `equivalence`) with declarative FDR family scope and method; emmeans (R)
  tabular adapter, Python permutation (map + cluster) adapter, edge/NBS adapter and
  directed-edge adapter; `--hypothesis NAME` selection. Auto-gating retired.
- `config.contrasts` derived from the design spec (stored bridge removed); a legacy
  `contrasts:` block is lifted into the spec.
- Module renames: `roi_pac` → `roi_cross_freq` (PAC + AAC + PPC), `roi_transfer_entropy` →
  `roi_directed` (TE + DTF); connectivity network split into `*_graph` + `*_nbs`.
- New modules: `vertex_cross_freq`, `vertex_directed` (ridge-MVAR DTF), `vertex_evoked`,
  `electrode_connectivity` (source-vs-sensor FC comparator).
- New kernels: wPLI, dPLI, AAC, n:m PPC with surrogate significance, Hipp-2012 AEC
  (vectorized), directed-aware FCD.
- Per-sub-output selection: `--metric` / `--band` / `--select`.
- `vertex_spatial` GLS statistics retired (R side).
- `ANALYSIS_METADATA` domains + `supplements`; MIT license; README rewritten around the
  source-localization handoff.

## v0.4.0 — 2026-04-25

### Compatibility

- Bump for `source-localization` v0.2.0 Allen32 ROI rename. Four composite ROI labels were renamed upstream for nomenclature accuracy:

| Identifier (v0.2.0) | Legacy alias | Reason |
|---|---|---|
| `Frontal_Anterior_{L,R}` | `Prefrontal_mPFC_{L,R}` | Composite includes ORB and FRP areas, which are lateral / ventral, not medial — the "mPFC" qualifier was misleading. |
| `Basal_Ganglia_{L,R}` | `Striatum_{L,R}` | Composite includes pallidum (globus pallidus), which is anatomically distinct from striatum. |
| `Amygdalar_Complex_{L,R}` | `Amygdala_{L,R}` | Composite includes claustrum and endopiriform nucleus, which are adjacent to but distinct from amygdala. |
| `Brainstem_Tectum_{L,R}` | `Brainstem_{L,R}` | Composite includes superior and inferior colliculi (dorsal-midbrain tectum), not just brainstem proper. |

10-region category `Prefrontal` → `Frontal-Anterior`. Region membership and label IDs (1–32) are unchanged — these are nomenclature clarifications only and have no effect on numerical results.

### No code changes

`source-analytics` consumes ROI labels dynamically from upstream YAML/JSON; no string-literal references to the renamed labels exist in this codebase. This bump documents the compatibility cut so downstream pinning has a clear boundary. Pre-v0.2.0 derivatives (e.g., the FORGE `ms1-v6-frozen` Zenodo deposit) can be read transparently via the `deprecated_aliases` block added in `source-localization` v0.2.0 / `roi_categories.yaml`.

## v0.3.0 — earlier

(no changelog entries before v0.4.0)
