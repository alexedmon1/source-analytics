# Changelog

## v0.8.2 — 2026-09-18 (results record what produced them)

### Added

- **`provenance.json`, written beside every analysis's tables.** `_version.py`'s
  docstring said the git-describe string was stamped into run manifests and compute
  keys; nothing consumed it, so `__version__` was resolved carefully and then thrown
  away. It is now recorded where it matters — next to the numbers it describes.

  The record carries the source-analytics version and git-describe string, the
  installed plugins (marking which provided the analysis), the paradigm/profile, the
  lifecycle steps that actually ran, the subjects and group counts, and the
  **localization settings the cohort shares** — source-localization version, preset,
  atlas, BEM, source space, spacing, inverse method, orientation, sampling mode, and
  the Monte Carlo draw parameters — carried forward from each subject's
  `config_resolved.yaml`. A different atlas, inverse or sampling mode is a different
  measurement, and a CSV cannot say which it was.

  - **Monte Carlo parcel caveats are copied in**, so the parcels a run rarely sampled
    or could not separate from a neighbour stay attached to the tables rather than
    only appearing in the log that produced them.
  - **Subjects localized before source-localization 0.4.2** carry no manifest and are
    counted as `n_unrecorded`, never guessed at.
  - **It cannot fail a run.** Written last, after the numbers exist; assembling and
    writing both swallow errors. `provenance.read_provenance(tbl_dir)` returns `None`
    when it is absent or corrupt.
  - It is `.json`, so source-lightbox's `*.csv` table glob does not mistake it for a
    stats table (verified against a real gallery build).

## v0.8.1 — 2026-09-18 (the version string stops borrowing a neighbour's repo)

### Fixed

- **`source_analytics.__version__` could report another project's version.**
  `_version.py` prefers `git describe` over installed metadata, because the dev
  workflow runs from a checkout where the metadata goes stale. It derived the repo
  root as `Path(__file__).parents[2]` and ran `git -C` there — but `git -C` does not
  fail on a directory that is not a repository, it walks *up* until it finds one. For
  an installed (non-editable) package that root is `<venv>/lib/pythonX.Y`, so any
  virtualenv sitting inside a git repository made source-analytics describe *that*
  repository. Observed while verifying the v0.8.0 re-pin: the source-analytics-vertex
  venv reported the plugin's commit, `76d899d-dirty`, as the source-analytics version.

  `git describe` is now run only when `rev-parse --show-toplevel` is the derived root
  itself. A real checkout (and a git worktree, whose toplevel is the worktree root)
  still describes itself; an installed copy falls through to installed metadata, which
  is the correct answer for it. `tests/test_version.py` covers both, and the
  enclosing-repo case fails without the check.

## v0.8.0 — 2026-09-18 (the vertex analyses move to a plugin; Monte Carlo runs recognised)

### Behaviour changes (read these before upgrading)

- **The vertex analyses are no longer part of source-analytics.** vertex_cluster,
  vertex_connectivity, vertex_cross_freq, vertex_directed, vertex_evoked, vertex_graph,
  vertex_nbs, vertex_network, vertex_signature, vertex_spatial, vertex_specparam, and
  fcd_comparison (which reads vertex_connectivity's output) moved to the private
  `source-analytics-vertex` package. So did `spectral.vertex`, `spectral.vertex_aperiodic`,
  the five `R/vertex_*.R` scripts, the vertex figure-registry schemas and the glass-brain
  summary figures. A config or `--analysis` that names one of them now fails with a
  message naming the plugin. To keep running them, install the plugin, or pin v0.7.1.
  Vertex maps depend on where one set of sources sits. The ROI analyses on Monte Carlo
  source operators replace them.
- `source_analytics.analyses` no longer exports the vertex classes or their old aliases
  (`WholebrainAnalysis`, `MVPAAnalysis`, `VertexMVPAAnalysis`, `SpecparamVertexAnalysis`,
  `SpatialLMMAnalysis`). The plugin exports them.

### Added

- **Monte Carlo runs are recognised, not just tolerated.**
  source-localization 0.5.0 added `source_space.source_sampling: monte_carlo`, which
  averages the ROI operator over many sparse source draws instead of solving one
  arbitrary grid. Its parcel series were already readable here — same
  `step6_roi_timeseries_signed.pkl`, same epoch-major layout, and every remaining
  analysis asks for `signed=True` — but nothing could tell such a run apart from a
  fixed-grid one. Now:
  - `source_analytics.io.run_manifest` reads `data/config_resolved.yaml`, the manifest
    source-localization 0.4.2+ writes beside its outputs: atlas, BEM, source space,
    inverse method, orientation and sampling mode. Absent for older runs, which read as
    unknown rather than as fixed.
  - `SubjectLoader.manifest`, `.is_monte_carlo`, `.monte_carlo_report` and
    `.parcel_caveats()`.
  - **`BaseAnalysis.run` refuses a cohort that was not localized the same way.** Pooling
    a fixed-grid subject with a Monte Carlo one, or two atlases, is a group statistic
    over two different measurements, and is otherwise silent: the arrays have the same
    shape and the parcel names line up. Subjects with no manifest are skipped, not
    guessed at.
  - **The Monte Carlo parcel caveats are logged before the numbers are produced.** A run
    flags parcels whose sensor topography it cannot separate from a neighbour's, and
    parcels it sampled in under half the draws. Both yield ordinary-looking table rows,
    so the warning is repeated where someone reads it.
  - Asking a Monte Carlo run for source-level data raises `MonteCarloRunError` naming the
    method, instead of `FileNotFoundError` advising a re-run that would produce the same
    absence. No grid is solved, so there is nothing below the parcels — by construction.
- **`source-analytics list` answers "what can this install run?"**, not only "which
  analyses". `--atlases` lists the registered parcellations with their parcel counts and
  brain coverage, read from source-localization's `registry.yaml` so the listing cannot
  drift from what is selectable. `--plugins` lists installed plugins and names the
  analyses that left core. `--all` prints everything.
- `atlas.atlas_meta(name)` exposes a registry entry's descriptive `meta:` block.

### Documentation

- **The README no longer advertises the vertex analyses.** They were removed from the
  package but left in the catalog tables, the study-config example, the core-concepts
  levels, the run-in-order script and the extras table — twelve modules documented as
  available that fail at `--analysis`. It now has a "Retired: the vertex level" section
  instead, naming the plugin and the version that reproduces published results.
  `tests/test_readme_catalog.py` asserts both directions: every registered analysis is
  documented, and no retired one appears outside the section explaining the retirement.
- The handoff section documents `config_resolved.yaml`, `monte_carlo_report.json`, and
  what a Monte Carlo run does and does not carry.

- **Analysis plugins** (`source_analytics.plugins`). A package adds analyses through the
  `source_analytics.plugins` entry-point group. It provides `ANALYSES`, `METADATA` and
  `ALIASES`, and optionally `register_figures(registry)`. `source-analytics run`/`list`/
  `figure` and `analysis_meta()` pick plugin analyses up. A plugin that fails to import
  is logged and skipped. One that reuses an existing analysis name raises.

### Fixed

- **nibabel is a core dependency.** `viz/__init__` imports `viz.brain_roi`, which
  imports nibabel at module level. Without the `atlas` extra, `import
  source_analytics.core` (and so the CLI) failed, although pyproject said the
  package imports without any extra. The ROI modules' atlas readers need it anyway.
  The `atlas` extra is kept, so existing install commands still work.

### Unchanged

- Kept in core because core modules use them: `spectral.vertex_connectivity`
  (electrode_connectivity's kernels), `viz.glass_brain`, `analyses._network_base`, the
  cluster-permutation statistics, and the `BaseAnalysis` helpers the vertex modules call
  (`_vertex_epoch_config`, `_label_vertex_regions`, cluster-state persistence).
- electrode_signature still compares against a `vertex_signature` table in its own
  paradigm, if the plugin wrote one.

## v0.7.1 — 2026-09-11 (R step failures fail the run; PAC mosaics)

### Behaviour changes (read these before re-running a study)

- **A failed or timed-out R statistics step now fails the run** (exit 1). A run over a
  whole paradigm still finishes its other modules first, then lists the failures. It
  used to log the failure and exit 0: on the FORGE treatment re-run, roi_directed (all
  three source arms) and roi_cross_freq's AAC/PPC tier were killed by a one-hour limit,
  and region tables from the previous code version went on looking current.
- **R statistics steps have no default time limit.** They were hard-coded to 3600 s
  (roi_psd, roi_aperiodic, roi_connectivity, roi_directed, roi_cross_freq,
  vertex_cluster) or 600 s (electrode and evoked modules). Set `r_timeout_sec` in a
  module's config block to impose one. The three vertex modules with their own limit
  lose it too; unlike the rest they still only log a failed step, which is left for
  the vertex split.

### Fixed

- **roi_cross_freq draws its PAC mosaics.** The mosaic call named legacy columns
  (`hedges_g`, `region`, `contrast`, `freq_pair`) that the native hypothesis table
  does not have, so none was ever drawn. It now reads `effect_size` / `spatial` /
  `hypothesis` / `band`, and the figures step draws them; `summary()` only did so
  when figures were requested in the same run.

## v0.7.0 — 2026-09-10 (audit remediation, per-atlas resolution, roi_signature)

A repo audit (2026-09-04) compared the README/CLAUDE.md against the code and
found 24 defects plus a dozen false README claims. All verified and fixed here.

### Behaviour changes (read these before re-running a study)

- **Atlases resolve by name to their own files.** `pipeline.atlas` is looked up in
  source-localization's `registry.yaml`, so allen26 and allen64 no longer pick up
  allen32's labels, mapping and `roi_categories.yaml` from the shared `allen/`
  directory. For allen26 studies this changes every region-level table
  (Frontal-Anterior and Olfactory were silently dropped; Deep Subcortical was built
  from 4 of its 8 parcels) and every ROI mosaic (the six merged parcels drew blank).
  ROI-level results are unchanged, and allen32/antwerp studies resolve to the same
  files as before. An atlas name that cannot be resolved is now an error, not a guess.
- **The R region tier uses the study's categories.** Every ROI R entry point replaced
  the study's (or a profile's) `roi_categories` with the atlas-directory file whenever
  one existed. Python now hands R the effective map and R prefers it
  (`resolve_roi_categories` in `stats_utils.R`); the file is only a fallback for a
  config that carries none.
- **The 10x voxel convention is read from the NIfTI header**, as source-localization
  does, not guessed from the filename. `Atlas_3DRoisLeftRight.Labels.nii` has stored
  true units since source-localization 2026-03-12 but was still shrunk 10x on the
  default-affine path. **No statistic changes**: ROI extraction and cluster/NBS region
  labels use the raw affine, which was always right. The only visible effect is
  cosmetic: the mm axis ranges and slice labels of `plot_brain_roi_mosaic` /
  `plot_brain_roi` when drawn on Antwerp, and no analysis module draws them on Antwerp.
  (`load_vertex_roi_labels`, the other default-affine consumer, has never run: it reads
  the mapping file's top-level keys as label ids and raises on every atlas, and
  `vertex_network` swallows the error and falls back to spatial node labels. Left for
  the vertex split.)
- **`electrode_signature` compares only within its own paradigm**, preferring
  `roi_signature` over `vertex_signature`. It used to take the first
  `vertex_signature_results.csv` anywhere under the results tree, which can be a
  stale table from another run. `signature_source_vs_sensor.csv` gains a
  `source_module` column.
- **Signature fits run single-threaded** (`threadpoolctl`, installed with scikit-learn).
  A run fits a tiny model LOOCV x (1 + n_permutations) times, and a multithreaded BLAS
  spent that time synchronising threads: logistic fits ran 60-90x slower (FORGE
  electrode features: 105-128 s vs 1.4-1.7 s per 21 LOOCV passes, same accuracy).
  Results are unchanged; a signature module that took days now takes about an hour.

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

### Added

- **`roi_signature`**: ROI-level neural signature (decoding on per-parcel relative band
  power), the source-side counterpart of `electrode_signature` with the identical
  feature estimator. It needs no vertex estimate, so it runs on any ROI output,
  including Monte Carlo operators. `sensor_paradigm:` compares it against an
  `electrode_signature` run in another paradigm.
- **`resolve_atlas` / `AtlasSpec`**, and `atlas_files:` in the study config for atlases
  that are not registered. Also `header_is_inflated` and `registered_atlases`.

- **`<module>_subnetwork_edges.csv`** next to `roi_nbs_hypotheses.csv` (ROI edge modules):
  one row per supra-threshold edge of every NBS component (`hypothesis, band, dv,
  component_id, component_p, significant, node_i, node_j, roi_i, roi_j, stat`). The
  component rows only carry counts; this sidecar is the edge-level membership behind them,
  and is what `source-lightbox` now draws its connectivity circos from (the retired
  `roi_connectivity_posthoc_region_pair.csv` used to play that role). Vertex modules, whose
  nodes have no names, do not write it.
- `plot_significance_circos(sig_label=...)` names what the opaque edges are in the axis
  title (default unchanged: "region pairs p < 0.05 uncorrected").

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
