# Analysis grouping & primary/secondary model

Branch: **`analysis-grouping`** (mirrored in source-analytics + source-lightbox).
We run everything from this branch until the manuscript is accepted, then merge to
`main`. Both repos are editable installs, so checking out this branch is enough for
the venv to use it.

## Why

The gallery listed ~12 separate analysis pages, and the relationship between
analyses was invisible — e.g. the **network** analysis (NBS + graph metrics) is
computed *on* the **connectivity** output but appeared as an unrelated page. We
want related analyses to come together, and we want it obvious — in both the
analysis engine (source-analytics) and the gallery — which analyses are primary
and which are secondary.

## The model (locked)

Two orthogonal concepts:

- **Domain** — where an analysis is *listed/grouped* (by the data it uses):
  **Spectral**, **Connectivity**, **Cross-frequency**, **Sensor-level**.
- **`supplements`** — a *secondary* flag: an analysis that can **only run after a
  primary finishes**, because it consumes the primary's output. Verified by code:
  - `roi_network` reads `roi_connectivity`'s edges → secondary → `roi_connectivity`.
  - `vertex_network` uses `vertex_connectivity`'s matrices → secondary → `vertex_connectivity`.
  - Everything else (`roi_aperiodic`, `roi_transfer_entropy`, `roi_pac`,
    `vertex_mvpa`, `vertex_spatial`, `vertex_specparam`, …) computes from **raw
    data** (timeseries/STC) → **primary**, even when it lives in the same domain
    (e.g. aperiodic decomposes the PSD but can run before it).

| Domain | Primary | Secondary (runs-after) |
|---|---|---|
| Spectral | roi_psd, roi_aperiodic, vertex_cluster, vertex_specparam, vertex_mvpa, vertex_spatial | — |
| Connectivity | roi_connectivity, roi_transfer_entropy, vertex_connectivity | **roi_network → roi_connectivity**, **vertex_network → vertex_connectivity** |
| Cross-frequency | roi_pac | — |
| Sensor-level | electrode_psd, electrode_aperiodic | electrode_comparison (validation) |

Domain is presentation; `supplements` is a real dependency (nest + gate the
secondary under its primary).

## Workstreams

### 1. source-analytics — make the relationship intrinsic
- Add `ANALYSIS_META: {name: {domain, supplements}}` next to `ANALYSIS_REGISTRY`
  in `core.py` (single source of truth, co-located with where analyses are
  defined). Optionally expose `analysis_meta()` so other tools can read it.
- Network layer runs on **all 5 connectivity metrics** (not just imag-coherence):
  config-driven via `roi_network.connectivity_metrics` (already supported; the
  analysis loops it for both graph metrics and NBS). Same for vertex.

### 2. source-analytics — re-run network on all metrics
- `study_treatment.yaml` sets `roi_network.connectivity_metrics: [imag_coherence,
  dwpli, pli, aec, coherence]`.
- Re-run: `source-analytics run --study study_treatment.yaml --paradigm resting
  --analysis roi_network` (slow — NBS permutations × 5 metrics). Vertex similarly
  if desired.

### 3. source-lightbox — group the gallery by domain
- Read `ANALYSIS_META` (via the source-analytics venv, same subprocess pattern as
  brain mosaics / circos) and attach `domain` + `supplements` to each analysis in
  the manifest.
- Nav + pages group analyses into **one page per domain**; **secondary analyses
  nest under their primary** with sub-tabs (Connectivity page → Edges (circos) /
  Network (NBS + graph) / Directed (TE)). Combined Summary, Figures sub-tabs,
  Tables.

## Sequence / status

1. [done] all-5-metrics config + roi_network re-run. Output verified:
   `roi_nbs_results.csv` 658 rows over all 5 metrics (aec, coherence, dwpli,
   imag_coherence, pli), 40 FDR-significant components; `roi_network_stats.csv`
   graph metrics likewise span all 5. (Config fix: `connectivity_metrics` must
   live under `paradigms.resting.analyses.roi_network`, since the CLI reads the
   per-analysis block via `for_paradigm_analysis`, not the top-level section.)
2. [done] `domain` + `supplements` added to `ANALYSIS_METADATA` in `core.py`;
   `analysis_meta()` accessor exposes it (canonical names only).
3. [done] lightbox reads `analysis_meta()` from the SA venv (`builder._read_analysis_meta`)
   → `manifest` attaches `meta {domain, supplements, description}` per analysis.
4. [done] lightbox nav + merged domain pages: nav lists domains (not 12 flat
   analyses); `renderDomain` shows one page per (paradigm × domain) with an
   analysis pill bar, each secondary flagged "supplemental" and nested right
   after its primary; pill content (Summary/Figures/Tables) fills lazily.
5. [done] rebuild gallery; verify Connectivity = connectivity + network (+ TE)
   together, Spectral = psd + aperiodic, etc.

### vertex follow-up (deferred)
`vertex_network` reads a single `metric` (default imag_coherence) and computes
connectivity from the STC itself — it does NOT loop `connectivity_metrics` the
way `roi_network` does. All-5-metrics at vertex level therefore needs a loop
refactor of `vertex_network_analysis.py` plus an expensive vertex-resolution NBS
run (5 metrics × 5000 perms). Not in the ROI-driven MS2 figures, so deferred.

## Verification

- `source-analytics` registry dump lists each analysis with its domain/supplements.
- Gallery Analytics nav shows domains (Spectral / Connectivity / …), not 12 flat
  analyses; the Connectivity page shows circos + NBS + graph together; the network
  figures cover all 5 metrics.

## Merge / cleanup

When the manuscript is accepted, merge `analysis-grouping` → `main` in both repos
and delete the branch.
