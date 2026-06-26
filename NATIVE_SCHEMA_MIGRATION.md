# Native-schema migration plan (drop the legacy hypothesis-CSV aliases)

**Status:** SCOPING ONLY — no code changes yet. This documents every consumer of
the legacy alias columns, the safe migration order, and the test checkpoints, so
the cross-repo refactor can be executed (or sequenced) deliberately.

**Goal:** stop emitting the legacy alias columns on `<module>_hypotheses.csv` and
have all figure/gallery consumers read the **native** hypothesis schema directly.

## 1. The aliases and their native sources

Added today by `R/hypothesis.R::.add_legacy_aliases()` (R producers) and
`src/source_analytics/hypothesis/tabular.py` (Python producers). Mapping:

| Legacy alias | Native column | Notes |
|---|---|---|
| `contrast`   | `hypothesis`        | the declared hypothesis name |
| `roi`        | `spatial`           | the spatial unit (ROI/edge/region/node) |
| `power_type` | `dv`                | dependent variable (only some modules emit `dv`) |
| `t_ratio`    | `stat` (where `stat_type=="t"`) | NA for F/z/tost rows |
| `hedges_g`   | `effect_size` (where `effect_size_type=="hedges_g"`) | NA for ω²/β rows |
| `p_fdr`      | `q_value`           | within-family FDR-adjusted p |

Native schema also carries: `kind`, `role`, `label`, `test`, `band`, `SE`, `df`,
`df_num`, `stat_type`, `p_value`, `estimate`, `estimate_lcl/ucl`,
`effect_size_type`, `group_a`, `group_b`, `significant`, `fdr_family`.

## 2. Consumer inventory

### 2a. source-lightbox (separate repo — the primary blast radius)

Lightbox is **column-sniffing**: `render.py::select_renderer()` dispatches by which
columns a table has (`_has(headers, ...)`). Renderers keyed on aliases:

| File / site | Reads | Native replacement |
|---|---|---|
| `render.py:289` ROI posthoc forest `matches` | `contrast, roi, band, hedges_g` | `hypothesis, spatial, band, effect_size` |
| `render.py:300` (value extraction) | `roi`, `hedges_g` | `spatial`, `effect_size` |
| `render.py:327` graph-nodal `matches` | `contrast, roi, band, graph_metric, t` | `hypothesis, spatial, band, graph_metric, stat` |
| `render.py:350` (value) | `roi`, `t` | `spatial`, `stat` |
| `render.py:332` | `p_fdr` | `q_value` |
| `render.py:502/513` effect heatmap | `band, max_abs_hedges_g` | derived col — keep or recompute from `effect_size` |
| `render.py:533/543` | `hedges_g` | `effect_size` |
| `_brain_render_worker.py:44/47/62/66/72/79/83/89/90` | `power_type, contrast, roi, hedges_g` | `dv, hypothesis, spatial, effect_size` |
| `brain_mosaic.py:50/67` | `power_type` | `dv` |
| `summarize.py:52/63/109` | `roi/vertex_idx, contrast, p_fdr` | `spatial, hypothesis, q_value` |
| `config.py` | `brain_power_type` config key | (config knob — keep, maps to `dv`) |
| `_circos_render_worker.py` | (verify cols) | TBD — audit before editing |
| `builder.py:95` | `power_type` passthrough | `dv` |

Renderers ALREADY native (no change): NBS subnetwork
(`key, component, n_edges, p_corrected` @ 413), cluster (`band, cluster_stat,
p_corrected` @ 483).

### 2b. source-analytics R figure scripts

`grep` shows ~127 alias references across 14 R files, **but most are NOT
hypotheses.csv alias readers** — they are:
- **legacy diagnostic functions** in `stats_utils.R` and the `run_omnibus_lmm*` /
  `run_posthoc_*` helpers, where `power_type`/`contrasts` are **function parameters**
  operating on the legacy long band tables (`relative`/`absolute` columns), not the
  `_hypotheses.csv`. These are kept as diagnostics and are independent of the alias
  drop. (Confirm per file before editing.)
- **figure scripts** (`plot_psd.R`, `report.R`, per-module `*_analysis.R`) that read
  the **rebuilt legacy tables** (`*_posthoc_*.csv`, `*_stats.csv`) — which carry their
  own native-legacy schema — OR read `_hypotheses.csv` via the aliases.

Action: per R file, classify each alias use as (i) legacy-fn parameter [leave],
(ii) reads rebuilt legacy table [leave until those tables are retired], or
(iii) reads `_hypotheses.csv` alias [migrate to native]. Only (iii) blocks the drop.

### 2c. source-analytics Python

`hypothesis/tabular.py` is a **producer** of the aliases (not a consumer). No Python
figure code consumes them (vertex/graph figures are matplotlib off native arrays).

## 3. Safe migration order (consumers before producers)

1. **Audit & classify** (no edits): per R file, label each alias use (i/ii/iii
   above); audit `_circos_render_worker.py` columns. Produces the exact edit list.
2. **Migrate source-lightbox renderers** to read native columns, accepting BOTH
   names during transition (e.g. `rec.get("effect_size", rec.get("hedges_g"))`,
   `_has(headers, "spatial"|"roi", ...)`). Commit + run the lightbox test suite +
   a gallery build. Lightbox keeps working whether or not aliases are present.
3. **Migrate SA R type-(iii) figure readers** to native columns (same dual-read
   tolerance). Commit + per-module figure regen.
4. **Drop alias generation**: remove `.add_legacy_aliases()` calls (R) and the alias
   block in `tabular.py` (Python). Commit.
5. **Remove the dual-read fallbacks** in lightbox + R once aliases are gone. Commit.
6. **Optional**: retire the rebuilt legacy tables (`*_posthoc_*.csv` etc.) if nothing
   else reads them — separate follow-up.

Rationale: steps 2–3 make consumers tolerant FIRST; step 4 (the breaking change)
then changes nothing observable; step 5 cleans up. At every step the gallery renders.

## 4. Test checkpoints

- **R/Python hypothesis suites**: `Rscript tests/test_hypothesis.R`,
  `pytest tests/test_tabular.py tests/test_edge.py tests/test_directed.py`.
- **Lightbox**: `pytest` in source-lightbox (`test_render.py`, `test_summarize.py`,
  `test_manifest.py`) + a full `scripts/build_gallery.sh` on the FORGE
  `results_treatment/` tree; diff the rendered figure set before/after (count + names).
- **SA per-module figure regen**: re-run `--steps figures` for roi_psd, roi_aperiodic,
  roi_cross_freq, roi_graph, electrode_psd on FORGE; confirm no missing/blank figures.
- **Golden check**: pick one module (roi_psd) and byte-compare its rendered figures
  before vs after the full migration (should be identical — same data, native cols).

## 5. Risk & rollback

- **Risk**: a lightbox renderer silently stops matching (figure disappears) if a
  `_has(...)` signature is migrated wrong. Mitigation: dual-read in step 2; the
  before/after figure-set diff in §4 catches drops.
- **Risk**: an unaudited R type-(iii) reader is missed → blank figure after step 4.
  Mitigation: the per-module figure regen in §4.
- **Rollback**: the alias generation is two small blocks (R `.add_legacy_aliases`,
  Python `tabular.py`); reverting step 4 restores aliases instantly. Steps 2–3/5 are
  independent commits, individually revertable.

## 6. Effort estimate

- Audit (§1): ~1 session, no risk.
- Lightbox migration (§3.2): the bulk — 7 files, ~1 session, medium risk (cross-repo).
- SA R migration (§3.3): small once classified (likely a handful of type-(iii) sites).
- Drop + cleanup (§3.4–5): small.

Two repos, ~4–5 commits. Sequence lightbox first; it is the gate.
