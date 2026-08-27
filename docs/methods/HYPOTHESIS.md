# The `hypothesis` layer — usage & reference

Declarative hypotheses, tested one at a time, by hand. This is the practical guide;
for the *why* and the design decisions see **`DESIGN_SPEC.md`**.

---

## 1. What it is

`hypothesis` is a **shared inference layer** — a third peer to `R/stats_utils.R` and
`src/source_analytics/stats/`. It is **not** a registry analysis module: it processes no
subjects, makes no figures, and never appears in `core.py`'s `ANALYSIS_METADATA`. Analysis
modules *call into* it from their `statistics()` step.

You **declare** your hypotheses once (in the study YAML), then **run them one at a time, by
name**, and read the result. Nothing auto-fires — there is no gating, no chained verdicts, no
"run everything and adjudicate." The scientific judgment stays with you.

| piece | where | status |
|---|---|---|
| spec loader + emmeans adapter + runner + module helper | `R/hypothesis.R` | ✅ built |
| `Hypothesis` / `DesignSpec` dataclasses, `StudyConfig.design_spec` | `src/source_analytics/config.py` | ✅ built |
| permutation adapter (vertex / connectivity maps) | `src/source_analytics/hypothesis/permutation.py` | ✅ built (vertex_cluster wired) |
| edge / NBS adapter (connectivity matrices → subnetworks) | `src/source_analytics/hypothesis/edge.py` | ✅ built (roi_nbs / vertex_nbs wired) |
| kept primitives `tost_equivalent()` / `.equivalence_margin()` | `R/stats_utils.R` | ✅ reused |

## 2. Declaring hypotheses (the spec)

Two YAML blocks in the study config:

```yaml
design:
  factor: group              # the categorical factor hypotheses are taken over
  reference: WT_VEH          # reference level (effect orientation)
  levels: [WT_VEH, KO_VEH, KO_HD_ICV, KO_HD_IV]   # optional explicit order
  covariates: [n_epochs]     # optional nuisance covariates (mean-centered)

hypotheses:
  - name: group_omnibus
    kind: omnibus
    groups: [WT_VEH, KO_VEH, KO_HD_ICV, KO_HD_IV]   # default: all design levels
    role: phenotype

  - name: disease_effect
    kind: contrast
    weights: { KO_VEH: 1, WT_VEH: -1 }              # the portable contrast payload
    role: phenotype

  - name: dose_response
    kind: regression
    predictor: dose          # continuous column; slope tested
    role: exploratory

  - name: hd_icv_normalization
    kind: equivalence
    weights: { KO_HD_ICV: 1, WT_VEH: -1 }
    margin: { mode: sd, value: 0.25 }
    role: normalization
```

**Backward compatible.** A study still using the legacy `contrasts:` block (with
`group_a`/`group_b`) works unchanged — each contrast is lifted to a `kind: contrast` whose
weights are `{group_a: +1, group_b: -1}`. You do not have to migrate to start using the layer.

### Field reference

| field | applies to | meaning |
|---|---|---|
| `name` | all | unique id; the handle you pass to `--hypothesis` |
| `kind` | all | `omnibus` \| `contrast` \| `regression` \| `equivalence` (default `contrast`) |
| `weights` | contrast, equivalence | `{level: weight}`; a linear combination of group means |
| `groups` | omnibus | the levels the F-test spans (default: all `design.levels`) |
| `predictor` | regression | continuous column whose slope is tested |
| `by` | regression | optional factor for per-group slopes |
| `margin` | equivalence | `{mode: sd|gap_fraction, value: ...}` (see §6) |
| `label`, `role` | all | display only; `role` groups output, never controls flow |

## 3. The four kinds

| kind | question | payload | effect size |
|---|---|---|---|
| **omnibus** | "do these groups differ at all?" | `groups` | partial ω² |
| **contrast** | a specific linear comparison (post-hoc) | `weights` | Hedges g |
| **regression** | slope of a continuous predictor | `predictor` (+`by`) | standardized β |
| **equivalence** | is a contrast within a margin? (TOST) | `weights` + `margin` | — |

The omnibus → contrast flow is the standard ANOVA → post-hoc workflow, **driven by you**: run
the omnibus, read it, then run whichever post-hoc contrasts *you* judge to follow. No gate
decides that for you.

## 4. Running them

`--hypothesis NAME[,NAME]` runs one (or a few) by name; with no flag, the module runs **all**
declared hypotheses. It composes with the existing `--metric` / `--band` / `--select` axes.

```bash
# one hypothesis
uv run --no-sync source-analytics run --study study_treatment.yaml \
    --paradigm resting --analysis roi_psd --hypothesis disease_effect

# a band-scoped single cell
... --analysis roi_psd --hypothesis disease_effect --band "Low Gamma"
```

Output is **additive**: a new `<module>_hypotheses.csv` under the module's tables dir, written
*alongside* the legacy omnibus/posthoc tables (which are untouched). Schema:

```
hypothesis, kind, role, band, spatial, estimate, SE, df, df_num, stat, stat_type,
p_value, estimate_lcl, estimate_ucl, effect_size, effect_size_type, label, test,
q_value, significant, fdr_family, dv
```

One row per **band × spatial cell × DV**. `q_value`/`significant` are **within-run** BH-FDR,
corrected across that hypothesis's band × spatial family (named in `fdr_family`). Cross-
hypothesis correction is your call, made explicitly — never silent.

## 5. Adapters & result contracts

The same hypothesis declaration runs under whichever **adapter** matches where a module's
statistic is computed — the divide is the inference machinery, **not** the spatial level.

| adapter | modules | result contract |
|---|---|---|
| **emmeans** (R LMM) ✅ | roi_psd, roi_aperiodic, electrode_psd, electrode_aperiodic, … | **tabular** — per-cell estimate/CI/p/FDR/effect (the schema above) |
| **permutation** (Python) ✅ | all per-unit-map vertex_*, electrode_connectivity, vertex_connectivity/directed/specparam | **map + clusters** — per-unit statistic map + cluster extent/mass/cluster-p (max-stat/TFCE corrected, *not* per-cell FDR) |
| **edge / NBS** (Python) ✅ | roi_nbs, vertex_nbs (+ the combined roi_network/vertex_network aliases) | **subnetwork table** — per-edge statistic matrix → connected supra-threshold components with edge-count, mass, peak, component-p (NBS max-component corrected) |

`electrode_connectivity` runs on the **permutation** adapter (it is a per-channel map),
alongside `vertex_connectivity` — not with `electrode_psd`. The **edge/NBS** adapter is the
third contract: it clusters supra-threshold *edges* into subnetworks rather than supra-threshold
*units* into spatial clusters, so it consumes connectivity *matrices* (the NBS family). A
pairwise contrast routes through the legacy `nbs_permutation_test` verbatim (bit-exact);
omnibus uses a per-edge F-matrix. Method knobs (cluster/NBS threshold, TFCE, adjacency,
`n_permutations`) live in **module config**, never in the hypothesis.

## 6. Equivalence (TOST)

`kind: equivalence` runs a TOST verdict for the chosen contrast — it adjudicates nothing else.
Margin modes:

- `sd`: `value × residual SD` — self-contained; recommended.
- `gap_fraction`: `value × |reference estimate|` — needs a reference effect; declare it
  explicitly (`margin: { mode: gap_fraction, value: 0.25, ref: disease_effect }`), never a
  hidden gate. (The reference wiring is added when a study uses it.)

## 7. API reference

**R (`R/hypothesis.R`)**

```r
spec <- parse_design_spec(config)        # parse design:/hypotheses: (config = read_yaml'd)

run_hypothesis(data, hyp, spec,          # run ONE hypothesis -> tidy data.frame
               dv_col = "dv", spatial_col = "roi",
               band_col = "band", bands = NULL,
               fit_scope = "shared",     # or "per_contrast" (reproduces legacy per-pair SEs)
               fdr_method = "BH")        # hyp = a parsed def or a name in spec$hypotheses

write_module_hypotheses(df, config, tbl_dir, prefix, dv_cols,   # the module one-liner
                        spatial_col = "roi", band_col = "band",
                        hypothesis = NULL, fit_scope = "shared")
```

`fit_scope = "shared"` (default) fits all design groups in one model — required for a ≥3-group
omnibus, and gives post-hoc contrasts a common error term. `"per_contrast"` subsets to the
contrast's groups and reproduces the legacy per-pair estimates bit-exact.

**Python (`config.py`)**

```python
cfg = StudyConfig.from_yaml("study.yaml")
cfg.design_spec                 # DesignSpec | None  (parsed design:/hypotheses:, or lifted contrasts:)
cfg.design_spec.hypotheses      # list[Hypothesis]   (.name/.kind/.weights/.groups/.predictor/.margin)
cfg.referenced_groups()         # set[str] — every group named by a contrast/hypothesis (subject discovery)
```

## 8. Wiring a new (emmeans-tabular) module

Four steps — see `roi_psd_analysis.R` / `electrode_aperiodic_analysis.R` for live examples:

1. **R:** `source(file.path(script_dir, "hypothesis.R"))` near the other sources.
2. **R:** add `parser$add_argument("--hypothesis", default = NULL)`.
3. **R:** after the module's stat-table exports, call `write_module_hypotheses(df, config,
   tbl_dir, prefix = "<module>", dv_cols = c(...), spatial_col = "...", band_col = "band"|NULL,
   hypothesis = args$hypothesis)`.
4. **Python:** add `"hypothesis": "declared hypothesis"` to the module's `SELECTABLE`, and pass
   `--hypothesis` through to the Rscript command from `self._selection.get("hypothesis")`.

This fits modules with a modest spatial cardinality (≤ ~32 ROIs / ~30 channels). Per-unit
**maps** (vertex/electrode connectivity, directed, specparam) belong to the permutation adapter,
and connectivity-**matrix** subnetworks (roi_nbs / vertex_nbs) belong to the edge/NBS adapter — a
`group × edge` LMM is infeasible and wrong for either (~496 edges → ~2500 params).

**Long-DV modules take the Python tabular adapter instead.** When a module exports one `value`
column plus a label column saying what it holds (the evoked pair: `value` + `measure_name`),
there is no `dv_cols` vector to pass. Hand the label column to `write_module_hypotheses_tabular`
as a **facet** — `facet_cols=("measure_name",)` — which runs an independent FDR family per facet
across the band × spatial grid. Facet ≡ DV: R gives each `dv_col` its own `run_hypothesis()` call
and so its own family, and a facet is that same family boundary expressed in a long table.
`analyses/_evoked_hypotheses.py` is the live example, shared by both evoked modules.

## 9. Status

- ✅ **emmeans adapter** built + verified (bit-exact vs legacy on real data); wired into
  `roi_psd`, `roi_aperiodic`, `electrode_psd`, `electrode_aperiodic` — every applicable
  emmeans-tabular module.
- ✅ **permutation adapter** (`hypothesis/permutation.py`): contrast (pairwise = legacy-exact;
  general weighted), omnibus-F, equivalence (TOST summary), Freedman–Lane covariates, map/cluster
  contract. **Wired into `vertex_cluster`, `vertex_connectivity` (FCD), `vertex_directed`
  (outflow/inflow/netflow), `vertex_specparam` (exponent/offset), and `electrode_connectivity`
  (per-channel FCD — the source-vs-sensor sensor side, montage adjacency auto-scaled).** All
  verified on real FORGE data via `write_module_hypotheses_perm()`.
- ✅ **edge / NBS adapter** (`hypothesis/edge.py`): contrast (pairwise = legacy-exact via
  `nbs_permutation_test`; general weighted), omnibus per-edge F, equivalence (per-edge TOST
  summary), Freedman–Lane covariates, subnetwork (component-table) contract. **Wired into `roi_nbs`,
  `vertex_nbs`, and the combined `roi_network`/`vertex_network` aliases** via
  `write_module_hypotheses_edge()` (additive `<module>_hypotheses.csv`). Verified bit-exact vs the
  legacy NBS on real FORGE data (Low Gamma / imag_coherence / KO_VEH vs WT_VEH).
- ✅ **evoked (tabular adapter, measure as facet):** wired into `roi_evoked` (spatial `roi`) and
  `electrode_evoked` (spatial `channel`) via the shared `analyses/_evoked_hypotheses.py`. These
  two export a long DV — one `value` column faceted by `measure_name`, not one column per
  measure — so the measure is passed as a **facet**, giving an independent FDR family per measure
  across the spatial grid. That is the only defensible family: the measures are on incomparable
  scales (ITC 0-1, ERSP dB, ERP amplitude in signal units, latency in seconds). There is **no band
  axis** — each measure definition already fixes its own band and time window — so the band
  coordinate is null. Additive: the descriptive `group * roi` LMM in the R scripts is unchanged
  and still runs. Verified on planted synthetic signal in `tests/test_evoked_hypotheses.py`.
- ⏳ **specials:** `vertex_mvpa` (decoding), `vertex_spatial` (GLS), `electrode_comparison`
  (agreement — may not take hypotheses).
- ⏳ **migration / retirement:** move `study_treatment.yaml` to `design:`/`hypotheses:` and
  delete the retired gating code (`apply_hypothesis_gating`, `build_rescue_verdicts`,
  `gate_on`).

See `DESIGN_SPEC.md` for the full architecture and the retired auto-gating rationale.
